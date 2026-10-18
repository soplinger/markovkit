#!/usr/bin/env python3
"""markovkit: exact discrete-time Markov chain toolkit driven by plain-text chain files.

Chain file format (one statement per line, '#' starts a comment):

    start Sunny 1             initial distribution (optional, may repeat)
    Sunny -> Rainy 1/10       transition probability (fraction or decimal)

Probabilities are parsed as exact fractions, so lint, info, step, power,
stationary, absorb and hit results are exact. mix and simulate use floats.

Exit codes: 0 ok, 1 chain file has errors or the analysis does not apply,
2 usage error, 3 a gate check failed.
"""

import argparse
import json
import math
import random
import sys
from bisect import bisect_right
from collections import deque
from fractions import Fraction
from itertools import accumulate
from pathlib import Path

EXIT_OK = 0
EXIT_CHAIN_ERRORS = 1
EXIT_USAGE = 2
EXIT_GATE_FAILED = 3


class ChainError(Exception):
    """The chain file has errors, or the chain cannot answer the question."""


class UsageError(Exception):
    """A command-line argument is wrong."""


class SingularSystemError(Exception):
    """A linear system has no unique solution."""


class Chain:
    def __init__(self, states, matrix, start):
        self.states = states
        self.index = {name: i for i, name in enumerate(states)}
        self.matrix = matrix
        self.start = start
        self.has_start = any(p != 0 for p in start)


# ---------------------------------------------------------------- parsing

def parse_probability(token):
    try:
        value = Fraction(token)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"'{token}' is not a probability")
    if value < 0 or value > 1:
        raise ValueError(f"probability {token} is outside 0..1")
    return value


def parse_chain(text):
    """Return (chain or None, errors, warnings). Errors and warnings are (line, message) pairs."""
    errors, warnings = [], []
    order, first_line, rows, row_line, start = [], {}, {}, {}, {}
    start_line = None

    def note(name, lineno):
        if name not in first_line:
            first_line[name] = lineno
            order.append(name)

    for lineno, raw in enumerate(text.splitlines(), start=1):
        parts = raw.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] == "start" and len(parts) == 3:
            state, token = parts[1], parts[2]
            try:
                p = parse_probability(token)
            except ValueError as exc:
                errors.append((lineno, str(exc)))
                continue
            if state in start:
                errors.append((lineno, f"duplicate start state '{state}'"))
                continue
            start[state] = p
            note(state, lineno)
            if start_line is None:
                start_line = lineno
        elif len(parts) == 4 and parts[1] == "->":
            src, dst, token = parts[0], parts[2], parts[3]
            try:
                p = parse_probability(token)
            except ValueError as exc:
                errors.append((lineno, str(exc)))
                continue
            row = rows.setdefault(src, {})
            if dst in row:
                errors.append((lineno, f"duplicate transition {src} -> {dst}"))
                continue
            row[dst] = p
            row_line.setdefault(src, lineno)
            note(src, lineno)
            note(dst, lineno)
            if p == 0:
                warnings.append((lineno, f"zero-probability transition {src} -> {dst} has no effect"))
        else:
            errors.append((lineno, "expected 'start <state> <probability>' or '<from> -> <to> <probability>'"))

    if not order:
        errors.append((0, "chain has no states"))

    for name in order:
        if name not in rows:
            errors.append((first_line[name], f"state '{name}' has no outgoing transitions "
                                             f"(add '{name} -> {name} 1' for an absorbing state)"))
    for src, row in rows.items():
        total = sum(row.values(), Fraction(0))
        if total != 1:
            errors.append((row_line[src], f"outgoing probabilities from '{src}' sum to {total}, not 1"))
    start_total = sum(start.values(), Fraction(0))
    if start and start_total != 1:
        errors.append((start_line, f"initial probabilities sum to {start_total}, not 1"))
    if errors:
        return None, errors, warnings

    n = len(order)
    index = {name: i for i, name in enumerate(order)}
    matrix = [[Fraction(0)] * n for _ in range(n)]
    for src, row in rows.items():
        for dst, p in row.items():
            matrix[index[src]][index[dst]] = p
    chain = Chain(order, matrix, [start.get(name, Fraction(0)) for name in order])

    if chain.has_start:
        support = [i for i, p in enumerate(chain.start) if p]
        reached = reachable_from(chain, support)
        for i, name in enumerate(order):
            if i not in reached:
                warnings.append((first_line[name], f"state '{name}' is unreachable from the start distribution"))
    return chain, errors, warnings


# ---------------------------------------------------------------- graph structure

def successors(chain, i):
    return [j for j, p in enumerate(chain.matrix[i]) if p]


def reachable_from(chain, sources):
    seen = set(sources)
    queue = deque(sources)
    while queue:
        i = queue.popleft()
        for j in successors(chain, i):
            if j not in seen:
                seen.add(j)
                queue.append(j)
    return seen


def reach_sets(chain):
    return [reachable_from(chain, [i]) for i in range(len(chain.states))]


def is_closed(chain, members):
    member_set = set(members)
    return all(j in member_set for i in members for j in successors(chain, i))


def class_period(chain, members):
    root = members[0]
    member_set = set(members)
    level = {root: 0}
    queue = deque([root])
    while queue:
        i = queue.popleft()
        for j in successors(chain, i):
            if j in member_set and j not in level:
                level[j] = level[i] + 1
                queue.append(j)
    period = 0
    for i in members:
        for j in successors(chain, i):
            if j in member_set:
                period = math.gcd(period, abs(level[i] + 1 - level[j]))
    return period


def analyse(chain):
    """Communicating classes with recurrence, period and absorbing flags."""
    reach = reach_sets(chain)
    classes, assigned = [], set()
    for i in range(len(chain.states)):
        if i in assigned:
            continue
        members = sorted(j for j in reach[i] if i in reach[j])
        assigned.update(members)
        closed = is_closed(chain, members)
        classes.append({
            "members": members,
            "recurrent": closed,
            "period": class_period(chain, members) if closed else None,
            "absorbing": closed and len(members) == 1 and chain.matrix[members[0]][members[0]] == 1,
        })
    return classes


def names_of(chain, members):
    return [chain.states[i] for i in members]


def class_label(chain, members):
    return "{" + ", ".join(names_of(chain, members)) + "}"


def transient_states(classes):
    return sorted(i for c in classes if not c["recurrent"] for i in c["members"])


# ---------------------------------------------------------------- exact linear algebra

def solve(A, B):
    """Solve A X = B exactly with Gauss-Jordan elimination. A is n x n, B is n x m."""
    n = len(A)
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if M[r][col] != 0), None)
        if pivot is None:
            raise SingularSystemError("linear system is singular")
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]
    return [row[n:] for row in M]


def stationary_of_class(chain, members):
    """Exact stationary distribution of a closed class, keyed by state index."""
    k = len(members)
    A = [[Fraction(0)] * k for _ in range(k)]
    for r, j in enumerate(members):
        for c, i in enumerate(members):
            A[r][c] = chain.matrix[i][j] - (1 if i == j else 0)
    A[k - 1] = [Fraction(1)] * k
    rhs = [[Fraction(0)] for _ in range(k)]
    rhs[k - 1] = [Fraction(1)]
    solution = solve(A, rhs)
    return {members[c]: solution[c][0] for c in range(k)}


def stationary_map(chain, classes):
    pi = {}
    for c in classes:
        if c["recurrent"]:
            pi.update(stationary_of_class(chain, c["members"]))
    return pi


def absorption(chain, classes):
    """Expected steps to absorption and absorption probabilities per recurrent class."""
    recurrent = [c for c in classes if c["recurrent"]]
    transient = transient_states(classes)
    if not transient:
        return {"transient": [], "recurrent": recurrent, "X": []}
    A = []
    for r, s in enumerate(transient):
        row = [-chain.matrix[s][q] for q in transient]
        row[r] += 1
        A.append(row)
    B = []
    for s in transient:
        row = [Fraction(1)]
        for c in recurrent:
            row.append(sum((chain.matrix[s][j] for j in c["members"]), Fraction(0)))
        B.append(row)
    return {"transient": transient, "recurrent": recurrent, "X": solve(A, B)}


def hitting_times(chain, target):
    """Expected steps from every state to reach target; None where the target may never be hit."""
    n = len(chain.states)
    reach = reach_sets(chain)
    can_reach_target = {i for i in range(n) if target in reach[i]}
    finite = [i for i in range(n) if reach[i] <= can_reach_target]
    unknowns = [i for i in finite if i != target]
    A = []
    for r, s in enumerate(unknowns):
        row = [-chain.matrix[s][q] for q in unknowns]
        row[r] += 1
        A.append(row)
    X = solve(A, [[Fraction(1)] for _ in unknowns]) if unknowns else []
    times = {}
    for i in range(n):
        if i == target:
            times[i] = Fraction(0)
        elif i in unknowns:
            times[i] = X[unknowns.index(i)][0]
        else:
            times[i] = None
    return times


def vec_mat(vec, matrix):
    n = len(vec)
    return [sum((vec[i] * matrix[i][j] for i in range(n) if vec[i]), Fraction(0)) for j in range(n)]


def mat_mul(A, B):
    n = len(A)
    cols = range(len(B[0]))
    return [[sum((A[i][k] * B[k][j] for k in range(n) if A[i][k]), Fraction(0)) for j in cols]
            for i in range(n)]


# ---------------------------------------------------------------- helpers

def require_start(chain, command):
    if not chain.has_start:
        raise ChainError(f"'{command}' needs a 'start <state> <probability>' line in the chain file")


def lookup(chain, name):
    if name not in chain.index:
        raise UsageError(f"unknown state '{name}'")
    return chain.index[name]


def parse_fraction(text, what):
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise UsageError(f"{what} must be a number, got '{text}'")


def table(headers, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()

    return [fmt_row(headers), fmt_row(["-" * w for w in widths])] + [fmt_row(r) for r in rows]


def decimal(x):
    return f"{float(x):.6f}"


def diagnostics(path, items, label):
    out = []
    for line, message in items:
        location = f"{path}:{line}" if line else f"{path}"
        out.append(f"{location}: {label}: {message}")
    return out


# ---------------------------------------------------------------- commands
# Each command returns (data, text_lines, exit_code).

def cmd_info(chain, args):
    classes = analyse(chain)
    count = sum(1 for row in chain.matrix for p in row if p)
    irreducible = len(classes) == 1
    data = {
        "states": chain.states,
        "transitions": count,
        "irreducible": irreducible,
        "classes": [{"states": names_of(chain, c["members"]), "recurrent": c["recurrent"],
                     "period": c["period"], "absorbing": c["absorbing"]} for c in classes],
        "transient": names_of(chain, transient_states(classes)),
    }
    lines = [f"States ({len(chain.states)}): {', '.join(chain.states)}",
             f"Transitions: {count}",
             f"Irreducible: {'yes' if irreducible else 'no'}",
             "Communicating classes:"]
    for c in classes:
        if not c["recurrent"]:
            kind = "transient"
        elif c["absorbing"]:
            kind = "recurrent, absorbing"
        else:
            kind = f"recurrent, period {c['period']}"
        lines.append(f"  {class_label(chain, c['members'])}  {kind}")
    lines.append(f"Transient states: {', '.join(data['transient']) or 'none'}")
    return data, lines, EXIT_OK


def cmd_lint_result(path, chain, errors, warnings):
    lines = diagnostics(path, errors, "error") + diagnostics(path, warnings, "warning")
    if errors:
        lines.append(f"{len(errors)} error(s), {len(warnings)} warning(s)")
        return lines, EXIT_CHAIN_ERRORS
    count = sum(1 for row in chain.matrix for p in row if p)
    lines.append(f"ok: {len(chain.states)} states, {count} transitions, {len(warnings)} warning(s)")
    return lines, EXIT_OK


def cmd_step(chain, args):
    require_start(chain, "step")
    if args.steps < 0:
        raise UsageError("steps must be zero or more")
    dist = list(chain.start)
    for _ in range(args.steps):
        dist = vec_mat(dist, chain.matrix)
    data = {"steps": args.steps, "distribution": {n: str(dist[i]) for i, n in enumerate(chain.states)}}
    rows = [[n, str(dist[i]), decimal(dist[i])] for i, n in enumerate(chain.states)]
    lines = [f"Distribution after {args.steps} step(s):"] + table(["state", "exact", "decimal"], rows)
    return data, lines, EXIT_OK


def cmd_power(chain, args):
    if args.steps < 0:
        raise UsageError("steps must be zero or more")
    n = len(chain.states)
    result = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for _ in range(args.steps):
        result = mat_mul(result, chain.matrix)
    data = {"steps": args.steps, "matrix": {s: {t: str(result[i][j]) for j, t in enumerate(chain.states)}
                                             for i, s in enumerate(chain.states)}}
    rows = [[s] + [str(result[i][j]) for j in range(n)] for i, s in enumerate(chain.states)]
    lines = [f"P^{args.steps} (row = from, column = to):"] + table([""] + chain.states, rows)
    return data, lines, EXIT_OK


def cmd_stationary(chain, args):
    classes = analyse(chain)
    pi = stationary_map(chain, classes)
    data = {"classes": [], "transient": names_of(chain, transient_states(classes))}
    lines = []
    for c in classes:
        if not c["recurrent"]:
            continue
        entry = {"states": names_of(chain, c["members"]), "stationary": {}, "return_time": {}}
        rows = []
        for i in c["members"]:
            name = chain.states[i]
            ret = 1 / pi[i]
            entry["stationary"][name] = str(pi[i])
            entry["return_time"][name] = str(ret)
            rows.append([name, str(pi[i]), decimal(pi[i]), str(ret), f"{float(ret):.4f}"])
        data["classes"].append(entry)
        lines.append(f"Recurrent class {class_label(chain, c['members'])}:")
        lines += table(["state", "stationary", "decimal", "return time", "steps"], rows)
        lines.append("")
    if data["transient"]:
        lines.append("Transient states (zero long-run probability): " + ", ".join(data["transient"]))
    return data, lines, EXIT_OK


def cmd_absorb(chain, args):
    classes = analyse(chain)
    ab = absorption(chain, classes)
    if not ab["transient"]:
        return {"transient": {}}, ["No transient states: every state belongs to a recurrent class."], EXIT_OK
    labels = [class_label(chain, c["members"]) for c in ab["recurrent"]]
    data = {"transient": {}}
    rows = []
    for r, s in enumerate(ab["transient"]):
        name = chain.states[s]
        X = ab["X"]
        data["transient"][name] = {"expected_steps": str(X[r][0]),
                                   "absorption": {label: str(X[r][k + 1]) for k, label in enumerate(labels)}}
        rows.append([name, str(X[r][0]), f"{float(X[r][0]):.4f}"] +
                    [str(X[r][k + 1]) for k in range(len(labels))])
    lines = table(["state", "expected steps", "decimal"] + [f"into {lab}" for lab in labels], rows)
    if chain.has_start:
        X = ab["X"]
        from_start = sum((chain.start[s] * X[r][0] for r, s in enumerate(ab["transient"])), Fraction(0))
        data["from_start_expected_steps"] = str(from_start)
        lines += ["", f"Expected steps to absorption from the start distribution: {from_start} "
                      f"(~{float(from_start):.4f})"]
    return data, lines, EXIT_OK


def cmd_hit(chain, args):
    target = lookup(chain, args.target)
    times = hitting_times(chain, target)
    data = {"target": args.target,
            "hitting_times": {chain.states[i]: (str(t) if t is not None else None) for i, t in times.items()}}
    rows = []
    for i, name in enumerate(chain.states):
        t = times[i]
        rows.append([name, str(t) if t is not None else "infinite", f"{float(t):.4f}" if t is not None else "-"])
    lines = [f"Expected steps to reach '{args.target}':"] + table(["from", "exact", "decimal"], rows)
    return data, lines, EXIT_OK


def cmd_mix(chain, args):
    classes = analyse(chain)
    if len(classes) != 1:
        raise ChainError("mix needs an irreducible chain (a single communicating class)")
    if not 0 < args.epsilon < 1:
        raise UsageError("--epsilon must be between 0 and 1")
    if args.max_steps < 0:
        raise UsageError("--max-steps must be zero or more")
    n = len(chain.states)
    pi = [float(x) for x in stationary_of_class(chain, list(range(n))).values()]
    P = [[float(p) for p in row] for row in chain.matrix]
    dists = [[1.0 if j == s else 0.0 for j in range(n)] for s in range(n)]
    history = []
    mixing_step = None
    for k in range(args.max_steps + 1):
        worst = max(0.5 * sum(abs(d[j] - pi[j]) for j in range(n)) for d in dists)
        history.append(worst)
        if worst <= args.epsilon:
            mixing_step = k
            break
        dists = [[sum(d[i] * P[i][j] for i in range(n)) for j in range(n)] for d in dists]
    period = classes[0]["period"]
    data = {"epsilon": args.epsilon, "period": period, "mixing_step": mixing_step,
            "worst_tv_by_step": [round(v, 6) for v in history]}
    if mixing_step is not None:
        lines = [f"Mixing time (worst start, epsilon {args.epsilon}): {mixing_step} step(s)"]
    elif period > 1:
        lines = [f"Chain is periodic (period {period}): total variation never falls below {args.epsilon}."]
    else:
        lines = [f"No mixing within {args.max_steps} steps; raise --max-steps."]
    shown = sorted({k for k in range(min(len(history), 10))} | ({mixing_step} if mixing_step is not None else set()))
    rows = [[k, f"{history[k]:.6f}"] for k in shown]
    lines += table(["step", "worst TV distance"], rows)
    return data, lines, EXIT_OK


def cmd_simulate(chain, args):
    require_start(chain, "simulate")
    if args.steps < 0:
        raise UsageError("--steps must be zero or more")
    n = len(chain.states)
    rng = random.Random(args.seed)
    start_states = [i for i in range(n) if chain.start[i]]
    start_cum = list(accumulate(float(chain.start[i]) for i in start_states))
    succ = [successors(chain, i) for i in range(n)]
    cum_rows = [list(accumulate(float(chain.matrix[i][j]) for j in succ[i])) for i in range(n)]

    def pick(options, cum):
        k = bisect_right(cum, rng.random() * cum[-1])
        return options[min(k, len(options) - 1)]

    state = pick(start_states, start_cum)
    counts = [0] * n
    counts[state] += 1
    for _ in range(args.steps):
        state = pick(succ[state], cum_rows[state])
        counts[state] += 1
    total = args.steps + 1
    freq = [c / total for c in counts]
    data = {"seed": args.seed, "steps": args.steps,
            "frequencies": {chain.states[i]: round(freq[i], 6) for i in range(n)}}
    rows = [[name, f"{freq[i]:.6f}"] for i, name in enumerate(chain.states)]
    lines = [f"Simulated {args.steps} step(s), seed {args.seed}:"]
    classes = analyse(chain)
    if len(classes) == 1:
        pi = stationary_of_class(chain, list(range(n)))
        deviation = max(abs(freq[i] - float(pi[i])) for i in range(n))
        data["max_deviation_from_stationary"] = round(deviation, 6)
        rows = [[name, f"{freq[i]:.6f}", decimal(pi[i]), f"{abs(freq[i] - float(pi[i])):.6f}"]
                for i, name in enumerate(chain.states)]
        lines += table(["state", "simulated", "exact", "abs diff"], rows)
        lines.append(f"Largest deviation from exact stationary: {deviation:.6f}")
    else:
        lines += table(["state", "simulated"], rows)
        lines.append("Chain is reducible: no single stationary distribution to compare against.")
    return data, lines, EXIT_OK


def cmd_dot(chain, args):
    classes = analyse(chain)
    absorbing = {i for c in classes if c["absorbing"] for i in c["members"]}

    def q(name):
        return '"' + name.replace('"', '\\"') + '"'

    lines = ["digraph markov {", "  rankdir=LR;"]
    for i, name in enumerate(chain.states):
        shape = "doublecircle" if i in absorbing else "circle"
        lines.append(f"  {q(name)} [shape={shape}];")
    for i, name in enumerate(chain.states):
        for j in successors(chain, i):
            lines.append(f"  {q(name)} -> {q(chain.states[j])} [label=\"{chain.matrix[i][j]}\"];")
    lines.append("}")
    return {"dot": "\n".join(lines)}, lines, EXIT_OK


def cmd_gate(chain, args):
    wanted = args.require_irreducible or args.require_aperiodic or args.min_stationary or \
        args.max_absorb_steps is not None
    if not wanted:
        raise UsageError("gate needs at least one check")
    classes = analyse(chain)
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    if args.require_irreducible:
        check("irreducible", len(classes) == 1, f"{len(classes)} communicating class(es)")
    if args.require_aperiodic:
        periods = [c["period"] for c in classes if c["recurrent"]]
        check("aperiodic", all(p == 1 for p in periods),
              "recurrent periods: " + ", ".join(str(p) for p in periods))
    if args.min_stationary:
        pi = stationary_map(chain, classes)
        for spec in args.min_stationary:
            name, sep, value = spec.rpartition("=")
            if not sep or not name:
                raise UsageError(f"--min-stationary expects STATE=PROB, got '{spec}'")
            i = lookup(chain, name)
            threshold = parse_fraction(value, "--min-stationary probability")
            if i not in pi:
                check(f"stationary {name} >= {value}", False, "state is transient")
            else:
                check(f"stationary {name} >= {value}", pi[i] >= threshold, f"stationary {pi[i]}")
    if args.max_absorb_steps is not None:
        require_start(chain, "gate")
        threshold = parse_fraction(args.max_absorb_steps, "--max-absorb-steps")
        ab = absorption(chain, classes)
        expected = sum((chain.start[s] * ab["X"][r][0] for r, s in enumerate(ab["transient"])), Fraction(0))
        check(f"expected steps to absorption <= {args.max_absorb_steps}", expected <= threshold,
              f"expected {expected} (~{float(expected):.4f})")

    passed = all(c["passed"] for c in checks)
    lines = [("PASS " if c["passed"] else "FAIL ") + f"{c['check']}: {c['detail']}" for c in checks]
    lines.append("gate passed" if passed else "gate FAILED")
    return {"passed": passed, "checks": checks}, lines, EXIT_OK if passed else EXIT_GATE_FAILED


COMMANDS = {
    "info": cmd_info,
    "step": cmd_step,
    "power": cmd_power,
    "stationary": cmd_stationary,
    "absorb": cmd_absorb,
    "hit": cmd_hit,
    "mix": cmd_mix,
    "simulate": cmd_simulate,
    "dot": cmd_dot,
    "gate": cmd_gate,
}


# ---------------------------------------------------------------- CLI

def build_parser():
    parser = argparse.ArgumentParser(prog="markovkit",
                                     description="Exact discrete-time Markov chain toolkit for plain-text chain files.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name, help_text, json_flag=True):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("file", help="plain-text chain file")
        if json_flag:
            p.add_argument("--json", action="store_true", help="print JSON instead of text")
        return p

    add("lint", "check the chain file and report line-numbered errors and warnings", json_flag=False)
    add("info", "states, communicating classes, periodicity and absorbing states")
    p = add("step", "distribution after N steps from the start line")
    p.add_argument("steps", type=int)
    p = add("power", "exact N-step transition matrix P^N")
    p.add_argument("steps", type=int)
    add("stationary", "exact stationary distribution and return times per recurrent class")
    add("absorb", "expected steps to absorption and absorption probabilities")
    p = add("hit", "expected steps from every state to reach a target state")
    p.add_argument("target")
    p = add("mix", "worst-case total-variation distance to stationarity and mixing time")
    p.add_argument("--epsilon", type=float, default=0.01, help="TV threshold (default 0.01)")
    p.add_argument("--max-steps", type=int, default=200, help="give up after this many steps")
    p = add("simulate", "seeded Monte Carlo run compared with exact stationary probabilities")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    add("dot", "Graphviz DOT drawing of the chain", json_flag=False)
    p = add("gate", "CI gate: exit 3 when a chain property check fails")
    p.add_argument("--require-irreducible", action="store_true")
    p.add_argument("--require-aperiodic", action="store_true")
    p.add_argument("--min-stationary", action="append", default=[], metavar="STATE=PROB")
    p.add_argument("--max-absorb-steps", metavar="N", help="max expected steps to absorption from start")
    return parser


def run(args):
    path = Path(args.file)
    text = path.read_text(encoding="utf-8")
    chain, errors, warnings = parse_chain(text)

    if args.command == "lint":
        lines, code = cmd_lint_result(path, chain, errors, warnings)
        print("\n".join(lines))
        return code

    if errors:
        for line in diagnostics(path, errors, "error"):
            print(line, file=sys.stderr)
        raise ChainError(f"{len(errors)} error(s) in chain file; run 'markovkit lint {path}' for details")

    data, lines, code = COMMANDS[args.command](chain, args)
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2))
    else:
        print("\n".join(lines))
    return code


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except UsageError as exc:
        print(f"markovkit: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ChainError as exc:
        print(f"markovkit: {exc}", file=sys.stderr)
        return EXIT_CHAIN_ERRORS
    except SingularSystemError as exc:
        print(f"markovkit: {exc}", file=sys.stderr)
        return EXIT_CHAIN_ERRORS
    except OSError as exc:
        print(f"markovkit: cannot read chain file: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
