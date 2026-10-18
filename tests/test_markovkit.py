import json
from fractions import Fraction
from pathlib import Path

import pytest

import markovkit

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
WEATHER = EXAMPLES / "weather.chain"
QUEUE = EXAMPLES / "queue.chain"
BOUNCE = EXAMPLES / "bounce.chain"


def parse_ok(text):
    chain, errors, warnings = markovkit.parse_chain(text)
    assert errors == []
    return chain, warnings


def parse_errors(text):
    chain, errors, _ = markovkit.parse_chain(text)
    assert chain is None
    return errors


def test_parse_reads_exact_fractions_and_decimals():
    chain, _ = parse_ok("start A 1\nA -> A 0.9\nA -> B 1/10\nB -> B 1\n")
    assert chain.states == ["A", "B"]
    assert chain.matrix[0] == [Fraction(9, 10), Fraction(1, 10)]
    assert chain.start == [Fraction(1), Fraction(0)]


def test_parse_reports_bad_probability_with_line_number():
    errors = parse_errors("start A 1\nA -> A 1.5\n")
    assert (2, "probability 1.5 is outside 0..1") in errors


def test_parse_reports_row_that_does_not_sum_to_one_on_its_first_line():
    text = "start A 1\nA -> B 0.5\nA -> C 0.5\nB -> A 1\nC -> C 0.7\n"
    errors = parse_errors(text)
    assert (5, "outgoing probabilities from 'C' sum to 7/10, not 1") in errors


def test_parse_flags_missing_outgoing_row_and_bad_syntax():
    errors = parse_errors("A -> B 1\n")
    assert (1, "state 'B' has no outgoing transitions (add 'B -> B 1' for an absorbing state)") in errors
    errors = parse_errors("A -> B 1\nA B 1\n")
    assert errors[0][0] == 2
    assert "expected" in errors[0][1]


def test_parse_flags_duplicate_transition_and_bad_start_sum():
    errors = parse_errors("start A 0.5\nA -> A 1\nA -> A 0\n")
    messages = dict(errors)
    assert messages[1] == "initial probabilities sum to 1/2, not 1"
    assert messages[3] == "duplicate transition A -> A"


def test_unreachable_state_is_a_warning_not_an_error():
    chain, warnings = parse_ok("start A 1\nA -> A 1\nB -> B 1\n")
    assert (3, "state 'B' is unreachable from the start distribution") in warnings


def test_two_state_stationary_distribution_is_exact():
    chain, _ = parse_ok("start A 1\nA -> A 1/2\nA -> B 1/2\nB -> A 1/4\nB -> B 3/4\n")
    pi = markovkit.stationary_map(chain, markovkit.analyse(chain))
    assert pi == {0: Fraction(1, 3), 1: Fraction(2, 3)}


def test_weather_stationary_satisfies_pi_P_equals_pi():
    chain, _ = parse_ok(WEATHER.read_text())
    pi = markovkit.stationary_map(chain, markovkit.analyse(chain))
    assert sum(pi.values()) == 1
    for j in range(len(chain.states)):
        assert sum(pi[i] * chain.matrix[i][j] for i in range(3)) == pi[j]


def test_classes_and_periods():
    weather, _ = parse_ok(WEATHER.read_text())
    (cls,) = markovkit.analyse(weather)
    assert cls["recurrent"] and cls["period"] == 1
    bounce, _ = parse_ok(BOUNCE.read_text())
    (cls,) = markovkit.analyse(bounce)
    assert cls["period"] == 2
    queue, _ = parse_ok(QUEUE.read_text())
    recurrent = [c for c in markovkit.analyse(queue) if c["recurrent"]]
    assert len(recurrent) == 2 and all(c["absorbing"] for c in recurrent)


def test_absorption_expected_steps_and_probabilities_are_exact():
    chain, _ = parse_ok(QUEUE.read_text())
    ab = markovkit.absorption(chain, markovkit.analyse(chain))
    waiting = ab["transient"][0]
    row = ab["X"][0]
    assert row[0] == 2  # geometric with success 1/2
    labels = {tuple(c["members"]): k for k, c in enumerate(ab["recurrent"])}
    served = labels[(chain.index["Served"],)]
    lost = labels[(chain.index["Lost"],)]
    assert row[served + 1] == Fraction(3, 5)
    assert row[lost + 1] == Fraction(2, 5)
    assert waiting == chain.index["Waiting"]


def test_hitting_times_solve_first_step_equations():
    chain, _ = parse_ok("start A 1\nA -> B 1\nB -> A 1/2\nB -> C 1/2\nC -> C 1\n")
    times = markovkit.hitting_times(chain, chain.index["C"])
    assert times[chain.index["A"]] == 4
    assert times[chain.index["B"]] == 3
    assert times[chain.index["C"]] == 0


def test_hitting_time_is_infinite_when_target_can_be_missed():
    chain, _ = parse_ok("start A 1\nA -> B 1\nB -> A 1\nC -> C 1\n")
    times = markovkit.hitting_times(chain, chain.index["C"])
    assert times[chain.index["A"]] is None
    assert times[chain.index["B"]] is None


def test_step_and_power_are_exact():
    chain, _ = parse_ok(BOUNCE.read_text())
    dist = list(chain.start)
    dist = markovkit.vec_mat(dist, chain.matrix)
    assert dist == [Fraction(0), Fraction(1)]
    square = markovkit.mat_mul(chain.matrix, chain.matrix)
    assert square == [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]


def test_solve_raises_on_singular_system():
    with pytest.raises(markovkit.SingularSystemError):
        markovkit.solve([[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]],
                        [[Fraction(1)], [Fraction(1)]])


def test_mix_converges_for_aperiodic_chain_and_never_for_periodic(capsys):
    assert markovkit.main(["mix", str(WEATHER), "--epsilon", "0.01"]) == 0
    assert "Mixing time" in capsys.readouterr().out
    assert markovkit.main(["mix", str(BOUNCE), "--max-steps", "30"]) == 0
    assert "periodic" in capsys.readouterr().out


def test_simulate_is_deterministic_for_a_seed(capsys):
    markovkit.main(["simulate", str(WEATHER), "--steps", "500", "--seed", "7", "--json"])
    first = json.loads(capsys.readouterr().out)
    markovkit.main(["simulate", str(WEATHER), "--steps", "500", "--seed", "7", "--json"])
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert abs(sum(first["frequencies"].values()) - 1) < 1e-4


def test_lint_reports_line_numbered_errors(tmp_path, capsys):
    bad = tmp_path / "bad.chain"
    bad.write_text("start A 1\nA -> B 0.5\nA -> B 0.5\nB -> A 1\n")
    assert markovkit.main(["lint", str(bad)]) == 1
    out = capsys.readouterr().out
    assert f"{bad}:3: error: duplicate transition A -> B" in out


def test_lint_passes_on_example(capsys):
    assert markovkit.main(["lint", str(QUEUE)]) == 0
    assert "ok: 3 states" in capsys.readouterr().out


def test_gate_exit_codes(capsys):
    assert markovkit.main(["gate", str(WEATHER), "--require-irreducible", "--require-aperiodic"]) == 0
    assert markovkit.main(["gate", str(QUEUE), "--require-irreducible"]) == markovkit.EXIT_GATE_FAILED
    assert markovkit.main(["gate", str(QUEUE), "--min-stationary", "Waiting=1/10"]) == markovkit.EXIT_GATE_FAILED
    assert markovkit.main(["gate", str(QUEUE), "--max-absorb-steps", "3"]) == 0
    assert markovkit.main(["gate", str(QUEUE)]) == markovkit.EXIT_USAGE


def test_stationary_json_probabilities_sum_to_one(capsys):
    assert markovkit.main(["stationary", str(WEATHER), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    total = sum(Fraction(v) for v in data["classes"][0]["stationary"].values())
    assert total == 1


def test_power_json_matches_exact_square(capsys):
    assert markovkit.main(["power", str(BOUNCE), "2", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["matrix"]["A"] == {"A": "1", "B": "0"}


def test_hit_and_dot_commands_run(capsys):
    assert markovkit.main(["hit", str(QUEUE), "Served", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["hitting_times"]["Lost"] is None
    assert markovkit.main(["dot", str(QUEUE)]) == 0
    assert "digraph markov" in capsys.readouterr().out


def test_unknown_state_is_a_usage_error(capsys):
    assert markovkit.main(["hit", str(QUEUE), "Nowhere"]) == markovkit.EXIT_USAGE
    assert "unknown state 'Nowhere'" in capsys.readouterr().err
