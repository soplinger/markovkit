# markovkit

**Exact Markov chain analysis from one plain-text file. Offline, stdlib only.**

markovkit reads a discrete-time Markov chain from a small plain-text file and answers the usual questions with exact fractions: where the chain settles, how fast it mixes, how long until it is absorbed, and how long it takes to reach a given state. Lint errors point at the exact line, and a CI gate fails the build when a chain property breaks.

No servers, accounts, API keys or network access. Python 3 standard library only.

## Features

- **Plain-text chain format.** One `from -> to probability` line per transition and optional `start state probability` lines. Probabilities may be fractions (`1/10`) or decimals (`0.9`), and both are kept exact.
- **Line-numbered lint.** Reports bad syntax, probabilities outside 0..1, duplicate transitions, rows that do not sum to 1, states with no outgoing row, and a start distribution that does not sum to 1. Unreachable states are warnings.
- **Structure report (`info`).** Communicating classes, recurrent and transient states, absorbing states, periods and irreducibility.
- **Exact n-step distribution (`step`).** Where the chain is after N steps from its start distribution, as fractions and decimals.
- **Exact matrix power (`power`).** The full N-step transition matrix P^N.
- **Stationary distribution (`stationary`).** Exact long-run probabilities for every recurrent class, with expected return times (1/π).
- **Absorption analysis (`absorb`).** Expected steps to absorption and the probability of ending in each recurrent class, from every transient state and from the start distribution.
- **Hitting times (`hit`).** Expected steps from every state to a chosen target. States that might never reach it are reported as infinite.
- **Mixing time (`mix`).** Worst-case total-variation distance to stationarity over all starting states, with the first step that falls below your epsilon. Periodic chains are identified as never converging.
- **Seeded simulation (`simulate`).** A reproducible Monte Carlo run, compared with the exact stationary distribution.
- **Graphviz export (`dot`).** A DOT drawing with exact edge labels and absorbing states marked.
- **CI gate (`gate`).** Checks irreducibility, aperiodicity, minimum stationary probability per state and a maximum expected time to absorption. Exits 3 when a check fails.
- **JSON output.** `--json` on the analysis commands gives machine-readable exact values as strings.
- **Stable exit codes.** 0 ok, 1 chain file has errors or the analysis does not apply, 2 usage error, 3 gate failed.

## Quick start

The chain file format:

```text
# examples/weather.chain
start Sunny 1

Sunny -> Sunny 0.8
Sunny -> Cloudy 1/5
Cloudy -> Sunny 1/4
Cloudy -> Cloudy 1/2
Cloudy -> Rainy 1/4
Rainy -> Cloudy 1/3
Rainy -> Rainy 2/3
```

Check it, then ask questions of it:

```sh
python3 markovkit.py lint examples/weather.chain
python3 markovkit.py info examples/weather.chain
python3 markovkit.py stationary examples/weather.chain
python3 markovkit.py mix examples/weather.chain --epsilon 0.01
```

For this chain the stationary distribution is exactly 5/12 Sunny, 1/3 Cloudy and 1/4 Rainy, so the expected return time to Sunny is 12/5 steps. `lint` exits 0 when the file is valid.

An absorbing example, a customer waiting at a service desk (`examples/queue.chain`):

```sh
python3 markovkit.py absorb examples/queue.chain
python3 markovkit.py gate examples/queue.chain --max-absorb-steps 3
```

From Waiting the expected number of steps to absorption is exactly 2, and the chain ends in Served with probability 3/5 and in Lost with probability 2/5.

Draw it:

```sh
python3 markovkit.py dot examples/queue.chain > queue.dot
dot -Tsvg queue.dot -o queue.svg   # needs Graphviz installed
```

## How to run

```text
python3 markovkit.py COMMAND FILE [options]

COMMAND is one of:
  lint        check the file and report line-numbered errors and warnings
  info        classes, periods, absorbing and transient states
  step N      distribution after N steps from the start line
  power N     exact N-step transition matrix
  stationary  exact stationary distribution and return times
  absorb      expected steps to absorption and absorption probabilities
  hit TARGET  expected steps from every state to TARGET
  mix         worst-case TV distance and mixing time (--epsilon, --max-steps)
  simulate    seeded Monte Carlo (--steps, --seed) vs exact stationary
  dot         Graphviz DOT output
  gate        CI checks (--require-irreducible, --require-aperiodic,
              --min-stationary STATE=PROB, --max-absorb-steps N)
```

Add `--json` to the analysis commands for JSON output. Examples for a CI pipeline:

```sh
python3 markovkit.py gate chains/weather.chain --require-irreducible --require-aperiodic \
    --min-stationary Rainy=1/5 || echo "chain no longer meets its targets"
```

The chain file is read as UTF-8. Lines starting with `#` are comments. State names are any run of non-space characters, except that `start` followed by two tokens is always an initial-distribution line.

## How to test

The tests use pytest, read the sample chains in `examples/`, and need no network access:

```sh
python3 -m pytest
```

They cover parsing and line-numbered errors, exact stationary distributions, absorption and hitting times, periodicity, mixing, seeded simulation, the CLI exit codes and the JSON output.

## Project layout

```text
markovkit.py            the whole tool: parser, exact linear algebra, commands, CLI
examples/               sample chains (weather, queue, bounce)
tests/                  pytest suite
```

## Notes on exactness

Lint, info, step, power, stationary, absorb, hit and gate use exact `Fraction` arithmetic, so their results have no rounding error. `mix` and `simulate` use floating point; `mix` reports distances to six decimal places.
