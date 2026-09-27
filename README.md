# Reachability Audit Service（深海浮标可达性审计）

Exact maximum-reachability audit for finite Markov decision processes over
arbitrary-precision rationals.  Given a buoy failure model — controllable
states, their actions and exact reduced-fraction transition distributions —
the service returns the maximum probability of reaching a rescued terminal
state, the canonical optimal action for every controllable state, and
per-state certificates that let the caller recompute every selection or
exclusion decision.

No floating point, no bounded random simulation, no enumeration of action
policies and no truncated value iteration is used anywhere.

## Method

* The maximum rescue probability is the **least fixpoint** of the Bellman
  optimality operator, computed exactly by solving the equivalent linear
  program `min Σ v(s) s.t. v ≥ T(v), v ≥ 0` via its dual packing LP with an
  exact rational simplex method (`fractions.Fraction`, Bland's rule, so it
  always terminates).  Closed loops that can never reach any terminal get
  value `0` automatically (the least fixpoint, not a truncated iterate).
* An action is **truly optimal** iff it is Bellman-equal *and* — for
  positive-value states — rescued remains reachable when the state is
  committed to that action and all other states use Bellman-equal actions.
  This filters out actions that only look optimal because they sit on a
  zero-progress closed loop.
* The **canonical action** is the lexicographically smallest truly optimal
  action identifier.
* Certificates list, per controllable state, the exact state value and the
  exact expected value of every action, plus the reason each action was
  selected or excluded.

## API

### `POST /api/reachability-audit`

Request body:

```json
{
  "states": ["s0", "r", "l"],
  "start": "s0",
  "terminals": { "rescued": ["r"], "lost": ["l"] },
  "actions": {
    "s0": {
      "go": { "r": "1/3", "l": "2/3" }
    }
  }
}
```

* `states` — unique state identifiers (duplicates are rejected).
* `start` — the initial state; must be declared in `states`.
* `terminals.rescued` / `terminals.lost` — disjoint subsets of `states`.
* `actions` — exactly one entry per **controllable** (non-terminal) state;
  each action maps target states to **reduced fractions** (`"1/3"`, `"2/3"`,
  `"1"`, …; JSON floats such as `0.5` are rejected).  Each distribution must
  sum to exactly `1`.  Terminal states must not carry actions.

Success response (`200`):

```json
{
  "status": "ok",
  "start_state": "s0",
  "max_rescue_probability": "1/3",
  "max_rescue_probability_num": 1,
  "max_rescue_probability_den": 3,
  "canonical_policy": { "s0": "go" },
  "state_values": { "l": "0", "r": "1", "s0": "1/3" },
  "zero_value_states": [],
  "terminal_unreachable_states": [],
  "certificates": [
    {
      "state": "s0",
      "value": "1/3",
      "canonical_action": "go",
      "actions": [
        {
          "action": "go",
          "expected_value": "1/3",
          "optimal": true,
          "selected": true,
          "reason": "selected_lexicographically_least_optimal"
        }
      ]
    }
  ]
}
```

Exclusion reasons: `expected_value_below_state_value` (the action's
expectation is strictly below the state value) and
`equal_value_but_no_optimal_continuation` (Bellman-equal but committing to
it can trap the process in a closed loop away from every terminal).

Validation failure (`400`, no certificate is produced) — every error is
locatable via a JSON path:

```json
{
  "status": "error",
  "errors": [
    {
      "path": "$.actions[\"s0\"][\"go\"]",
      "code": "PROBABILITY_SUM",
      "message": "transition probabilities of action 'go' sum to 5/6, expected exactly 1"
    }
  ]
}
```

Error codes include `UNKNOWN_STATE`, `MISSING_ACTIONS`,
`TERMINAL_HAS_ACTIONS`, `PROBABILITY_SUM`, `INVALID_PROBABILITY`,
`DUPLICATE_STATE`, `DUPLICATE_KEY`, `DUPLICATE_TERMINAL`, `EMPTY_ACTIONS`,
`EMPTY_DISTRIBUTION`, `INVALID_TYPE`, `INVALID_JSON`.

### `GET /healthz`

Liveness/readiness probe, returns `{"status": "ok"}`.

## Run with Docker Compose

```bash
docker compose up --build web          # serve on host port 8080
HOST_PORT=9090 docker compose up web   # host port is configurable
docker compose up --exit-code-from verify --abort-on-container-exit
```

The `verify` service waits for `web` to be healthy, then runs

1. **code tests** — the unit suite (exact solver, validation, and a
   brute-force cross-check that lives only in the tests);
2. **build checks** — every source file byte-compiles and imports;
3. **API/HTTP smoke** — posts the must-rescue, `1/3`-`2/3`, equal-value
   action and closed-loop models and asserts the exact fraction
   probabilities, the lexicographic canonical action with equal expected
   values, and the zero rescue probability of the closed loop, plus a
   locatable `400` failure for a bad probability sum;

then prints a report and exits with code `0` (pass) or `1` (fail); the exit
code is reported both as the process exit status and on the
`VERIFY EXIT CODE:` line.

## Run locally (Python ≥ 3.9, stdlib only)

```bash
python -m unittest discover -s tests -v   # tests
PORT=8080 python -m app.server            # service
python -m verify.verify                   # full verify against localhost
```
