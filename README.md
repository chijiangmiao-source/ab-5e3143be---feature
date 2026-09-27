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

### Optional termination-latency audit

Setting `"termination_latency_audit": true` in the request additionally
fixes the maximum rescue probability and then, minimising only over truly
optimal actions, computes the exact minimum expected number of link
actions until a rescued or lost terminal is reached:

* States that cannot terminate almost surely are isolated exactly — as a
  maximum-reachability solve (probability `1` of reaching *any* terminal)
  on the model restricted to truly optimal actions.  Their minimum
  latency is certified as `"infinity"`.
* On the remaining states the minimum expected hitting time is the least
  fixpoint of `d = T(d)`, `T(d)(s) = min_a [1 + Σ_t P(s,a,t)·d(t)]`,
  computed exactly as the unique optimum of the LP
  `max Σ d(s)  s.t.  d ≤ T(d), d ≥ 0` with the same rational simplex.
* Every candidate action is certified with its exact action-value
  `1 + Σ_t P(s,a,t)·d(t)` — or with `"latency": "infinity"` and reason
  `infinite_latency_non_terminating_loop` when its support can enter a
  non-terminating closed loop with positive probability.  Infinite
  candidates are never ranked beside finite ones; the canonical action
  is the least exact latency, ties broken lexicographically.

Requests that do not enable the option (or set it to `false`) produce
exactly the legacy response fields.

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
* `termination_latency_audit` — optional boolean (default `false`).  When
  `true`, the response gains a `termination_latency` section (see below);
  when absent or `false` the response is exactly the legacy schema.

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

With `"termination_latency_audit": true` the response additionally
contains a `termination_latency` section, e.g. for a model where a slow
direct action (`1 + (2/3)·d(s0) = 7/3` expected link actions) competes
with a faster deterministic relay (`2`):

```json
{
  "objective": "min_expected_link_actions_to_terminal",
  "start_state": "s0",
  "start_min_latency": "2",
  "start_min_latency_num": 2,
  "start_min_latency_den": 1,
  "start_finite": true,
  "canonical_policy": { "s0": "z_relay", "s1": "go" },
  "infinite_latency_states": [],
  "states": [
    {
      "state": "s0",
      "rescue_probability": "1",
      "min_latency": "2",
      "min_latency_num": 2,
      "min_latency_den": 1,
      "finite": true,
      "canonical_action": "z_relay",
      "actions": [
        {
          "action": "a_direct",
          "latency": "7/3",
          "latency_num": 7,
          "latency_den": 3,
          "finite": true,
          "selected": false,
          "reason": "latency_above_minimum"
        },
        {
          "action": "z_relay",
          "latency": "2",
          "latency_num": 2,
          "latency_den": 1,
          "finite": true,
          "selected": true,
          "reason": "selected_min_latency_lexicographically_least"
        }
      ]
    }
  ]
}
```

Per state the section lists the rescue probability, the exact minimum
termination latency, every candidate action's exact latency expectation
and the canonical decision, so the caller can recompute
`min_latency = min over candidates of (1 + Σ P·d)`.  A candidate that
keeps the maximum rescue probability but can fall into a non-terminating
closed loop with positive probability is certified as
`"latency": "infinity", "finite": false` with reason
`infinite_latency_non_terminating_loop` and is never ranked beside finite
candidates; states whose every candidate loops forever appear in
`infinite_latency_states` with `"min_latency": "infinity"`.  Other
reasons: `latency_above_minimum`,
`min_latency_but_identifier_not_lexicographically_least`, and
`selected_min_latency_lexicographically_least`.

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
   locatable `400` failure for a bad probability sum; it also exercises
   the termination-latency audit (exact faster relay over a slower direct
   action, an infinite-latency closed-loop candidate, an unchanged legacy
   response, and a rejected non-boolean flag);

then prints a report and exits with code `0` (pass) or `1` (fail); the exit
code is reported both as the process exit status and on the
`VERIFY EXIT CODE:` line.

## Run locally (Python ≥ 3.9, stdlib only)

```bash
python -m unittest discover -s tests -v   # tests
PORT=8080 python -m app.server            # service
python -m verify.verify                   # full verify against localhost
```
