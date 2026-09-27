"""Exact maximum-reachability audit for finite Markov decision processes.

All arithmetic is carried out with ``fractions.Fraction`` (arbitrary
precision rationals).  The maximum probability of reaching a "rescued"
terminal state is computed as the *least fixpoint* of the Bellman
optimality operator, obtained by solving the equivalent linear program

    (P)  min  sum_s v(s)
         s.t. v(s) - sum_t P(s,a,t) * v(t) >= beta(s,a)   for every
              controllable state s and action a,
              v(s) >= 0,

where beta(s,a) is the probability mass that action a moves directly
into a rescued terminal.  By Knaster-Tarski the least fixpoint is the
componentwise-minimal pre-fixpoint, hence the unique optimum of (P).

Instead of solving (P) directly we solve its dual

    (D)  max  beta^T y  s.t.  B^T y <= 1, y >= 0

with an exact rational simplex method (Bland's rule, guaranteed to
terminate).  (D) has a trivially feasible origin, so no phase-1 is
needed, and the optimal objective-row coefficients of the slack
variables are exactly the optimal primal solution v*.

No floating point, no bounded random simulation, no enumeration of
action policies and no truncated value iteration is used anywhere.

When the request enables ``termination_latency_audit``, the service
additionally computes — with the maximum rescue probability fixed — the
exact minimum expected number of link actions until a rescued or lost
terminal is reached, minimised only over truly optimal actions.  This is
again an exact least fixpoint: states that cannot terminate almost
surely are isolated by an exact max-reachability solve on the
restricted model, and the finite minimum expected hitting times are the
unique optimum of the LP  max sum d(s)  s.t.  d <= T(d), d >= 0.  A
candidate action whose support can enter a non-terminating closed loop
is certified with infinite latency, never ranked beside finite
candidates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from math import gcd


# ---------------------------------------------------------------------------
# JSON document representation (keeps object pairs so duplicate keys and
# non-exact numbers can be reported with a precise location).
# ---------------------------------------------------------------------------


class JsonObj:
    """A JSON object preserved as an ordered list of key/value pairs."""

    __slots__ = ("pairs",)

    def __init__(self, pairs):
        self.pairs = pairs


class JsonFloat:
    """A JSON number with a fractional/exponent part (rejected: inexact)."""

    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


class JsonConstant:
    """A non-finite JSON constant such as NaN/Infinity (rejected)."""

    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


class AuditError(Exception):
    """Validation failure carrying a list of locatable errors."""

    def __init__(self, errors):
        super().__init__("model validation failed")
        self.errors = errors


def _err(errors, path, code, message):
    errors.append({"path": path, "code": code, "message": message})


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------


@dataclass
class Model:
    states: list  # ordered list of unique state identifiers
    start: str
    rescued: frozenset
    lost: frozenset
    # state -> action -> {target: Fraction}  (controllable states only,
    # zero-probability entries removed)
    actions: dict
    # when true, also run the exact minimum-termination-latency audit
    termination_latency: bool = False


def load_and_validate(text):
    """Parse the request body and return a validated :class:`Model`.

    Raises :class:`AuditError` with locatable errors on any violation.
    """
    try:
        doc = json.loads(
            text,
            object_pairs_hook=JsonObj,
            parse_float=JsonFloat,
            parse_constant=JsonConstant,
        )
    except json.JSONDecodeError as exc:
        raise AuditError(
            [
                {
                    "path": "$",
                    "code": "INVALID_JSON",
                    "message": "request body is not valid JSON: %s" % exc,
                }
            ]
        )
    errors = []
    model = _validate(doc, errors)
    if errors:
        raise AuditError(errors)
    return model


def _fields(jobj, path, errors):
    """Return first-value-per-key dict of a JsonObj, flagging duplicates."""
    out = {}
    for key, value in jobj.pairs:
        if key in out:
            _err(errors, path, "DUPLICATE_KEY", "duplicate key '%s'" % key)
            continue
        out[key] = value
    return out


def _is_name(value):
    return isinstance(value, str) and len(value) > 0


def _validate(doc, errors):
    if not isinstance(doc, JsonObj):
        _err(errors, "$", "INVALID_TYPE", "request body must be a JSON object")
        return None

    root = _fields(doc, "$", errors)

    # ---- states -----------------------------------------------------------
    states = []
    state_set = set()
    raw_states = root.get("states")
    if raw_states is None:
        _err(errors, "$.states", "MISSING_FIELD", "field 'states' is required")
    elif not isinstance(raw_states, list):
        _err(errors, "$.states", "INVALID_TYPE", "'states' must be an array")
    elif not raw_states:
        _err(errors, "$.states", "EMPTY_LIST", "'states' must not be empty")
    else:
        for i, item in enumerate(raw_states):
            path = "$.states[%d]" % i
            if not _is_name(item):
                _err(errors, path, "INVALID_NAME",
                     "state identifier must be a non-empty string")
                continue
            if item in state_set:
                _err(errors, path, "DUPLICATE_STATE",
                     "duplicate state identifier '%s'" % item)
                continue
            state_set.add(item)
            states.append(item)

    # ---- start ------------------------------------------------------------
    start = None
    raw_start = root.get("start")
    if raw_start is None:
        _err(errors, "$.start", "MISSING_FIELD", "field 'start' is required")
    elif not _is_name(raw_start):
        _err(errors, "$.start", "INVALID_NAME",
             "'start' must be a non-empty string")
    elif raw_start not in state_set:
        _err(errors, "$.start", "UNKNOWN_STATE",
             "start state '%s' is not declared in 'states'" % raw_start)
    else:
        start = raw_start

    # ---- terminals --------------------------------------------------------
    rescued = set()
    lost = set()
    raw_terminals = root.get("terminals")
    if raw_terminals is None:
        _err(errors, "$.terminals", "MISSING_FIELD",
             "field 'terminals' is required")
    elif not isinstance(raw_terminals, JsonObj):
        _err(errors, "$.terminals", "INVALID_TYPE",
             "'terminals' must be an object with 'rescued' and 'lost' arrays")
    else:
        tfields = _fields(raw_terminals, "$.terminals", errors)
        for key, target in (("rescued", rescued), ("lost", lost)):
            raw_list = tfields.get(key)
            path = "$.terminals.%s" % key
            if raw_list is None:
                _err(errors, path, "MISSING_FIELD",
                     "field 'terminals.%s' is required" % key)
                continue
            if not isinstance(raw_list, list):
                _err(errors, path, "INVALID_TYPE",
                     "'terminals.%s' must be an array" % key)
                continue
            for i, item in enumerate(raw_list):
                ipath = "%s[%d]" % (path, i)
                if not _is_name(item):
                    _err(errors, ipath, "INVALID_NAME",
                         "terminal identifier must be a non-empty string")
                    continue
                if item not in state_set:
                    _err(errors, ipath, "UNKNOWN_STATE",
                         "terminal state '%s' is not declared in 'states'"
                         % item)
                    continue
                if item in rescued or item in lost:
                    _err(errors, ipath, "DUPLICATE_TERMINAL",
                         "state '%s' is declared as a terminal more than once"
                         % item)
                    continue
                target.add(item)

    terminals = rescued | lost

    # ---- actions ----------------------------------------------------------
    actions = {}
    raw_actions = root.get("actions")
    if raw_actions is None:
        _err(errors, "$.actions", "MISSING_FIELD",
             "field 'actions' is required")
        raw_actions = JsonObj([])
    if not isinstance(raw_actions, JsonObj):
        _err(errors, "$.actions", "INVALID_TYPE", "'actions' must be an object")
        raw_actions = JsonObj([])

    action_blocks = _fields(raw_actions, "$.actions", errors)
    for state, block in action_blocks.items():
        spath = '$.actions["%s"]' % state
        if state not in state_set:
            _err(errors, spath, "UNKNOWN_STATE",
                 "actions declared for unknown state '%s'" % state)
            continue
        if state in terminals:
            _err(errors, spath, "TERMINAL_HAS_ACTIONS",
                 "terminal state '%s' must not carry actions" % state)
            continue
        if not isinstance(block, JsonObj):
            _err(errors, spath, "INVALID_TYPE",
                 "actions of state '%s' must be an object" % state)
            continue
        action_map = _fields(block, spath, errors)
        if not action_map:
            _err(errors, spath, "EMPTY_ACTIONS",
                 "controllable state '%s' must declare at least one action"
                 % state)
            continue
        parsed_actions = {}
        for action, dist in action_map.items():
            apath = '%s["%s"]' % (spath, action)
            if not _is_name(action):
                _err(errors, apath, "INVALID_NAME",
                     "action identifier must be a non-empty string")
                continue
            if not isinstance(dist, JsonObj):
                _err(errors, apath, "INVALID_TYPE",
                     "transition distribution of action '%s' must be an object"
                     % action)
                continue
            entries = _fields(dist, apath, errors)
            if not entries:
                _err(errors, apath, "EMPTY_DISTRIBUTION",
                     "transition distribution of action '%s' is empty" % action)
                continue
            parsed_dist = {}
            total = Fraction(0)
            ok = True
            for target, raw_p in entries.items():
                tpath = '%s["%s"]' % (apath, target)
                if target not in state_set:
                    _err(errors, tpath, "UNKNOWN_STATE",
                         "transition target '%s' is not declared in 'states'"
                         % target)
                    ok = False
                    continue
                prob = _parse_probability(raw_p, tpath, errors)
                if prob is None:
                    ok = False
                    continue
                if prob > 0:
                    parsed_dist[target] = prob
                total += prob
            if not ok:
                continue
            if total != 1:
                _err(
                    errors,
                    apath,
                    "PROBABILITY_SUM",
                    "transition probabilities of action '%s' sum to %s, "
                    "expected exactly 1" % (action, _frac_str(total)),
                )
                continue
            parsed_actions[action] = parsed_dist
        if parsed_actions:
            actions[state] = parsed_actions

    # ---- termination latency audit option ----------------------------------
    termination_latency = False
    raw_option = root.get("termination_latency_audit")
    if raw_option is not None:
        if isinstance(raw_option, bool):
            termination_latency = raw_option
        else:
            _err(errors, "$.termination_latency_audit", "INVALID_TYPE",
                 "'termination_latency_audit' must be a boolean")

    # ---- every controllable state must carry actions ----------------------
    # Only report states that declared no action block at all; states whose
    # actions were declared but failed validation already carry locatable
    # errors above.
    for state in states:
        if state not in terminals and state not in action_blocks:
            _err(
                errors,
                '$.actions["%s"]' % state,
                "MISSING_ACTIONS",
                "controllable (non-terminal) state '%s' has no actions"
                % state,
            )

    if errors:
        return None
    return Model(
        states=states,
        start=start,
        rescued=frozenset(rescued),
        lost=frozenset(lost),
        actions=actions,
        termination_latency=termination_latency,
    )


def _parse_probability(value, path, errors):
    """Parse an exact reduced-fraction probability; return Fraction or None."""
    if isinstance(value, bool):
        _err(errors, path, "INVALID_PROBABILITY",
             "probability must be an exact fraction, not a boolean")
        return None
    if isinstance(value, int):
        num, den = value, 1
    elif isinstance(value, JsonFloat):
        _err(errors, path, "INVALID_PROBABILITY",
             "floating-point number '%s' is not accepted; use an exact "
             "fraction string such as \"1/3\"" % value.text)
        return None
    if isinstance(value, JsonConstant):
        _err(errors, path, "INVALID_PROBABILITY",
             "non-finite constant '%s' is not a valid probability" % value.text)
        return None
    if isinstance(value, str):
        text = value.strip()
        parts = text.split("/")
        if len(parts) > 2:
            _err(errors, path, "INVALID_PROBABILITY",
                 "malformed fraction '%s'" % value)
            return None
        try:
            num = int(parts[0], 10)
            den = int(parts[1], 10) if len(parts) == 2 else 1
        except ValueError:
            _err(errors, path, "INVALID_PROBABILITY",
                 "malformed fraction '%s'" % value)
            return None
    else:
        _err(errors, path, "INVALID_PROBABILITY",
             "probability must be an exact fraction string (e.g. \"2/3\") "
             "or an integer")
        return None

    if den <= 0:
        _err(errors, path, "INVALID_PROBABILITY",
             "denominator must be positive")
        return None
    if num < 0 or num > den:
        _err(errors, path, "INVALID_PROBABILITY",
             "probability %d/%d is outside [0, 1]" % (num, den))
        return None
    if gcd(abs(num), den) != 1:
        _err(errors, path, "INVALID_PROBABILITY",
             "fraction %d/%d is not in lowest terms (既约分数 required)"
             % (num, den))
        return None
    return Fraction(num, den)


# ---------------------------------------------------------------------------
# Exact rational simplex method (Bland's rule) for  max c.x, A x <= b, x >= 0
# with b >= 0 so the slack basis is feasible.  Returns (x, dual_prices, opt).
# ---------------------------------------------------------------------------


def _simplex_max(c, A, b):
    m = len(A)
    n = len(c)
    width = n + m
    T = [[Fraction(0) for _ in range(width + 1)] for _ in range(m + 1)]
    for i in range(m):
        row_a = A[i]
        row_t = T[i]
        for j in range(n):
            v = row_a[j]
            if v:
                row_t[j] = v
        row_t[n + i] = Fraction(1)
        row_t[width] = b[i]
    for j in range(n):
        v = c[j]
        if v:
            T[m][j] = -v
    basis = [n + i for i in range(m)]

    while True:
        obj = T[m]
        enter = -1
        for j in range(width):
            if obj[j] < 0:
                enter = j
                break
        if enter < 0:
            break  # optimal
        leave = -1
        best_ratio = None
        for i in range(m):
            a = T[i][enter]
            if a > 0:
                ratio = T[i][width] / a
                if (
                    best_ratio is None
                    or ratio < best_ratio
                    or (ratio == best_ratio and basis[i] < basis[leave])
                ):
                    best_ratio = ratio
                    leave = i
        if leave < 0:
            raise ArithmeticError("unbounded LP (impossible for reachability)")
        pivot_row = T[leave]
        pivot = pivot_row[enter]
        if pivot != 1:
            pivot_row = [v / pivot for v in pivot_row]
            T[leave] = pivot_row
        for i in range(m + 1):
            if i == leave:
                continue
            factor = T[i][enter]
            if factor:
                T[i] = [a - factor * p for a, p in zip(T[i], pivot_row)]
        basis[leave] = enter

    x = [Fraction(0)] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][width]
    dual_prices = [T[m][n + i] for i in range(m)]
    return x, dual_prices, T[m][width]


# ---------------------------------------------------------------------------
# Exact solver
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    values: dict  # every state -> Fraction (rescued=1, lost=0)
    canonical: dict  # controllable state -> chosen action id
    optimal_actions: dict  # controllable state -> list of truly optimal ids
    expected: dict  # controllable state -> {action: Fraction}
    zero_value_states: list
    terminal_unreachable_states: list
    max_rescue_probability: Fraction


def _expected_value(model, dist, values):
    total = Fraction(0)
    for target, prob in dist.items():
        total += prob * values[target]
    return total


def solve(model):
    """Compute exact maximum reachability values and an optimal policy."""
    nonterm = [s for s in model.states
               if s not in model.rescued and s not in model.lost]

    # Dual packing LP: one variable y[s,a] per action, one constraint per
    # controllable state:  sum_a y[s,a] - sum_{(t,a)} P(t,a,s) y[t,a] <= 1.
    y_index = {}
    for s in nonterm:
        for a in model.actions[s]:
            y_index[(s, a)] = len(y_index)
    n = len(y_index)
    t_index = {t: i for i, t in enumerate(nonterm)}
    m = len(nonterm)

    values = {}
    if n:
        A = [[Fraction(0)] * n for _ in range(m)]
        c = [Fraction(0)] * n
        b = [Fraction(1)] * m
        for (s, a), j in y_index.items():
            dist = model.actions[s][a]
            beta = Fraction(0)
            for target, prob in dist.items():
                if target in model.rescued:
                    beta += prob
                elif target in t_index:
                    A[t_index[target]][j] -= prob
            c[j] = beta
            A[t_index[s]][j] += 1
        _, dual_prices, _ = _simplex_max(c, A, b)
        for t in nonterm:
            values[t] = dual_prices[t_index[t]]
    for r in model.rescued:
        values[r] = Fraction(1)
    for l in model.lost:
        values[l] = Fraction(0)

    # Expected value of every action under the optimal value vector.
    expected = {}
    for s in nonterm:
        expected[s] = {
            a: _expected_value(model, dist, values)
            for a, dist in model.actions[s].items()
        }

    # Self-check: values must be an exact fixpoint of the Bellman operator.
    for s in nonterm:
        best = max(expected[s].values())
        if best != values[s]:
            raise ArithmeticError(
                "internal error: value vector is not a fixpoint at '%s'" % s)

    # Bellman-equal candidate actions per state.
    bellman_equal = {
        s: sorted(a for a, ev in expected[s].items() if ev == values[s])
        for s in nonterm
    }

    # Truly optimal actions: a Bellman-equal action a at a positive-value
    # state s is optimal iff rescued is still reachable when s is committed
    # to a and every other state may use any Bellman-equal action.  (This
    # filters out actions that only look optimal because they sit on a
    # zero-progress closed loop.)  At zero-value states every Bellman-equal
    # action is optimal.
    optimal_actions = {}
    for s in nonterm:
        candidates = bellman_equal[s]
        if values[s] == 0:
            optimal_actions[s] = candidates
            continue
        good = []
        for a in candidates:
            if _rescued_reachable_committing(model, bellman_equal, s, a):
                good.append(a)
        optimal_actions[s] = good

    canonical = {s: optimal_actions[s][0] for s in nonterm}

    zero_value_states = sorted(s for s in nonterm if values[s] == 0)
    terminal_unreachable_states = _terminal_unreachable(model, nonterm)

    return SolveResult(
        values=values,
        canonical=canonical,
        optimal_actions=optimal_actions,
        expected=expected,
        zero_value_states=zero_value_states,
        terminal_unreachable_states=terminal_unreachable_states,
        max_rescue_probability=values[model.start],
    )


def _rescued_reachable_committing(model, bellman_equal, state, action):
    """True if a rescued terminal is reachable when ``state`` must use
    ``action`` and every other controllable state may use any of its
    Bellman-equal actions."""
    seen = {state}
    stack = [state]
    while stack:
        u = stack.pop()
        if u in model.rescued:
            return True
        if u in model.lost:
            continue
        if u == state:
            succ = model.actions[u][action].keys()
        else:
            succ = []
            for a2 in bellman_equal[u]:
                succ.extend(model.actions[u][a2].keys())
        for w in succ:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return False


def _terminal_unreachable(model, nonterm):
    """Controllable states from which no terminal state is reachable at all
    (closed loops that can never be rescued nor lost)."""
    can_reach = set(model.rescued) | set(model.lost)
    changed = True
    while changed:
        changed = False
        for s in nonterm:
            if s in can_reach:
                continue
            for dist in model.actions[s].values():
                if any(t in can_reach for t in dist):
                    can_reach.add(s)
                    changed = True
                    break
    return sorted(s for s in nonterm if s not in can_reach)


# ---------------------------------------------------------------------------
# Exact minimum termination latency (optional audit)
# ---------------------------------------------------------------------------


@dataclass
class LatencyResult:
    latency: dict  # safe controllable state -> Fraction (min expected steps)
    infinite_states: list  # controllable states whose min latency is infinite
    action_latency: dict  # controllable state -> {action: Fraction or None}
    canonical: dict  # controllable state -> latency-canonical action id


def solve_latency(model, result):
    """Exact minimum expected number of link actions until a terminal
    (rescued or lost) state is reached, minimised only over truly optimal
    actions, i.e. those that preserve the maximum rescue probability.

    A candidate action has *infinite* latency iff committing to it can,
    with positive probability, trap the process in a non-terminating
    closed loop.  This is decided exactly, not by simulation:

    * Stage 1 computes, on the restricted model that keeps only truly
      optimal actions, the maximum probability of reaching ANY terminal.
      States reaching a terminal with probability 1 are "safe"; every
      other state has infinite minimum latency (no proper policy exists).
    * Stage 2 computes the minimum expected hitting time of the terminal
      set on the safe states as the exact optimum of the LP

          max  sum_s d(s)
          s.t. d(s) - sum_t P(s,a,t) d(t) <= 1   for every truly optimal
               action a whose support stays inside the safe set,
               d(s) >= 0,

      which is the least fixpoint of the Bellman minimality operator
      d = T(d), T(d)(s) = min_a [1 + sum_t P(s,a,t) d(t)].  The LP is
      bounded: every safe state owns a proper policy, so no nonzero
      subharmonic direction exists.  Any feasible d satisfies d <= T(d),
      hence d <= J* (the least fixpoint), and J* itself is feasible, so
      the unique optimum is exactly J*.
    """
    nonterm = [s for s in model.states
               if s not in model.rescued and s not in model.lost]
    terminals = set(model.rescued) | set(model.lost)

    # Stage 1: exact max reachability of the terminal set within the
    # restricted (truly optimal actions only) model.
    sub_actions = {
        s: {a: model.actions[s][a] for a in result.optimal_actions[s]}
        for s in nonterm
    }
    sub_model = Model(
        states=model.states,
        start=model.start,
        rescued=frozenset(terminals),
        lost=frozenset(),
        actions=sub_actions,
    )
    sub_values = solve(sub_model).values
    safe = set(s for s in nonterm if sub_values[s] == 1)

    def stays_safe(dist):
        return all(t in terminals or t in safe for t in dist)

    # Stage 2: exact minimum expected hitting time on the safe states.
    safe_states = [s for s in nonterm if s in safe]
    latency = {}
    if safe_states:
        index = {s: i for i, s in enumerate(safe_states)}
        n = len(safe_states)
        A = []
        for s in safe_states:
            for a in result.optimal_actions[s]:
                dist = model.actions[s][a]
                if not stays_safe(dist):
                    continue
                row = [Fraction(0)] * n
                row[index[s]] = Fraction(1)
                for target, prob in dist.items():
                    if target in index:
                        row[index[target]] -= prob
                A.append(row)
        x, _, _ = _simplex_max(
            [Fraction(1)] * n, A, [Fraction(1)] * len(A))
        for s in safe_states:
            latency[s] = x[index[s]]

    # Per-candidate-action exact latency expectation 1 + E[d(successor)];
    # infinite (None) when the action can move, with positive probability,
    # into a state from which no policy terminates almost surely.
    action_latency = {}
    for s in nonterm:
        entry = {}
        for a in result.optimal_actions[s]:
            dist = model.actions[s][a]
            if s in safe and stays_safe(dist):
                total = Fraction(1)
                for target, prob in dist.items():
                    if target not in terminals:
                        total += prob * latency[target]
                entry[a] = total
            else:
                entry[a] = None
        action_latency[s] = entry

    # Canonical action: least exact latency, ties broken by identifier.
    # (On infinite-latency states every candidate is infinite; fall back
    # to the lexicographically least truly optimal action.)
    canonical = {}
    for s in nonterm:
        entry = action_latency[s]
        finite = [(lat, a) for a, lat in entry.items() if lat is not None]
        if finite:
            best = min(lat for lat, _ in finite)
            canonical[s] = min(a for lat, a in finite if lat == best)
        else:
            canonical[s] = min(entry)

    # Self-check: the LP optimum reproduces the per-action minimum exactly.
    for s in safe_states:
        best = min(lat for lat in action_latency[s].values()
                   if lat is not None)
        if best != latency[s]:
            raise ArithmeticError(
                "internal error: latency LP is not a fixpoint at '%s'" % s)

    return LatencyResult(
        latency=latency,
        infinite_states=sorted(s for s in nonterm if s not in safe),
        action_latency=action_latency,
        canonical=canonical,
    )


# ---------------------------------------------------------------------------
# Certificate rendering
# ---------------------------------------------------------------------------


def _frac_str(frac):
    return str(frac) if frac.denominator != 1 else str(frac.numerator)


def audit(model):
    """Run the exact audit and return the JSON-serialisable certificate."""
    result = solve(model)

    certificates = []
    for s in sorted(result.canonical):
        optimal = set(result.optimal_actions[s])
        entries = []
        for a in sorted(result.expected[s]):
            ev = result.expected[s][a]
            if a in optimal:
                if a == result.canonical[s]:
                    reason = "selected_lexicographically_least_optimal"
                else:
                    reason = "optimal_but_identifier_not_lexicographically_least"
                is_optimal = True
            else:
                if ev < result.values[s]:
                    reason = "expected_value_below_state_value"
                else:
                    reason = "equal_value_but_no_optimal_continuation"
                is_optimal = False
            entries.append(
                {
                    "action": a,
                    "expected_value": _frac_str(ev),
                    "expected_value_num": ev.numerator,
                    "expected_value_den": ev.denominator,
                    "optimal": is_optimal,
                    "selected": a == result.canonical[s],
                    "reason": reason,
                }
            )
        certificates.append(
            {
                "state": s,
                "value": _frac_str(result.values[s]),
                "value_num": result.values[s].numerator,
                "value_den": result.values[s].denominator,
                "canonical_action": result.canonical[s],
                "actions": entries,
            }
        )

    v = result.max_rescue_probability
    out = {
        "status": "ok",
        "start_state": model.start,
        "max_rescue_probability": _frac_str(v),
        "max_rescue_probability_num": v.numerator,
        "max_rescue_probability_den": v.denominator,
        "canonical_policy": {s: result.canonical[s]
                             for s in sorted(result.canonical)},
        "state_values": {s: _frac_str(result.values[s])
                         for s in sorted(result.values)},
        "zero_value_states": result.zero_value_states,
        "terminal_unreachable_states": result.terminal_unreachable_states,
        "certificates": certificates,
    }
    if model.termination_latency:
        out["termination_latency"] = _latency_certificate(model, result)
    return out


def _latency_certificate(model, result):
    """Render the per-state termination-latency certificate section."""
    lat = solve_latency(model, result)

    states = []
    for s in sorted(result.canonical):
        best = lat.latency.get(s)  # None -> infinite minimum latency
        entries = []
        for a in sorted(lat.action_latency[s]):
            action_lat = lat.action_latency[s][a]
            selected = a == lat.canonical[s]
            if action_lat is None:
                entries.append(
                    {
                        "action": a,
                        "latency": "infinity",
                        "finite": False,
                        "selected": selected,
                        "reason": (
                            "selected_min_latency_lexicographically_least"
                            if selected
                            else "infinite_latency_non_terminating_loop"
                        ),
                    }
                )
            else:
                if selected:
                    reason = "selected_min_latency_lexicographically_least"
                elif action_lat == best:
                    reason = ("min_latency_but_identifier_not_"
                              "lexicographically_least")
                else:
                    reason = "latency_above_minimum"
                entries.append(
                    {
                        "action": a,
                        "latency": _frac_str(action_lat),
                        "latency_num": action_lat.numerator,
                        "latency_den": action_lat.denominator,
                        "finite": True,
                        "selected": selected,
                        "reason": reason,
                    }
                )
        value = result.values[s]
        state_entry = {
            "state": s,
            "rescue_probability": _frac_str(value),
            "rescue_probability_num": value.numerator,
            "rescue_probability_den": value.denominator,
            "canonical_action": lat.canonical[s],
            "actions": entries,
        }
        if best is None:
            state_entry["min_latency"] = "infinity"
            state_entry["finite"] = False
        else:
            state_entry["min_latency"] = _frac_str(best)
            state_entry["min_latency_num"] = best.numerator
            state_entry["min_latency_den"] = best.denominator
            state_entry["finite"] = True
        states.append(state_entry)

    # Minimum termination latency of the start state (0 when already
    # terminal, "infinity" when every maximal-rescue policy can loop).
    if model.start in model.rescued or model.start in model.lost:
        start_lat = Fraction(0)
    else:
        start_lat = lat.latency.get(model.start)

    section = {
        "objective": "min_expected_link_actions_to_terminal",
        "start_state": model.start,
        "canonical_policy": {s: lat.canonical[s]
                             for s in sorted(lat.canonical)},
        "infinite_latency_states": lat.infinite_states,
        "states": states,
    }
    if start_lat is None:
        section["start_min_latency"] = "infinity"
        section["start_finite"] = False
    else:
        section["start_min_latency"] = _frac_str(start_lat)
        section["start_min_latency_num"] = start_lat.numerator
        section["start_min_latency_den"] = start_lat.denominator
        section["start_finite"] = True
    return section
