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
    # opt-in exact termination-delay audit (off unless requested)
    termination_delay_audit: bool = False


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

    # ---- termination-delay audit option (opt-in) --------------------------
    delay_audit = False
    raw_option = root.get("termination_delay_audit")
    if raw_option is not None:
        if isinstance(raw_option, bool):
            delay_audit = raw_option
        else:
            _err(errors, "$.termination_delay_audit", "INVALID_TYPE",
                 "'termination_delay_audit' must be a boolean")

    if errors:
        return None
    return Model(
        states=states,
        start=start,
        rescued=frozenset(rescued),
        lost=frozenset(lost),
        actions=actions,
        termination_delay_audit=delay_audit,
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
    delay: object = None  # DelayResult when the audit is enabled


@dataclass
class DelayResult:
    # controllable state -> exact minimum expected number of link actions
    # before reaching ANY terminal, among policies keeping the maximum
    # rescue probability (None at states where no such policy terminates
    # almost surely)
    min_delay: dict
    canonical: dict  # controllable state -> delay-optimal action id
    # controllable state -> {action: Fraction | None} exact expected
    # termination delay under the delay-optimal continuation with that
    # action committed; None marks a positive-probability non-terminating
    # closed loop (infinite delay)
    action_delay: dict
    # controllable state -> {action: sorted list of states certifying a
    # non-terminating closed loop reachable with positive probability}
    infinite_evidence: dict
    infinite_states: list  # states whose min delay is infinite


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

    result = SolveResult(
        values=values,
        canonical=canonical,
        optimal_actions=optimal_actions,
        expected=expected,
        zero_value_states=zero_value_states,
        terminal_unreachable_states=terminal_unreachable_states,
        max_rescue_probability=values[model.start],
    )
    if model.termination_delay_audit:
        result.delay = _solve_delay(
            model, nonterm, values, expected, optimal_actions)
    return result


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
# Exact minimum termination-delay audit
#
# The rescue-probability value vector is fixed first; only *truly optimal*
# actions (the actions of policies attaining the maximum rescue
# probability) are admissible here -- choosing any other would lower the
# fixed probability.  Among those we seek the policy minimising the
# expected number of link actions before ANY terminal (rescued or lost) is
# hit.  Everything is exact rational arithmetic; no simulation and no
# truncated value iteration is used.
#
# Winning region W (states from which an admissible policy terminates
# almost surely): the nested attractor fixed point
#     X_0 = every controllable state
#     R_0 = terminals
#     R   = existential attractor inside X (s joins once some admissible
#           action keeps its whole support in X union terminals and moves
#           positive probability onto R)
#     X <- R minus terminals, repeat until stable.
# Geometric-exit actions (e.g. s -> s with 2/3, r with 1/3) stay winning;
# states eliminated at the end are the polytrap: every admissible
# continuation traps them with positive probability.
#
# Least expected delay on W is the optimum of the exact
# stochastic-shortest-path linear program in max form
#     max  sum_s h(s)
#     s.t. h(s) <= 1 + sum_t P(s,a,t) h(t)
#          for every admissible action a whose support stays in
#          W union terminals,   h >= 0,   h(terminal) = 0.
# The feasible region is the set of sub-fixpoints; h = 0 is feasible and
# its optimum is the least Bellman fixpoint, i.e. the true minimum
# expected link-action count.  It is bounded exactly because W admits a
# proper (almost-sure terminating) policy; candidate actions that merely
# coordinate into loops only add redundant inequalities.  Solved exactly
# with the same rational Bland simplex used for the rescue LP.
#
# Per candidate action a at state s, COMMITMENT semantics apply: s is
# pinned to a on every revisit while every other state may re-optimise.
# The same LP is solved on the winning region of the pinned sub-MDP; it is
# unbounded exactly when the commitment traps with positive probability,
# in which case the delay is infinite and a locatable positive-probability
# walk into a bottom closed class (stationary actions never leave it, no
# terminal reachable) is attached as evidence.
# ---------------------------------------------------------------------------


def _allowed(preserving, forced, x):
    """Admissible actions at x under a commitment (the forced state may
    use only its committed action)."""
    if forced is not None and x == forced[0]:
        return [forced[1]]
    return preserving[x]


def _winning_region(model, nonterm, terminals, preserving, forced=None):
    """States from which almost-sure termination is forceable with
    admissible actions (``forced`` pins one state to one action)."""
    xset = set(nonterm)
    while True:
        reached = set(terminals)
        while True:
            added = set()
            for s in nonterm:
                if s not in xset or s in reached:
                    continue
                for a in _allowed(preserving, forced, s):
                    dist = model.actions[s][a]
                    if all(t in terminals or t in xset for t in dist) and \
                            any(t in reached for t in dist):
                        added.add(s)
                        break
            if not added:
                break
            reached |= added
        nxt = reached - set(terminals)
        if nxt == xset:
            return xset
        xset = nxt


def _delay_lp(model, win, terminals, preserving, forced=None):
    """Exact least expected termination delay on the winning region.

    Returns {state: Fraction}, or None when no admissible policy under the
    commitment terminates almost surely (the LP is unbounded)."""
    states = [s for s in model.states if s in win]
    idx = {s: i for i, s in enumerate(states)}
    rows = []
    for s in states:
        for a in _allowed(preserving, forced, s):
            dist = model.actions[s][a]
            if all(t in terminals or t in win for t in dist):
                rows.append((s, a))
    n = len(states)
    m = len(rows)
    A = [[Fraction(0)] * n for _ in range(m)]
    c = [Fraction(1)] * n
    b = [Fraction(1)] * m
    for i, (s, a) in enumerate(rows):
        A[i][idx[s]] = Fraction(1)
        for t, p in model.actions[s][a].items():
            if t in idx:
                A[i][idx[t]] -= p
    try:
        x, _, _ = _simplex_max(c, A, b)
    except ArithmeticError:
        return None  # unbounded: positive probability of never ending
    h = {s: x[idx[s]] for s in states}

    # Self-check: h is the exact least Bellman fixpoint over usable
    # admissible actions, computed with rational arithmetic.
    for s in states:
        best = None
        for a in _allowed(preserving, forced, s):
            dist = model.actions[s][a]
            if not all(t in terminals or t in win for t in dist):
                continue
            d = Fraction(1)
            for t, p in dist.items():
                if t in h:
                    d += p * h[t]
            if best is None or d < best:
                best = d
        if best != h[s]:
            raise ArithmeticError(
                "internal error: delay vector is not a fixpoint at '%s'" % s)
    return h


def _parent_depth(parent, entry, state):
    d = 0
    cur = state
    while cur != entry:
        if parent[cur] is None:
            return d
        cur = parent[cur][0]
        d += 1
    return d


def _closed_class_evidence(model, nonterm, terminals, preserving, win,
                           forced, entry):
    """Locatable evidence of positive-probability non-termination: a
    positive-probability admissible walk from ``entry`` into a bottom
    closed class whose stationary admissible actions never leave it."""
    region = {s for s in nonterm if s not in win}

    # States reachable from entry without passing through a terminal.
    reachable = {entry}
    parent = {entry: None}
    stack = [entry]
    while stack:
        x = stack.pop()
        for a in _allowed(preserving, forced, x):
            for t in model.actions[x][a]:
                if t in region and t not in reachable:
                    reachable.add(t)
                    parent[t] = (x, a)
                    stack.append(t)

    # Greatest sub-core in which every state keeps an allowed action whose
    # entire support stays inside the core (a closed end component).
    core = set(reachable)
    while True:
        stay = {}
        nxt = set()
        for x in core:
            for a in _allowed(preserving, forced, x):
                if all(t in core for t in model.actions[x][a]):
                    stay[x] = a
                    nxt.add(x)
                    break
        if nxt == core:
            break
        core = nxt

    # Bottom SCC of the graph induced by the staying actions: none of its
    # stationary actions ever moves probability outside the SCC.
    succ = {x: list(model.actions[x][stay[x]]) for x in core}
    index = {}
    low = {}
    stack_t = []
    on = set()
    sccs = []
    counter = [0]

    def strongconnect(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack_t.append(v)
        on.add(v)
        for w in succ[v]:
            if w not in index:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack_t.pop()
                on.discard(w)
                comp.add(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in sorted(core):
        if v not in index:
            strongconnect(v)
    scc_of = {x: i for i, comp in enumerate(sccs) for x in comp}
    bottom = [i for i, comp in enumerate(sccs)
              if all(scc_of[w] == i for x in comp for w in succ[x])]
    if not core or not bottom:
        raise ArithmeticError(
            "internal error: no closed class found in non-terminating "
            "region from '%s'" % entry)
    closed_states = sccs[bottom[0]]

    # Walk from entry to the nearest class state via BFS parents.
    goal = min(closed_states, key=lambda z: _parent_depth(parent, entry, z))
    path = []
    cur = goal
    while cur != entry:
        prev, a = parent[cur]
        path.append({
            "from": prev,
            "action": a,
            "to": cur,
            "probability": _frac_str(model.actions[prev][a][cur]),
        })
        cur = prev
    path.reverse()
    return {
        "non_terminating_region_states": sorted(region),
        "walk_to_closed_class": path,
        "closed_class_states": sorted(closed_states),
        "stationary_actions": {
            x: {
                "action": stay[x],
                "support": sorted(model.actions[x][stay[x]]),
            }
            for x in sorted(closed_states)
        },
        "explanation": "under the commitment every state of the non-"
                       "terminating region keeps positive probability "
                       "inside it; the exhibited walk reaches a bottom "
                       "closed class with strictly positive probability, "
                       "and the stationary action of every class state "
                       "keeps its whole transition support inside the "
                       "class, so no terminal is ever reached and the "
                       "expected termination delay is infinite",
    }


def _committed_delay(model, nonterm, terminals, preserving, s, a):
    """Exact expected termination delay when s is pinned to action a and
    every other state follows the best delay-preserving continuation.
    Returns (Fraction | None, evidence | None)."""
    forced = (s, a)
    win = _winning_region(model, nonterm, terminals, preserving, forced)
    if s not in win:
        entry = next(t for t in model.actions[s][a]
                     if t not in terminals and t not in win)
        evidence = _closed_class_evidence(
            model, nonterm, terminals, preserving, win, forced, entry)
        first = {
            "from": s,
            "action": a,
            "to": entry,
            "probability": _frac_str(model.actions[s][a][entry]),
        }
        evidence["walk_to_closed_class"] = [first] + evidence[
            "walk_to_closed_class"]
        evidence["committed_first_step"] = first
        return None, evidence
    h = _delay_lp(model, win, terminals, preserving, forced)
    if h is None:
        # s is in the existential winning region yet the LP is unbounded:
        # defensive -- treat as non-terminating and build evidence anyway.
        entry = next((t for t in model.actions[s][a]
                      if t not in terminals and t not in win), None)
        if entry is None:
            raise ArithmeticError(
                "internal error: delay LP unbounded at winning state '%s'"
                % s)
        evidence = _closed_class_evidence(
            model, nonterm, terminals, preserving, win, forced, entry)
        return None, evidence
    return h[s], None


def _solve_delay(model, nonterm, values, expected, optimal_actions):
    terminals = set(model.rescued) | set(model.lost)
    preserving = {s: sorted(optimal_actions[s]) for s in nonterm}

    win = _winning_region(model, nonterm, terminals, preserving)
    h_free = _delay_lp(model, win, terminals, preserving) if win else {}
    if h_free is None:
        raise ArithmeticError(
            "internal error: delay LP unbounded on the almost-sure "
            "winning core")

    action_delay = {s: {} for s in nonterm}
    evidence = {s: {} for s in nonterm}
    min_delay = {}
    canonical = {}
    for s in nonterm:
        min_delay[s] = h_free.get(s)  # None == infinite
        canonical[s] = None
        best_finite = None
        for a in preserving[s]:
            d, ev = _committed_delay(
                model, nonterm, terminals, preserving, s, a)
            action_delay[s][a] = d
            if ev is not None:
                evidence[s][a] = ev
            elif best_finite is None or d < best_finite[1] or \
                    (d == best_finite[1] and a < best_finite[0]):
                best_finite = (a, d)
        if best_finite is not None:
            canonical[s] = best_finite[0]

    return DelayResult(
        min_delay=min_delay,
        canonical=canonical,
        action_delay=action_delay,
        infinite_evidence=evidence,
        infinite_states=sorted(s for s in nonterm if min_delay[s] is None),
    )


# ---------------------------------------------------------------------------
# Certificate rendering
# ---------------------------------------------------------------------------


def _frac_str(frac):
    return str(frac) if frac.denominator != 1 else str(frac.numerator)


def _delay_frac(d):
    if d is None:
        return None
    return _frac_str(d)


def audit(model):
    """Run the exact audit and return the JSON-serialisable certificate."""
    result = solve(model)
    delay = result.delay

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
            entry = {
                "action": a,
                "expected_value": _frac_str(ev),
                "expected_value_num": ev.numerator,
                "expected_value_den": ev.denominator,
                "optimal": is_optimal,
                "selected": a == result.canonical[s],
                "reason": reason,
            }
            if delay is not None:
                d = delay.action_delay[s].get(a)
                if a not in optimal:
                    entry["termination_delay_audited"] = False
                    entry["termination_delay_excluded_reason"] = (
                        "action_does_not_preserve_max_rescue_probability")
                elif d is None:
                    entry["termination_delay"] = "infinity"
                    entry["termination_delay_infinite"] = True
                    entry["delay_selected"] = False
                    entry["delay_reason"] = (
                        "infinite_positive_probability_nonterminating_loop")
                    entry["infinite_delay_evidence"] = \
                        delay.infinite_evidence[s][a]
                else:
                    entry["termination_delay"] = _frac_str(d)
                    entry["termination_delay_num"] = d.numerator
                    entry["termination_delay_den"] = d.denominator
                    entry["termination_delay_infinite"] = False
                    if a == delay.canonical[s]:
                        entry["delay_selected"] = True
                        entry["delay_reason"] = (
                            "selected_least_expected_termination_delay")
                    else:
                        entry["delay_selected"] = False
                        entry["delay_reason"] = (
                            "higher_expected_termination_delay")
            entries.append(entry)
        state_cert = {
            "state": s,
            "value": _frac_str(result.values[s]),
            "value_num": result.values[s].numerator,
            "value_den": result.values[s].denominator,
            "canonical_action": result.canonical[s],
            "actions": entries,
        }
        if delay is not None:
            md = delay.min_delay[s]
            state_cert["min_termination_delay"] = (
                "infinity" if md is None else _frac_str(md))
            state_cert["min_termination_delay_infinite"] = md is None
            if md is not None:
                state_cert["min_termination_delay_num"] = md.numerator
                state_cert["min_termination_delay_den"] = md.denominator
            state_cert["canonical_delay_action"] = delay.canonical[s]
        certificates.append(state_cert)

    v = result.max_rescue_probability
    payload = {
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
    if delay is not None:
        payload["termination_delay_audit"] = {
            "enabled": True,
            "metric": "expected_number_of_link_actions_before_any_terminal",
            "subject_to": "max_rescue_probability_preserved",
            "state_min_termination_delays": {
                s: ("infinity" if delay.min_delay[s] is None
                    else _frac_str(delay.min_delay[s]))
                for s in sorted(delay.min_delay)
            },
            "infinite_delay_states": delay.infinite_states,
            "minimum_delay_policy": {
                s: delay.canonical[s]
                for s in sorted(delay.canonical)
                if delay.canonical[s] is not None
            },
        }
    return payload
