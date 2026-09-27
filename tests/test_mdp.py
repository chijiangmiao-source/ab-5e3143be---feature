"""Unit tests for the exact reachability audit.

The brute-force cross-check enumerates every memoryless policy and solves
its Markov chain exactly with rational Gaussian elimination.  That
machinery lives ONLY in the tests; the service itself never enumerates
policies, never iterates values to a tolerance and never uses floats.
"""

import json
import random
import unittest
from fractions import Fraction

from app import mdp


def build(payload):
    """Validate a model dict, asserting success, and return the Model."""
    return mdp.load_and_validate(json.dumps(payload))


def audit(payload):
    return mdp.audit(build(payload))


def thirds_model():
    return {
        "states": ["s0", "r", "l"],
        "start": "s0",
        "terminals": {"rescued": ["r"], "lost": ["l"]},
        "actions": {
            "s0": {
                "go": {"r": "1/3", "l": "2/3"},
            }
        },
    }


class RequiredScenariosTest(unittest.TestCase):
    def test_must_rescue(self):
        result = audit(
            {
                "states": ["s0", "r", "l"],
                "start": "s0",
                "terminals": {"rescued": ["r"], "lost": ["l"]},
                "actions": {"s0": {"go": {"r": "1"}}},
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["max_rescue_probability"], "1")
        self.assertEqual(result["canonical_policy"], {"s0": "go"})
        cert = result["certificates"][0]
        self.assertEqual(cert["state"], "s0")
        self.assertEqual(cert["value"], "1")
        self.assertEqual(cert["actions"][0]["expected_value"], "1")
        self.assertTrue(cert["actions"][0]["selected"])

    def test_thirds_exact_fraction(self):
        result = audit(thirds_model())
        self.assertEqual(result["max_rescue_probability"], "1/3")
        self.assertEqual(result["max_rescue_probability_num"], 1)
        self.assertEqual(result["max_rescue_probability_den"], 3)
        cert = result["certificates"][0]
        self.assertEqual(cert["value"], "1/3")
        self.assertEqual(cert["actions"][0]["expected_value"], "1/3")

    def test_equal_value_actions_pick_lexicographically_least(self):
        payload = {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {
                    "zeta": {"r": "1/2", "l": "1/2"},
                    "alpha": {"r": "1/2", "l": "1/2"},
                    "mid": {"r": "1/2", "l": "1/2"},
                }
            },
        }
        result = audit(payload)
        self.assertEqual(result["canonical_policy"], {"s0": "alpha"})
        cert = result["certificates"][0]
        expected_values = {a["action"]: a["expected_value"] for a in cert["actions"]}
        self.assertEqual(expected_values, {"alpha": "1/2", "mid": "1/2", "zeta": "1/2"})
        for entry in cert["actions"]:
            self.assertTrue(entry["optimal"])
            self.assertEqual(entry["selected"], entry["action"] == "alpha")

    def test_closed_loop_zero_rescue(self):
        payload = {
            "states": ["s0", "k1", "k2", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {"drift": {"k1": "1"}},
                "k1": {"spin": {"k2": "1"}},
                "k2": {"spin": {"k1": "1"}},
            },
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "0")
        self.assertEqual(result["zero_value_states"], ["k1", "k2", "s0"])
        self.assertEqual(result["terminal_unreachable_states"], ["k1", "k2", "s0"])
        values = {c["state"]: c["value"] for c in result["certificates"]}
        self.assertEqual(values, {"k1": "0", "k2": "0", "s0": "0"})

    def test_start_terminal(self):
        result = audit(
            {
                "states": ["s0", "r"],
                "start": "r",
                "terminals": {"rescued": ["r"], "lost": []},
                "actions": {"s0": {"go": {"r": "1"}}},
            }
        )
        self.assertEqual(result["max_rescue_probability"], "1")


class SolverCorrectnessTest(unittest.TestCase):
    def test_bellman_equal_but_not_optimal_is_excluded(self):
        # a2 is Bellman-equal (its expected value equals the state value)
        # but committing to it traps the process in the s1<->s2 loop, so
        # it must NOT be reported as optimal.
        payload = {
            "states": ["s1", "s2", "r", "l"],
            "start": "s1",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s1": {"a1": {"s2": "1"}},
                "s2": {
                    "a2": {"s1": "1"},
                    "b2": {"r": "1/2", "l": "1/2"},
                },
            },
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "1/2")
        self.assertEqual(result["canonical_policy"], {"s1": "a1", "s2": "b2"})
        cert = {c["state"]: c for c in result["certificates"]}["s2"]
        by_action = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(by_action["a2"]["expected_value"], "1/2")
        self.assertFalse(by_action["a2"]["optimal"])
        self.assertEqual(by_action["a2"]["reason"],
                         "equal_value_but_no_optimal_continuation")
        self.assertTrue(by_action["b2"]["optimal"])
        self.assertTrue(by_action["b2"]["selected"])

    def test_suboptimal_action_reason(self):
        payload = {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {
                    "good": {"r": "2/3", "l": "1/3"},
                    "bad": {"r": "1/3", "l": "2/3"},
                }
            },
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "2/3")
        cert = result["certificates"][0]
        by_action = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(by_action["bad"]["reason"],
                         "expected_value_below_state_value")
        self.assertFalse(by_action["bad"]["optimal"])

    def test_multistep_exact_fraction(self):
        # s0 -> s1 (1/2) / l (1/2); s1 -> r (2/3) / s0 (1/3)
        # v0 = v1/2, v1 = 2/3 + v0/3  =>  v0 = 2/5, v1 = 4/5
        payload = {
            "states": ["s0", "s1", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {"go": {"s1": "1/2", "l": "1/2"}},
                "s1": {"go": {"r": "2/3", "s0": "1/3"}},
            },
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "2/5")
        values = {c["state"]: c["value"] for c in result["certificates"]}
        self.assertEqual(values["s1"], "4/5")

    def test_self_loop_with_exit(self):
        # v = 1/3 * 1 + 2/3 * v  =>  v = 1
        payload = {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {"s0": {"persist": {"r": "1/3", "s0": "2/3"}}},
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "1")


class TerminationLatencyTest(unittest.TestCase):
    def relay_model(self):
        # a_direct reaches r with probability 1 but is slow: its exact
        # action-value is 1 + (2/3) d(s0) = 7/3 link actions; z_relay
        # reaches r via s1 in exactly 2.  The rescue-probability canonical
        # policy picks a_direct (lexicographic tie-break); the latency
        # audit must pick z_relay instead.
        return {
            "states": ["s0", "s1", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "termination_latency_audit": True,
            "actions": {
                "s0": {
                    "a_direct": {"s0": "2/3", "r": "1/3"},
                    "z_relay": {"s1": "1"},
                },
                "s1": {"go": {"r": "1"}},
            },
        }

    def loop_candidate_model(self):
        # risky keeps the maximum rescue probability 1/2 but with
        # probability 1/2 falls into the non-terminating k1 loop.
        return {
            "states": ["s0", "k1", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "termination_latency_audit": True,
            "actions": {
                "s0": {
                    "risky": {"r": "1/2", "k1": "1/2"},
                    "safe": {"r": "1/2", "l": "1/2"},
                },
                "k1": {"spin": {"k1": "1"}},
            },
        }

    def test_faster_relay_beats_slower_direct(self):
        result = audit(self.relay_model())
        self.assertEqual(result["max_rescue_probability"], "1")
        # stage 1 (rescue probability only) keeps the lexicographic choice
        self.assertEqual(result["canonical_policy"], {"s0": "a_direct", "s1": "go"})
        lat = result["termination_latency"]
        self.assertEqual(lat["objective"], "min_expected_link_actions_to_terminal")
        self.assertEqual(lat["start_min_latency"], "2")
        self.assertEqual(lat["start_min_latency_num"], 2)
        # stage 2 re-ranks by exact latency: the relay is faster
        self.assertEqual(lat["canonical_policy"], {"s0": "z_relay", "s1": "go"})
        self.assertEqual(lat["infinite_latency_states"], [])
        cert = {c["state"]: c for c in lat["states"]}["s0"]
        self.assertEqual(cert["rescue_probability"], "1")
        self.assertEqual(cert["min_latency"], "2")
        self.assertTrue(cert["finite"])
        by_action = {a["action"]: a for a in cert["actions"]}
        # exact action-value 1 + (2/3) * d(s0) = 1 + (2/3) * 2 = 7/3
        self.assertEqual(by_action["a_direct"]["latency"], "7/3")
        self.assertEqual(by_action["a_direct"]["latency_num"], 7)
        self.assertEqual(by_action["a_direct"]["latency_den"], 3)
        self.assertEqual(by_action["a_direct"]["reason"], "latency_above_minimum")
        self.assertFalse(by_action["a_direct"]["selected"])
        self.assertEqual(by_action["z_relay"]["latency"], "2")
        self.assertTrue(by_action["z_relay"]["selected"])
        self.assertEqual(
            by_action["z_relay"]["reason"],
            "selected_min_latency_lexicographically_least",
        )

    def test_infinite_loop_candidate_is_marked_and_finite_policy_returned(self):
        result = audit(self.loop_candidate_model())
        self.assertEqual(result["max_rescue_probability"], "1/2")
        lat = result["termination_latency"]
        self.assertEqual(lat["infinite_latency_states"], ["k1"])
        self.assertEqual(lat["start_min_latency"], "1")
        self.assertEqual(lat["canonical_policy"]["s0"], "safe")
        cert = {c["state"]: c for c in lat["states"]}["s0"]
        self.assertEqual(cert["rescue_probability"], "1/2")
        self.assertEqual(cert["min_latency"], "1")
        by_action = {a["action"]: a for a in cert["actions"]}
        risky = by_action["risky"]
        self.assertEqual(risky["latency"], "infinity")
        self.assertFalse(risky["finite"])
        self.assertFalse(risky["selected"])
        self.assertEqual(risky["reason"], "infinite_latency_non_terminating_loop")
        safe = by_action["safe"]
        self.assertEqual(safe["latency"], "1")
        self.assertTrue(safe["finite"])
        self.assertTrue(safe["selected"])
        # the doomed loop state itself is certified with infinite latency
        k1 = {c["state"]: c for c in lat["states"]}["k1"]
        self.assertEqual(k1["min_latency"], "infinity")
        self.assertFalse(k1["finite"])
        self.assertEqual(k1["canonical_action"], "spin")
        self.assertEqual(k1["actions"][0]["latency"], "infinity")

    def test_fractional_latency_is_exact(self):
        # d0 = 1 + (2/3) d1, d1 = 1  =>  d0 = 5/3
        result = audit(
            {
                "states": ["s0", "s1", "r", "l"],
                "start": "s0",
                "terminals": {"rescued": ["r"], "lost": ["l"]},
                "termination_latency_audit": True,
                "actions": {
                    "s0": {"go": {"r": "1/3", "s1": "2/3"}},
                    "s1": {"go": {"r": "1"}},
                },
            }
        )
        cert = {c["state"]: c for c in
                result["termination_latency"]["states"]}["s0"]
        self.assertEqual(cert["min_latency"], "5/3")
        self.assertEqual(cert["min_latency_num"], 5)
        self.assertEqual(cert["min_latency_den"], 3)

    def test_equal_latency_actions_pick_lexicographically_least(self):
        result = audit(
            {
                "states": ["s0", "r", "l"],
                "start": "s0",
                "terminals": {"rescued": ["r"], "lost": ["l"]},
                "termination_latency_audit": True,
                "actions": {
                    "s0": {
                        "zeta": {"r": "1"},
                        "alpha": {"r": "1"},
                    }
                },
            }
        )
        lat = result["termination_latency"]
        self.assertEqual(lat["canonical_policy"], {"s0": "alpha"})
        cert = lat["states"][0]
        by_action = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(by_action["zeta"]["latency"], "1")
        self.assertEqual(
            by_action["zeta"]["reason"],
            "min_latency_but_identifier_not_lexicographically_least",
        )

    def test_option_disabled_keeps_response_unchanged(self):
        result = audit(thirds_model())
        self.assertNotIn("termination_latency", result)

    def test_option_false_keeps_response_unchanged(self):
        payload = thirds_model()
        payload["termination_latency_audit"] = False
        result = audit(payload)
        self.assertNotIn("termination_latency", result)

    def test_flag_must_be_boolean(self):
        payload = thirds_model()
        payload["termination_latency_audit"] = "yes"
        with self.assertRaises(mdp.AuditError) as ctx:
            mdp.load_and_validate(json.dumps(payload))
        err = ctx.exception.errors[0]
        self.assertEqual(err["code"], "INVALID_TYPE")
        self.assertEqual(err["path"], "$.termination_latency_audit")


class ValidationTest(unittest.TestCase):
    def assert_error(self, payload, code, raw=None):
        with self.assertRaises(mdp.AuditError) as ctx:
            mdp.load_and_validate(raw if raw is not None else json.dumps(payload))
        codes = [e["code"] for e in ctx.exception.errors]
        self.assertIn(code, codes)
        for e in ctx.exception.errors:
            self.assertTrue(e["path"].startswith("$"))
            self.assertTrue(e["message"])
        return ctx.exception.errors

    def test_unknown_transition_target(self):
        payload = thirds_model()
        payload["actions"]["s0"]["go"]["ghost"] = "1/3"
        payload["actions"]["s0"]["go"]["r"] = "1/3"
        payload["actions"]["s0"]["go"]["l"] = "1/3"
        errors = self.assert_error(payload, "UNKNOWN_STATE")
        err = [e for e in errors if e["code"] == "UNKNOWN_STATE"][0]
        self.assertIn("ghost", err["path"])

    def test_unknown_start(self):
        payload = thirds_model()
        payload["start"] = "nowhere"
        self.assert_error(payload, "UNKNOWN_STATE")

    def test_missing_actions(self):
        payload = thirds_model()
        payload["states"].append("dangling")
        self.assert_error(payload, "MISSING_ACTIONS")

    def test_terminal_with_actions(self):
        payload = thirds_model()
        payload["actions"]["r"] = {"stay": {"r": "1"}}
        self.assert_error(payload, "TERMINAL_HAS_ACTIONS")

    def test_probability_sum_not_one(self):
        payload = thirds_model()
        payload["actions"]["s0"]["go"] = {"r": "1/2", "l": "1/3"}
        errors = self.assert_error(payload, "PROBABILITY_SUM")
        err = [e for e in errors if e["code"] == "PROBABILITY_SUM"][0]
        self.assertIn("5/6", err["message"])

    def test_non_reduced_fraction(self):
        payload = thirds_model()
        payload["actions"]["s0"]["go"] = {"r": "2/6", "l": "4/6"}
        self.assert_error(payload, "INVALID_PROBABILITY")

    def test_float_probability_rejected(self):
        raw = json.dumps(thirds_model()).replace('"1/3"', "0.5").replace(
            '"2/3"', '"1/2"')
        self.assert_error({}, "INVALID_PROBABILITY", raw=raw)

    def test_duplicate_state(self):
        payload = thirds_model()
        payload["states"].append("s0")
        self.assert_error(payload, "DUPLICATE_STATE")

    def test_duplicate_json_key(self):
        raw = (
            '{"states": ["s0", "r", "l"], "start": "s0",'
            ' "terminals": {"rescued": ["r"], "lost": ["l"]},'
            ' "actions": {"s0": {"go": {"r": "1/3", "r": "1/3", "l": "1/3"}}}}'
        )
        self.assert_error({}, "DUPLICATE_KEY", raw=raw)

    def test_terminal_overlap(self):
        payload = thirds_model()
        payload["terminals"]["lost"].append("r")
        self.assert_error(payload, "DUPLICATE_TERMINAL")

    def test_invalid_json(self):
        with self.assertRaises(mdp.AuditError) as ctx:
            mdp.load_and_validate("{not json")
        self.assertEqual(ctx.exception.errors[0]["code"], "INVALID_JSON")


# ---------------------------------------------------------------------------
# Brute-force cross-check (test-only): enumerate all memoryless policies and
# solve each induced Markov chain exactly.
# ---------------------------------------------------------------------------


def _policy_values(model, policy):
    nonterm = [s for s in model.states
               if s not in model.rescued and s not in model.lost]
    # states that can reach a rescued terminal under this policy
    reach = set(model.rescued)
    changed = True
    while changed:
        changed = False
        for s in nonterm:
            if s not in reach and any(
                t in reach for t in model.actions[s][policy[s]]
            ):
                reach.add(s)
                changed = True
    solvable = [s for s in nonterm if s in reach]
    idx = {s: i for i, s in enumerate(solvable)}
    n = len(solvable)
    # (I - Q) x = beta via exact Gaussian elimination
    mat = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for s in solvable:
        i = idx[s]
        mat[i][i] = Fraction(1)
        for t, p in model.actions[s][policy[s]].items():
            if t in model.rescued:
                mat[i][n] += p
            elif t in idx:
                mat[i][idx[t]] -= p
    for col in range(n):
        piv = next(r for r in range(col, n) if mat[r][col] != 0)
        mat[col], mat[piv] = mat[piv], mat[col]
        pv = mat[col][col]
        mat[col] = [v / pv for v in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[col])]
    values = {s: (mat[idx[s]][n] if s in idx else Fraction(0)) for s in nonterm}
    for r in model.rescued:
        values[r] = Fraction(1)
    for l in model.lost:
        values[l] = Fraction(0)
    return values


def _random_model(rng):
    n_states = rng.randint(1, 3)
    ctr = ["s%d" % i for i in range(n_states)]
    states = ctr + ["r", "l"]
    actions = {}
    for s in ctr:
        n_actions = rng.randint(1, 2)
        amap = {}
        for k in range(n_actions):
            targets = [t for t in states if rng.random() < 0.7] or ["l"]
            weights = [rng.randint(1, 5) for _ in targets]
            total = sum(weights)
            amap["a%d" % k] = {
                t: str(Fraction(w, total)) for t, w in zip(targets, weights)
            }
        actions[s] = amap
    return {
        "states": states,
        "start": "s0",
        "terminals": {"rescued": ["r"], "lost": ["l"]},
        "actions": actions,
    }


class CrossCheckTest(unittest.TestCase):
    def test_against_brute_force_policy_enumeration(self):
        rng = random.Random(20260926)
        for trial in range(60):
            payload = _random_model(rng)
            model = build(payload)
            result = mdp.solve(model)
            nonterm = [s for s in model.states
                       if s not in model.rescued and s not in model.lost]
            act_lists = [sorted(model.actions[s]) for s in nonterm]

            # enumerate every policy and record its exact value vector
            policies = [[]]
            for acts in act_lists:
                policies = [p + [a] for p in policies for a in acts]
            evaluated = []
            for combo in policies:
                policy = dict(zip(nonterm, combo))
                vals = _policy_values(model, policy)
                evaluated.append((policy, tuple(vals[s] for s in nonterm)))
            best = [Fraction(-1)] * len(nonterm)
            for _, vec in evaluated:
                best = [max(a, b) for a, b in zip(best, vec)]
            for i, s in enumerate(nonterm):
                self.assertEqual(
                    result.values[s], best[i],
                    "trial %d state %s payload %s" % (trial, s, payload),
                )
            # canonical action: least action used by some policy attaining
            # the componentwise optimum everywhere
            optimal_policies = [
                policy for policy, vec in evaluated
                if all(vec[j] == best[j] for j in range(len(nonterm)))
            ]
            self.assertTrue(optimal_policies)
            for i, s in enumerate(nonterm):
                optimal_here = {p[s] for p in optimal_policies}
                self.assertEqual(
                    set(result.optimal_actions[s]), optimal_here,
                    "trial %d state %s" % (trial, s),
                )
                self.assertEqual(result.canonical[s], min(optimal_here))


def _policy_termination_latency(model, policy):
    """Exact expected number of link actions until a terminal is hit under
    ``policy``; ``None`` (infinity) for states whose chain can avoid every
    terminal forever with positive probability.  Test-only machinery."""
    nonterm = [s for s in model.states
               if s not in model.rescued and s not in model.lost]
    terminals = set(model.rescued) | set(model.lost)
    succ = {s: set(model.actions[s][policy[s]]) for s in nonterm}

    # transitive closure of the chain graph
    reach = {}
    for s in nonterm:
        seen = set()
        stack = list(succ[s])
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            if u in succ:
                stack.extend(succ[u])
        reach[s] = seen

    # t sits inside a terminal-free closed class iff every state reachable
    # from t reaches t back and no terminal is reachable from t.
    def traps(t):
        return (
            not (reach[t] & terminals)
            and all(t in reach[u] for u in reach[t] if u in succ)
        )

    infinite = set()
    for s in nonterm:
        candidates = [s] + sorted(reach[s])
        if any(u in succ and traps(u) for u in candidates):
            infinite.add(s)

    finite = [s for s in nonterm if s not in infinite]
    idx = {s: i for i, s in enumerate(finite)}
    n = len(finite)
    # (I - Q) x = 1 via exact Gaussian elimination
    mat = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for s in finite:
        i = idx[s]
        mat[i][i] = Fraction(1)
        mat[i][n] = Fraction(1)
        for t, p in model.actions[s][policy[s]].items():
            if t in idx:
                mat[i][idx[t]] -= p
    for col in range(n):
        piv = next(r for r in range(col, n) if mat[r][col] != 0)
        mat[col], mat[piv] = mat[piv], mat[col]
        pv = mat[col][col]
        mat[col] = [v / pv for v in mat[col]]
        for r in range(n):
            if r != col and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[col])]
    out = {}
    for s in nonterm:
        out[s] = mat[idx[s]][n] if s in idx else None
    return out


class LatencyCrossCheckTest(unittest.TestCase):
    def test_against_brute_force_policy_enumeration(self):
        rng = random.Random(20260927)
        for trial in range(60):
            payload = _random_model(rng)
            model = build(payload)
            result = mdp.solve(model)
            lat = mdp.solve_latency(model, result)
            nonterm = [s for s in model.states
                       if s not in model.rescued and s not in model.lost]
            act_lists = [result.optimal_actions[s] for s in nonterm]

            # enumerate every policy over truly optimal actions
            policies = [[]]
            for acts in act_lists:
                policies = [p + [a] for p in policies for a in acts]
            evaluated = []
            for combo in policies:
                policy = dict(zip(nonterm, combo))
                evaluated.append((policy,
                                  _policy_termination_latency(model, policy)))
            best = {s: None for s in nonterm}  # None means +infinity
            for _, values in evaluated:
                for s in nonterm:
                    v = values[s]
                    if v is not None and (best[s] is None or v < best[s]):
                        best[s] = v
            per_state_actions = {s: set() for s in nonterm}
            for policy, values in evaluated:
                for s in nonterm:
                    if best[s] is not None and values[s] == best[s]:
                        per_state_actions[s].add(policy[s])
            for s in nonterm:
                expected = best[s]
                actual = lat.latency.get(s)
                self.assertEqual(
                    actual, expected,
                    "trial %d state %s payload %s" % (trial, s, payload),
                )
                if expected is None:
                    # every truly optimal action loops: canonical falls
                    # back to the lexicographically least candidate
                    self.assertIn(s, lat.infinite_states)
                    self.assertEqual(
                        lat.canonical[s], min(result.optimal_actions[s]))
                    self.assertTrue(all(
                        v is None
                        for v in lat.action_latency[s].values()))
                else:
                    self.assertNotIn(s, lat.infinite_states)
                    self.assertEqual(
                        lat.canonical[s], min(per_state_actions[s]))
                    # per-action certificate matches 1 + E[d(successor)]
                    terminals = set(model.rescued) | set(model.lost)
                    for a, action_lat in lat.action_latency[s].items():
                        dist = model.actions[s][a]
                        if all(t in terminals or t in lat.latency
                               for t in dist):
                            want = Fraction(1) + sum(
                                p * lat.latency[t]
                                for t, p in dist.items()
                                if t not in terminals)
                            self.assertEqual(action_lat, want)
                        else:
                            self.assertIsNone(action_lat)


if __name__ == "__main__":
    unittest.main()
