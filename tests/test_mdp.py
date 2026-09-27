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


def thirds_delay_model():
    payload = thirds_model()
    payload["termination_delay_audit"] = True
    return payload


class TerminationDelayTest(unittest.TestCase):
    def test_disabled_request_has_no_delay_fields(self):
        result = audit(thirds_model())
        self.assertNotIn("termination_delay_audit", result)
        for cert in result["certificates"]:
            self.assertNotIn("min_termination_delay", cert)
            for a in cert["actions"]:
                self.assertNotIn("termination_delay", a)

    def test_simple_one_step_delay(self):
        result = audit(thirds_delay_model())
        block = result["termination_delay_audit"]
        self.assertTrue(block["enabled"])
        self.assertEqual(
            block["metric"],
            "expected_number_of_link_actions_before_any_terminal")
        self.assertEqual(block["state_min_termination_delays"], {"s0": "1"})
        cert = result["certificates"][0]
        self.assertEqual(cert["min_termination_delay"], "1")
        self.assertEqual(cert["min_termination_delay_num"], 1)
        self.assertEqual(cert["min_termination_delay_den"], 1)
        self.assertFalse(cert["min_termination_delay_infinite"])
        self.assertEqual(cert["canonical_delay_action"], "go")
        action = cert["actions"][0]
        self.assertEqual(action["termination_delay"], "1")
        self.assertTrue(action["delay_selected"])

    def test_faster_relay_beats_slow_direct(self):
        # direct: r with 1/3, retry with 2/3 -> geometric, exact delay 3
        # relay:  s -> m -> r, two deterministic link actions -> delay 2
        # both rescue with probability 1; the relay must be chosen.
        payload = {
            "states": ["s", "m", "r", "l"],
            "start": "s",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s": {
                    "direct": {"r": "1/3", "s": "2/3"},
                    "relay": {"m": "1"},
                },
                "m": {"fwd": {"r": "1"}},
            },
            "termination_delay_audit": True,
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "1")
        delays = result["termination_delay_audit"][
            "state_min_termination_delays"]
        self.assertEqual(delays, {"m": "1", "s": "2"})
        self.assertEqual(
            result["termination_delay_audit"]["minimum_delay_policy"],
            {"m": "fwd", "s": "relay"})
        cert = {c["state"]: c for c in result["certificates"]}["s"]
        by = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(by["direct"]["termination_delay"], "3")
        self.assertFalse(by["direct"]["delay_selected"])
        self.assertEqual(by["relay"]["termination_delay"], "2")
        self.assertTrue(by["relay"]["delay_selected"])
        # rescue-probability canonical policy is unchanged by the audit
        self.assertEqual(result["canonical_policy"]["s"], "direct")

    def test_faster_direct_beats_two_link_relay(self):
        # direct: r with 2/3, retry with 1/3 -> exact delay 3/2
        # relay:  s -> m -> r -> delay 2; the direct action wins on delay.
        payload = {
            "states": ["s", "m", "r", "l"],
            "start": "s",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s": {
                    "direct": {"r": "2/3", "s": "1/3"},
                    "relay": {"m": "1"},
                },
                "m": {"fwd": {"r": "1"}},
            },
            "termination_delay_audit": True,
        }
        result = audit(payload)
        cert = {c["state"]: c for c in result["certificates"]}["s"]
        by = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(by["direct"]["termination_delay"], "3/2")
        self.assertTrue(by["direct"]["delay_selected"])
        self.assertEqual(by["relay"]["termination_delay"], "2")
        self.assertFalse(by["relay"]["delay_selected"])

    def test_infinite_loop_candidate_is_marked_and_finite_policy_returned(self):
        # Both actions preserve the maximum rescue probability 1/2:
        #   a_trap -> r 1/2, k 1/2 ; k spins forever (positive-prob trap)
        #   d_safe -> r 1/2, l 1/2 (terminates after one link action)
        payload = {
            "states": ["s", "k", "r", "l"],
            "start": "s",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s": {
                    "a_trap": {"r": "1/2", "k": "1/2"},
                    "d_safe": {"r": "1/2", "l": "1/2"},
                },
                "k": {"spin": {"k": "1"}},
            },
            "termination_delay_audit": True,
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "1/2")
        block = result["termination_delay_audit"]
        self.assertEqual(block["state_min_termination_delays"],
                         {"k": "infinity", "s": "1"})
        self.assertEqual(block["infinite_delay_states"], ["k"])
        self.assertEqual(block["minimum_delay_policy"], {"s": "d_safe"})
        certs = {c["state"]: c for c in result["certificates"]}
        cert = certs["s"]
        by = {a["action"]: a for a in cert["actions"]}
        trap = by["a_trap"]
        self.assertTrue(trap["optimal"])
        self.assertEqual(trap["termination_delay"], "infinity")
        self.assertTrue(trap["termination_delay_infinite"])
        self.assertFalse(trap["delay_selected"])
        self.assertEqual(trap["delay_reason"],
                         "infinite_positive_probability_nonterminating_loop")
        evidence = trap["infinite_delay_evidence"]
        self.assertEqual(evidence["closed_class_states"], ["k"])
        self.assertEqual(evidence["walk_to_closed_class"][0]["probability"],
                         "1/2")
        self.assertEqual(evidence["walk_to_closed_class"][0]["to"], "k")
        self.assertEqual(
            evidence["stationary_actions"]["k"]["support"], ["k"])
        safe = by["d_safe"]
        self.assertEqual(safe["termination_delay"], "1")
        self.assertTrue(safe["delay_selected"])
        # the trap state itself has only an infinite candidate
        kentry = certs["k"]["actions"][0]
        self.assertEqual(kentry["termination_delay"], "infinity")
        self.assertIsNone(certs["k"]["canonical_delay_action"])

    def test_pure_closed_loop_all_delays_infinite(self):
        payload = {
            "states": ["s0", "k1", "k2", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {"drift": {"k1": "1"}},
                "k1": {"spin": {"k2": "1"}},
                "k2": {"spin": {"k1": "1"}},
            },
            "termination_delay_audit": True,
        }
        result = audit(payload)
        block = result["termination_delay_audit"]
        self.assertEqual(
            block["state_min_termination_delays"],
            {"k1": "infinity", "k2": "infinity", "s0": "infinity"})
        self.assertEqual(block["infinite_delay_states"],
                         ["k1", "k2", "s0"])
        self.assertEqual(block["minimum_delay_policy"], {})
        for cert in result["certificates"]:
            self.assertTrue(cert["min_termination_delay_infinite"])
            for a in cert["actions"]:
                self.assertEqual(a["termination_delay"], "infinity")
                self.assertIn("closed_class_states",
                              a["infinite_delay_evidence"])

    def test_zero_value_quit_vs_spin_prefers_terminating(self):
        payload = {
            "states": ["z", "r", "l"],
            "start": "z",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "z": {
                    "spin": {"z": "1"},
                    "quit": {"l": "1"},
                }
            },
            "termination_delay_audit": True,
        }
        result = audit(payload)
        self.assertEqual(result["max_rescue_probability"], "0")
        cert = result["certificates"][0]
        self.assertEqual(cert["min_termination_delay"], "1")
        self.assertEqual(cert["canonical_delay_action"], "quit")
        by = {a["action"]: a for a in cert["actions"]}
        self.assertEqual(by["quit"]["termination_delay"], "1")
        self.assertTrue(by["quit"]["delay_selected"])
        self.assertEqual(by["spin"]["termination_delay"], "infinity")

    def test_suboptimal_action_is_excluded_from_delay_audit(self):
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
            "termination_delay_audit": True,
        }
        result = audit(payload)
        entry = {a["action"]: a for a in result["certificates"][0]["actions"]}
        self.assertNotIn("termination_delay", entry["bad"])
        self.assertFalse(entry["bad"]["termination_delay_audited"])
        self.assertEqual(
            entry["bad"]["termination_delay_excluded_reason"],
            "action_does_not_preserve_max_rescue_probability")
        self.assertEqual(entry["good"]["termination_delay"], "1")

    def test_boolean_option_validation(self):
        raw = json.dumps(thirds_delay_model()).replace(
            '"termination_delay_audit": true',
            '"termination_delay_audit": "yes"')
        with self.assertRaises(mdp.AuditError) as ctx:
            mdp.load_and_validate(raw)
        err = [e for e in ctx.exception.errors
               if e["code"] == "INVALID_TYPE"
               and e["path"] == "$.termination_delay_audit"]
        self.assertEqual(len(err), 1)

    def test_option_false_equals_absent(self):
        off = dict(thirds_model(), termination_delay_audit=False)
        self.assertEqual(
            {k for k in audit(off)},
            {k for k in audit(thirds_model())})


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


def _policy_termination_delays(model, policy):
    """Exact expected number of link actions before ANY terminal under a
    fixed memoryless policy; Fraction where termination is almost sure,
    None where a terminal-free closed class is reachable (infinite)."""
    nonterm = [s for s in model.states
               if s not in model.rescued and s not in model.lost]
    terminals = model.rescued | model.lost
    # states that can reach a terminal at all under this policy
    can = set(terminals)
    changed = True
    while changed:
        changed = False
        for s in nonterm:
            if s not in can and any(
                    t in can for t in model.actions[s][policy[s]]):
                can.add(s)
                changed = True
    # almost-sure core: greatest fixpoint inside `can` whose whole support
    # stays in the core (geometric exits survive; positive-prob traps do not)
    good = {s for s in nonterm if s in can}
    while True:
        nxt = {s for s in good
               if all(t in terminals or t in good
                      for t in model.actions[s][policy[s]])}
        if nxt == good:
            break
        good = nxt
    idx = {s: i for i, s in enumerate(sorted(good))}
    n = len(idx)
    mat = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for s in good:
        i = idx[s]
        mat[i][i] = Fraction(1)
        mat[i][n] = Fraction(1)
        for t, p in model.actions[s][policy[s]].items():
            if t in idx:
                mat[i][idx[t]] -= p
    if n:
        for col in range(n):
            piv = next(r for r in range(col, n) if mat[r][col] != 0)
            mat[col], mat[piv] = mat[piv], mat[col]
            pv = mat[col][col]
            mat[col] = [v / pv for v in mat[col]]
            for r in range(n):
                if r != col and mat[r][col] != 0:
                    f = mat[r][col]
                    mat[r] = [a - f * b
                              for a, b in zip(mat[r], mat[col])]
    out = {}
    for s in nonterm:
        out[s] = mat[idx[s]][n] if s in idx else None
    return out


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

    def test_delays_against_brute_force_policy_enumeration(self):
        rng = random.Random(20260927)
        for trial in range(80):
            payload = _random_model(rng)
            payload["termination_delay_audit"] = True
            model = build(payload)
            result = mdp.solve(model)
            self.assertIsNotNone(result.delay)
            terminals = model.rescued | model.lost
            nonterm = [s for s in model.states if s not in terminals]
            act_lists = [sorted(model.actions[s]) for s in nonterm]
            policies = [{}]
            for s, acts in zip(nonterm, act_lists):
                policies = [dict(p, **{s: a}) for p in policies
                            for a in acts]
            best_vals = {s: result.values[s] for s in nonterm}
            delays_by_action = {s: {} for s in nonterm}
            for policy in policies:
                vals = _policy_values(model, policy)
                if any(vals[s] != best_vals[s] for s in nonterm):
                    continue  # does not preserve the max rescue vector
                dh = _policy_termination_delays(model, policy)
                for s in nonterm:
                    cur = delays_by_action[s].get(policy[s])
                    d = dh[s]
                    if cur is None or (d is not None and
                                       (cur[0] is None or d < cur[0])):
                        delays_by_action[s][policy[s]] = (d, policy)
            for s in nonterm:
                got = result.delay.min_delay[s]
                finite = [d for d, _ in delays_by_action[s].values()
                          if d is not None]
                want = min(finite) if finite else None
                self.assertEqual(
                    got, want,
                    "trial %d state %s min delay payload %s"
                    % (trial, s, payload))
                for a in result.optimal_actions[s]:
                    got_a = result.delay.action_delay[s][a]
                    want_a = delays_by_action[s][a][0]
                    self.assertEqual(
                        got_a, want_a,
                        "trial %d state %s action %s payload %s"
                        % (trial, s, a, payload))


if __name__ == "__main__":
    unittest.main()
