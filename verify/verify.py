"""Compose ``verify`` service: code tests + build checks + API/HTTP smoke.

Runs three stages and exits non-zero if any of them fails:

1. code tests   -- the unit test suite (exact solver, validation, and the
                   brute-force cross-check);
2. build checks -- every source file byte-compiles and the packages import;
3. API smoke    -- the four required models are POSTed to a live service:
                   must-rescue, 1/3-2/3 fractions, equal-value actions
                   (lexicographic tie-break) and the zero-rescue closed
                   loop, plus a locatable-failure case; then the
                   termination-latency audit is exercised (exact faster
                   relay over slower direct action, an infinite-latency
                   closed-loop candidate, an unchanged legacy response,
                   and a non-boolean flag rejection).

The stage that checks fraction probabilities and the zero-rescue closed
loop is the HTTP smoke stage.  The process prints a report and then exits
with a status code, which is also printed explicitly.
"""

import json
import os
import py_compile
import subprocess
import sys
import time
import urllib.error
import urllib.request

WEB_URL = os.environ.get("WEB_URL", "http://127.0.0.1:8080").rstrip("/")
AUDIT_URL = WEB_URL + "/api/reachability-audit"
HEALTH_URL = WEB_URL + "/healthz"

RESULTS = []


def record(stage, name, ok, detail=""):
    RESULTS.append((stage, name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    line = "[%s] %s - %s" % (mark, stage, name)
    if detail:
        line += " :: " + detail
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Stage 1: unit tests
# ---------------------------------------------------------------------------


def stage_unit_tests():
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        capture_output=True,
        text=True,
    )
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    summary = tail[-1] if tail else ""
    ok = proc.returncode == 0
    record("unit-tests", "python -m unittest discover -s tests", ok, summary)
    if not ok:
        print(proc.stdout, flush=True)
        print(proc.stderr, flush=True)
    return ok


# ---------------------------------------------------------------------------
# Stage 2: build checks
# ---------------------------------------------------------------------------


def stage_build_checks():
    ok = True
    files = []
    for root in ("app", "verify", "tests"):
        for dirpath, _, names in os.walk(root):
            for name in sorted(names):
                if name.endswith(".py"):
                    files.append(os.path.join(dirpath, name))
    for path in sorted(files):
        try:
            py_compile.compile(path, doraise=True)
            record("build", "compile %s" % path, True)
        except py_compile.PyCompileError as exc:
            record("build", "compile %s" % path, False, str(exc))
            ok = False
    for module in ("app.mdp", "app.server"):
        try:
            __import__(module)
            record("build", "import %s" % module, True)
        except Exception as exc:  # noqa: BLE001
            record("build", "import %s" % module, False, repr(exc))
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Stage 3: HTTP smoke
# ---------------------------------------------------------------------------


def wait_for_service(timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                if resp.status == 200:
                    record("smoke", "GET /healthz", True, "HTTP 200")
                    return True
        except Exception:  # noqa: BLE001 - service not up yet
            time.sleep(1.0)
    record("smoke", "GET /healthz", False,
           "service at %s not healthy within %ss" % (WEB_URL, timeout))
    return False


def post_audit(payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        AUDIT_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def check(name, condition, detail=""):
    record("smoke", name, condition, detail)
    return condition


def stage_http_smoke():
    ok = wait_for_service()
    if not ok:
        return False

    # 1. must-rescue model -> exact probability 1
    status, body = post_audit(
        {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {"s0": {"go": {"r": "1"}}},
        }
    )
    ok &= check(
        "must-rescue probability is exactly 1",
        status == 200
        and body.get("max_rescue_probability") == "1"
        and body.get("canonical_policy") == {"s0": "go"},
        "status=%s body=%s" % (status, body.get("max_rescue_probability")),
    )

    # 2. 1/3 and 2/3 transitions -> exact fraction 1/3
    status, body = post_audit(
        {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {"s0": {"go": {"r": "1/3", "l": "2/3"}}},
        }
    )
    ok &= check(
        "1/3-2/3 model yields exact fraction 1/3",
        status == 200
        and body.get("max_rescue_probability") == "1/3"
        and body.get("max_rescue_probability_num") == 1
        and body.get("max_rescue_probability_den") == 3,
        "status=%s value=%s" % (status, body.get("max_rescue_probability")),
    )

    # 3. equal-value actions -> lexicographically least, equal expectations
    status, body = post_audit(
        {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {
                    "zeta": {"r": "1/2", "l": "1/2"},
                    "alpha": {"r": "1/2", "l": "1/2"},
                }
            },
        }
    )
    cert = (body.get("certificates") or [{}])[0]
    acts = {a["action"]: a for a in cert.get("actions", [])}
    ok &= check(
        "equal-value actions: lexicographic choice with equal expectations",
        status == 200
        and body.get("canonical_policy") == {"s0": "alpha"}
        and acts.get("alpha", {}).get("expected_value") == "1/2"
        and acts.get("zeta", {}).get("expected_value") == "1/2"
        and acts.get("alpha", {}).get("optimal") is True
        and acts.get("zeta", {}).get("optimal") is True
        and acts.get("alpha", {}).get("selected") is True
        and acts.get("zeta", {}).get("selected") is False,
        "policy=%s" % (body.get("canonical_policy"),),
    )

    # 4. closed loop that can never reach any terminal -> zero rescue
    status, body = post_audit(
        {
            "states": ["s0", "k1", "k2", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {
                "s0": {"drift": {"k1": "1"}},
                "k1": {"spin": {"k2": "1"}},
                "k2": {"spin": {"k1": "1"}},
            },
        }
    )
    ok &= check(
        "closed loop yields zero rescue probability",
        status == 200
        and body.get("max_rescue_probability") == "0"
        and body.get("max_rescue_probability_num") == 0
        and sorted(body.get("terminal_unreachable_states", []))
        == ["k1", "k2", "s0"]
        and sorted(body.get("zero_value_states", [])) == ["k1", "k2", "s0"],
        "value=%s unreachable=%s"
        % (body.get("max_rescue_probability"),
           body.get("terminal_unreachable_states")),
    )

    # 5. invalid model (probabilities do not sum to one) -> locatable 400
    status, body = post_audit(
        {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {"s0": {"go": {"r": "1/2", "l": "1/3"}}},
        }
    )
    errors = body.get("errors", [])
    ok &= check(
        "probability-sum violation returns locatable failure, no certificate",
        status == 400
        and body.get("status") == "error"
        and any(e.get("code") == "PROBABILITY_SUM"
                and "actions" in e.get("path", "") for e in errors)
        and "certificates" not in body,
        "status=%s errors=%s" % (status, errors),
    )

    # 6. termination latency audit: exact faster relay beats slower direct
    status, body = post_audit(
        {
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
    )
    lat = body.get("termination_latency") or {}
    lat_states = {c["state"]: c for c in lat.get("states", [])}
    s0_actions = {a["action"]: a
                  for a in lat_states.get("s0", {}).get("actions", [])}
    ok &= check(
        "latency audit: exact faster relay chosen over slower direct action",
        status == 200
        and body.get("max_rescue_probability") == "1"
        and body.get("canonical_policy") == {"s0": "a_direct", "s1": "go"}
        and lat.get("start_min_latency") == "2"
        and lat.get("canonical_policy") == {"s0": "z_relay", "s1": "go"}
        and lat.get("infinite_latency_states") == []
        and lat_states.get("s0", {}).get("rescue_probability") == "1"
        and lat_states.get("s0", {}).get("min_latency") == "2"
        and s0_actions.get("a_direct", {}).get("latency") == "7/3"
        and s0_actions.get("a_direct", {}).get("reason")
        == "latency_above_minimum"
        and s0_actions.get("z_relay", {}).get("latency") == "2"
        and s0_actions.get("z_relay", {}).get("selected") is True,
        "status=%s start_min_latency=%s" % (status, lat.get("start_min_latency")),
    )

    # 7. infinite closed-loop candidate: marked, finite policy still returned
    status, body = post_audit(
        {
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
    )
    lat = body.get("termination_latency") or {}
    lat_states = {c["state"]: c for c in lat.get("states", [])}
    s0_actions = {a["action"]: a
                  for a in lat_states.get("s0", {}).get("actions", [])}
    risky = s0_actions.get("risky", {})
    ok &= check(
        "infinite-latency closed-loop candidate marked, finite policy kept",
        status == 200
        and body.get("max_rescue_probability") == "1/2"
        and lat.get("infinite_latency_states") == ["k1"]
        and lat.get("start_min_latency") == "1"
        and lat.get("canonical_policy", {}).get("s0") == "safe"
        and risky.get("latency") == "infinity"
        and risky.get("finite") is False
        and risky.get("selected") is False
        and risky.get("reason") == "infinite_latency_non_terminating_loop"
        and s0_actions.get("safe", {}).get("latency") == "1"
        and s0_actions.get("safe", {}).get("selected") is True
        and lat_states.get("k1", {}).get("min_latency") == "infinity"
        and lat_states.get("k1", {}).get("finite") is False,
        "status=%s risky=%s" % (status, risky),
    )

    # 8. request without the option: response schema unchanged
    status, body = post_audit(
        {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {"s0": {"go": {"r": "1/3", "l": "2/3"}}},
        }
    )
    ok &= check(
        "response without the latency option keeps legacy fields only",
        status == 200
        and body.get("max_rescue_probability") == "1/3"
        and "termination_latency" not in body,
        "status=%s keys=%s" % (status, sorted(body)),
    )

    # 9. non-boolean flag -> locatable 400
    status, body = post_audit(
        {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "termination_latency_audit": "yes",
            "actions": {"s0": {"go": {"r": "1"}}},
        }
    )
    errors = body.get("errors", [])
    ok &= check(
        "non-boolean latency flag returns locatable failure",
        status == 400
        and body.get("status") == "error"
        and any(e.get("code") == "INVALID_TYPE"
                and "termination_latency_audit" in e.get("path", "")
                for e in errors),
        "status=%s errors=%s" % (status, errors),
    )
    return ok


def main():
    print("verify: code tests, build checks and HTTP smoke against %s"
          % WEB_URL, flush=True)
    ok = True
    ok = stage_unit_tests() and ok
    ok = stage_build_checks() and ok
    ok = stage_http_smoke() and ok

    failed = [r for r in RESULTS if not r[2]]
    print("-" * 70, flush=True)
    print("verify: %d checks, %d failed" % (len(RESULTS), len(failed)),
          flush=True)
    code = 0 if ok and not failed else 1
    print("VERIFY RESULT: %s" % ("PASS" if code == 0 else "FAIL"), flush=True)
    print("VERIFY EXIT CODE: %d" % code, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
