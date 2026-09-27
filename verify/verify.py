"""Compose ``verify`` service: code tests + build checks + API/HTTP smoke.

Runs three stages and exits non-zero if any of them fails:

1. code tests   -- the unit test suite (exact solver, validation, and the
                   brute-force cross-check);
2. build checks -- every source file byte-compiles and the packages import;
3. API smoke    -- the required models are POSTed to a live service:
                   must-rescue, 1/3-2/3 fractions, equal-value actions
                   (lexicographic tie-break) and the zero-rescue closed
                   loop, a locatable-failure case, plus the opt-in
                   termination-delay audit (slower direct vs faster relay,
                   infinite-loop candidate alongside a finite policy, and
                   the unchanged response when the option is absent).

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

    # 6. termination-delay audit: slower direct vs faster relay
    status, body = post_audit(
        {
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
    )
    block = body.get("termination_delay_audit") or {}
    cert = {c["state"]: c for c in body.get("certificates", [])}.get("s", {})
    acts = {a["action"]: a for a in cert.get("actions", [])}
    ok &= check(
        "delay audit: exact smaller-delay relay beats slower direct",
        status == 200
        and body.get("max_rescue_probability") == "1"
        and block.get("state_min_termination_delays", {}).get("s") == "2"
        and acts.get("direct", {}).get("termination_delay") == "3"
        and acts.get("relay", {}).get("termination_delay") == "2"
        and acts.get("relay", {}).get("delay_selected") is True
        and acts.get("direct", {}).get("delay_selected") is False
        and block.get("minimum_delay_policy", {}).get("s") == "relay",
        "delays=%s policy=%s"
        % (block.get("state_min_termination_delays"),
           block.get("minimum_delay_policy")),
    )

    # 7. delay audit: an optimal candidate that traps with positive
    # probability must be marked infinite with evidence, while a finite
    # candidate is still returned
    status, body = post_audit(
        {
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
    )
    block = body.get("termination_delay_audit") or {}
    cert = {c["state"]: c for c in body.get("certificates", [])}.get("s", {})
    acts = {a["action"]: a for a in cert.get("actions", [])}
    trap = acts.get("a_trap", {})
    safe = acts.get("d_safe", {})
    evidence = trap.get("infinite_delay_evidence") or {}
    ok &= check(
        "delay audit: infinite loop candidate marked, finite policy kept",
        status == 200
        and body.get("max_rescue_probability") == "1/2"
        and trap.get("termination_delay") == "infinity"
        and trap.get("termination_delay_infinite") is True
        and trap.get("optimal") is True
        and trap.get("delay_selected") is False
        and evidence.get("closed_class_states") == ["k"]
        and safe.get("termination_delay") == "1"
        and safe.get("delay_selected") is True
        and block.get("infinite_delay_states") == ["k"]
        and block.get("minimum_delay_policy") == {"s": "d_safe"},
        "trap=%s safe=%s infinite=%s"
        % (trap.get("termination_delay"),
           safe.get("termination_delay"),
           block.get("infinite_delay_states")),
    )

    # 8. request without the option carries no delay fields at all
    status, body = post_audit(
        {
            "states": ["s0", "r", "l"],
            "start": "s0",
            "terminals": {"rescued": ["r"], "lost": ["l"]},
            "actions": {"s0": {"go": {"r": "1/3", "l": "2/3"}}},
        }
    )
    cert = (body.get("certificates") or [{}])[0]
    ok &= check(
        "delay audit is opt-in: absent request yields unchanged response",
        status == 200
        and "termination_delay_audit" not in body
        and "min_termination_delay" not in cert
        and all("termination_delay" not in a for a in cert.get("actions", [])),
        "keys=%s" % sorted(body.keys()),
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
