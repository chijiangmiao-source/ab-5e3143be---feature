"""HTTP front-end for the exact reachability audit (stdlib only)."""

from __future__ import annotations

import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import mdp

AUDIT_PATH = "/api/reachability-audit"
HEALTH_PATHS = ("/healthz", "/health")


class Handler(BaseHTTPRequestHandler):
    server_version = "ReachabilityAudit/1.0"
    protocol_version = "HTTP/1.1"

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self._send_json(
            404,
            {
                "status": "error",
                "errors": [
                    {
                        "path": self.path,
                        "code": "NOT_FOUND",
                        "message": "unknown endpoint; use POST %s" % AUDIT_PATH,
                    }
                ],
            },
        )

    def do_GET(self):
        if self.path in HEALTH_PATHS:
            self._send_json(200, {"status": "ok"})
        else:
            self._not_found()

    def do_POST(self):
        if self.path != AUDIT_PATH:
            self._not_found()
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            self._send_json(
                400,
                {
                    "status": "error",
                    "errors": [
                        {
                            "path": "$",
                            "code": "INVALID_JSON",
                            "message": "request body is not valid UTF-8",
                        }
                    ],
                },
            )
            return
        try:
            model = mdp.load_and_validate(text)
        except mdp.AuditError as exc:
            self._send_json(400, {"status": "error", "errors": exc.errors})
            return
        try:
            result = mdp.audit(model)
        except Exception:  # pragma: no cover - defensive
            traceback.print_exc()
            self._send_json(
                500,
                {
                    "status": "error",
                    "errors": [
                        {
                            "path": "$",
                            "code": "INTERNAL",
                            "message": "internal solver error",
                        }
                    ],
                },
            )
            return
        self._send_json(200, result)

    def log_message(self, fmt, *args):  # keep container logs clean
        pass


def main():
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("reachability-audit listening on port %d" % port, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
