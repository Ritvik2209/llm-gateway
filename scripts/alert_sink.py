"""A local sink that prints the notifications Alertmanager would deliver.

Slack delivery is a POST to a URL — a solved integration that carries no engineering risk.
What does carry risk is whether the notification template renders with real values or emits
`<no value>`, and whether the annotations say anything an on-call engineer can act on. This
prints exactly what Alertmanager produces, so that part is verifiable without a credential
and without posting test alerts into a real workspace.

    python scripts/alert_sink.py

Then trigger a condition — for example open the mock provider's circuit — and the alert
will appear here once Prometheus has evaluated it and Alertmanager has grouped it.
"""

import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 5001

# Without this the output is block-buffered when stdout is a pipe rather than a terminal,
# so notifications sit invisibly in the buffer and the sink looks like it received nothing.
sys.stdout.reconfigure(line_buffering=True)


class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(f"\n  unparseable payload: {raw[:400]!r}")
            return

        print()
        print("=" * 78)
        print(
            f"  {datetime.now():%H:%M:%S}  "
            f"{payload.get('status', '?').upper()}  "
            f"{payload.get('receiver', '?')}  "
            f"({len(payload.get('alerts', []))} alert(s))"
        )
        print("=" * 78)

        for alert in payload.get("alerts", []):
            labels = alert.get("labels", {})
            annotations = alert.get("annotations", {})
            print(f"\n  {labels.get('alertname', '?')}  [{labels.get('severity', '?')}]")
            scope = {
                key: value
                for key, value in labels.items()
                if key in ("team_id", "provider", "provider_name", "component")
            }
            if scope:
                print(f"    {scope}")
            print(f"    summary: {annotations.get('summary', '(none)')}")
            description = " ".join(annotations.get("description", "").split())
            if description:
                print(f"    detail : {description}")
            if annotations.get("dashboard"):
                print(f"    link   : {annotations['dashboard']}")

            # The check that matters: a template that failed to render shows up here.
            rendered = " ".join(annotations.values())
            if "<no value>" in rendered or "%!" in rendered:
                print("    *** TEMPLATE DID NOT RENDER ***")

    def log_message(self, *args) -> None:
        """Silence the default per-request logging; the payload output is the point."""


if __name__ == "__main__":
    print(f"listening for Alertmanager notifications on http://0.0.0.0:{PORT}/alerts")
    print("stop with ctrl-c\n")
    HTTPServer(("0.0.0.0", PORT), AlertHandler).serve_forever()
