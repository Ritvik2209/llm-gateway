# Deploying the public demo

This puts the gateway on one Linux server as a public demo: a page at `/` where visitors send
requests, simulate a Groq outage and hit the rate limit, plus read-only Grafana dashboards.
It runs the same code as the local stack, with demo mode switched on and the server locked down.

## What runs where

```
Internet ──443──▶ Caddy ──▶ gateway :8000 ──▶ Groq (hosted)
                    │           │  └────────▶ mock (in-process)
                    │           └──▶ Redis (volume, append-only)
                    └──▶ Grafana :3000 ◀── Prometheus ──▶ Alertmanager ──▶ Slack
```

Only Caddy is reachable from outside. Ollama is not deployed: a 2 GB server cannot run it, so
the demo tier falls back from Groq to mock instead.

| | Local stack | Laptop demo | Demo server |
|---|---|---|---|
| Compose files | `docker-compose.yml` | plus `docker-compose.demo.yml` | plus `docker-compose.prod.yml` |
| Interface | curl, `/docs`, `scripts/demo.py` | page at `/`, plus `/docs` | page at `/`, plus `/docs` |
| Providers | Groq, Ollama, mock | Groq, Ollama, mock | Groq, mock |
| Teams | `config/teams.yaml` (committed demo keys) | `config/teams.demo.yaml` (committed, six on the page) | `secrets/teams.yaml` (private, outside git) |
| Simulated outage | `/admin/mock/toggle-failure`, admin key | page buttons, Groq and Ollama, end after 60 s | page button, Groq, ends after 60 s |
| Open ports | all, on your machine | all, on your machine | 80 and 443 only |
| `/metrics` | open | open | 404 from outside; Prometheus reads it internally |
| Redis data | lost when the container is recreated | lost when the container is recreated | kept on a volume |

The laptop demo is the server's page with Ollama behind it as well; its two server teams are
copied unchanged, and a test keeps them that way. Each place has its own Prometheus and Grafana,
scraping only its own gateway.

## Requirements

- A Linux server with **2 GB of RAM**, Ubuntu 24.04. A DigitalOcean droplet works; the GitHub
  Student Developer Pack's credit covers one for about a year.
- A domain with two DNS `A` records pointing at the server, for example `api.example.com` and
  `grafana.example.com`. A free DuckDNS subdomain also works.
- A Groq API key.
- A Slack incoming webhook. Use a **separate channel** for the demo: the outage button opens
  Groq's circuit, and the circuit-open alert fires if nobody sends a request afterwards.

## 1. Prepare the server

Allow inbound **22, 80 and 443 only**, in the provider's cloud firewall. Then, over SSH:

```bash
# Swap, so building the image does not run out of memory on 2 GB.
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Docker Engine and the Compose plugin.
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # log out and back in afterwards

git clone https://github.com/Ritvik2209/llm-gateway.git
cd llm-gateway
```

## 2. Add the secrets

None of these files are in git. Create them on the server, or copy them up with `scp`.

```bash
# Environment: the Groq key, the two domains and Grafana's admin password.
cp .env.example .env
nano .env    # set GROQ_API_KEY, GATEWAY_DOMAIN, GRAFANA_DOMAIN, GRAFANA_ADMIN_PASSWORD

# Teams: the two public demo teams, and a private admin key you generate now.
mkdir -p secrets
cp deploy/teams.example.yaml secrets/teams.yaml
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
nano secrets/teams.yaml   # paste that value over REPLACE-WITH-A-LONG-RANDOM-KEY

# Slack: the demo channel's incoming-webhook URL, one line.
cp monitoring/alertmanager/slack_webhook_url.example monitoring/alertmanager/slack_webhook_url
nano monitoring/alertmanager/slack_webhook_url
```

The keys of `public_demo` teams are published on the page on purpose. Their limits (20
requests a minute, 4,000 tokens a minute, $1 a month) are what protect your Groq key. The
gateway refuses to load a team that is both `public_demo` and `is_admin`.

## 3. Start

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Caddy requests the HTTPS certificates on the first visit to each domain, once DNS points at the
server. `docker compose logs caddy` shows the result.

## 4. Verify

Run these from your own machine, replacing the domains.

```bash
curl https://api.example.com/health                                  # {"status":"ok"}
curl -o /dev/null -w '%{http_code}\n' https://api.example.com/metrics  # 404

# The committed demo keys must not work here.
curl -o /dev/null -w '%{http_code}\n' https://api.example.com/v1/chat \
  -H 'Authorization: Bearer demo-team-alpha-local-only' \
  -H 'Content-Type: application/json' \
  -d '{"model":"chat-general","messages":[{"role":"user","content":"hi"}]}'   # 401

# Neither may the placeholder admin key.
curl -o /dev/null -w '%{http_code}\n' https://api.example.com/admin/health \
  -H 'Authorization: Bearer REPLACE-WITH-A-LONG-RANDOM-KEY'                 # 401

# Internal services must be unreachable.
nc -zv -w 3 <server-ip> 6379    # fails
nc -zv -w 3 <server-ip> 9090    # fails
```

Then open `https://api.example.com/` and follow "What to try" on the page. Its "Live dashboards"
section links all four Grafana dashboards; `https://grafana.example.com/` itself should open on
the Operations dashboard without asking for a login.

## Updating

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Team limits can also be changed live through the admin API with your private key; the change is
written back into `secrets/teams.yaml`.

## Things to know

- **The page shows one outage button per provider the demo teams use.** On this server that is
  Groq only. The commented-out teams in `deploy/teams.example.yaml` add Ollama controls on a
  server that runs Ollama; the laptop demo runs them.
- **The outage button changes real state.** It makes the gateway treat Groq as down for 60 s
  (`DEMO_OUTAGE_SECONDS`), for every visitor. The first few requests take about 3.5 s longer
  while the gateway retries, then the circuit opens. After the outage, the circuit stays open
  until the next request tests Groq, which is why the circuit-open alert can fire on a quiet
  demo.
- **Limits are shared.** The rate limit is per team, so one visitor's burst rate-limits
  everyone on that team for up to a minute.
- **The fallback reply is canned.** Mock answers "mock response". The page labels it as a
  backup; on a server that can run Ollama, put Ollama second in the `chat-demo` tier and the
  fallback answer is real.
- **Ollama shows as degraded** on the dashboards, because the gateway probes it once at startup
  and finds nothing. No alert fires for that.
- **Slack alert links point at `localhost:3000`.** The dashboard links in
  `monitoring/alerts.yml` are written for the local stack.
- **Throughput is lower than the published figure.** The ~450 req/s ceiling was measured on a
  laptop; a 1 vCPU server will top out lower.
