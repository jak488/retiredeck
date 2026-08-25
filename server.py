"""Static file server for the dashboard + a live /api/market endpoint.

Serves index.html / prices.json / etc. exactly like `python -m http.server`,
and adds GET /api/market which fetches the latest S&P 500, 10yr and 30yr
Treasury quotes via yfinance. Results are cached for TTL seconds so repeated
page loads are fast and we don't spam the data source. The last good result is
also written to market.json as an offline fallback.

Run:  python3 server.py   (listens on 127.0.0.1:8777)
"""
import http.server, json, os, time, threading


def _load_dotenv(path=".env"):
    """Minimal .env loader — sets os.environ from KEY=VALUE lines (no dependency)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except FileNotFoundError:
        pass


TTL = 120  # seconds to cache a market snapshot
_CACHE = {"ts": 0.0, "data": None}
_LOCK = threading.Lock()

# --- chatbot daily spend guard (server-wide, resets at local midnight) ---
DAILY_USD_LIMIT = 10.0          # hard cap on Claude spend per calendar day
_PRICE = {                      # USD per token, claude-haiku-4-5
    "in": 1.0 / 1e6, "out": 5.0 / 1e6,
    "cache_write": 1.25 / 1e6, "cache_read": 0.1 / 1e6,
}
_BUDGET = {"date": None, "spent": 0.0}
_BUDGET_LOCK = threading.Lock()


def _budget_spent_today():
    import datetime
    today = datetime.date.today().isoformat()
    with _BUDGET_LOCK:
        if _BUDGET["date"] != today:          # new day → reset
            _BUDGET["date"], _BUDGET["spent"] = today, 0.0
        return _BUDGET["spent"]


def _budget_add(usage):
    cost = ((usage.input_tokens or 0) * _PRICE["in"]
            + (usage.output_tokens or 0) * _PRICE["out"]
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * _PRICE["cache_write"]
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * _PRICE["cache_read"])
    with _BUDGET_LOCK:
        _BUDGET["spent"] += cost
        total = _BUDGET["spent"]
    print("[chat] +$%.4f   daily total $%.2f / $%.2f" % (cost, total, DAILY_USD_LIMIT))


def fetch_market():
    import yfinance as yf

    def last2(tkr):
        s = yf.download(tkr, period="7d", interval="1d",
                        auto_adjust=True, progress=False)["Close"].dropna()
        s = s.iloc[:, 0] if hasattr(s, "columns") else s
        return float(s.iloc[-2]), float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d")

    ny = lambda v: v / 10.0 if v > 20 else v   # ^TNX/^TYX sometimes quoted as yield*10
    sp_p, sp_c, asof = last2("^GSPC")
    t10_p, t10_c, _ = last2("^TNX")
    t30_p, t30_c, _ = last2("^TYX")
    t10_p, t10_c = ny(t10_p), ny(t10_c)
    t30_p, t30_c = ny(t30_p), ny(t30_c)
    return {
        "asof": asof,
        "spx": {"level": round(sp_c, 2), "chg_pct": round((sp_c / sp_p - 1) * 100, 2)},
        "y10": {"level": round(t10_c, 2), "chg_bps": round((t10_c - t10_p) * 100)},
        "y30": {"level": round(t30_c, 2), "chg_bps": round((t30_c - t30_p) * 100)},
    }


def get_market():
    now = time.time()
    with _LOCK:
        if _CACHE["data"] is not None and now - _CACHE["ts"] <= TTL:
            return _CACHE["data"]
    try:
        data = fetch_market()
        with _LOCK:
            _CACHE["data"], _CACHE["ts"] = data, now
        try:
            json.dump(data, open("market.json", "w"))  # persist offline fallback
        except OSError:
            pass
        return data
    except Exception:
        if _CACHE["data"] is not None:
            return _CACHE["data"]          # serve stale on transient failure
        return json.load(open("market.json"))  # last resort: on-disk snapshot


SYSTEM_PROMPT = """You are the RetireDeck Assistant, an educational guide embedded in the \
RetireDeck retirement-planning dashboard. Help users understand and use the calculators on the \
page, and explain retirement-planning concepts at an educational level.

Rules:
- Educational information only — NOT personalized financial, investment, tax, or retirement advice. \
Never tell a user what they personally should do with their money; explain how the tools work and \
what the numbers mean, and suggest consulting a licensed advisor for personal decisions.
- Ground explanations of the tools in the page's actual content (provided below). Use the real \
calculator names, input labels, and options that appear on the page, and the methodology described \
in the captions.
- Be concise, friendly, and practical. Respond in plain text (no markdown symbols like ** or #).
- Never claim a fiduciary relationship or guarantee outcomes.

The dashboard's current HTML is provided so you know exactly which tools, inputs, and options exist \
and what their captions say."""


class ChatError(Exception):
    pass


def chat_reply(messages, context=None):
    try:
        import anthropic
    except ImportError:
        raise ChatError("The 'anthropic' package isn't installed on the server (pip install anthropic).")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ChatError("ANTHROPIC_API_KEY is not set on the server. Restart it with your key to enable the assistant.")
    client = anthropic.Anthropic()
    html = open("index.html", encoding="utf-8").read()
    system = [{
        "type": "text",
        "text": SYSTEM_PROMPT + "\n\n<page_html>\n" + html + "\n</page_html>",
        "cache_control": {"type": "ephemeral"},   # cache the big, stable HTML prefix — cheap on repeat turns
    }]
    if context:   # volatile: current inputs/results. Separate uncached block so it doesn't break the cache above.
        system.append({
            "type": "text",
            "text": ("The user's CURRENT calculator inputs and on-screen results are below. When they refer "
                     "to 'my' scenario, numbers, mix, or results, use these exact values. Results reflect what "
                     "is displayed right now.\n\n<dashboard_state>\n"
                     + json.dumps(context, indent=2) + "\n</dashboard_state>"),
        })
    msgs = [{"role": m["role"], "content": m["content"]}
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")][-20:]
    if _budget_spent_today() >= DAILY_USD_LIMIT:
        raise ChatError("The assistant has reached today's usage limit. Please check back tomorrow.")
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system,
        messages=msgs,
    )
    _budget_add(resp.usage)
    return "".join(b.text for b in resp.content if b.type == "text")


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/api/market":
            try:
                body = json.dumps(get_market()).encode()
            except Exception:
                self.send_error(503, "market data unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/chat":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            out = json.dumps({"reply": chat_reply(payload.get("messages", []), payload.get("context"))}).encode()
            status = 200
        except ChatError as e:
            out = json.dumps({"error": str(e)}).encode(); status = 400
        except Exception as e:
            out = json.dumps({"error": "server error: %s" % e}).encode(); status = 500
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass  # quiet


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    _load_dotenv()   # load ANTHROPIC_API_KEY (and any other vars) from .env if present
    host = "0.0.0.0"                            # bind all interfaces (required by hosts like Render)
    port = int(os.environ.get("PORT", 8777))   # Render provides $PORT; default 8777 for local dev
    httpd = http.server.ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print("serving RetireDeck on http://%s:%d  (live /api/market, TTL %ds)" % (host, port, TTL))
    httpd.serve_forever()
