## AI NIFTY Options Trader (Multi-Agent)

This project is a **semi-automated multi-agent trading signal system** for **NIFTY options** that runs locally on macOS.  
It mimics a small trading desk with specialized analyst agents and an orchestrator that generates intraday BUY/SELL signals and sends them to WhatsApp.  
Order execution is **manual** (e.g. in Zerodha Kite).

### Key Features

- **Multi-agent architecture** (chart, option chain, news/macro, volatility, regime, liquidity, trigger, risk, orchestrator)
- **Backtesting** of NIFTY intraday strategies
- **WhatsApp alerts** via Twilio
- **Structured logging** via `loguru`
- **Graceful error handling** and fallbacks
- **Automated tests** with `pytest`
- **Developer agents** for testing and code review (using `crewai`)

### Tech Stack

- Python 3.11+
- `kiteconnect`, `pandas`, `numpy`, `ta`, `requests`, `websocket-client`
- `langgraph`, `crewai`, `pytest`, `pydantic`, `schedule`, `loguru`

### Installation (macOS)

From the repository root (`/Users/kumarshashwat/src/apps/trade-alog-system`):

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env`, fill in only the values you need, and restrict the file permissions:

```bash
cp .env.example .env
chmod 600 .env
```

You can also export these variables in your shell instead of using a `.env` file:

```bash
export KITE_API_KEY="your_kite_api_key"
export KITE_API_SECRET="your_kite_api_secret"
export KITE_ACCESS_TOKEN="your_kite_access_token"
export KITE_LAST_LOGIN="2026-03-13T09:00:00+05:30"
export KITE_REDIRECT_URL="http://localhost:8000/callback"
export KITE_INSTRUMENT_TOKEN="256265"

export NEWS_API_KEY="your_news_api_key"
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="your_gemini_api_key"
export LLM_MODEL="gemini-2.0-flash"
export LLM_VALIDATION_ENABLED="false"

export TWILIO_ACCOUNT_SID="your_twilio_sid"
export TWILIO_AUTH_TOKEN="your_twilio_token"
export TWILIO_WHATSAPP_FROM="whatsapp:+1415xxxxxxx"
export WHATSAPP_TO="whatsapp:+91xxxxxxxxxx"
```

The application uses `pydantic` settings and reads secrets from environment variables or `.env`. Do not hardcode secrets in `config/settings.py` or commit them to the repository.

`KITE_INSTRUMENT_TOKEN` is optional. If omitted, the app will try to resolve the NIFTY index instrument token from Zerodha instruments metadata at runtime. For production, setting it explicitly is safer and avoids a symbol-lookup dependency during market hours.

### Zerodha Live Data Notes

The trading engine now uses Zerodha historical candles during market hours for chart and regime analysis.

- `ChartAgent` reads NIFTY intraday OHLCV candles from Kite.
- `RegimeAgent` uses the same candle stream for trend and volatility regime detection.
- If Kite credentials are missing, invalid, or the candle fetch fails, the system falls back to an empty DataFrame and those agents degrade to neutral/default behavior instead of crashing.

Before relying on the app in live market hours, make sure:

- your `KITE_ACCESS_TOKEN` is valid for the current session
- the account has access to the required market data
- `KITE_INSTRUMENT_TOKEN` is correct for the instrument you want to analyze, if you set it manually

For NIFTY 50, `256265` is the common NSE index instrument token, but you should still verify it against your Zerodha account/instruments dump before treating it as fixed infrastructure.

### Zerodha Login Flow

The project now includes a local authentication flow for Kite Connect.

Generate the login URL only:

```bash
python -m ai_trader.auth.kite_login --print-url-only
```

Run the full local login flow:

```bash
python -m ai_trader.auth.kite_login
```

This will:

- print the Zerodha login URL
- start a local callback listener on `KITE_REDIRECT_URL`
- wait for Zerodha to redirect back with `request_token`
- exchange it for an access token
- store `KITE_ACCESS_TOKEN` and `KITE_LAST_LOGIN` in your local `.env`

Notes:

- The redirect URL configured in Zerodha must exactly match `KITE_REDIRECT_URL`
- if you are using `http://localhost:8000/callback`, do not run the dashboard on port `8000` at the same time during login
- Zerodha access tokens are valid only for the trading day
- by default, the login URL is auto-opened in your default browser

### One-Click Daily Startup

For the supported daily workflow on macOS, use:

```bash
python -m ai_trader.start_trading_day
```

This command will:

- validate the stored Kite session
- if needed, auto-open the Zerodha login page in your browser
- auto-capture the callback on `KITE_REDIRECT_URL`
- store the fresh access token
- start the trading engine automatically

If you also want the dashboard started:

```bash
python -m ai_trader.start_trading_day --with-dashboard
```

If you want the login URL printed without auto-opening the browser:

```bash
python -m ai_trader.start_trading_day --no-open-browser
```

### macOS launchd

You can generate a `launchd` plist for pre-market startup:

```bash
python -m ai_trader.macos_launchd --hour 8 --minute 55 --with-dashboard
```

This prints the plist path. Then load it with:

```bash
launchctl load ~/Library/LaunchAgents/com.ai_trader.daily_start.plist
```

Reload after edits:

```bash
launchctl unload ~/Library/LaunchAgents/com.ai_trader.daily_start.plist
launchctl load ~/Library/LaunchAgents/com.ai_trader.daily_start.plist
```

Validate the current session locally:

```bash
python -c "from ai_trader.auth.token_manager import validate_session; print(validate_session())"
```

### Local Security Notes

- Keep `.env` local only. It is git-ignored and should never be committed.
- Prefer short-lived credentials where possible, especially `KITE_ACCESS_TOKEN`.
- Treat Twilio credentials, Gemini/OpenAI keys, News API keys, and broker tokens as secrets even on a local machine.
- Avoid pasting secrets into notebooks, screenshots, shell history, or logs.
- Restart the app after rotating credentials because API clients are created at process startup.

### Running the System

From the repository root:

```bash
source .venv/bin/activate
python -m ai_trader.main
```

This starts the scheduler that runs the trading loop every **2 minutes** during configured market hours.

On startup, the system now:

- validates the stored Kite access token
- triggers the local login flow automatically if the token is missing or expired
- injects the authenticated Kite client into the market-data runtime before scheduling starts

At each cycle, the engine:

- fetches live or fallback market inputs
- evaluates chart, option-chain, news, volatility, regime, and liquidity agents
- generates a directional trade signal only when the inputs align and pass risk checks
- sends the signal to WhatsApp if Twilio is configured

The system is still a signal engine, not an order execution bot. Orders remain manual unless you build a broker execution layer on top.

### Running Tests

```bash
source .venv/bin/activate
pytest -q
```

### Backtesting

Backtesting utilities live in `backtesting/backtester.py`. You can run a simple backtest via:

```bash
python -m ai_trader.backtesting.backtester
```
