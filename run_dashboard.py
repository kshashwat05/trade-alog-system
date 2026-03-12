from __future__ import annotations

import uvicorn

from ai_trader.dashboard.app import create_dashboard_app


if __name__ == "__main__":
    uvicorn.run(create_dashboard_app(), host="127.0.0.1", port=8000)
