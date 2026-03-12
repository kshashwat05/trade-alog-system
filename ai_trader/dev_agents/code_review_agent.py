from __future__ import annotations

import subprocess
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import sys

from loguru import logger

from ai_trader.config.settings import settings

try:  # Best-effort import; agent still works without crewai.
    from crewai import Agent, Task  # type: ignore
except Exception:  # noqa: BLE001
    Agent = None  # type: ignore[assignment]
    Task = None  # type: ignore[assignment]


@dataclass
class CodeReviewReport:
    issues: List[str]
    suggestions: List[str]


class CodeReviewAgent:
    """Developer agent responsible for reviewing code quality and safety.

    This implementation uses lightweight static checks (lint/format) to keep the
    environment self-contained but is structured to be extended with richer analysis.
    """

    def __init__(self, project_root: str | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.agent: Optional["Agent"] = None
        if Agent is not None:
            try:
                self.agent = Agent(
                    role="Code Review Agent",
                    goal="Review the codebase for correctness, safety, and maintainability.",
                    backstory="You are a senior Python quant reviewing an algo trading system.",
                    allow_delegation=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to initialize crewai Agent in CodeReviewAgent: {exc}")
                self.agent = None

    def run_static_checks(self) -> CodeReviewReport:
        issues: List[str] = []
        suggestions: List[str] = []

        # Example static check: run python -m py_compile on package
        try:
            logger.info("Running bytecode compilation check for ai_trader package.")
            # Use compileall so we can target a package directory.
            proc = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", "ai_trader"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                issues.append("Bytecode compilation failed for ai_trader.")
                issues.append(proc.stderr.strip())
        except OSError as exc:
            logger.error(f"Static check failed to run: {exc}")
            issues.append(str(exc))

        journal_db = self.project_root / settings.trade_journal_path
        if journal_db.exists():
            try:
                conn = sqlite3.connect(journal_db)
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
                ).fetchone()
                if row is None:
                    issues.append("Trade journal database exists but trades table is missing.")
                conn.close()
            except sqlite3.Error as exc:
                issues.append(f"Trade journal integrity check failed: {exc}")

        risk_source = (self.project_root / "ai_trader/agents/risk_agent.py").read_text()
        if "signal_cooldown_minutes" not in risk_source:
            issues.append("Risk agent does not appear to enforce signal cooldowns.")

        http_source = (self.project_root / "ai_trader/data/http_client.py").read_text()
        if "Retry(" not in http_source:
            issues.append("HTTP client retries are not configured.")

        main_source = (self.project_root / "ai_trader/main.py").read_text()
        if "trading_engine.log" not in main_source:
            suggestions.append("Consider consolidating all runtime logging into logs/trading_engine.log.")

        if not issues:
            suggestions.append("Consider adding a dedicated linter like ruff or flake8.")

        if self.agent is not None and Task is not None:
            try:
                Task(
                    description="Review latest static check results and summarize key issues.",
                    agent=self.agent,
                    expected_output="List of issues and recommendations for improvement.",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to create crewai Task in CodeReviewAgent: {exc}")

        return CodeReviewReport(issues=issues, suggestions=suggestions)
