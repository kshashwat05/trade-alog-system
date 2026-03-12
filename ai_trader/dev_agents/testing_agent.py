from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import sys

from loguru import logger

try:  # Best-effort import; agent still works without crewai.
    from crewai import Agent, Task  # type: ignore
except Exception:  # noqa: BLE001
    Agent = None  # type: ignore[assignment]
    Task = None  # type: ignore[assignment]


@dataclass
class TestRunResult:
    success: bool
    pytest_exit_code: int
    coverage_file: str | None
    coverage_summary: str | None
    output: str


class TestingAgent:
    """Developer agent responsible for running tests and reporting coverage."""

    def __init__(self, project_root: str | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.agent: Optional["Agent"] = None
        if Agent is not None:
            try:
                self.agent = Agent(
                    role="Testing Agent",
                    goal="Ensure the trading system passes tests and has coverage.",
                    backstory="You are a senior QA engineer automating test execution.",
                    allow_delegation=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to initialize crewai Agent in TestingAgent: {exc}")
                self.agent = None

    def run_pytest(self, extra_args: List[str] | None = None) -> TestRunResult:
        args = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "--disable-warnings",
            "--cov=ai_trader",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ]
        if extra_args:
            args.extend(extra_args)

        logger.info(f"Running pytest via TestingAgent: {' '.join(args)}")

        try:
            proc = subprocess.run(
                args,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            logger.error(f"Failed to run pytest: {exc}")
            return TestRunResult(
                success=False,
                pytest_exit_code=1,
                coverage_file=None,
                coverage_summary=None,
                output=str(exc),
            )

        output = proc.stdout + "\n" + proc.stderr
        success = proc.returncode == 0
        coverage_file = ".coverage" if (self.project_root / ".coverage").exists() else None
        coverage_summary = None
        for line in output.splitlines():
            if line.strip().startswith("TOTAL"):
                coverage_summary = line.strip()

        logger.info(f"Pytest completed with code {proc.returncode}")
        # Optionally register a crewai Task if available (no-op otherwise)
        if self.agent is not None and Task is not None:
            try:
                Task(
                    description="Summarize latest pytest run.",
                    agent=self.agent,
                    expected_output="High-level test status and coverage presence.",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to create crewai Task in TestingAgent: {exc}")

        return TestRunResult(
            success=success,
            pytest_exit_code=proc.returncode,
            coverage_file=coverage_file,
            coverage_summary=coverage_summary,
            output=output,
        )

    def recommended_test_targets(self) -> list[str]:
        return [
            "institutional positioning calculations",
            "gamma regime estimation",
            "liquidity sweep detection",
            "LLM validation fallback logic",
            "dashboard API endpoints",
        ]
