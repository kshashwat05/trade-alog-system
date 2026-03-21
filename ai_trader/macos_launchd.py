from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap


def build_launchd_plist(
    *,
    label: str,
    repo_root: Path,
    python_path: Path,
    log_path: Path,
    start_hour: int,
    start_minute: int,
    with_dashboard: bool,
) -> str:
    args = [
        str(python_path),
        "-m",
        "ai_trader.start_trading_day",
    ]
    if with_dashboard:
        args.append("--with-dashboard")
    arg_block = "\n".join(f"        <string>{arg}</string>" for arg in args)
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{label}</string>
            <key>ProgramArguments</key>
            <array>
{arg_block}
            </array>
            <key>WorkingDirectory</key>
            <string>{repo_root}</string>
            <key>RunAtLoad</key>
            <false/>
            <key>StartCalendarInterval</key>
            <dict>
                <key>Hour</key>
                <integer>{start_hour}</integer>
                <key>Minute</key>
                <integer>{start_minute}</integer>
            </dict>
            <key>StandardOutPath</key>
            <string>{log_path}</string>
            <key>StandardErrorPath</key>
            <string>{log_path}</string>
        </dict>
        </plist>
        """
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a macOS launchd plist for ai_trader daily startup.")
    parser.add_argument("--label", default="com.ai_trader.daily_start")
    parser.add_argument("--hour", type=int, default=9)
    parser.add_argument("--minute", type=int, default=0)
    parser.add_argument("--with-dashboard", action="store_true")
    parser.add_argument(
        "--output",
        default=str(Path.home() / "Library/LaunchAgents/com.ai_trader.daily_start.plist"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python_path = repo_root / ".venv" / "bin" / "python"
    log_path = repo_root / "logs" / "launchd_startup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    plist = build_launchd_plist(
        label=args.label,
        repo_root=repo_root,
        python_path=python_path,
        log_path=log_path,
        start_hour=args.hour,
        start_minute=args.minute,
        with_dashboard=args.with_dashboard,
    )
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(plist)
    print(output_path)


if __name__ == "__main__":
    main()
