#!/usr/bin/env python3
"""Launch script for the GOLD360 Streamlit Intelligence Platform."""
import subprocess
import sys
from pathlib import Path


def main():
    dashboard_dir = Path(__file__).resolve().parent / "gold360" / "dashboard"
    app_path = dashboard_dir / "app.py"
    if not app_path.exists():
        print(f"Dashboard app not found at {app_path}")
        sys.exit(1)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
