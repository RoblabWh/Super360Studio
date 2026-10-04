#!/usr/bin/env bash
# Startet Super360 Studio (kein ROS-Sourcing noetig).
# Interpreter: SUPER360_PYTHON, sonst die venv ~/.venvs/super360 (Ubuntu 24.04,
# s. README), sonst das python3 des Systems (Ubuntu 22.04).
set -e
cd "$(dirname "$(readlink -f "$0")")"
PY="${SUPER360_PYTHON:-}"
if [ -z "$PY" ] && [ -x "$HOME/.venvs/super360/bin/python3" ]; then
    PY="$HOME/.venvs/super360/bin/python3"
fi
exec "${PY:-python3}" app.py "$@"
