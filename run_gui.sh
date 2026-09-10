#!/usr/bin/env bash
# Startet Super360 Studio mit dem Host-Python (kein ROS-Sourcing noetig).
set -e
cd "$(dirname "$(readlink -f "$0")")"
exec python3 app.py "$@"
