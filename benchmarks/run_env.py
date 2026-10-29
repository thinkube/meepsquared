"""Run a benchmark script with environment variables set first.

    python run_env.py KEY=VALUE [KEY=VALUE ...] -- script.py [args...]
"""
import os
import runpy
import sys

split = sys.argv.index("--")
for pair in sys.argv[1:split]:
    key, value = pair.split("=", 1)
    os.environ[key] = value
script, *args = sys.argv[split + 1:]
sys.argv = [script, *args]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))  # as python <script> does
runpy.run_path(script, run_name="__main__")
