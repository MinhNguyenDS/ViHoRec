"""Backward-compatible entry point. The protocol is short-history, not cold-start.

Import and run ``short_history_eval`` instead. This file exists so older
commands in the README and paper still execute.
"""

from short_history_eval import (  # noqa: F401
    BUCKETS,
    aggregate,
    per_user_metrics,
    run,
)

if __name__ == "__main__":
    print("Note: this protocol is short-history (min train history = 2), "
          "not strict cold-start. Writing reports/short_history.json.")
    run()
