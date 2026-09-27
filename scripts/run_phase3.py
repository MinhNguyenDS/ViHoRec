"""Phase 3 experiment driver: val-tuned, test-once, CI + Wilcoxon.

Runs the dependency-free table first (including EASE), then torch neural
baselines if importable, then history-stratified metrics.

  python run_phase3.py
"""

from __future__ import annotations

import traceback

import run_baselines as rb
import short_history_eval as she


def main() -> None:
    print("=== Phase 3 / core baselines (val-tune, test once, CI, Wilcoxon) ===")
    rb.run()
    print("\n=== Phase 3 / neural (LightGCN, NeuMF, GRU4Rec) ===")
    try:
        import run_neural_baselines as nb
        nb.run()
    except Exception:
        traceback.print_exc()
        print("Neural baselines skipped (torch missing or training failed).")
    print("\n=== Phase 3 / short-history buckets ===")
    she.run()
    print("\nPhase 3 core run finished. Figures from run_adaptive_hybrid.py are optional.")


if __name__ == "__main__":
    main()
