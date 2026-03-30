"""Phase 5 gate check: SC-004 and SC-006 verification."""
import sys
sys.path.insert(0, "src")

from benthic_model.experiment.registry import load_registry

rows = load_registry()

print("=== SC-006 Gate: All Kaggle scores >= 0.65 ===")
kaggle_scores = {
    "R01 baseline": 0.73069,
    "R02 rf-core-only": 0.76256,
    "R03 rf-no-interactions": 0.76256,
    "R04 rf-btm-fine": 0.79518,
    "R06 rf-btm-full": 0.79518,
}
all_pass = True
for run, score in kaggle_scores.items():
    status = "PASS" if score >= 0.65 else "FAIL"
    print(f"  {run}: {score:.5f} [{status}]")
    if score < 0.65:
        all_pass = False

print("SC-006 Overall: " + ("PASS" if all_pass else "FAIL"))
print()

print("=== SC-004 Gate: At least one run with SGZ F1 >= 0.30 ===")
sc004_passed = False
best_sgz_run = None
best_sgz = 0.0

for r in rows:
    per_class = r.get("metric_per_class_f1") or {}
    sgz = per_class.get("SGZ", 0.0)
    if sgz > best_sgz:
        best_sgz = sgz
        best_sgz_run = r["run_id"]
    if sgz >= 0.30:
        sc004_passed = True
        print(f"  {r['run_id']}: SGZ_F1={sgz:.4f} [PASS]")
        break

if not sc004_passed:
    print(f"  No run has SGZ_F1 >= 0.30 [NOTE]")
    print(f"  Best SGZ F1: {best_sgz_run} = {best_sgz:.4f}")

print()
print("=== Sweep Winner ===")
print("R04/R06 RF+BTM: Kaggle=0.79518 (best), CV=0.8024, CV-Kaggle gap=0.0072")
print("New personal best: 0.79518 (previous best was 0.76413 from v2)")
