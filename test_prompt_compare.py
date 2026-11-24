import pandas as pd

from src.run_eval import run_evaluation, accuracy

VERSIONS = ["v1_basic", "v2_grounded", "v3_refuse"]

all_results = []
summary = []

for version in VERSIONS:
    print(f"\nRunning {version}...")
    results = run_evaluation("data", "tests_data/questions.csv", version)
    all_results.append(results)
    summary.append({
        "prompt_version": version,
        "accuracy_pct": round(accuracy(results), 1),
        "correct": int(results["correct"].sum()),
        "total": len(results),
    })

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv("tests_data/prompt_comparison.csv", index=False)

summary_df = pd.DataFrame(summary)
print("\nSummary:")
print(summary_df.to_string(index=False))
print("\nSaved detailed results to tests_data/prompt_comparison.csv")