from src.run_eval import run_evaluation, accuracy

results = run_evaluation("data", "tests_data/questions.csv", "v3_refuse")
print(results[["question", "answer", "correct"]].to_string())
print(f"\nAccuracy: {accuracy(results):.1f}%")

results.to_csv("tests_data/results_v3.csv", index=False)
print("Saved to tests_data/results_v3.csv")