from src.evaluation import load_questions, question_stats

df = load_questions("tests_data/questions.csv")
print(df)
print(question_stats(df))