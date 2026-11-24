from src.evaluation import load_questions, question_stats

df = load_questions("data/questions.csv")
print(df)
print(question_stats(df))