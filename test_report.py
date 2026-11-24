from src.format_report import build_readable_report, write_readable_report

report = build_readable_report()
print(report.to_string(index=False))

write_readable_report(report)
print("\nSaved to tests_data/readable_report.txt")