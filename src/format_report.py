import pandas as pd


def build_readable_report(path: str = "tests_data/prompt_comparison.csv") -> pd.DataFrame:
    """Turn the long comparison CSV into one row per question."""
    df = pd.read_csv(path)

    # Empty keyword means "should say I don't know"; label it so pivot keeps the row
    df["expected_keyword"] = df["expected_keyword"].fillna("").replace("", "(should refuse)")
    df["result"] = df["correct"].map({True: "PASS", False: "FAIL"})

    wide = df.pivot_table(
        index=["question", "expected_keyword"],
        columns="prompt_version",
        values="result",
        aggfunc="first",
    ).reset_index()

    # Keep the question order from the test file
    order = list(dict.fromkeys(df["question"]))
    wide["_order"] = wide["question"].map({q: i for i, q in enumerate(order)})
    wide = wide.sort_values("_order").drop(columns="_order")

    wide.columns.name = None
    return wide


def write_readable_report(report: pd.DataFrame, out_path: str = "tests_data/readable_report.txt") -> None:
    """Write the report as a plain-text table with | separators."""
    headers = list(report.columns)
    rows = [[str(v) for v in row] for row in report.values.tolist()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(values):
        return " | ".join(v.ljust(w) for v, w in zip(values, widths))

    lines = [fmt(headers), "-+-".join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")