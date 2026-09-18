from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "metadata" / "extracted"
OUT_DIR = ROOT / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "DXSUM": DATA_DIR / "DXSUM_18Sep2026.csv",
    "REGISTRY": DATA_DIR / "REGISTRY_18Sep2026.csv",
    "ROSTER": DATA_DIR / "ROSTER_18Sep2026.csv",
}

tables = {}

for name, path in FILES.items():
    df = pd.read_csv(path, low_memory=False)

    if "RID" in df.columns:
        df["RID"] = pd.to_numeric(df["RID"], errors="coerce").astype("Int64")

    if "EXAMDATE" in df.columns:
        df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

    tables[name] = df


def basic_summary(name, df):
    out = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_rids": (
            int(df["RID"].nunique(dropna=True))
            if "RID" in df.columns else None
        ),
        "missing_rid": (
            int(df["RID"].isna().sum())
            if "RID" in df.columns else None
        ),
    }

    if "PHASE" in df.columns:
        out["phases"] = sorted(
            [str(x) for x in df["PHASE"].dropna().unique()]
        )

    if "EXAMDATE" in df.columns:
        valid = df["EXAMDATE"].dropna()
        out["examdate_min"] = (
            str(valid.min().date()) if len(valid) else None
        )
        out["examdate_max"] = (
            str(valid.max().date()) if len(valid) else None
        )
        out["missing_examdate"] = int(df["EXAMDATE"].isna().sum())

    key = [
        c for c in ["PHASE", "RID", "VISCODE2"]
        if c in df.columns
    ]

    if len(key) >= 2:
        out["duplicate_phase_rid_visit_rows"] = int(
            df.duplicated(key, keep=False).sum()
        )

    return out


summary = {
    name: basic_summary(name, df)
    for name, df in tables.items()
}


# ---------------------------------------------------
# Diagnosis inventory
# ---------------------------------------------------

dx = tables["DXSUM"]

print("\n" + "=" * 72)
print("DXSUM DIAGNOSIS DISTRIBUTION")
print("=" * 72)

if "DIAGNOSIS" in dx.columns:
    dx["DIAGNOSIS_NUM"] = pd.to_numeric(
        dx["DIAGNOSIS"], errors="coerce"
    ).astype("Int64")

    total_counts = (
        dx["DIAGNOSIS_NUM"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nOverall:")
    print(total_counts.to_string())

    if "PHASE" in dx.columns:
        phase_counts = pd.crosstab(
            dx["PHASE"],
            dx["DIAGNOSIS_NUM"],
            dropna=False
        )

        print("\nBy phase:")
        print(phase_counts.to_string())

        summary["DXSUM"]["diagnosis_by_phase"] = {
            str(idx): {
                str(col): int(value)
                for col, value in row.items()
            }
            for idx, row in phase_counts.iterrows()
        }

    summary["DXSUM"]["diagnosis_counts"] = {
        str(k): int(v)
        for k, v in total_counts.items()
    }


# ---------------------------------------------------
# RID overlap
# ---------------------------------------------------

rid_sets = {}

for name, df in tables.items():
    if "RID" in df.columns:
        rid_sets[name] = set(
            int(x)
            for x in df["RID"].dropna().unique()
        )

if all(k in rid_sets for k in ["DXSUM", "REGISTRY", "ROSTER"]):
    summary["RID_OVERLAP"] = {
        "all_three": len(
            rid_sets["DXSUM"]
            & rid_sets["REGISTRY"]
            & rid_sets["ROSTER"]
        ),
        "dxsum_not_roster": len(
            rid_sets["DXSUM"] - rid_sets["ROSTER"]
        ),
        "dxsum_not_registry": len(
            rid_sets["DXSUM"] - rid_sets["REGISTRY"]
        ),
    }


# ---------------------------------------------------
# RID <-> PTID consistency
# ---------------------------------------------------

for name, df in tables.items():
    if {"RID", "PTID"}.issubset(df.columns):

        tmp = df[["RID", "PTID"]].dropna().drop_duplicates()

        rid_conflicts = (
            tmp.groupby("RID")["PTID"]
            .nunique()
            .gt(1)
            .sum()
        )

        ptid_conflicts = (
            tmp.groupby("PTID")["RID"]
            .nunique()
            .gt(1)
            .sum()
        )

        summary[name]["rid_with_multiple_ptid"] = int(rid_conflicts)
        summary[name]["ptid_with_multiple_rid"] = int(ptid_conflicts)


# ---------------------------------------------------
# Console report
# ---------------------------------------------------

print("\n" + "=" * 72)
print("TABLE INVENTORY")
print("=" * 72)

for name in ["DXSUM", "REGISTRY", "ROSTER"]:
    print(f"\n{name}")
    for key, value in summary[name].items():
        if key != "diagnosis_by_phase":
            print(f"  {key}: {value}")


print("\nRID overlap:")
print(summary.get("RID_OVERLAP", {}))


# ---------------------------------------------------
# Save only aggregate information
# No participant identifiers are exported.
# ---------------------------------------------------

output = OUT_DIR / "metadata_inventory_18Sep2026.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nSaved aggregate inventory to:\n{output}")
