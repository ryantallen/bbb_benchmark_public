import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


def parse_tabular_text(text) -> dict:
    """
    Parse a tab‑delimited two‑row table held in a string.
    Returns {} if the input is None/blank or malformed.
    """
    if not text or not isinstance(text, str):      # <- ADDED
        return {}

    lines = [ln.rstrip("\r") for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        headers = [h.strip() for h in lines[0].split("\t")]
        values  = [v.strip() for v in lines[1].split("\t")]
        if len(headers) == len(values):
            return dict(zip(headers, values))
    return {}


def classify_run(best_scores_dict: dict, num_valid_years: int) -> tuple[str, int | None]:

    status_raw = (best_scores_dict.get("Status") or "").strip()

    # --- Complete -----------------------------------------------------------
    if status_raw == "Complete":
        return "Complete", 10

    # --- Fired --------------------------------------------------------------
    if status_raw.startswith("Fired"):
        if m := re.search(r"Year\s*(\d+)", status_raw):
            return "Fired", int(m.group(1))
        return "Fired", None

    # --- LLM error ----------------------------------------------------------
    if status_raw.startswith("Playing"):
        if m := re.search(r"Year\s*(\d+)", status_raw):
            yr = int(m.group(1))
            if yr > 3:
                return "LLM error", yr
        return "Program Error", None

    # --- Fallback rules -----------------------------------------------------
    if num_valid_years == 0:          # never got off the ground
        return "Program Error", None

    # has some data but we can't classify → treat as LLM error
    return "Invalid", None


def process_json_files(file_paths_with_providers):
    """
    Read every run in every results‑file and return one tidy row per run.
    Adds columns:  status  |  year_ended  |  provider  |  is_open_source
    """
    all_rows = []

    for tup in file_paths_with_providers:
        # Backward/forward compatible tuple unpacking
        # Expected: (file_path, provider, is_open_source)
        if len(tup) == 3:
            file_path, provider, is_open_source = tup
        else:
            file_path, provider = tup
            is_open_source = False
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        header = data.get("header", {})
        model  = header.get("MODEL") or header.get("model")
        difficulty = header.get("difficulty") or (
            "Basic" if "Basic" in file_path
            else "Advanced" if "Advanced" in file_path
            else "Intermediate"
        )

        runs = data.get("runs", {})
        for run_id, run_data in runs.items():
            # ------- count valid year‑entries --------------------------------
            num_valid_years = 0
            for yr in range(3, 11):
                entry = run_data.get(f"year_{yr}")
                if isinstance(entry, dict) and "error" not in entry:
                    num_valid_years += 1

            # ------- grab run‑level tables -----------------------------------
            best_scores_txt = (run_data.get("best_scores") or "")          
            sim_summary_txt = (run_data.get("simulation_summary") or "")    

            best_scores = parse_tabular_text(best_scores_txt)
            sim_summary = parse_tabular_text(sim_summary_txt)

            # ------- classify this run ---------------------------------------
            status, year_ended = classify_run(best_scores, num_valid_years)

            row = {
                "file_name": os.path.basename(file_path),
                "model": model,
                "instructions": header.get("instructions"),
                "difficulty": difficulty,
                "run_id": run_id,
                "num_valid_years": num_valid_years,
                "num_runs": header.get("NUM_RUNS"),
                "instruction_set": header.get("instruction_set"),
                "background": header.get("background"),
                "format_instructions": header.get("format_instructions"),
                "timestamp": header.get("timestamp"),
                "market_segment": header.get("market_segment"),
                # ---- new fields ----
                "provider": provider,
                "status": status,
                "year_ended": year_ended,
                "is_open_source": is_open_source,
                # Inferred from filename tokens (back-compat with older JSON headers)
                "mask_mode": (
                    "unmasked" if "unmasked" in os.path.basename(file_path).lower()
                    else "masked" if "masked" in os.path.basename(file_path).lower()
                    else "unknown"
                ),
            }

            # Heuristic/metadata override: if header explicitly flags open source, trust it
            header_open_source = (
                bool(header.get("is_open_source"))
                or bool(header.get("open_source"))
                or (isinstance(header.get("license"), str) and any(
                    kw in header.get("license").lower() for kw in [
                        "apache", "mit", "bsd", "mpl", "gpl", "cc-by", "open"]
                ))
            )
            if header_open_source:
                row["is_open_source"] = True

            # add wide columns for AGM units and price by year (Year 3..10)
            for yr in range(3, 11):
                year_key = f"year_{yr}"
                ydata = run_data.get(year_key, {})
                if isinstance(ydata, dict):
                    row[f"{year_key}_agm_units"] = ydata.get("agm_units")
                    row[f"{year_key}_agm_price"] = ydata.get("agm_price")
                else:
                    row[f"{year_key}_agm_units"] = None
                    row[f"{year_key}_agm_price"] = None

            for yr in range(3, 11):
                year_key = f"year_{yr}"
                ydata = run_data.get(year_key, {})
                if isinstance(ydata, dict):
                    row[f"{year_key}_supercapacitor_units"] = ydata.get("supercapacitor_units")
                    row[f"{year_key}_supercapacitor_price"] = ydata.get("supercapacitor_price")
                else:
                    row[f"{year_key}_supercapacitor_units"] = None
                    row[f"{year_key}_supercapacitor_price"] = None

            # copy parsed scores / summaries
            for k, v in best_scores.items():
                row[f"best_scores_{k}"] = v
            for k, v in sim_summary.items():
                row[f"simulation_summary_{k}"] = v

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Convert raw_results JSON files into analysis-ready tidy CSV.")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Where to write tidy_results.csv (default: analysis/data/tidy_results.csv).",
    )
    parser.add_argument(
        "--raw_results_dir",
        default=None,
        help="Directory containing raw_results/* (default: <repo_root>/raw_results).",
    )
    parser.add_argument(
        "--include_unmasked",
        action="store_true",
        help="Include 'unmasked' rows if any are present. Default is masked-only.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    output_csv = Path(args.output_csv) if args.output_csv else (script_dir / "data" / "tidy_results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Find all provider folders and collect JSON files with provider info
    raw_results_dir = Path(args.raw_results_dir) if args.raw_results_dir else (repo_root / "raw_results")
    
    # Find all folders that match the pattern "all_results_*"
    provider_folders = [p for p in raw_results_dir.glob("all_results_*") if p.is_dir()]
    
    if not provider_folders:
        print(f"No provider folders found in {raw_results_dir} directory.")
        print("Please ensure folders matching 'all_results_*' pattern exist.")
        return
    
    # Collect all JSON files with their provider information
    file_paths_with_providers = []
    for folder in provider_folders:
        folder_name = folder.name
        provider_root = folder_name.replace("all_results_", "")

        # Special handling for OpenRouter: recurse one+ levels and treat first subfolder as provider
        if provider_root == "openrouter":
            # e.g., .../all_results_openrouter/<subprovider>/**/*.json
            for subdir in [d for d in folder.iterdir() if d.is_dir()]:
                sub_provider = subdir.name
                for json_file in subdir.glob("**/*.json"):
                    file_paths_with_providers.append((str(json_file), sub_provider, True))
        else:
            # Non-openrouter providers: direct JSONs under the folder
            for json_file in folder.glob("*.json"):
                file_paths_with_providers.append((str(json_file), provider_root, False))
    
    # Check if any JSON files were found
    if not file_paths_with_providers:
        print(f"No JSON files found in any provider folders under {raw_results_dir}.")
        print("Please ensure the directories exist and contain *.json files.")
        return
    
    print(f"Found {len(file_paths_with_providers)} JSON files to process from {len(provider_folders)} provider folders.")
    # Summarize counts per resolved provider label (including openrouter subproviders)
    provider_counts = {}
    for tup in file_paths_with_providers:
        if len(tup) == 3:
            _, prov, _ = tup
        else:
            _, prov = tup
        provider_counts[prov] = provider_counts.get(prov, 0) + 1
    for prov, cnt in sorted(provider_counts.items()):
        print(f"  {prov}: {cnt} files")
    
    df_runs = process_json_files(file_paths_with_providers)
    
    # Check if the DataFrame has any rows
    if df_runs.empty:
        print("No data was extracted from the JSON files.")
        print("DataFrame is empty, saving empty CSV and exiting.")
        df_runs.to_csv(output_csv, index=False)
        return

    # Default: masked-only analysis (unmasked results are deprecated in this repo)
    if not args.include_unmasked and "mask_mode" in df_runs.columns:
        df_runs = df_runs.loc[df_runs["mask_mode"].astype(str).str.lower().ne("unmasked")].copy()
        
    print(f"Processed data contains {len(df_runs)} rows.")
    print(df_runs.head())  # quick peek
    df_runs.to_csv(output_csv, index=False)

    # -----------------------------------------------------------------------
    # Run‑count report – ignore only the Program Error rows
    # -----------------------------------------------------------------------
    print("\nCalculating run counts (Complete, Fired, or LLM error only)…")

    # Check if 'status' column exists
    if 'status' not in df_runs.columns:
        print("Error: 'status' column not found in the processed data.")
        return
        
    valid_df = df_runs[df_runs["status"] != "Program Error"]

    if valid_df.empty:
        print("No qualifying runs found.")
        return

    # include every provider / model / instruction_set / difficulty combination
    all_combos = df_runs[["provider", "model", "instruction_set", "difficulty"]].drop_duplicates()
    counts = (
        valid_df.groupby(["provider", "model", "instruction_set", "difficulty"])
        .size()
        .reset_index(name="run_count")
    )
    final = all_combos.merge(counts, how="left").fillna({"run_count": 0}).astype({"run_count": int})
    final = final.sort_values(["provider", "model", "difficulty", "instruction_set"])

    print("\n--- Qualifying Run Counts ---")
    for _, r in final.iterrows():
        instr = (r["instruction_set"][:75] + "…") if r["instruction_set"] and len(r["instruction_set"]) > 78 else r["instruction_set"]
        print(f"Provider: {r['provider']}, Model: {r['model']}, Difficulty: {r['difficulty']}")
        print(f"  Instruction Set: {instr}")
        print(f"  Count: {r['run_count']}")
        print("-" * 20)

    print("--- End of Report ---")


if __name__ == "__main__":
    main()
