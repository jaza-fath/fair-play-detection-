import sys
import os
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = [
    "data/raw/gameplay_data.csv",
    "data/processed/train_data.csv",
    "data/processed/test_data.csv",
    "models/saved_models/scaler.pkl",
    "models/anomaly_detection/isolation_forest_model.pkl",
    "models/classification/random_forest_model.pkl",
    "data/file_hashes/file_hashes.json",
    "data/database/fairplay.db"
]

MODULES_TO_RUN = [
    "src/data_collection/generate_gameplay_data.py",
    "src/preprocessing/preprocess_gameplay_data.py",
    "src/anomaly_detection/train_anomaly_model.py",
    "src/file_integrity/file_integrity_checker.py",
    "src/classification/train_classification_model.py",
    "src/explainable_ai/explain_player_prediction.py",
    "src/database/database_manager.py"
]


def file_exists(path):
    return os.path.exists(os.path.join(PROJECT_ROOT, path))


def check_required_files():
    print("=" * 70)
    print("FINAL INTEGRATION TEST - FILE CHECK")
    print("=" * 70)

    missing = []
    for path in REQUIRED_FILES:
        if file_exists(path):
            print(f"[OK]   {path}")
        else:
            print(f"[MISS] {path}")
            missing.append(path)

    print("\nSummary:")
    print(f"Total required files : {len(REQUIRED_FILES)}")
    print(f"Missing files        : {len(missing)}")

    return missing


def run_script(script_path):
    full_path = os.path.join(PROJECT_ROOT, script_path)
    print("\n" + "=" * 70)
    print(f"RUNNING: {script_path}")
    print("=" * 70)

    if not os.path.exists(full_path):
        print(f"[SKIP] File not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, full_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )

        print(result.stdout)

        if result.stderr:
            print("[STDERR]")
            print(result.stderr)

        if result.returncode == 0:
            print(f"[SUCCESS] {script_path}")
            return True
        else:
            print(f"[FAILED] {script_path}")
            return False

    except Exception as e:
        print(f"[ERROR] Could not run {script_path}: {e}")
        return False


def generate_final_report(file_missing, module_results):
    report_path = os.path.join(PROJECT_ROOT, "final_test_report.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("AI-BASED FAIR PLAY DETECTION SYSTEM")
    lines.append("FINAL INTEGRATION TEST REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("FILE CHECK RESULTS")
    lines.append("-" * 70)
    if file_missing:
        lines.append("Missing Files:")
        for item in file_missing:
            lines.append(f"- {item}")
    else:
        lines.append("All required files are present.")

    lines.append("")
    lines.append("MODULE EXECUTION RESULTS")
    lines.append("-" * 70)
    for script, success in module_results.items():
        status = "PASS" if success else "FAIL"
        lines.append(f"{status}: {script}")

    lines.append("")
    lines.append("FINAL STATUS")
    lines.append("-" * 70)

    if not file_missing and all(module_results.values()):
        lines.append("SYSTEM STATUS: READY")
        lines.append("All core modules executed successfully.")
    else:
        lines.append("SYSTEM STATUS: ISSUES FOUND")
        lines.append("Some files or modules need attention.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nFinal report saved to: {report_path}")


def main():
    print("=" * 70)
    print("AI-BASED FAIR PLAY DETECTION SYSTEM - FINAL INTEGRATION TEST")
    print("=" * 70)

    # 1. Check required files
    missing_files = check_required_files()

    # 2. Run modules in sequence
    module_results = {}
    for script in MODULES_TO_RUN:
        module_results[script] = run_script(script)

    # 3. Generate final report
    generate_final_report(missing_files, module_results)

    # 4. Print final result
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if not missing_files and all(module_results.values()):
        print("✅ All systems tested successfully.")
        print("✅ Fair play detection pipeline is functioning.")
        print("✅ AI models, file integrity, database, and reporting are ready.")
    else:
        print("⚠ Some components need attention. Check the report file.")


if __name__ == "__main__":
    main()