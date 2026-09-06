import os
import hashlib
import json
import datetime

HASH_STORAGE_PATH = "data/file_hashes/file_hashes.json"
GAME_FILES_PATH = "game_files/"
LOG_PATH = "logs/alert_logs/file_integrity_log.txt"


def calculate_hash(file_path):
    """
    Calculate SHA-256 hash of a file
    """
    sha256 = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    except PermissionError:
        print(f"Permission denied: {file_path}")
        return None


def scan_game_files(directory):
    """
    Scan all files in game directory
    and return their hashes
    """
    file_hashes = {}

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return file_hashes

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = calculate_hash(file_path)

            if file_hash:
                file_hashes[file_path] = {
                    "hash": file_hash,
                    "size": os.path.getsize(file_path),
                    "last_modified": str(
                        datetime.datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        )
                    )
                }

    return file_hashes


def save_hashes(file_hashes):
    """
    Save file hashes to JSON storage
    """
    os.makedirs(os.path.dirname(HASH_STORAGE_PATH), exist_ok=True)

    data = {
        "timestamp": str(datetime.datetime.now()),
        "total_files": len(file_hashes),
        "files": file_hashes
    }

    with open(HASH_STORAGE_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Hashes saved to: {HASH_STORAGE_PATH}")
    print(f"Total files scanned: {len(file_hashes)}")


def load_stored_hashes():
    """
    Load previously stored hashes
    """
    if not os.path.exists(HASH_STORAGE_PATH):
        print("No stored hashes found.")
        print("Run baseline scan first.")
        return None

    with open(HASH_STORAGE_PATH, "r") as f:
        data = json.load(f)

    return data


def compare_hashes(stored_data, current_hashes):
    """
    Compare stored hashes with current hashes
    to detect file modifications
    """
    results = {
        "modified_files": [],
        "new_files": [],
        "deleted_files": [],
        "clean_files": [],
        "total_suspicious": 0
    }

    stored_files = stored_data["files"]

    # Check for modified or deleted files
    for file_path, stored_info in stored_files.items():
        if file_path not in current_hashes:
            results["deleted_files"].append(file_path)
        else:
            current_hash = current_hashes[file_path]["hash"]
            stored_hash = stored_info["hash"]

            if current_hash != stored_hash:
                results["modified_files"].append({
                    "file": file_path,
                    "stored_hash": stored_hash,
                    "current_hash": current_hash,
                    "original_size": stored_info["size"],
                    "current_size": current_hashes[file_path]["size"]
                })
            else:
                results["clean_files"].append(file_path)

    # Check for new files
    for file_path in current_hashes:
        if file_path not in stored_files:
            results["new_files"].append(file_path)

    # Count suspicious files
    results["total_suspicious"] = (
        len(results["modified_files"]) +
        len(results["new_files"]) +
        len(results["deleted_files"])
    )

    return results


def log_results(results):
    """
    Log file integrity check results
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    timestamp = str(datetime.datetime.now())

    log_entry = f"""
========================================
File Integrity Check
Timestamp: {timestamp}
========================================
Modified Files : {len(results['modified_files'])}
New Files      : {len(results['new_files'])}
Deleted Files  : {len(results['deleted_files'])}
Clean Files    : {len(results['clean_files'])}
Total Suspicious: {results['total_suspicious']}
========================================
"""

    if results["modified_files"]:
        log_entry += "\nMODIFIED FILES:\n"
        for item in results["modified_files"]:
            log_entry += f"  - {item['file']}\n"
            log_entry += f"    Stored Hash : {item['stored_hash']}\n"
            log_entry += f"    Current Hash: {item['current_hash']}\n"

    if results["new_files"]:
        log_entry += "\nNEW FILES DETECTED:\n"
        for f in results["new_files"]:
            log_entry += f"  - {f}\n"

    if results["deleted_files"]:
        log_entry += "\nDELETED FILES:\n"
        for f in results["deleted_files"]:
            log_entry += f"  - {f}\n"

    with open(LOG_PATH, "a") as f:
        f.write(log_entry)

    print(log_entry)


def baseline_scan():
    """
    Perform initial baseline scan
    and store file hashes
    """
    print("Starting baseline scan...")
    print(f"Scanning directory: {GAME_FILES_PATH}")

    file_hashes = scan_game_files(GAME_FILES_PATH)

    if not file_hashes:
        print("No files found to scan.")
        return

    save_hashes(file_hashes)
    print("Baseline scan complete!")


def integrity_check():
    """
    Compare current file hashes
    with stored baseline hashes
    """
    print("Starting integrity check...")

    stored_data = load_stored_hashes()

    if not stored_data:
        print("Please run baseline scan first.")
        return

    print(f"Baseline timestamp: {stored_data['timestamp']}")
    print(f"Total files in baseline: {stored_data['total_files']}")

    print("Scanning current files...")
    current_hashes = scan_game_files(GAME_FILES_PATH)

    print("Comparing hashes...")
    results = compare_hashes(stored_data, current_hashes)

    log_results(results)

    if results["total_suspicious"] == 0:
        print("All files are clean. No tampering detected.")
    else:
        print(f"WARNING: {results['total_suspicious']} suspicious files detected!")

    return results


def create_dummy_game_files():
    """
    Create dummy game files for testing
    """
    os.makedirs(GAME_FILES_PATH, exist_ok=True)

    dummy_files = {
        "game_files/game_engine.dll": "original game engine content",
        "game_files/player_config.cfg": "player configuration data",
        "game_files/anti_cheat.dll": "anti cheat module content",
        "game_files/game_assets.pak": "game assets package data",
        "game_files/network_module.dll": "network module content"
    }

    for file_path, content in dummy_files.items():
        with open(file_path, "w") as f:
            f.write(content)

    print(f"Created {len(dummy_files)} dummy game files.")


def simulate_file_tampering():
    """
    Simulate file tampering for testing
    """
    tampered_file = "game_files/anti_cheat.dll"

    if os.path.exists(tampered_file):
        with open(tampered_file, "w") as f:
            f.write("TAMPERED CONTENT - HACK INJECTED")

        print(f"Simulated tampering on: {tampered_file}")
    else:
        print("Game files not found. Run create_dummy_game_files first.")


def main():
    print("=" * 50)
    print("File Integrity Verification System")
    print("=" * 50)

    # Step 1: Create dummy game files for testing
    print("\n[1] Creating dummy game files...")
    create_dummy_game_files()

    # Step 2: Run baseline scan
    print("\n[2] Running baseline scan...")
    baseline_scan()

    # Step 3: Simulate tampering
    print("\n[3] Simulating file tampering...")
    simulate_file_tampering()

    # Step 4: Run integrity check
    print("\n[4] Running integrity check...")
    results = integrity_check()

    print("\nFile Integrity Verification Complete!")


if __name__ == "__main__":
    main()