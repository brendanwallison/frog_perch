import csv
import re
from pathlib import Path

def shift_audio_timestamps(directory_path: str | Path, log_file: str = "rename_log.csv"):
    target_dir = Path(directory_path)
    
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"❌ Error: Directory '{target_dir}' does not exist.")
        return

    # Regex breaks the filename into three parts:
    # Group 1: Prefix and Date (e.g., "P2__20241102_")
    # Group 2: The Hour (e.g., "11")
    # Group 3: The Minutes, Seconds, and Suffix (e.g., "0000_SYNC.WAV")
    pattern = re.compile(r"^(.*?\d{8}_)(\d{2})(\d{4}_SYNC\.[a-zA-Z0-9]+)$")
    
    log_records = []
    success_count = 0
    skip_count = 0
    
    print(f"⏳ Scanning directory: {target_dir}...")

    for file_path in target_dir.iterdir():
        if not file_path.is_file():
            continue
            
        match = pattern.match(file_path.name)
        if match:
            prefix = match.group(1)
            hour_str = match.group(2)
            suffix = match.group(3)
            
            # Shift the hour forward by 1 and pad with a leading zero if necessary
            new_hour = int(hour_str) + 1
            new_hour_str = f"{new_hour:02d}"
            
            new_filename = f"{prefix}{new_hour_str}{suffix}"
            new_file_path = target_dir / new_filename
            
            # Safety check to prevent overwriting existing files
            if new_file_path.exists():
                print(f"⚠️ Skipping {file_path.name}: Target {new_filename} already exists.")
                skip_count += 1
                continue
                
            # Perform the filesystem rename
            file_path.rename(new_file_path)
            
            log_records.append({
                "original_filename": file_path.name,
                "new_filename": new_filename
            })
            success_count += 1

    # Write the audit log to CSV
    if log_records:
        log_path = target_dir / log_file
        with open(log_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["original_filename", "new_filename"])
            writer.writeheader()
            writer.writerows(log_records)
        print(f"✅ Successfully renamed {success_count} files.")
        print(f"📄 Audit log saved to: {log_path}")
    else:
        print("ℹ️ No files matching the pattern were found to rename.")
        
    if skip_count > 0:
        print(f"⚠️ Skipped {skip_count} files to prevent overwriting.")

if __name__ == "__main__":
    # --- Set your target directory here ---
    AUDIO_DIRECTORY = "/home/breallis/datasets/frog_calls/gabon_full/P2_nn_features" 
    
    shift_audio_timestamps(AUDIO_DIRECTORY)