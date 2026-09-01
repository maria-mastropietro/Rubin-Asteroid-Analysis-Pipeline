import re
from pathlib import Path

def split_found_objects(input_file: Path, threshold: int = 165):
    """Split found_objects.txt into choose_physical.txt and choose_lsm.txt based on total obs threshold."""
    
    physical_objects = []
    lsm_objects = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("=") or "Provid" in line or "Permid" in line:
                continue
            
            # Parse line like: "  2007 VT346 ({'r': 59, 'i': 35, 'g': 24}) Total obs = 118 (obs_sbn_X05_2026-06-04.parquet)"
            match = re.search(r"Total obs = (\d+)", line)
            if match:
                total_obs = int(match.group(1))

                # Extract object name (everything before the parenthesis)
                obj_name = line.split("(")[0].strip()

                # Extract parquet file name from the end of the line
                parquet_match = re.search(r"\((obs_sbn_[^)]+\.parquet)\)", line)
                parquet_path = parquet_match.group(1) if parquet_match else None

                if total_obs >= threshold:
                    physical_objects.append((obj_name, parquet_path))
                else:
                    lsm_objects.append((obj_name, parquet_path))

    # Save physical objects
    physical_file = input_file.parent / "choose_physical.txt"
    with open(physical_file, "w", encoding="utf-8") as f:
        for obj, parquet in physical_objects:
            if parquet:
                f.write(f"{obj}\t{parquet}\n")
            else:
                f.write(f"{obj}\n")

    # Save LSM objects
    lsm_file = input_file.parent / "choose_lsm.txt"
    with open(lsm_file, "w", encoding="utf-8") as f:
        for obj, parquet in lsm_objects:
            if parquet:
                f.write(f"{obj}\t{parquet}\n")
            else:
                f.write(f"{obj}\n")

    print(f"Saved {len(physical_objects)} objects to {physical_file}")
    print(f"Saved {len(lsm_objects)} objects to {lsm_file}")

if __name__ == "__main__":
    input_file = Path(__file__).resolve().parent / "found_objects.txt"
    split_found_objects(input_file, threshold=165)
