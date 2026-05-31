import os
import sys
import json
import csv
from nav_pipeline import NavPipeline

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_TASK_CSV = os.path.join(REPO_ROOT, "dataset", "relative_coordinates_export.csv")
UE_ORIGIN_X = 1910.0
UE_ORIGIN_Y = -458.0
UE_ORIGIN_Z = 100.0


def ue_world_to_airsim_ned(x, y, z):
    """
    Convert UE world coordinates into the AirSim local NED frame used by Model1/Model6.
    This dataset is stored in UE world coordinates (centimeter-scale values).
    """
    rel_x = (float(x) - UE_ORIGIN_X) / 100.0
    rel_y = (float(y) - UE_ORIGIN_Y) / 100.0
    rel_z = -((float(z) - UE_ORIGIN_Z) / 100.0)
    return [rel_x, rel_y, rel_z]


def normalize_task_point(x, y, z):
    """
    Auto-detect task coordinate scale.
    Large magnitudes are treated as UE world coordinates and converted.
    Smaller values are assumed to already be AirSim-local.
    """
    values = [float(x), float(y), float(z)]
    if max(abs(v) for v in values) > 500.0:
        return ue_world_to_airsim_ned(*values), "ue_world"
    return values, "airsim_local"

def run_batch_from_csv(csv_path):
    """
    Run batch test from a CSV file.
    CSV format: start_x, start_y, start_z, end_x, end_y, end_z
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    task_list = []
    source_modes = set()
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start, start_mode = normalize_task_point(row['start_x'], row['start_y'], row['start_z'])
            end, end_mode = normalize_task_point(row['end_x'], row['end_y'], row['end_z'])
            source_modes.update([start_mode, end_mode])
            task_list.append({
                'start': start,
                'end': end,
            })

    if not task_list:
        print("Error: No tasks found in CSV.")
        return

    if source_modes == {"ue_world"}:
        print(
            f"Detected UE-world task coordinates in {csv_path}; "
            f"converted to AirSim local NED using origin "
            f"({UE_ORIGIN_X}, {UE_ORIGIN_Y}, {UE_ORIGIN_Z}) and scale /100."
        )
    elif "ue_world" in source_modes:
        print(f"Mixed coordinate modes detected in {csv_path}; converted large-magnitude rows automatically.")

    pipeline = NavPipeline(planner_mode="occupancy")
    
    print("Initializing environment...")
    # pipeline.launch_airsim() # Uncomment if you want the script to launch AirSim
    success, msg = pipeline.init_navigation()
    if not success:
        print(f"Initialization failed: {msg}")
        return

    # Run batch
    summary = pipeline.run_batch_test(task_list, status_callback=print)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "batch_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "="*30)
    print("BATCH TEST SUMMARY")
    print(f"Total Tasks:  {summary['total']}")
    print(f"Success:      {summary['arrived']}")
    print(f"Arrival Rate: {summary['arrival_rate']}")
    print(f"Full results saved to: {output_path}")
    print("="*30)

if __name__ == "__main__":
    run_batch_from_csv(DEFAULT_TASK_CSV)
