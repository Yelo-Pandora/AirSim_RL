import os
import sys
import json
import csv
from nav_pipeline import NavPipeline

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def run_batch_from_csv(csv_path):
    """
    Run batch test from a CSV file.
    CSV format: start_x, start_y, start_z, end_x, end_y, end_z
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    task_list = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_list.append({
                'start': [float(row['start_x']), float(row['start_y']), float(row['start_z'])],
                'end': [float(row['end_x']), float(row['end_y']), float(row['end_z'])]
            })

    if not task_list:
        print("Error: No tasks found in CSV.")
        return

    pipeline = NavPipeline()
    
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
    # Example: Create a dummy CSV for testing if it doesn't exist
    test_csv = os.path.join(os.path.dirname(__file__), "test_tasks.csv")
    if not os.path.exists(test_csv):
        with open(test_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z'])
            writer.writerow([0, 0, -2, 20, 20, -2])
            writer.writerow([0, 0, -2, 40, 0, -2])
            writer.writerow([10, 10, -5, -10, -10, -5])
        print(f"Created example CSV: {test_csv}")

    run_batch_from_csv(test_csv)
