import subprocess
import sys
import time
from datetime import datetime


def run_script(script_path, data_path, num_runs=10):
    for i in range(num_runs):
        print("\n" + "=" * 50)
        print(f"Starting run {i+1} of {num_runs}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50 + "\n")

        try:
            subprocess.run([sys.executable, script_path, data_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in run {i+1}: {e}")

        print("\n" + "=" * 50)
        print(f"Completed run {i+1}")
        print("=" * 50 + "\n")

        if i < num_runs - 1:
            time.sleep(2)


if __name__ == "__main__":
    script_path = sys.argv[1]
    data_path = sys.argv[2]
    run_script(script_path, data_path, 20)
