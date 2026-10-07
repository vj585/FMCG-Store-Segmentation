import os
import sys
import subprocess

def run_script(script_path):
    print(f"========================================")
    print(f"Running {script_path}...")
    print(f"========================================")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"ERROR: {script_path} failed with code {result.returncode}")
        exit(1)
        
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
            
    run_script(os.path.join("src", "a2_segmentation.py"))
    
    print("Assessment 2 Pipeline Complete!")
