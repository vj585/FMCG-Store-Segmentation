import os
import subprocess

def run_script(script_path):
    print(f"========================================")
    print(f"Running {script_path}...")
    print(f"========================================")
    result = subprocess.run(["python", script_path])
    if result.returncode != 0:
        print(f"ERROR: {script_path} failed with code {result.returncode}")
        exit(1)
        
if __name__ == "__main__":
    if not os.path.basename(os.getcwd()) == "Assessment-2-Store-Segmentation":
        if os.path.exists("Assessment-2-Store-Segmentation"):
            os.chdir("Assessment-2-Store-Segmentation")
        else:
            print("Please run this script from the root project directory or the Assessment-2-Store-Segmentation directory.")
            exit(1)
            
    run_script(os.path.join("src", "a2_segmentation.py"))
    
    print("Assessment 2 Pipeline Complete!")
