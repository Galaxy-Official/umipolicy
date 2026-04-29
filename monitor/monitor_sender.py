import json
import time
import os
from pathlib import Path
import subprocess
import datetime

MONITOR_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(MONITOR_DIR, ".."))
INPUT_JSON = os.path.join(MONITOR_DIR, "monitor_input.json")
OUTPUT_DIR = os.path.join(MONITOR_DIR, "monitor_output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "state_sum.json")
OBS_DEST = "obs://sai.liyl/lihong/monitor_output"

def find_latest_log(script_path):
    script_name = os.path.basename(script_path).replace('.sh', '')
    logs = list(Path(WORKSPACE_DIR).rglob(f"*{script_name}*.log"))
    if not logs:
        return None
    valid_logs = [f for f in logs if f.is_file() and os.path.getsize(f) > 0]
    if not valid_logs:
        return max(logs, key=os.path.getmtime)
    
    latest_log = max(valid_logs, key=os.path.getmtime)
    return latest_log

def parse_log(log_path):
    state = {
        "status": "Unknown",
        "progress": "",
        "error_trace": "",
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        if os.path.getsize(log_path) == 0:
            state["status"] = "Running (Empty Log)"
            return state
            
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        if not lines:
            state["status"] = "Running (Empty Log)"
            return state
            
        error_lines = []
        in_error = False
        tail_lines = lines[-200:]
        for line in tail_lines:
            if "Traceback (most recent call last)" in line or "Error:" in line or "Exception:" in line:
                in_error = True
            if in_error:
                error_lines.append(line.strip())
                
        if in_error:
            state["status"] = "Error"
            state["error_trace"] = "\n".join(error_lines)
            return state
            
        progress_line = ""
        for line in reversed(lines):
            line_lower = line.lower()
            if "epoch" in line_lower or "step" in line_lower or "loss" in line_lower or "it/s" in line_lower or "train_" in line_lower:
                progress_line = line.strip()
                break
                
        if progress_line:
            state["progress"] = progress_line
        else:
            for line in reversed(lines):
                if line.strip():
                    state["progress"] = line.strip()
                    break
        
        if any("done" in line.lower() or "completed" in line.lower() for line in tail_lines[-20:]):
             state["status"] = "Completed"
        else:
             state["status"] = "Running"
             
    except Exception as e:
        state["status"] = "Parse Error"
        state["error_trace"] = str(e)
        
    return state

def run_monitor():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    while True:
        try:
            scripts_to_monitor = []
            poll_interval = 300
            
            if os.path.exists(INPUT_JSON):
                with open(INPUT_JSON, 'r') as f:
                    config = json.load(f)
                if isinstance(config, dict) and "tasks" in config:
                    for task in config.get("tasks", []):
                        if task.get("enabled", True):
                            scripts_to_monitor.append(task.get("script"))
                    poll_interval = config.get("poll_interval_seconds", 300)
                elif isinstance(config, list):
                    scripts_to_monitor = config
            else:
                print(f"Warning: {INPUT_JSON} not found.")
                
            summary = {}
            for script in scripts_to_monitor:
                log_path = find_latest_log(script)
                if log_path:
                    state = parse_log(log_path)
                    try:
                        state["log_file"] = str(log_path.relative_to(WORKSPACE_DIR))
                    except ValueError:
                        state["log_file"] = str(log_path)
                    summary[script] = state
                else:
                    summary[script] = {
                        "status": "Not Started / Log Not Found",
                        "progress": "",
                        "error_trace": "",
                        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(summary, f, indent=4)
                
            print(f"[{datetime.datetime.now()}] Saved local state to {OUTPUT_JSON}")
            
            # Sync to OBS from MONITOR_DIR so "monitor_output" folder is found
            print(f"[{datetime.datetime.now()}] Syncing to OBS...")
            subprocess.run(["obsutil", "cp", "-r", "monitor_output", OBS_DEST], cwd=MONITOR_DIR, check=False)
            
        except Exception as e:
            print(f"[{datetime.datetime.now()}] Monitor loop error: {e}")
            
        time.sleep(poll_interval)

if __name__ == "__main__":
    run_monitor()
