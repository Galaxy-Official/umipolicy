import time
import subprocess
import os
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer
import datetime
import sys

MONITOR_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: we use monitor_output as the LOCAL_DEST folder name inside MONITOR_DIR
LOCAL_DEST = os.path.join(MONITOR_DIR, "monitor_output")
OBS_SRC = "obs://sai.liyl/lihong/monitor_output"
PORT = 8080

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/monitor_dashboard.html'
        return super().do_GET()

def sync_obs_loop():
    while True:
        print(f"[{datetime.datetime.now()}] Syncing from OBS...")
        try:
            # Sync to the local monitor directory
            subprocess.run(["obsutil", "cp", "-r", OBS_SRC, LOCAL_DEST], cwd=MONITOR_DIR, check=False)
        except Exception as e:
            print(f"[{datetime.datetime.now()}] Sync error: {e}")
        time.sleep(300)

if __name__ == "__main__":
    if not os.path.exists(LOCAL_DEST):
        os.makedirs(LOCAL_DEST)
        
    print(f"Starting Background Sync from {OBS_SRC} to {LOCAL_DEST} every 5 mins.")
    sync_thread = threading.Thread(target=sync_obs_loop, daemon=True)
    sync_thread.start()
    
    print(f"Starting Local Dashboard Server on http://localhost:{PORT}")
    # Change working directory so SimpleHTTPRequestHandler serves files from here
    os.chdir(MONITOR_DIR)
    server = HTTPServer(("", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        sys.exit(0)
