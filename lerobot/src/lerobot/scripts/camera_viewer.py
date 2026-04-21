import multiprocessing as mp
import queue
import cv2
import traceback
import sys

def viewer_process(q: mp.Queue):
    print(">>> [Process] Camera Viewer Started! Waiting for images...", flush=True)
    current_img_dict = {}
    while True:
        try:
            item = q.get(timeout=0.03)
            if item == "STOP":
                print(">>> [Process] Camera Viewer Stopped.", flush=True)
                break
            current_img_dict = item
        except queue.Empty:
            pass
        except Exception as e:
            print(">>> [Process] VIEWER PROCESS EXCEPTION:", flush=True)
            traceback.print_exc()

        if current_img_dict:
            for k, img in current_img_dict.items():
                cv2.imshow(k, img)
        cv2.waitKey(10)
        
    cv2.destroyAllWindows()
