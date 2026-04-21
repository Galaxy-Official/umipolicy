import threading
import queue
import cv2
import traceback

def viewer_thread_func(q: queue.Queue):
    print(">>> [Thread] Camera Viewer Started! Waiting for images...", flush=True)
    current_img_dict = {}
    while True:
        try:
            item = q.get(timeout=0.03)
            if item == "STOP":
                print(">>> [Thread] Camera Viewer Stopped.", flush=True)
                break
            current_img_dict = item
        except queue.Empty:
            pass
        except Exception as e:
            print(">>> [Thread] VIEWER THREAD EXCEPTION:", flush=True)
            traceback.print_exc()

        if current_img_dict:
            for k, img in current_img_dict.items():
                cv2.imshow(k, img)
        cv2.waitKey(10)  # Pump the GUI event loop in the background thread
        
    cv2.destroyAllWindows()
