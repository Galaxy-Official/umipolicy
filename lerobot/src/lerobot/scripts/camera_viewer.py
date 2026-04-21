import multiprocessing as mp
import cv2
import time

def viewer_process(queue: mp.Queue):
    while True:
        try:
            item = queue.get(timeout=1.0)
            if item == "STOP":
                break
            
            for k, img in item.items():
                cv2.imshow(k, img)
            cv2.waitKey(1)
        except Exception:
            pass
    cv2.destroyAllWindows()
