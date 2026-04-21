import multiprocessing as mp
import queue
import cv2
import traceback

def viewer_process(q: mp.Queue):
    current_img_dict = {}
    while True:
        try:
            # Short timeout so we can constantly pump cv2.waitKey()
            item = q.get(timeout=0.03)
            if item == "STOP":
                break
            current_img_dict = item  # update latest images
        except queue.Empty:
            pass
        except Exception as e:
            print("VIEWER PROCESS EXCEPTION:")
            traceback.print_exc()

        # Render continuously with the last known images
        if current_img_dict:
            for k, img in current_img_dict.items():
                cv2.imshow(k, img)
        cv2.waitKey(10)  # Pump the GUI event loop
        
    cv2.destroyAllWindows()
