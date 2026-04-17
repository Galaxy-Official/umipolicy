import json
import numpy as np

target_objects = ["clapper",  "document_box", "bucket", "drawer"]
SR_m = json.load(open("/home/rhos/xiaoyang/pred_keypoints/capture_0301/SR_m.json", "r"))
SR_p = json.load(open("/home/rhos/xiaoyang/pred_keypoints/capture_0301/SR_p.json", "r"))

mapping = {
    "S": 1,
    "F-G": -1,
    "F-A": -2,
    "F-A,F-G": -3,
    "F-G,F-A": -3
}

for obj in target_objects:
    print(obj)
    m_evaluations = np.array([mapping[v] for v in SR_m[obj].values()])
    m_fail = len(np.where(m_evaluations < 0)[0])
    m_A = len(np.where(m_evaluations < -1)[0]) # -2 & -3
    m_G = len(np.where(m_evaluations==-1)[0])\
            + len(np.where(m_evaluations==-3)[0])# -1, -3
    print("m-fail", "total", m_fail, 
            "A", m_A, 
            "G", m_G,
            )
    print("m-fail",  
            "A", np.round(100*m_A/m_fail, 2),
            "G", np.round(100*m_G/m_fail, 2),
            "\n"
            )

    p_evaluations = np.array([mapping[v] for v in SR_p[obj].values()])
    p_fail = len(np.where(p_evaluations < 0)[0])
    p_A = len(np.where(p_evaluations < -1)[0]) # -2 & -3
    p_G = len(np.where(p_evaluations==-1)[0])\
            + len(np.where(p_evaluations==-3)[0])# -1, -3
    print("p-fail", "total", p_fail, 
            "A", p_A, 
            "G", p_G,
            )
    print("p-fail",  
            "A", np.round(100*p_A/p_fail, 2),
            "G", np.round(100*p_G/p_fail, 2),
            )

    

    print("-----------")


