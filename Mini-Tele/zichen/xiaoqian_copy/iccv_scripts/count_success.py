import json
import numpy as np

target_objects = [
"notebook", 
"box", 
"clapper", 
"door", 
"drawer", 
"bucket", 
"document_box", 
"folder",
"lamp"
]

SR_m = json.load(open("/home/rhos/xiaoyang/pred_keypoints/capture_0301/SR_m.json", "r"))
SR_p = json.load(open("/home/rhos/xiaoyang/pred_keypoints/capture_0301/SR_p.json", "r"))

mapping = {
    "S": 1,
    "F-G": -1,
    "F-A": -2,
    "F-A,F-G": -3,
    "F-G,F-A": -3
}

m_tot_success = 0
p_tot_success = 0
km_tot_success = 0
for obj in target_objects:
    print(obj)
    m_evaluations = np.array([mapping[v] for v in SR_m[obj].values()])
    print("SR-m", 
                len(np.where(m_evaluations[:10]==1)[0]), 
                len(np.where(m_evaluations[10:20]==1)[0]), 
                len(np.where(m_evaluations[20:]==1)[0]), 
                )
    p_evaluations = np.array([mapping[v] for v in SR_p[obj].values()])
    print("SR-p", 
                len(np.where(p_evaluations[:10]==1)[0]), 
                len(np.where(p_evaluations[10:20]==1)[0]), 
                len(np.where(p_evaluations[20:]==1)[0]), 
                )
    
    if obj == "drawer":
        print("SR-km", 
                10, 10, 10
                )
        km_tot_success += 30
    else:
        print("SR-km", 
                    len(np.where(p_evaluations[:10]>-2)[0]), 
                    len(np.where(p_evaluations[10:20]>-2)[0]), 
                    len(np.where(p_evaluations[20:]>-2)[0]), 
                    )
        km_tot_success += len(np.where(p_evaluations>-2)[0])

    m_tot_success += len(np.where(m_evaluations==1)[0])
    p_tot_success += len(np.where(p_evaluations==1)[0])
    

    print("-----------")

print(m_tot_success)
print(p_tot_success)
print(km_tot_success)

m_tot_success /= (30 * len(target_objects))
p_tot_success /= (30 * len(target_objects))
km_tot_success /= (30 * len(target_objects))
print(m_tot_success)
print(p_tot_success)
print(km_tot_success)