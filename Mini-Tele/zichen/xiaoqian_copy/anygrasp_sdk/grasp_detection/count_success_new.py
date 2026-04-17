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

SR = json.load(open("/Disk3/xiaoqian/pred_keypoints/capture_0301/SR_p_gt.json", "r"))

mapping = {
    "S": 1,
    "F": -1,
}

m_tot_success = 0
p_tot_success = 0
km_tot_success = 0
for obj in target_objects:
    print(obj)
    p_evaluations = np.array([mapping[v] for v in SR[obj].values()])
    print("SR-p", 
                len(np.where(p_evaluations[:10]==1)[0]), 
                len(np.where(p_evaluations[10:20]==1)[0]), 
                len(np.where(p_evaluations[20:]==1)[0]), 
                np.round(len(np.where(p_evaluations==1)[0]) / 30 * 100, 2)
                )
    
    p_tot_success += len(np.where(p_evaluations==1)[0])
    

    print("-----------")

print(p_tot_success)

p_tot_success /= (30 * len(target_objects))
print(p_tot_success)