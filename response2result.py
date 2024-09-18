import os, pickle, tqdm
import os.path as osp

'''
    This file is for manually converting the specific video results
'''

root_dir = "./tal_output/thumos14/time-inst-v1"
response_dir = osp.join(root_dir, "response")
result_dir = osp.join(root_dir, "result")
error_file_path = osp.join(root_dir, "error_vids.txt")

filenames = sorted(os.listdir(response_dir))
for fn in tqdm.tqdm(filenames):
    vid = fn.replace(".pkl", "")
    response_fp = osp.join(response_dir, fn)
    result_fp = osp.join(result_dir, f"{vid}.txt")
    if osp.exists(result_fp):
        continue
    
    with open(response_fp, "rb") as fp:
        response = pickle.load(fp)
        
    if response._error:
        print(f"No result: {vid}")
        with open(error_file_path, "a") as fp:
            fp.write(f"{vid}\n")
        continue
    
    result = response.text
    
    with open(result_fp, "w") as fp:
        fp.write(result)