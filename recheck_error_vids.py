import pickle, os
import os.path as osp


tgt_output_dir = "tal_output/fineaction/gemini-1.5-pro-001/time-inst-v1/gen-all"

error_vids_filepath = osp.join(tgt_output_dir, "error_vids.txt")
with open(error_vids_filepath, "r") as fp:
    error_vids = [l.strip() for l in fp.readlines()]
    
    
tgt_vids = []
left_vids = []
vid2responses = {}
for idx, vid in enumerate(error_vids):
    response_filepath = osp.join(tgt_output_dir, f"response/{vid}.pkl")
    with open(response_filepath, "rb") as fp:
        r = pickle.load(fp)
    vid2responses[vid] = r
    
    # if r is None:
    #     tgt_vids.append(vid)
    #     os.remove(response_filepath)
    #     print(f"Remove {vid}")
    # if r == "video loading error":
    #     tgt_vids.append(vid)
    #     os.remove(response_filepath)
    #     print(f"Remove {vid}")
    if r == "video loading error" or  r is None:
        tgt_vids.append(vid)
        os.remove(response_filepath)
        print(f"Remove {vid}")
    else:
        left_vids.append(vid)

print(f"Remove {len(tgt_vids)} error response")
print(f"Left {len(left_vids)} error response")
tgt_vids

with open(error_vids_filepath, "w") as fp:
    fp.writelines("\n".join(list(left_vids)) + "\n")