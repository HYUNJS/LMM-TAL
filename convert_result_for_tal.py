import os, json, re, argparse
import os.path as osp
import pandas as pd


'''
{"results": {
    vid: List(Dict(
        "label_id": 0,
        "segment": [start, end],
        "score: 0.11
    ))
}
}
'''

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", default="flash", choices=["flash", "pro"])
argparser.add_argument("--dataset_name", default="thumos14", choices=["thumos14", "fineaction"])
argparser.add_argument("--time_inst_ver", default=1, type=int, choices=[0, 1])
argparser.add_argument("--anno_ver", default=0, type=int, choices=[0,1,2], help="0: generalized | 1: constrained-base | 2: constrained-novel")
args = argparser.parse_args()

model_name = args.model_name
dataset_name = args.dataset_name
time_inst_ver = args.time_inst_ver
anno_ver = args.anno_ver
MODEL_ID = "gemini-1.5-pro-001" if model_name == "pro" else "gemini-1.5-flash-001" 
if anno_ver == 0:
    cls_filepath = f"./dataset/{dataset_name}/annotations/{dataset_name}_labels.csv"
    ANNO_VER = "gen-all"
elif anno_ver == 1:
    cls_filepath = f"./dataset/{dataset_name}/annotations/{dataset_name}_K400_overlap_labels.csv"
    ANNO_VER = "con-base"
elif anno_ver == 2:
    cls_filepath = f"./dataset/{dataset_name}/annotations/{dataset_name}_K400_nonoverlap_labels.csv"
    ANNO_VER = "con-novel"

OUTPUT_DIR = f"./tal_output/{dataset_name}"
TGT_VERSION = f"time-inst-v{time_inst_ver}"
RESULT_DIR = osp.join(OUTPUT_DIR, MODEL_ID, TGT_VERSION, ANNO_VER, "result")
filenames = sorted(os.listdir(RESULT_DIR))
result_filename = f"{MODEL_ID}_{TGT_VERSION}_{dataset_name}-{ANNO_VER}_results.json"
formatted_result_filepath = osp.join(OUTPUT_DIR, MODEL_ID, TGT_VERSION, ANNO_VER, result_filename)
print(result_filename)
cls_ids = pd.read_csv(cls_filepath)["id"].values
 
# Use regular expressions to extract the tuples from the string
pattern = re.compile(r'\((\d+.\d+), (\d+.\d+), (\d+), (\d+.\d+)\)')

results = {}
for fn in filenames:
    vid = fn.replace(".txt", "")
    with open(osp.join(RESULT_DIR, fn), "r") as fp:
        gemini_result = fp.read()

    matches = pattern.findall(gemini_result)
    annos = []
    # Populate the lists by parsing the tuples
    for match in matches:
        start_time, end_time, class_idx, confidence_score = match
        start_time, end_time = float(start_time), float(end_time)
        confidence_score = float(confidence_score)
        class_idx = int(class_idx)
        if class_idx  not in cls_ids:
            print(f"Invalid class index {class_idx}")
            continue
            
        instance = {
                        "label_id": class_idx,
                        "segment": [start_time, end_time],
                        "score": confidence_score
                    }
        annos.append(instance)

    results[vid] = annos

formatted_results = {"results": results}
with open(formatted_result_filepath, "w") as fp:
    json.dump(formatted_results, fp)
