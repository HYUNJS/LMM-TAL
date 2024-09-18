import os, traceback, decord, glob, math, tqdm, pickle, argparse, json
from PIL.Image import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Any, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import os.path as osp
from PIL import Image

import google.generativeai as genai
from google.generativeai.types.content_types import to_blob # for checking the message size


MAX_MESSAGE_SIZE = 20971520
MAX_IMG_SIZE = 512
MAX_OUTPUT_TOKEN = 4096
safety_settings = [
  {
    "category": "HARM_CATEGORY_DANGEROUS",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]


def count_message_in_byte(str_message_list, vid_filepath, frame_idxs):
    str_message_bytes = 0
    for m in str_message_list:
        str_message_bytes += len(m.encode('utf-8'))
    
    frame = load_frames(vid_filepath, [0])[0]
    per_frame_message_bytes = len(to_blob(frame).data)
    frame_message_bytes = per_frame_message_bytes * len(frame_idxs)
    message_bytes = str_message_bytes + frame_message_bytes
    
    return message_bytes

def set_google_key(key: Optional[str] = None) -> None:
    if key is None:
        assert "GOOGLE_API_KEY" in os.environ
        key = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=key)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(4))
def call_google_api(model, messages, stream=False):
    try:
        response = model.generate_content(messages, stream=stream)
        response.resolve()
        return response
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        # raise e
        return None
  
def gen_time_instruction(frame_idxs_in_sec, video_time, version):
    if version == 0:
        formatted_frame_idxs = [f"{x:.2f}" for x in frame_idxs_in_sec]
        num_frames = len(frame_idxs_in_sec)
        frame_time = ",".join(formatted_frame_idxs)
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}."
        time_instruction = [time_instruction]
    elif version == 1:
        time_instruction = [f"This frame is sampled at {t:.2f} second." for t in frame_idxs_in_sec]
        time_instruction = [f"The video lasts for {video_time:.2f} seconds"] + time_instruction
    else:
        raise NotImplementedError(f"Time instruction ver.{version} is not yet implemented")

    return time_instruction

def add_time_instruction(frames, time_instruction, version):
    if version == 0:
        time_added_message = frames + time_instruction
    elif version == 1:
        time_added_message = [None] * (2*len(frames)+1)
        time_added_message[0] = time_instruction[0]
        time_added_message[1::2] = time_instruction[1:]
        time_added_message[2::2] = frames
    else:
        raise NotImplementedError(f"Time instruction ver.{version} is not yet implemented")

    return time_added_message

def get_video_metadata(vid_filepath, tgt_fps=2):
    vr = decord.VideoReader(vid_filepath, num_threads=1) # multi thread cannot handle some corrupted videos
    vid_fps = vr.get_avg_fps()
    intv = vid_fps / tgt_fps
    num_frames = len(vr)
    frame_idxs = np.ceil(np.arange(math.ceil((num_frames-1) / intv) + 1) * intv)
    frame_idxs[-1] = num_frames-1
    frame_idxs_in_sec = [i/vid_fps for i in frame_idxs]
    video_time = (num_frames-1) / vid_fps
    
    return frame_idxs, frame_idxs_in_sec, video_time

def load_frames(vid_filepath, frame_idxs):
    vr = decord.VideoReader(vid_filepath, num_threads=1, fault_tol=0.1) # multi thread cannot handle some corrupted videos
    h, w, _ = vr[0].shape
    max_side = max(h, w)
    if max_side > MAX_IMG_SIZE:
        if h >= w:
            new_w = int(w / (h / MAX_IMG_SIZE))
            new_h = MAX_IMG_SIZE
            vr = decord.VideoReader(vid_filepath, width=new_w, height=new_h, fault_tol=0.1)    
        else:
            new_w = MAX_IMG_SIZE
            new_h = int(h / (w / MAX_IMG_SIZE))
            vr = decord.VideoReader(vid_filepath, width=new_w, height=new_h, fault_tol=0.1)    
    else:
        vr = decord.VideoReader(vid_filepath, fault_tol=0.1)
    frames_ = vr.get_batch(frame_idxs).asnumpy()
    frames = [Image.fromarray(img) for img in frames_]
    
    return frames

def call_model_per_video(vid, tal_prompt, time_inst_ver, stream=False):
    prefix, suffix = tal_prompt.split("[INPUT_VIDEO]")

    vid_filepath = glob.glob(f"{VIDEO_ROOT_DIR}/{vid}.*")[0]
    frame_idxs, frame_idxs_in_sec, video_time = get_video_metadata(vid_filepath, tgt_fps=2)
    time_instruction = gen_time_instruction(frame_idxs_in_sec, video_time, time_inst_ver)
    try:
        message_size = count_message_in_byte([prefix, suffix, *time_instruction], vid_filepath, frame_idxs)
    except Exception as e:
        print(f"{vid} loading error")
        return "video loading error"
    
    if message_size > MAX_MESSAGE_SIZE:
        print(f"{vid} exceeds the message size limit - {message_size}")
        return "exceed size limit"
    
    try:
        frames = load_frames(vid_filepath, frame_idxs)
    except Exception as e:
        print(f"{vid} loading error")
        return "video loading error"
    
    messages = [prefix, *add_time_instruction(frames, time_instruction, time_inst_ver), suffix]
    # print("# Tokens: ", model.count_tokens(messages))
    response = call_google_api(model, messages, stream)
        
    return response

if __name__ == "__main__":
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
    prompt_path = "./prompts/tal_gemini_prompt.txt"
    
    with open("./api_key/gemini.txt", "r") as fp:
        GEMINI_APIKEY = fp.readline()
    set_google_key(key=GEMINI_APIKEY)
    
    generation_config = genai.GenerationConfig(max_output_tokens=MAX_OUTPUT_TOKEN)
    model = genai.GenerativeModel(MODEL_ID, generation_config=generation_config, safety_settings=safety_settings)
    
    if anno_ver == 0:
        anno_filepath = f"./dataset/{dataset_name}/annotations/validation_tal.json"
        cls_filepath = f"./dataset/{dataset_name}/annotations/{dataset_name}_labels.csv"
        ANNO_VER = "gen-all"
    elif anno_ver == 1:
        anno_filepath = f"./dataset/{dataset_name}/annotations/validation_K400_tal.json"
        cls_filepath = f"./dataset/{dataset_name}/annotations/{dataset_name}_K400_overlap_labels.csv"
        ANNO_VER = "con-base"
    elif anno_ver == 2:
        anno_filepath = f"./dataset/{dataset_name}/annotations/validation_nonK400_tal.json"
        cls_filepath = f"./dataset/{dataset_name}/annotations/{dataset_name}_K400_nonoverlap_labels.csv"
        ANNO_VER = "con-novel"

    with open(anno_filepath, 'r') as fp:
        annos = json.load(fp)['database']
    vids = list(annos.keys())
    
    VIDEO_ROOT_DIR = f"./dataset/{dataset_name}/raw_videos"
    OUTPUT_DIR = f"./tal_output/{dataset_name}"
    TGT_VERSION = f"time-inst-v{time_inst_ver}"
    RESPONSE_DIR = osp.join(OUTPUT_DIR, MODEL_ID, TGT_VERSION, ANNO_VER, "response")
    RESULT_DIR = osp.join(OUTPUT_DIR, MODEL_ID, TGT_VERSION, ANNO_VER, "result")
    error_file_path = osp.join(OUTPUT_DIR, MODEL_ID, TGT_VERSION, ANNO_VER, "error_vids.txt")
    os.makedirs(RESPONSE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
   
    with Path(prompt_path).open("r") as f:
        tal_prompt_tmpl = f.read().strip()
    cls_df = pd.read_csv(cls_filepath).loc[:, ['id', 'name']]
    cls_data = np.array(cls_df.to_records(index=False))
    action_cls_map = ""
    for cls_idx, cls_name in cls_data:
        action_cls_map += f"{cls_name}: {cls_idx} \n"
        
    tal_prompt = tal_prompt_tmpl.format(action_cls_map=action_cls_map)     
    error_vids = []
    
    # vids = ["v_00000213"]
    # vids = ["video_test_0000045"]
    for idx, vid in enumerate(tqdm.tqdm(vids)):
        response_fp = osp.join(RESPONSE_DIR, f"{vid}.pkl")
        result_fp = osp.join(RESULT_DIR, f"{vid}.txt")
        print(f"[video] {vid}")
        if osp.exists(response_fp):
            print(f"Skip {vid} - alredy existing")
            continue
        
        response = call_model_per_video(vid, tal_prompt, time_inst_ver)
        with open(response_fp, "wb") as fp:
            pickle.dump(response, fp)
        
        if response is None or type(response) is str or response._error:
            print(f"No result: {vid}")
            error_vids.append(vid)
            with open(error_file_path, "a") as fp:
                fp.write(f"{vid}\n")
            continue
        
        result = response.text
        print(result)
        with open(result_fp, "w") as fp:
            fp.write(result)
    
    print("=======================================")
    print(f"[Finished] Error vids: {len(error_vids)}")
    print(error_vids)