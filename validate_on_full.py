#!/usr/bin/env python3
# validate_on_full.py

import os
import glob
import json
import time
import base64
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import List, Tuple, Dict
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env

# ─── CONFIG ─────────────────────────────────────────────────────────────
RUNPOD_URL      = "http://194.68.245.137:22119/v1"
CHAT_MODEL_PATH = (
    "/workspace/PDF-AI/hf/hub/models--Qwen--Qwen2.5-VL-72B-Instruct-AWQ/"
    "snapshots/c8b87d4b81f34b6a147577a310d7e75f0698f6c2"
)
EMBED_MODEL     = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

BASE_DIR        = "data"
FULL_JSON_GLOB  = os.path.join(BASE_DIR, "full", "*.json")
SYSTEM_PROMPT   = "chart_analysis_system_prompt.md"
USER_PROMPT     = "prompt.txt"

SLEEP_PER_CALL  = 0.3  # seconds

# ─── CLIENT SETUP ────────────────────────────────────────────────────────
client     = OpenAI(api_key="EMPTY",       base_url=RUNPOD_URL)
emb_client = OpenAI(api_key=OPENAI_API_KEY)  # for embeddings

# ─── HELPERS ────────────────────────────────────────────────────────────
def load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def load_base64(rel_path: str) -> str:
    full = os.path.join(BASE_DIR, rel_path)
    with open(full, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def make_image_part(b64: str) -> Dict:
    return {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}

def format_debug(debug: Dict) -> str:
    return (
        f"price: {debug['price']}, trend: {debug['trend']}, ao: {debug['ao']}, "
        f"%K: {debug['%K']}, %D: {debug['%D']}"
    )

def parse_response(output: str) -> Tuple[str,List[str]]:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    label = lines[0] if lines else ""
    reasons=[]
    for ln in lines[1:]:
        reasons.append(ln.lstrip("- ").strip() if ln.startswith("-") else ln)
        if len(reasons)==3:
            break
    while len(reasons)<3:
        reasons.append("")
    return label, reasons

def cosine_sim(a:np.ndarray, b:np.ndarray)->float:
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

# ─── LOAD PROMPTS ───────────────────────────────────────────────────────
system_text = load_prompt(SYSTEM_PROMPT)
user_text   = load_prompt(USER_PROMPT)

# ─── VALIDATION LOOP ────────────────────────────────────────────────────
records=[]
for path in glob.glob(FULL_JSON_GLOB):
    with open(path) as f:
        ex = json.load(f)

    # ground truth
    gt_label   = ex["label"].strip()
    gt_reasons = ex["reasoning"][:]  # copy
    while len(gt_reasons)<3:
        gt_reasons.append("")

    debug_txt  = format_debug(ex["debug"])
    imgs_rel   = [ex["images"]["main"], ex["images"]["ao"], ex["images"]["rsi"]]

    # build chat messages
    sys_msg = {"role":"system","content":[{"type":"text","text":system_text}]}
    user_content = [{"type":"text","text":user_text},
                    {"type":"text","text":debug_txt}]
    for rel in imgs_rel:
        user_content.append(make_image_part(load_base64(rel)))
    user_msg = {"role":"user","content":user_content}

    # call VLM
    resp = client.chat.completions.create(
        model=CHAT_MODEL_PATH,
        messages=[sys_msg,user_msg],
        max_tokens=512
    )
    out = resp.choices[0].message.content.strip()
    pred_label, pred_reasons = parse_response(out)

    ok_structure = bool(pred_label and all(pred_reasons))
    ok_label     = (pred_label.lower()==gt_label.lower())

    while len(pred_reasons)<3:
        pred_reasons.append("")

    # compute similarity only for non-empty pairs
    sims=[]
    for gt,pr in zip(gt_reasons,pred_reasons):
        if not gt or not pr:
            sims.append(0.0)
        else:
            try:
                pair = emb_client.embeddings.create(
                    model=EMBED_MODEL,
                    input=[gt,pr]
                )
                v0 = np.array(pair.data[0].embedding)
                v1 = np.array(pair.data[1].embedding)
                sims.append(cosine_sim(v0,v1))
            except Exception as e:
                print(f"⚠️ Embedding error for '{gt}' vs '{pr}' in {path}: {e}")
                sims.append(0.0)

    avg_sim = sum(sims)/3.0

    records.append({
        "file":       os.path.basename(path),
        "gt_label":   gt_label,
        "pred_label": pred_label,
        "ok_structure": ok_structure,
        "ok_label":     ok_label,
        "sim1":        sims[0],
        "sim2":        sims[1],
        "sim3":        sims[2],
        "avg_sim":     avg_sim
    })

    time.sleep(SLEEP_PER_CALL)

# ─── REPORT ────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
print(f"Structure OK   : {df['ok_structure'].mean():.2%}")
print(f"Label Accuracy : {df['ok_label'].mean():.2%}")
print(f"Mean Sim       : {df['avg_sim'].mean():.4f}")

df.to_csv("validation_full_results.csv", index=False)
print("Saved validation_full_results.csv")
