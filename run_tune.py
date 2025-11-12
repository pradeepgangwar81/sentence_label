import os, yaml, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from utils import (clean_text, build_label_glossary, stratified_split, sample_few_shot,
                   micro_f1_at_threshold, safe_json, keyword_prior, conflict_nudge)
from prompt import build_prompt, DEFAULT_DEFS
from llm_client import call_ollama

load_dotenv()
SEED = int(os.getenv("SEED","42"))
N_PASSES = int(os.getenv("N_PASSES","5"))

cfg = yaml.safe_load(open("config.yaml","r"))
LABELS = cfg["labels"]

def average_scores(outputs, labels):
    
    agg = {lab: [] for lab in labels}
    for out in outputs:
        sc = out.get("scores", {})
        for lab in labels:
            if lab in sc:
                agg[lab].append(float(sc[lab]))
    return {lab: (sum(v)/len(v) if v else 0.0) for lab,v in agg.items()}

def llm_predict_scores(item_text, labels, label_defs, glossary, fewshots, temperatures):
    from utils import safe_json
    outs = []
    prompt = build_prompt(labels, label_defs, glossary, fewshots, item_text)
    for t in temperatures:
        raw = call_ollama(prompt, temperature=float(t))
        js = safe_json(raw)
      
        if "scores" not in js:
            
            labs = js.get("labels",[])
            js["scores"] = {lab: (0.8 if lab in labs else 0.0) for lab in labels}
        outs.append(js)
    return average_scores(outs, labels)

def main():
    np.random.seed(SEED)
    train_path = "data/bodywash-train.xlsx"
    df = pd.read_excel(train_path)
    df = df.rename(columns={"Core Item":"text","Level 1 Factors":"label"})
    df["text_clean"] = df["text"].map(clean_text)

    glossary = build_label_glossary(df, "label", "text", LABELS, topk=15)

    tr, va = stratified_split(df, "label", test_size=cfg["val_size"], seed=SEED)

    few = sample_few_shot(tr, "label", "text", LABELS, k=cfg["few_shot_per_label"], seed=SEED)

    temps = cfg["temperatures"]

 
    S, Y = [], []
    for _, row in tqdm(va.iterrows(), total=len(va), desc="LLM scoring (val)"):
        scores = llm_predict_scores(row["text"], LABELS, DEFAULT_DEFS, glossary, few, temps)

        txtc = row["text_clean"]
        arr = []
        for lab in LABELS:
            s = scores.get(lab, 0.0)
            s = keyword_prior(txtc, lab, glossary, s, cfg["keyword_bonus"])
            s = conflict_nudge(txtc, lab, s, cfg["feel_finish_bonus"])
            arr.append(min(max(s,0.0),1.0))
        S.append(arr)

        y = [1 if row["label"] == lab else 0 for lab in LABELS]
        Y.append(y)

    S = np.array(S); Y = np.array(Y)

    best = (-1, None, None, None)
    tau = cfg["grid_threshold"]
    grid = np.arange(tau["start"], tau["stop"]+1e-9, tau["step"])
    for t in grid:
        f1, p, r = micro_f1_at_threshold(S, Y, t)
        if f1 > best[0]:
            best = (f1, t, p, r)

    print(f"Best micro-F1={best[0]:.4f} at τ={best[1]:.2f} (P={best[2]:.4f}, R={best[3]:.4f})")

    artifacts = {
        "tau_star": float(best[1]),
        "label_defs": DEFAULT_DEFS,
        "glossary": glossary
    }
    with open("artifacts.json","w") as f:
        json.dump(artifacts, f, indent=2)

if __name__ == "__main__":
    main()
