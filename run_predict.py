import os, json, yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from utils import clean_text, safe_json, keyword_prior, conflict_nudge
from prompt import build_prompt, DEFAULT_DEFS
from llm_client import call_ollama

load_dotenv()
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
    
    with open("artifacts.json","r") as f:
        art = json.load(f)
    tau_star = float(art["tau_star"])
    glossary = art["glossary"]
    label_defs = DEFAULT_DEFS

    
    train_df = pd.read_excel("data/bodywash-train.xlsx").rename(columns={"Core Item":"text","Level 1 Factors":"label"})
    fewshots = []
    for lab in LABELS:
        ex = train_df[train_df["label"]==lab].head(3)  # 3 examples per label
        for _, r in ex.iterrows():
            fewshots.append({"text": str(r["text"]).strip(), "labels":[lab]})

    
    test_df = pd.read_excel("data/bodywash-test.xlsx")
    
    text_col = "Core Item" if "Core Item" in test_df.columns else test_df.columns[0]
    test_df["text"] = test_df[text_col].astype(str)
    test_df["text_clean"] = test_df["text"].map(clean_text)

    temps = cfg["temperatures"]

    preds = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predict (test)"):
        scores = llm_predict_scores(row["text"], LABELS, label_defs, glossary, fewshots, temps)

        txtc = row["text_clean"]
        final_scores = {}
        for lab in LABELS:
            s = scores.get(lab, 0.0)
            s = keyword_prior(txtc, lab, glossary, s, cfg["keyword_bonus"])
            s = conflict_nudge(txtc, lab, s, cfg["feel_finish_bonus"])
            final_scores[lab] = float(min(max(s,0.0),1.0))

        chosen = [lab for lab,sc in final_scores.items() if sc >= tau_star]
        preds.append({
            "item_index": idx,
            "text": row["text"],
            "predicted_labels": "|".join(chosen),
            "scores_json": json.dumps(final_scores)
        })

    out = pd.DataFrame(preds)
    out.to_csv("predictions_test.csv", index=False)
    print("Wrote predictions_test.csv")

if __name__ == "__main__":
    main()
