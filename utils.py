import re, json, random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_label_glossary(df: pd.DataFrame, label_col: str, text_col: str,
                         labels: List[str], topk: int = 15):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=20000)
    X = vec.fit_transform(df[text_col].map(clean_text))
    vocab = np.array(vec.get_feature_names_out())
    glossary = {}

    for lab in labels:

        mask = (df[label_col] == lab).to_numpy()

        n = int(mask.sum())
        if n == 0:
            glossary[lab] = []
            continue

  
        mat = X[mask]

        mean_tfidf = np.asarray(mat.mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:topk]
        glossary[lab] = [vocab[i] for i in top_idx if mean_tfidf[i] > 0]

    return glossary


def stratified_split(df: pd.DataFrame, y_col: str, test_size=0.2, seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train, idx_val = next(sss.split(df, df[y_col]))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_val].reset_index(drop=True)

def sample_few_shot(df: pd.DataFrame, label_col: str, text_col: str, labels: List[str], k: int, seed: int=42):
    rng = random.Random(seed)
    shots = []
    for lab in labels:
        cand = df[df[label_col]==lab][[text_col,label_col]].dropna().sample(
            min(k, (df[label_col]==lab).sum()), random_state=rng.randint(0,10**9)
        )
        for _, r in cand.iterrows():
            shots.append({"text": str(r[text_col]).strip(), "labels": [lab]})
    rng.shuffle(shots)
    return shots

def micro_f1_at_threshold(S: np.ndarray, Y: np.ndarray, tau: float) -> Tuple[float,float,float]:
    Yhat = (S >= tau).astype(int)
    TP = (Yhat & Y).sum()
    FP = (Yhat & (1 - Y)).sum()
    FN = ((1 - Yhat) & Y).sum()
    P = TP / (TP + FP + 1e-12)
    R = TP / (TP + FN + 1e-12)
    F1 = 2*P*R/(P+R+1e-12)
    return F1, P, R

def safe_json(s: str):

    try:
        return json.loads(s)
    except:
        s = s.strip()
        s = s[s.find("{") : s.rfind("}")+1] if "{" in s and "}" in s else s
        try:
            return json.loads(s)
        except:
            return {}

def keyword_prior(text: str, label: str, glossary: Dict[str, List[str]], base: float, bonus: float) -> float:
    cues = glossary.get(label, [])
    hits = sum(1 for c in cues if c in text)
    return base + bonus if hits >= 2 else base

def conflict_nudge(text: str, label: str, base: float, feel_bonus: float) -> float:
    if label != "Feel / Finish": return base
    # simple post-rinse feel cues
    cues = ["non drying","soft skin","silky","smooth","not sticky","residue free","moisturized"]
    return base + feel_bonus if any(c in text for c in cues) else base
