import os, json, time, requests
from dotenv import load_dotenv
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","mistral")

def call_ollama(prompt: str, temperature: float=0.3, max_tokens: int=800) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": temperature,
        "options": {"num_predict": max_tokens}
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
   
    text = ""
    for line in r.text.splitlines():
        try:
            obj = json.loads(line)
            text += obj.get("response","")
        except:
            pass
    return text
