from typing import List, Dict

def build_prompt(labels: List[str],
                 label_defs: Dict[str,str],
                 glossary: Dict[str, List[str]],
                 fewshots: List[dict],
                 item_text: str) -> str:
    parts = []
    parts.append("You are a multi-label classifier. Output ONLY valid JSON.")
    parts.append("Task: Read the BODYWASH ITEM and assign ALL applicable Level-1 Factors.")
    parts.append("Return: {\"labels\":[...], \"scores\":{\"Label\":0-1}}. No extra text.")

    parts.append("\nFactors (definition | key terms):")
    for lab in labels:
        defs = label_defs.get(lab, "").strip()
        cues = ", ".join(glossary.get(lab, [])[:12])
        parts.append(f"- {lab}: {defs} | {cues}")

    parts.append("\nExamples:")
    for ex in fewshots[:min(3*len(labels), 48)]:  
        labs = ex["labels"]
        parts.append(f'INPUT: "{ex["text"]}"')
        parts.append(f'OUTPUT: {{"labels": {labs}, "scores": {{"{labs[0]}": 0.8}}}}')  

    parts.append("\nINFER ON this ITEM:")
    parts.append(f'\"\"\"{item_text}\"\"\"')
    parts.append("JSON only.")
    return "\n".join(parts)

DEFAULT_DEFS = {
 "Accessibility":"Availability, easy to find, store carries it",
 "Brand Accountability":"Brand’s responsibility, ethics, responding to issues",
 "Brand For Me":"Personal fit, brand identity matches me",
 "Brand Value":"Reputation, trust, worth paying more",
 "Cleansing":"Cleaning power, removes dirt/sweat",
 "Companion Approval":"Partner/family likes it",
 "Efficacy":"It works as promised (non-specific)",
 "Feel / Finish":"Skin feel AFTER rinse (soft, smooth, non-drying)",
 "Fragrance":"Smell, scent, perfume, aroma, longevity of scent",
 "Packaging":"Bottle, pump, cap, leaks, look",
 "Price":"Cost, expensive/cheap, value for money",
 "Product Texture":"During use: lather/foam/creamy/gel/thickness/slip",
 "Purchase Experience":"Buying process, delivery, store service",
 "Safety":"No irritation, gentle, hypoallergenic",
 "Skin Care":"Moisturizing, nourishing, skin benefits while using",
 "Skin Texture Improvement":"Improves skin texture over time"
}
