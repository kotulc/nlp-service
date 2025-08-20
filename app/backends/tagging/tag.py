import re

from typing import List, Dict

from .extract import extract_candidates, extract_topics


def to_hashtags(phrases:List[str])->List[str]:
    tags = set()
    for p in phrases:
        base = re.sub(r"[^A-Za-z0-9 ]+", "", p).strip()
        if not base: continue
        parts = base.split()
        if len(parts) == 1:
            tags.add("#" + parts[0][:30])
        else:
            tags.add("#" + "".join(w.capitalize() for w in parts)[:40])
            # common abbreviations
            abbr = "".join(w[0].upper() for w in parts if w)
            if 2 <= len(abbr) <= 6: tags.add("#" + abbr)

    # simple length & dedupe
    ranked = sorted(tags, key=lambda t: (-len(t) <= 20, len(t), t))  # favor compact tags

    return ranked[:12]


def tag_document(title:str, desc:str, body:str)->Dict:
    doc_text = " ".join([title, desc, body])
    candidates = extract_candidates(title, desc, body)
    topics = extract_topics(candidates, doc_text, k=5)

    # collect top phrases for hashtagging
    top_phrases = []
    for t in topics:
        top_phrases.append(t["topic"])
        top_phrases.extend(t["keywords"][:2])

    hashtags = to_hashtags(list(dict.fromkeys(top_phrases)))  # preserve order, dedupe

    return {"topics": topics, "hashtags": hashtags}
