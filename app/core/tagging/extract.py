import re
import numpy
import spacy
import yake

from typing import List, Dict

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans


# Define module level variables
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def maximal_marginal_relevance(doc_emb, cand_embs, candidates, lambda_=0.6, top_n=20):
    selected, selected_idx = [], []
    sim_to_doc = util.cos_sim(cand_embs, doc_emb).cpu().numpy().ravel()
    sim_matrix = util.cos_sim(cand_embs, cand_embs).cpu().numpy()
    idxs = list(range(len(candidates)))

    while len(selected) < min(top_n, len(candidates)):
        if not selected:
            i = int(numpy.argmax(sim_to_doc))
            selected.append(candidates[i]); selected_idx.append(i); idxs.remove(i)
            continue

        mmr_scores = []
        for i in idxs:
            diversity = max(sim_matrix[i, selected_idx]) if selected_idx else 0
            score = lambda_ * sim_to_doc[i] - (1 - lambda_) * diversity
            mmr_scores.append((score, i))
            
        _, best_i = max(mmr_scores)
        selected.append(candidates[best_i]); selected_idx.append(best_i); idxs.remove(best_i)
    
    return selected

def extract_candidates(title:str, desc:str, body:str, top_k:int=40)->List[str]:
    text = " ".join([title, desc, body])
    doc = nlp(text)

    # noun chunks + entities
    phrases = {chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text) > 2}
    phrases |= {ent.text.strip() for ent in doc.ents if len(ent.text) > 2}

    # YAKE keywords (adds recall)
    kw = yake.KeywordExtractor(n=3, top=top_k).extract_keywords(text)
    phrases |= {k for k, _ in kw}

    # boost title/desc
    phrases |= {t.strip() for t in nlp(title).noun_chunks}
    phrases |= {t.strip() for t in nlp(desc).noun_chunks}

    # normalize & filter
    clean = []
    for p in phrases:
        p = re.sub(r"\s+", " ", p).strip()
        if 3 <= len(p) <= 60 and not p.isnumeric():
            clean.append(p)

    return list(set(clean))

def extract_topics(candidates:List[str], doc_text:str, k:int=5)->List[Dict]:
    doc_emb = embedder.encode([doc_text], convert_to_tensor=True)
    cand_embs = embedder.encode(candidates, convert_to_tensor=True)

    # pick diverse, relevant subset first
    shortlist = mmr(doc_emb, cand_embs, candidates, top_n=min(30, len(candidates)))
    if len(shortlist) < k: k = max(1, len(shortlist))
    X = embedder.encode(shortlist)
    k = min(k, len(shortlist))
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)

    topics = []
    for cluster_id in range(k):
        members = [shortlist[i] for i, c in enumerate(km.labels_) if c == cluster_id]
        if not members: continue
        # label = best member closest to cluster centroid
        centroid = km.cluster_centers_[cluster_id]
        sims = util.cos_sim(embedder.encode(members), centroid).cpu().numpy().ravel()
        label = members[int(numpy.argmax(sims))]
        topics.append({"topic": label, "keywords": sorted(set(members))})

    return topics
