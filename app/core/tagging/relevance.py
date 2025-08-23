import numpy

from sentence_transformers import util


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
