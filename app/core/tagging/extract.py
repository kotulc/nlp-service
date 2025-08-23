import numpy
import spacy
import yake

from relevance import maximal_marginal_relevance as mmr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from keybert._mmr import mmr
from keybert import KeyBERT


# Define module level variables
nlp = spacy.load("en_core_web_lg")


# Example usage and testing function
def demo_tagger():
    """Test the tagging functionality with different parameters."""

    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to 
    natural intelligence displayed by animals including humans. Leading AI textbooks define 
    the field as the study of "intelligent agents": any system that perceives its environment 
    and takes actions that maximize its chance of achieving its goals. Some popular accounts 
    use the term "artificial intelligence" to describe machines that mimic "cognitive" 
    functions that humans associate with the human mind, such as "learning" and "problem solving".
    As machines become increasingly capable, tasks considered to require "intelligence" are 
    often removed from the definition of AI, a phenomenon known as the AI effect. For instance, 
    optical character recognition is frequently excluded from things considered to be AI, 
    having become a routine technology.
    """
    
    print("=== Basic Tagging ===")
    key_bert = KeyBERT(model='all-MiniLM-L6-v2')
    bert_phrases = key_bert.extract_keywords(sample_text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10, use_mmr=False)
    bert_phrases = [phrase for phrase, score in bert_phrases]
    print("\nKeyBERT Keywords:", bert_phrases)

    # Spacy noun chunks + entities
    doc = nlp(sample_text)
    spacy_entities = list({entity.text.strip() for entity in doc.ents})
    print("\nEntities:", spacy_entities)

    # YAKE keywords (adds recall)
    yake_phrases = yake.KeywordExtractor(n=1, top=10).extract_keywords(sample_text)
    yake_phrases = [phrase for phrase, score in yake_phrases]
    print("\nYAKE keywords:", yake_phrases)

    candidates = []
    for keys in (bert_phrases, yake_phrases, spacy_entities):
        candidates.extend([k.lower() for k in keys])
    candidates = list(set(candidates))
    print("\nCandidate Keywords:", candidates)

    
    def rank_words_semantic(text_body, word_list, model_name='all-MiniLM-L6-v2', top_n=10):
        """Rank words by semantic similarity to text using embeddings"""
        model = SentenceTransformer(model_name)
        
        # Create embeddings
        text_embedding = model.encode([text_body])
        word_embeddings = model.encode(word_list)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(text_embedding, word_embeddings)[0]
        
        # Create word-score pairs
        word_scores = list(zip(word_list, similarities))
        
        return sorted(word_scores, key=lambda x: x[1], reverse=True)[:top_n]


    ranked_keywords = rank_words_semantic(sample_text, candidates)
    print("\nRanked by Semantic Similarity:", ranked_keywords)

    # model = SentenceTransformer("all-MiniLM-L6-v2")     # standard SBERT model
    # doc_embedding  = model.encode([sample_text], normalize_embeddings=True)
    # candidate_embedding = model.encode(candidates, normalize_embeddings=True)
    # selected = mmr(doc_embedding, candidate_embedding, candidates, top_n=8, diversity=0.6)
    # print("\nMMR Keywords:", selected)

    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    result = generator(f"In as few words as possible summarize the following text: {sample_text}")
    print("\nGoogle summarize:", result[0]['generated_text']) 

    result = generator(f"List key concepts of the following text: {sample_text}")
    print("\nGoogle concepts:", result[0]['generated_text']) 

    result = generator(f"Apply a label to the following text: {sample_text}")
    print("\nGoogle label:", result[0]['generated_text']) 

    result = generator(f"In as few words as possible, list tangentially related concepts:\n\nText: {sample_text}\n\nRelated Topics:")
    print("\nGoogle theme:", result[0]['generated_text']) 


    import torch
    import transformers

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Specify the model you want to use
    model_id = "microsoft/Phi-4-mini-instruct"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Use the model for text generation
    generator = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    sequences = generator(
        f"In as few words as possible, list tangentially related concepts:\n\nText: {sample_text}\n\nRelated Topics:",
        do_sample=True, max_new_tokens=32, temperature=0.7
    )
    print("\nPhi4:", sequences[0]["generated_text"])

    # sequences = generator(
    #     f"List key concepts of the following text: {sample_text}",
    #     do_sample=True, max_new_tokens=32, temperature=0.7
    # )
    # print("\nPhi4:", sequences[0]["generated_text"])

    # sequences = generator(
    #     f"What is the theme of the following text: {sample_text}",
    #     do_sample=True, max_new_tokens=32, temperature=0.7
    # )
    # print("\nPhi4:", sequences[0]["generated_text"])


    # Specify the model you want to use
    model_id = "google/gemma-3-1b-it"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Use the model for text generation
    generator = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    sequences = generator(
        f"In as few words as possible, list tangentially related concepts:\n\nText: {sample_text}\n\nRelated Topics:",
        do_sample=True, max_new_tokens=32, temperature=0.7
    )
    print("\Gemma3:", sequences[0]["generated_text"])


if __name__ == "__main__":
    demo_tagger()
