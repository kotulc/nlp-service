from enum import Enum

from app.core.tags.extract import extract_entities, extract_keywords, extract_related


TAG_TYPES = {
    "entities": extract_entities,   # Named entities such as people, organizations, locations, etc.
    "keywords": extract_keywords,   # Important keywords and phrases that capture the main topics
    "related": extract_related      # Related topics and concepts derived from the content
}


def get_tags(content: str, min_length: int=1, max_length: int=3, top_n: int=10) -> dict:
    """Return a dictionary of entities, keywords, and related topic tags"""
    entities, entity_scores = extract_entities(content, top_n)
    keywords, keyword_scores = extract_keywords(content, top_n)
    related, related_scores = extract_related(content, min_length, max_length, top_n)

    tags = dict(entities=entities, keywords=keywords, related=related)
    scores = dict(entities=entity_scores, keywords=keyword_scores, related=related_scores)

    return dict(tags=tags, scores=scores)
