from enum import Enum

from app.core.summary.headings import get_title, get_subtitle, get_description, get_outline


SUMMARY_TYPES = {
    "title": get_title,              # Suggested content titles
    "subtitle": get_subtitle,        # Candidate content subtitles 
    "description": get_description,  # Basic content summaries
    "outline": get_outline           # A list of content section key points or themes
}


def get_summary(content: str, summary: str, n_sections=3, top_n: int=10) -> tuple:
    """Return a dictionary of entities, keywords, and related topic tags"""
    if summary in SUMMARY_TYPES:
        summary_function = SUMMARY_TYPES[summary]

        if summary == "outline":
            scores = summary_function(content, n_sections=n_sections, top_n=1)
            scores = [s[0] for s in scores]  # Unpack the section lists
        else:
            scores = summary_function(content, top_n=top_n)

        # Return a dict of lists (summaries, scores)
        return dict(summaries=scores[0], scores=scores[1])
    else:
        return dict(summaries=[], scores=[])
