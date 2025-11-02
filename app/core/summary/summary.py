from app.core.summary.headings import get_title, get_subtitle, get_description, get_outline


SUMMARY_TYPES = {
    "title": get_title,              # Suggested content titles
    "subtitle": get_subtitle,        # Candidate content subtitles 
    "description": get_description,  # Basic content summaries
    "outline": get_outline           # A list of content section key points or themes
}


def get_summary(content: str, summary: str, **kwargs) -> tuple:
    """Return a dictionary of entities, keywords, and related topic tags"""
    summaries, scores = [], []
    
    if summary in SUMMARY_TYPES:
        summary_function = SUMMARY_TYPES[summary]
        summaries, scores = summary_function(content, **kwargs)

    # Return a dict of lists (summaries, scores)
    return dict(summaries=summaries, scores=scores)

