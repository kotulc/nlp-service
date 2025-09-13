from app.core.summary.headings import get_title, get_subtitle, get_description, get_outline


class SummaryType(str, Enum):
    title = headings.get_title              # Suggested content titles
    subtitle = headings.get_subtitle        # Candidate content subtitles 
    description = headings.get_description  # Basic content summaries
    outline = headings.get_outline          # A list of content section key points or themes


def get_summary(content: str, summary: str, n_sections=3, top_n: int=10) -> tuple:
    """Return a dictionary of entities, keywords, and related topic tags"""
    summary_function = SummaryType(summary)
    if summary == "outline":
        scores = summary_function(content, n_sections=n_sections, top_n=1)[0]
    else:
        scores = summary_function(content, top_n=top_n)
    summary, scores = zip(*scores)

    return tags, scores
