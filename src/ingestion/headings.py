"""Finding a document's title page, and turning a short title into a heading for every chunk.

A cover page that holds only a title ("EXTENUATING CIRCUMSTANCES GUIDELINES FOR EXAMINATIONS")
becomes a chunk with no answer in it. It still ranks high, because it names the topic, so it
takes a place in the top k that a chunk with the answer should have. The chunks with the
answers, meanwhile, often never repeat the topic's own words.

TitlePageHeading finds the leading short pages of a document (the title page). They are never
chunked on their own: they hold no content, and the file name is already in the metadata.
When the title is short enough to be a heading, DocumentProcessor puts it in front of every
chunk of the document. A longer cover page (ministry names, intake years) would only add the
same noise to every chunk, so it is dropped without becoming a heading.
"""
from __future__ import annotations

Page = tuple[int, str]


class TitlePageHeading:
    """Leading pages of at most `max_title_words` words, followed by a longer page, are the title page.

    The title becomes the heading only if it has at most `max_heading_words` words.
    """

    def __init__(self, max_title_words: int = 20, max_heading_words: int = 10):
        if max_title_words < 1 or max_heading_words < 1:
            raise ValueError("max_title_words and max_heading_words must be at least 1.")
        self.max_title_words = max_title_words
        self.max_heading_words = max_heading_words

    def split(self, pages: list[Page]) -> tuple[str | None, list[Page]]:
        """Returns (heading or None, the pages that are still chunked)."""
        first_long = next(
            (i for i, (_, text) in enumerate(pages) if len(text.split()) > self.max_title_words), None
        )
        if not first_long:                      # None: no long page at all; 0: no title page
            return None, pages
        words = [word for _, text in pages[:first_long] for word in text.split()]
        heading = " ".join(words) if 0 < len(words) <= self.max_heading_words else None
        return heading, pages[first_long:]
