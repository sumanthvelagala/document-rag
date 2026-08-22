import re

# universal text cleaner — works on any document type (legal, medical, financial, academic)
def clean_text(text):
    # fix hyphenated line breaks (word wrap artifacts from PDF extraction)
    text = re.sub(r'-\n', '', text)
    # remove page numbers sitting alone on a line
    text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
    # remove "Page X of Y" patterns
    text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    # collapse multiple spaces and normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # collapse multiple newlines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

