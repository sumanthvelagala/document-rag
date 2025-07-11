import re
# cleaning header and footer
def is_header_or_footer(element, page_height=1600):
    try:
        coords = element.metadata.coordinates.points
        y_vals = [pt[1] for pt in coords]
        min_y = min(y_vals)
        max_y = max(y_vals)

        header_threshold = 0.08 * page_height  
        footer_threshold = 0.92 * page_height  

        is_header = min_y < header_threshold
        is_footer = max_y > footer_threshold

        return is_header or is_footer
    except Exception:
        return False

# cleaning authors and stuff between title and abstract
def clean_between_title_and_abstract(elements):
    title_idx = -1
    abstract_idx = -1
    for i, element in enumerate(elements):
        if title_idx == -1 and element.category == "Title":
            title_idx = i
        elif abstract_idx == -1 and (element.text.lower().startswith("abstract") or element.text.lower().startswith("abstract:")):
            abstract_idx = i
    if title_idx != -1 and abstract_idx != -1 and title_idx < abstract_idx:
        cleaned =  elements[:title_idx] + elements[abstract_idx:]
        return cleaned
    return elements

def clean_citations(elements):
    for element in elements:
        element.text = re.sub(r'\(\d+\)', '', element.text)
        element.text = re.sub(r'\b[A-Z][a-z]+ ?\(?\d+\)?', '', element.text)
    return elements

