import pypdf

def extract_text_from_pdf(file_path):
    """Extract text from each page of a PDF."""
    chunk_text = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunk_text.append(text.strip())  # Each page is one chunk
    return chunk_text
