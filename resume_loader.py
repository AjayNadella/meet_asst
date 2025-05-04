# resume_loader.py

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def load_resume_and_jd():
    """Loads both resume and job description text."""
    resume_path = "resume_data/your_resume.pdf"
    jd_path = "resume_data/job_description.txt"

    resume_text = extract_text_from_pdf(resume_path)
    jd_text = open(jd_path, "r", encoding="utf-8").read().strip()

    return resume_text, jd_text
