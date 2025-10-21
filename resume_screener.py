"""
Resume Screening System - Backend Module
Refactored for Streamlit compatibility
"""

import os
import re
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import docx2txt
import spacy
from sentence_transformers import SentenceTransformer, util

# Initialize models (cached)
_nlp = None
_model = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

# ============ TEXT EXTRACTION ============
def extract_text_from_pdf(file_obj) -> str:
    """Extract text from PDF file object or path"""
    text = []
    try:
        with fitz.open(stream=file_obj.read() if hasattr(file_obj, 'read') else file_obj, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        raise ValueError(f"Error reading PDF: {e}")
    return "\n".join(text)

def extract_text_from_docx(file_obj) -> str:
    """Extract text from DOCX file object or path"""
    try:
        if hasattr(file_obj, 'read'):
            return docx2txt.process(file_obj)
        else:
            return docx2txt.process(file_obj)
    except Exception as e:
        raise ValueError(f"Error reading DOCX: {e}")

def extract_text(file_obj, filename: str) -> str:
    """Extract text from uploaded file"""
    ext = filename.split('.')[-1].lower()
    
    if ext == "pdf":
        return extract_text_from_pdf(file_obj)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_obj)
    elif ext == "txt":
        content = file_obj.read()
        return content.decode('utf-8', errors='ignore') if isinstance(content, bytes) else content
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ============ ANONYMIZATION ============
def anonymize_text(text: str, remove_dates=True) -> str:
    """Remove PII to reduce bias"""
    nlp = get_nlp()
    doc = nlp(text)
    
    # Replace PERSON entities
    tokens = []
    person_spans = {ent.start_char: ent.end_char for ent in doc.ents if ent.label_ == "PERSON"}
    
    i = 0
    while i < len(text):
        if i in person_spans:
            end = person_spans[i]
            tokens.append("[NAME]")
            i = end
        else:
            tokens.append(text[i])
            i += 1
    
    anonymized = "".join(tokens)
    
    # Remove contact info
    anonymized = re.sub(r'\S+@\S+', '[EMAIL]', anonymized)
    anonymized = re.sub(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{6,12}', '[PHONE]', anonymized)
    anonymized = re.sub(r'http\S+|www\.\S+', '[URL]', anonymized)
    
    if remove_dates:
        anonymized = re.sub(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b', '[DATE]', anonymized)
    
    # Remove gendered words
    anonymized = re.sub(r'\b(male|female|man|woman|he|she|his|her)\b', '[GENDER]', anonymized, flags=re.IGNORECASE)
    
    return anonymized

# ============ SKILL EXTRACTION ============
import re
import spacy
from difflib import get_close_matches

# Load SpaCy model once
nlp = spacy.load("en_core_web_sm")

# Expanded engineering skill list
SKILL_LIST = [
    "electrical systems", "mechanical systems", "electronics", "fabrication",
    "troubleshooting", "diagnostics", "equipment installation", "preventive maintenance",
    "root cause analysis", "system testing", "quality assurance", "technical documentation",
    "solidworks", "autocad", "plc", "oracle", "oscilloscope", "multimeter",
    "ms office", "powerpoint", "excel", "visio", "sap", "labview",
    "six sigma", "iso", "lean manufacturing", "safety compliance", "osha", "mep",
    "team leadership", "training", "project scheduling", "vendor coordination",
    "maintenance planning", "field service", "military operations", "navy systems"
]

def extract_skills(text: str, threshold: float = 0.8) -> list:
    """Extract skills using fuzzy matching and noun chunking"""
    text_lower = text.lower()
    found = set()

    # Exact match first
    for skill in SKILL_LIST:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found.add(skill)

    # Fuzzy match via noun chunks
    doc = nlp(text)
    noun_phrases = [chunk.text.lower().strip() for chunk in doc.noun_chunks]

    for phrase in noun_phrases:
        matches = get_close_matches(phrase, SKILL_LIST, n=1, cutoff=threshold)
        if matches:
            found.add(matches[0])

    return sorted(found)


# ============ RESUME PROCESSING ============
def process_resume(file_obj, filename: str) -> Dict:
    """Process a single resume file"""
    try:
        raw_text = extract_text(file_obj, filename)
        anonymized = anonymize_text(raw_text)
        skills = extract_skills(raw_text)
        
        return {
            "filename": filename,
            "raw_text": raw_text,
            "anonymized_text": anonymized,
            "skills": skills,
            "status": "success"
        }
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "error": str(e)
        }

def process_multiple_resumes(uploaded_files) -> pd.DataFrame:
    """Process multiple uploaded resume files"""
    results = []
    
    for uploaded_file in uploaded_files:
        result = process_resume(uploaded_file, uploaded_file.name)
        results.append(result)
    
    df = pd.DataFrame(results)
    # Filter out errors
    df = df[df['status'] == 'success'].reset_index(drop=True)
    return df

# ============ EMBEDDING & RANKING ============
def compute_embeddings(texts: List[str]):
    """Compute sentence embeddings"""
    model = get_model()
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def prepare_embeddings(df_resumes: pd.DataFrame, job_desc_text: str):
    """Prepare embeddings for resumes and job description"""
    resume_texts = []
    
    for _, row in df_resumes.iterrows():
        skill_summary = " ".join(row["skills"]) if row["skills"] else ""
        # Combine anonymized text with skills (cap at 10k chars)
        combined = row["anonymized_text"][:10000] + "\n\nSkills: " + skill_summary
        resume_texts.append(combined)
    
    # Prepare job description
    jd_anonymized = anonymize_text(job_desc_text, remove_dates=False)
    jd_skills = " ".join(extract_skills(job_desc_text))
    jd_combined = jd_anonymized[:10000] + "\n\nRequired Skills: " + jd_skills
    
    all_texts = resume_texts + [jd_combined]
    embeddings = compute_embeddings(all_texts)
    
    resume_embs = embeddings[:-1]
    jd_emb = embeddings[-1]
    
    return resume_embs, jd_emb, jd_skills

def rank_resumes(df_resumes: pd.DataFrame, job_desc_text: str, 
                 top_k=10, skill_weight=0.4, semantic_weight=0.6) -> pd.DataFrame:
    """Rank resumes against job description"""
    
    if len(df_resumes) == 0:
        return pd.DataFrame()
    
    # Compute embeddings
    resume_embs, jd_emb, jd_skills_str = prepare_embeddings(df_resumes, job_desc_text)
    
    # Semantic similarity scores
    cos_scores = util.cos_sim(resume_embs, jd_emb)[:, 0].cpu().numpy()
    
    # Skill overlap scores
    jd_skills = set(extract_skills(job_desc_text))
    skill_scores = []
    
    for skills in df_resumes["skills"]:
        if not skills:
            skill_scores.append(0.0)
        else:
            overlap = len(set(skills).intersection(jd_skills))
            denom = max(1, len(jd_skills))
            skill_scores.append(overlap / denom)
    
    skill_scores = np.array(skill_scores)
    
    # Normalize semantic scores to 0-1
    cos_min, cos_max = cos_scores.min(), cos_scores.max()
    cos_range = cos_max - cos_min
    if cos_range > 0:
        normalized_cos = (cos_scores - cos_min) / cos_range
    else:
        normalized_cos = cos_scores
    
    # Combined score
    final_scores = semantic_weight * normalized_cos + skill_weight * skill_scores
    
    # Add scores to dataframe
    df = df_resumes.copy()
    df["semantic_score"] = cos_scores
    df["skill_score"] = skill_scores
    df["final_score"] = final_scores
    
    # Sort and return top K
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    return df.head(top_k)
