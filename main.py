from fastapi import FastAPI, UploadFile, HTTPException
from pathlib import Path
import pytesseract
from PIL import Image
import io
import re
import json
import os
import pdfplumber
from pdf2image import convert_from_bytes
import tempfile
from huggingface_hub import InferenceClient
import requests
from typing import List, Dict
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Initialize Hugging Face client with token from environment variable
client = InferenceClient(token=os.getenv('HUGGINGFACE_API_TOKEN'))

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using Hugging Face API"""
    try:
        # Using the embeddings endpoint
        embeddings = client.feature_extraction(texts)
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []

def calculate_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    dot_product = sum(a * b for a, b in zip(emb1, emb2))
    norm1 = sum(a * a for a in emb1) ** 0.5
    norm2 = sum(b * b for b in emb2) ** 0.5
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

def extract_text_from_pdf(pdf_bytes):
    try:
        # First try normal PDF text extraction
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # If we got meaningful text, return it
            if len(text.strip()) > 50:  # Arbitrary threshold to check if we got meaningful text
                return text

        # If normal extraction didn't yield good results, fall back to OCR
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def extract_text_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed (some images might be in RGBA or other formats)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills using semantic search and embeddings"""
    # Common skill categories for context
    skill_contexts = {
        "programming": "programming languages and software development",
        "frameworks": "software frameworks and libraries",
        "tools": "development tools and platforms",
        "databases": "database technologies and data storage",
        "cloud": "cloud computing and services",
        "soft_skills": "professional soft skills and methodologies"
    }
    
    # Extract potential skill phrases
    # Look for capitalized terms, terms after bullets, and other patterns
    potential_skills = set()
    
    # Extract terms after bullet points or dashes
    bullet_items = re.findall(r'[•\-]\s*([^•\n]+)', text)
    for item in bullet_items:
        # Look for technical terms and multi-word phrases
        terms = re.findall(r'\b[A-Z][A-Za-z0-9#+\-\.]*(?:\s+[A-Z][A-Za-z0-9#+\-\.]*)*\b', item)
        potential_skills.update(terms)
    
    # Extract capitalized terms that might be technologies
    tech_terms = re.findall(r'\b[A-Z][A-Za-z0-9#+\-\.]*\b', text)
    potential_skills.update(tech_terms)
    
    # Extract terms that follow common skill indicators
    skill_indicators = r'(?:proficient in|experience with|knowledge of|skilled in|expertise in)\s+([^\.]+)'
    indicator_matches = re.findall(skill_indicators, text.lower())
    for match in indicator_matches:
        terms = re.findall(r'\b[A-Za-z0-9#+\-\.]+\b', match)
        potential_skills.update(terms)
    
    # Filter and validate skills using embeddings
    validated_skills = set()
    if potential_skills:
        # Create skill descriptions for comparison
        skill_texts = [f"This is a technical skill or technology: {skill}" for skill in potential_skills]
        
        # Get embeddings for skill descriptions and contexts
        context_texts = [f"This describes {desc}" for desc in skill_contexts.values()]
        
        try:
            skill_embeddings = get_embeddings(skill_texts)
            context_embeddings = get_embeddings(context_texts)
            
            # Compare each potential skill with skill contexts
            for i, skill in enumerate(potential_skills):
                # Calculate similarity with each context
                max_similarity = max(
                    calculate_similarity(skill_embeddings[i], context_emb)
                    for context_emb in context_embeddings
                )
                
                # If the skill is similar enough to any context, add it
                if max_similarity > 0.5:  # Threshold can be adjusted
                    validated_skills.add(skill)
                
                # Rate limit to avoid API throttling
                time.sleep(0.1)
        
        except Exception as e:
            print(f"Error in skill validation: {e}")
            # Fallback: use basic filtering if API fails
            validated_skills = {
                skill for skill in potential_skills
                if len(skill) > 1 and not skill.lower() in {'and', 'the', 'with', 'for', 'in', 'on', 'at'}
            }
    
    return sorted(list(validated_skills))

def extract_name(text):
    """Extract name from resume text"""
    # Look for name in the first few lines
    first_lines = text.split('\n')[:3]
    for line in first_lines:
        # Look for a line with 2-3 words, all capitalized or proper case
        words = line.strip().split()
        if 2 <= len(words) <= 3:
            if all(word[0].isupper() for word in words):
                return line.strip()
    return None

def parse_resume_text(text):
    parsed_data = {
        "contact_info": {},
        "education": [],
        "experience": [],
        "skills": [],
        "projects": []
    }
    
    # Extract skills first since it's most improved
    parsed_data["skills"] = extract_skills_from_text(text)
    
    # Extract name
    name = extract_name(text)
    if name:
        parsed_data["contact_info"]["name"] = name
    
    # Extract contact information
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        parsed_data["contact_info"]["email"] = emails[0]
    
    phone_pattern = r'\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
    phones = re.findall(phone_pattern, text)
    if phones:
        parsed_data["contact_info"]["phone"] = phones[0]
    
    linkedin_pattern = r'linkedin\.com/in/[\w-]+'
    linkedin = re.findall(linkedin_pattern, text.lower())
    if linkedin:
        parsed_data["contact_info"]["linkedin"] = linkedin[0]
    
    # Process sections
    lines = text.split('\n')
    current_section = None
    section_content = []
    education_keywords = ["education", "academic", "qualification", "university", "college", "school", "bachelor", "master", "phd"]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_lower = line.lower()
        
        # Improved section detection
        if any(edu in line_lower for edu in education_keywords):
            if current_section and section_content:
                _add_section_content(parsed_data, current_section, section_content)
            current_section = "education"
            section_content = []
            if not line_lower == "education":
                section_content.append(line)
        elif any(exp in line_lower for exp in ["experience", "employment", "work history"]):
            if current_section and section_content:
                _add_section_content(parsed_data, current_section, section_content)
            current_section = "experience"
            section_content = []
        elif "projects" in line_lower:
            if current_section and section_content:
                _add_section_content(parsed_data, current_section, section_content)
            current_section = "projects"
            section_content = []
        else:
            if _is_education_line(line_lower):
                if current_section != "education":
                    if current_section and section_content:
                        _add_section_content(parsed_data, current_section, section_content)
                    current_section = "education"
                    section_content = []
                section_content.append(line)
            elif current_section:
                section_content.append(line)
    
    # Add the final section
    if current_section and section_content:
        _add_section_content(parsed_data, current_section, section_content)
    
    return parsed_data

def _is_education_line(line: str) -> bool:
    """Helper function to detect education-related lines"""
    edu_patterns = [
        r'\b(bachelor|master|phd|b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?sc|m\.?sc|diploma)\b',
        r'\b(20\d{2})\s*-\s*(20\d{2}|present)\b',
        r'\bcgpa\s*:?\s*\d+\.?\d*\b',
        r'\buniversity|college|institute\b'
    ]
    return any(re.search(pattern, line) for pattern in edu_patterns)

def _add_section_content(parsed_data: Dict, section: str, content: List[str]):
    """Helper function to add content to the appropriate section"""
    if content:
        # Clean up the content before adding
        cleaned_content = [
            line for line in content
            if line and not any(
                keyword in line.lower()
                for keyword in ["education", "experience", "employment", "projects"]
            )
        ]
        if cleaned_content:
            parsed_data[section].extend(cleaned_content)

@app.post("/upload")
async def upload_file(file: UploadFile):
    content = await file.read()
    
    # Create output directory if it doesn't exist
    output_dir = Path("parsed_resumes")
    output_dir.mkdir(exist_ok=True)
    
    # Determine file type and extract text
    file_ext = Path(file.filename).suffix.lower()
    if file_ext in ['.pdf']:
        extracted_text = extract_text_from_pdf(content)
    elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        extracted_text = extract_text_from_image(content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Parse the extracted text
    parsed_data = parse_resume_text(extracted_text)
    
    # Create filenames for both raw text and structured data
    original_name = Path(file.filename).stem
    text_filename = output_dir / f"resume-{original_name}.txt"
    json_filename = output_dir / f"resume-{original_name}.json"
    
    # Save raw text
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    # Save structured data
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
    
    return {
        "message": "Resume processed successfully",
        "text_file": str(text_filename),
        "json_file": str(json_filename),
        "parsed_data": parsed_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
