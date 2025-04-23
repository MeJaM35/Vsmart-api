# AI-Powered Resume Parser

An advanced resume parsing system that extracts structured information from resumes in various formats (PDF, images) using AI and NLP techniques.

## Features

- Robust section detection with multiple fallback strategies
- Advanced skills classification with semantic matching
- Structured parsing of education, experience, and projects
- Support for PDF and image (OCR) processing
- Performance metrics and evaluation system
- Caching system for embeddings
- Dynamic skills taxonomy that auto-updates
- REST API with FastAPI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
HUGGINGFACE_API_TOKEN=your_token_here
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

3. Install Tesseract OCR:
- Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

4. Run the server:
```bash
uvicorn main:app --reload
```

## API Endpoints

### Upload Resume
```http
POST /upload
```
Upload and process a resume file (PDF or image)

### Get Metrics
```http
GET /metrics
```
Get parser performance metrics

### Evaluate Resume
```http
POST /evaluate/{resume_id}
```
Compare parser output with manual annotations

## Architecture

- FastAPI for REST API
- HuggingFace for embeddings and semantic matching
- pdfplumber and pdf2image for PDF processing
- Tesseract OCR for image processing
- scikit-learn for similarity calculations
- Async processing with background tasks
- Disk-persistent caching system

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request