from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
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
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import time
from dotenv import load_dotenv
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging
import aiofiles
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import uuid
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resume_parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("resume_parser")

# Load environment variables
load_dotenv()

# Paths
DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "parsed_resumes"
EVAL_DIR = DATA_DIR / "evaluation"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR, OUTPUT_DIR, EVAL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class EvaluationMetrics:
    """Class for evaluating the performance of the resume parser"""
    def __init__(self, eval_dir: Path = EVAL_DIR):
        self.eval_dir = eval_dir
        self.metrics_file = eval_dir / "metrics.json"
        self.ground_truth_dir = eval_dir / "ground_truth"
        self.ground_truth_dir.mkdir(exist_ok=True)
        self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics if available"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
                self.metrics = self._initialize_metrics()
        else:
            self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict:
        """Initialize empty metrics structure"""
        return {
            "overall": {
                "total_processed": 0,
                "success_rate": 0.0,
                "average_processing_time": 0.0
            },
            "sections": {
                section: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0} 
                for section in ["contact_info", "summary", "education", "experience", 
                               "skills", "projects", "certifications", "leadership"]
            },
            "skills": {
                "precision": 0.0,
                "recall": 0.0, 
                "f1": 0.0,
                "average_confidence": 0.0
            },
            "errors": defaultdict(int)
        }
    
    def save_ground_truth(self, resume_id: str, parsed_data: Dict, manual_data: Dict):
        """Save ground truth data for a resume"""
        try:
            gt_file = self.ground_truth_dir / f"{resume_id}_ground_truth.json"
            data = {
                "resume_id": resume_id,
                "parsed_data": parsed_data,
                "manual_data": manual_data,
                "timestamp": datetime.now().isoformat()
            }
            with open(gt_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving ground truth: {e}")
    
    def update_metrics(self, resume_id: str, start_time: float, 
                      parsed_data: Dict, error: Optional[str] = None):
        """Update metrics with results from a parse operation"""
        processing_time = time.time() - start_time
        
        # Update overall metrics
        self.metrics["overall"]["total_processed"] += 1
        
        if error:
            self.metrics["errors"][error] = self.metrics["errors"].get(error, 0) + 1
            success = False
        else:
            success = True
        
        # Calculate success rate
        success_count = self.metrics["overall"]["total_processed"] - sum(self.metrics["errors"].values())
        self.metrics["overall"]["success_rate"] = success_count / self.metrics["overall"]["total_processed"]
        
        # Update average processing time with exponential moving average
        prev_avg = self.metrics["overall"]["average_processing_time"]
        if prev_avg == 0:
            self.metrics["overall"]["average_processing_time"] = processing_time
        else:
            self.metrics["overall"]["average_processing_time"] = 0.9 * prev_avg + 0.1 * processing_time
        
        # Save updated metrics
        self._save_metrics()
        
        # Return processing time for this operation
        return {
            "resume_id": resume_id,
            "success": success,
            "processing_time": processing_time,
            "error": error
        }
    
    def evaluate_against_ground_truth(self, resume_id: str):
        """Evaluate parser performance against ground truth data"""
        gt_file = self.ground_truth_dir / f"{resume_id}_ground_truth.json"
        if not gt_file.exists():
            return {"error": "Ground truth not found for this resume"}
        
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            parsed = data["parsed_data"]
            manual = data["manual_data"]
            
            results = {}
            
            # Evaluate each section
            for section in ["contact_info", "summary", "education", "experience", 
                           "skills", "projects", "certifications", "leadership"]:
                if section in parsed and section in manual:
                    metrics = self._calculate_section_metrics(parsed[section], manual[section])
                    results[section] = metrics
                    
                    # Update running metrics
                    section_metrics = self.metrics["sections"][section]
                    count = section_metrics["count"]
                    section_metrics["precision"] = (section_metrics["precision"] * count + metrics["precision"]) / (count + 1)
                    section_metrics["recall"] = (section_metrics["recall"] * count + metrics["recall"]) / (count + 1)
                    section_metrics["f1"] = (section_metrics["f1"] * count + metrics["f1"]) / (count + 1)
                    section_metrics["count"] += 1
            
            # Special handling for skills
            if "skills" in parsed and "skills" in manual:
                skill_metrics = self._calculate_skills_metrics(parsed["skills"], manual["skills"])
                results["skills_detailed"] = skill_metrics
                
                # Update running metrics
                self.metrics["skills"]["precision"] = (self.metrics["skills"]["precision"] * 
                                                    self.metrics["sections"]["skills"]["count"] + 
                                                    skill_metrics["precision"]) / (self.metrics["sections"]["skills"]["count"] + 1)
                self.metrics["skills"]["recall"] = (self.metrics["skills"]["recall"] * 
                                                   self.metrics["sections"]["skills"]["count"] + 
                                                   skill_metrics["recall"]) / (self.metrics["sections"]["skills"]["count"] + 1)
                self.metrics["skills"]["f1"] = (self.metrics["skills"]["f1"] * 
                                              self.metrics["sections"]["skills"]["count"] + 
                                              skill_metrics["f1"]) / (self.metrics["sections"]["skills"]["count"] + 1)
                self.metrics["skills"]["average_confidence"] = (self.metrics["skills"]["average_confidence"] * 
                                                             self.metrics["sections"]["skills"]["count"] + 
                                                             skill_metrics["average_confidence"]) / (self.metrics["sections"]["skills"]["count"] + 1)
            
            # Save updated metrics
            self._save_metrics()
            
            return results
        
        except Exception as e:
            logger.error(f"Error evaluating against ground truth: {e}")
            return {"error": str(e)}
    
    def _calculate_section_metrics(self, parsed_data: Any, manual_data: Any) -> Dict:
        """Calculate precision, recall, F1 for a section"""
        if isinstance(parsed_data, dict) and isinstance(manual_data, dict):
            # For dictionary sections like contact_info
            parsed_set = set(parsed_data.items())
            manual_set = set(manual_data.items())
        elif isinstance(parsed_data, list) and isinstance(manual_data, list):
            # For list sections
            parsed_set = set(str(item).lower() for item in parsed_data)
            manual_set = set(str(item).lower() for item in manual_data)
        else:
            # Fallback for mixed types
            parsed_set = {str(parsed_data).lower()}
            manual_set = {str(manual_data).lower()}
        
        true_positives = len(parsed_set.intersection(manual_set))
        precision = true_positives / len(parsed_set) if parsed_set else 0
        recall = true_positives / len(manual_set) if manual_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _calculate_skills_metrics(self, parsed_skills: Dict, manual_skills: Dict) -> Dict:
        """Calculate detailed metrics for skills section"""
        # Extract flat list of skill names from parsed data
        parsed_skill_names = set()
        confidence_sum = 0
        confidence_count = 0
        
        for category, skills in parsed_skills.items():
            for skill in skills:
                if isinstance(skill, dict) and "name" in skill:
                    parsed_skill_names.add(skill["name"].lower())
                    if "confidence" in skill:
                        confidence_sum += skill["confidence"]
                        confidence_count += 1
                else:
                    parsed_skill_names.add(str(skill).lower())
        
        # Extract flat list of skill names from manual data
        manual_skill_names = set()
        for category, skills in manual_skills.items():
            for skill in skills:
                if isinstance(skill, dict) and "name" in skill:
                    manual_skill_names.add(skill["name"].lower())
                else:
                    manual_skill_names.add(str(skill).lower())
        
        # Calculate metrics
        true_positives = len(parsed_skill_names.intersection(manual_skill_names))
        precision = true_positives / len(parsed_skill_names) if parsed_skill_names else 0
        recall = true_positives / len(manual_skill_names) if manual_skill_names else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_confidence": avg_confidence,
            "parsed_skills_count": len(parsed_skill_names),
            "manual_skills_count": len(manual_skill_names),
            "common_skills_count": true_positives
        }
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary of current metrics"""
        return self.metrics

# Initialize evaluation metrics
evaluation = EvaluationMetrics()

class EmbeddingCache:
    """Efficient caching system for embeddings with disk persistence"""
    def __init__(self, cache_dir: Path = CACHE_DIR, max_size: int = 10000):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "embedding_cache.pkl"
        self.max_size = max_size
        self.cache: Dict[str, Tuple[List[float], datetime]] = {}
        self.lock = threading.Lock()
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if exists"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                # Clean old entries
                self._cleanup()
                logger.info(f"Loaded embedding cache with {len(self.cache)} entries")
            else:
                logger.info("No embedding cache found, starting with empty cache")
                self.cache = {}
        except Exception as e:
            logger.error(f"Cache load error: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    def _cleanup(self):
        """Remove old entries and maintain cache size"""
        now = datetime.now()
        # Remove entries older than 30 days
        self.cache = {
            k: v for k, v in self.cache.items()
            if (now - v[1]) < timedelta(days=30)
        }
        # If still too large, remove oldest entries
        if len(self.cache) > self.max_size:
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            self.cache = dict(sorted_items[-self.max_size:])
    
    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        with self.lock:
            if key in self.cache:
                embedding, _ = self.cache[key]
                return embedding
        return None
    
    def set(self, key: str, embedding: List[float]):
        """Set embedding in cache"""
        with self.lock:
            self.cache[key] = (embedding, datetime.now())
            if len(self.cache) >= self.max_size:
                self._cleanup()
            # Periodically save cache
            if len(self.cache) % 100 == 0:
                self._save_cache()

# Initialize embedding cache
embedding_cache = EmbeddingCache()

# Initialize Hugging Face client
client = InferenceClient(
    token=os.getenv('HUGGINGFACE_API_TOKEN', ''),
    model=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
)

class SkillsLoader:
    """Dynamic skills taxonomy loader with auto-updates"""
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.skills_file = models_dir / "skills_taxonomy.json"
        self.update_interval = timedelta(days=7)  # Check for updates weekly
        self.skills_taxonomy = {}
        self.last_updated = None
        self.skill_vectors = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize skills taxonomy"""
        if not self._initialized:
            await self._load_or_update_taxonomy()
            self._initialized = True
    
    async def _load_or_update_taxonomy(self):
        """Load existing taxonomy or update from sources"""
        try:
            if self.skills_file.exists():
                # Check if update is needed
                mod_time = datetime.fromtimestamp(self.skills_file.stat().st_mtime)
                if datetime.now() - mod_time < self.update_interval:
                    # Load existing taxonomy
                    async with aiofiles.open(self.skills_file, 'r') as f:
                        content = await f.read()
                        self.skills_taxonomy = json.loads(content)
                    self.last_updated = mod_time
                    logger.info(f"Loaded skills taxonomy with {sum(len(skills) for skills in self.skills_taxonomy.values())} skills")
                    # Generate vectors for skills
                    await self._generate_skill_vectors()
                    return
            
            # Need to update taxonomy
            await self._update_taxonomy()
            
        except Exception as e:
            logger.error(f"Error loading skills taxonomy: {e}")
            # If failed to load or update, initialize with minimal taxonomy
            self._initialize_minimal_taxonomy()
    
    async def _update_taxonomy(self):
        """Update skills taxonomy from various sources"""
        try:
            # Start with base categories
            updated_taxonomy = {
                "programming": [],
                "frameworks": [],
                "data": [],
                "cloud": [],
                "tools": [],
                "soft_skills": [],
                "domain_knowledge": [],
                "methodologies": []
            }
            
            # Try to fetch from GitHub or other sources
            # This would typically include fetching from a maintained repository
            sources = [
                self._fetch_from_github,
                self._fetch_from_stackoverflow,
                self._fetch_from_local_source
            ]
            
            for source_func in sources:
                try:
                    source_data = await source_func()
                    if source_data:
                        # Merge data from this source
                        for category, skills in source_data.items():
                            if category in updated_taxonomy:
                                # Add new skills, avoid duplicates
                                existing = set(skill.lower() for skill in updated_taxonomy[category])
                                updated_taxonomy[category].extend(
                                    skill for skill in skills 
                                    if skill.lower() not in existing
                                )
                except Exception as e:
                    logger.warning(f"Error fetching from source {source_func.__name__}: {e}")
            
            # Save updated taxonomy
            self.skills_taxonomy = updated_taxonomy
            async with aiofiles.open(self.skills_file, 'w') as f:
                await f.write(json.dumps(updated_taxonomy, indent=2))
            
            self.last_updated = datetime.now()
            logger.info(f"Updated skills taxonomy with {sum(len(skills) for skills in updated_taxonomy.values())} skills")
            
            # Generate vectors for skills
            await self._generate_skill_vectors()
            
        except Exception as e:
            logger.error(f"Error updating skills taxonomy: {e}")
            self._initialize_minimal_taxonomy()
    
    async def _fetch_from_github(self) -> Dict[str, List[str]]:
        """Fetch skills from GitHub repositories"""
        # Placeholder for actual implementation
        # Would fetch from a curated repository of skills
        return {}
    
    async def _fetch_from_stackoverflow(self) -> Dict[str, List[str]]:
        """Fetch trending skills from StackOverflow tags"""
        # Placeholder for actual implementation
        return {}
    
    async def _fetch_from_local_source(self) -> Dict[str, List[str]]:
        """Fetch skills from local backup file"""
        try:
            local_file = self.models_dir / "backup_skills.json"
            if local_file.exists():
                async with aiofiles.open(local_file, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
        except Exception as e:
            logger.warning(f"Error loading from local backup: {e}")
        
        # Return a minimal set as fallback
        return self._get_minimal_taxonomy()
    
    def _initialize_minimal_taxonomy(self):
        """Initialize with a minimal taxonomy if all else fails"""
        self.skills_taxonomy = self._get_minimal_taxonomy()
        logger.info(f"Initialized minimal skills taxonomy with {sum(len(skills) for skills in self.skills_taxonomy.values())} skills")
    
    def _get_minimal_taxonomy(self) -> Dict[str, List[str]]:
        """Get a minimal taxonomy of common skills"""
        return {
            "programming": ["Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Swift"],
            "frameworks": ["React", "Angular", "Vue", "Django", "Flask", "Spring", "ASP.NET", "Express", "TensorFlow", "PyTorch"],
            "data": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Elasticsearch", "Redis", "Kafka", "Hadoop", "Spark", "Power BI", "Tableau"],
            "cloud": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CloudFormation", "Lambda", "EC2", "S3"],
            "tools": ["Git", "GitHub", "GitLab", "Jenkins", "CircleCI", "Travis CI", "Jira", "Confluence", "VS Code", "IntelliJ"],
            "soft_skills": ["Communication", "Teamwork", "Problem Solving", "Critical Thinking", "Time Management"],
            "domain_knowledge": ["Finance", "Healthcare", "E-commerce", "Marketing", "Education"],
            "methodologies": ["Agile", "Scrum", "Kanban", "DevOps", "CI/CD", "TDD", "BDD"]
        }
    
    async def _generate_skill_vectors(self):
        """Generate embedding vectors for all skills"""
        all_skills = []
        for category, skills in self.skills_taxonomy.items():
            all_skills.extend(skills)
        
        # Deduplicate
        unique_skills = list(set(all_skills))
        
        # Track success rate for API calls
        api_success = 0
        api_failures = 0
        
        # Generate vectors in batches
        batch_size = 16  # Reduced from 32 to improve reliability
        max_retries = 3
        
        for i in range(0, len(unique_skills), batch_size):
            batch = unique_skills[i:i+batch_size]
            
            # Check cache first
            batch_embeddings = []
            uncached_skills = []
            uncached_indices = []
            
            for j, skill in enumerate(batch):
                cached_embedding = embedding_cache.get(skill)
                if cached_embedding is not None:
                    batch_embeddings.append((j, cached_embedding))
                else:
                    uncached_skills.append(skill)
                    uncached_indices.append(j)
            
            # Get embeddings for uncached skills
            if uncached_skills:
                retry_count = 0
                while retry_count < max_retries and uncached_skills:
                    try:
                        # Use exponential backoff for retries
                        if retry_count > 0:
                            wait_time = 2 ** retry_count
                            await asyncio.sleep(wait_time)
                            logger.info(f"Retry {retry_count}/{max_retries} for skill vectors after {wait_time}s delay")
                        
                        embeddings = client.feature_extraction(uncached_skills)
                        
                        # If we get here, API call was successful
                        for j, (idx, skill) in enumerate(zip(uncached_indices, uncached_skills)):
                            embedding_cache.set(skill, embeddings[j])
                            batch_embeddings.append((idx, embeddings[j]))
                        
                        # Clear lists since we processed all skills
                        uncached_skills = []
                        uncached_indices = []
                        api_success += 1
                        
                    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                        # Specific handling for connection and HTTP errors
                        retry_count += 1
                        api_failures += 1
                        logger.error(f"Error generating skill vectors: {e}")
                        
                        if retry_count >= max_retries:
                            # Use fallback after max retries
                            logger.warning("Max retries reached, using fallback for skill vectors")
                            self._use_tfidf_fallback(uncached_skills, uncached_indices, batch_embeddings)
                            uncached_skills = []
                            uncached_indices = []
                    
                    except Exception as e:
                        # Other exceptions
                        api_failures += 1
                        logger.error(f"Unexpected error generating skill vectors: {e}")
                        retry_count += 1
                        
                        if retry_count >= max_retries:
                            # Use fallback after max retries
                            self._use_tfidf_fallback(uncached_skills, uncached_indices, batch_embeddings)
                            uncached_skills = []
                            uncached_indices = []
            
            # Sort by original index and extract just the embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            for j, skill in enumerate(batch):
                if j < len(batch_embeddings):
                    self.skill_vectors[skill] = batch_embeddings[j][1]
        
        # Log summary of generation process
        total_attempts = api_success + api_failures
        success_rate = (api_success / total_attempts * 100) if total_attempts > 0 else 0
        logger.info(f"Generated vectors for {len(self.skill_vectors)} skills (API success rate: {success_rate:.1f}%)")
        
        # If we have very low success rate with the API, log a warning
        if total_attempts > 0 and success_rate < 30:
            logger.warning("Very low API success rate. Consider checking API connectivity or switching to local models")
    
    def _use_tfidf_fallback(self, skills, indices, batch_embeddings):
        """Use TF-IDF as a fallback method when API is unavailable"""
        try:
            logger.warning("Using TF-IDF fallback for skill vectors")
            # Create a TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=768)  # Match the embedding dimension
            
            # Fit and transform the skills
            tfidf_matrix = vectorizer.fit_transform(skills)
            
            # Convert sparse matrix to dense and pad if necessary
            dense_matrix = tfidf_matrix.toarray()
            
            # If dimensions are less than 768, pad with zeros
            if dense_matrix.shape[1] < 768:
                pad_width = ((0, 0), (0, 768 - dense_matrix.shape[1]))
                dense_matrix = np.pad(dense_matrix, pad_width, 'constant')
            # If dimensions are more than 768, truncate
            elif dense_matrix.shape[1] > 768:
                dense_matrix = dense_matrix[:, :768]
                
            # Normalize the vectors
            from sklearn.preprocessing import normalize
            dense_matrix = normalize(dense_matrix)
            
            # Add to batch embeddings
            for j, idx in enumerate(indices):
                batch_embeddings.append((idx, dense_matrix[j].tolist()))
                embedding_cache.set(skills[j], dense_matrix[j].tolist())
                
            logger.info(f"Generated {len(skills)} TF-IDF fallback vectors")
        except Exception as e:
            logger.error(f"Error in TF-IDF fallback: {e}")
            # Last resort fallback: use random vectors
            for j, idx in enumerate(indices):
                random_vector = (np.random.random(768) * 2 - 1).tolist()
                batch_embeddings.append((idx, random_vector))
                embedding_cache.set(skills[j], random_vector)
    
    def find_matching_skills(self, text: str, threshold: float = 0.75) -> List[Dict[str, Union[str, float, str]]]:
        """Find matching skills in text using embeddings and pattern matching"""
        if not self._initialized or not self.skill_vectors:
            return []
        
        matches = []
        
        # Direct pattern matching for exact matches
        for category, skills in self.skills_taxonomy.items():
            for skill in skills:
                # Create pattern that matches whole words only
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    matches.append({
                        "name": skill,
                        "category": category,
                        "confidence": 1.0,
                        "method": "exact_match"
                    })
        
        # Semantic matching for phrases
        try:
            # Split text into sentences or chunks
            chunks = [s.strip() for s in re.split(r'[,.\n]', text) if s.strip()]
            
            for chunk in chunks:
                # Skip if too short
                if len(chunk.split()) < 2:
                    continue
                
                # Get embedding for chunk
                chunk_embedding = client.feature_extraction([chunk])[0]
                
                # Compare with skill vectors
                for skill, skill_vector in self.skill_vectors.items():
                    similarity = cosine_similarity([chunk_embedding], [skill_vector])[0][0]
                    if similarity > threshold:
                        # Find category
                        category = next((cat for cat, skills in self.skills_taxonomy.items() 
                                       if skill in skills), "other")
                        
                        # Check if already matched exactly
                        if not any(m["name"] == skill for m in matches):
                            matches.append({
                                "name": skill,
                                "category": category,
                                "confidence": float(similarity),
                                "method": "semantic_match"
                            })
        except Exception as e:
            logger.error(f"Error in semantic skill matching: {e}")
        
        # Remove duplicates, keeping highest confidence
        unique_matches = {}
        for match in matches:
            name = match["name"].lower()
            if name not in unique_matches or match["confidence"] > unique_matches[name]["confidence"]:
                unique_matches[name] = match
        
        return list(unique_matches.values())
    
    def categorize_skills(self, skills: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize skills into taxonomy categories"""
        categorized = defaultdict(list)
        
        for skill in skills:
            if "category" in skill:
                categorized[skill["category"]].append(skill)
            else:
                # Try to find category
                skill_name = skill["name"]
                category = None
                
                # Check if skill exists in taxonomy
                for cat, cat_skills in self.skills_taxonomy.items():
                    if any(s.lower() == skill_name.lower() for s in cat_skills):
                        category = cat
                        break
                
                if not category:
                    # Use semantic similarity to find closest category
                    category = "other"
                
                skill["category"] = category
                categorized[category].append(skill)
        
        return dict(categorized)

# Initialize skills loader
skills_loader = SkillsLoader()

class BatchProcessor:
    """Efficient batch processor for API calls"""
    def __init__(self, batch_size: int = 32, wait_time: float = 0.1):
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.current_batch = []
        self.results = []
    
    async def add_item(self, item: str) -> int:
        """Add item to current batch and process if batch is full"""
        self.current_batch.append(item)
        if len(self.current_batch) >= self.batch_size:
            await self.process_batch()
        return len(self.results) - 1
    
    async def process_batch(self):
        """Process current batch of items"""
        if not self.current_batch:
            return
        
        try:
            # Get embeddings for batch
            embeddings = await self._get_embeddings_batch(self.current_batch)
            self.results.extend(embeddings)
            self.current_batch = []
            await asyncio.sleep(self.wait_time)  # Rate limiting
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        results = []
        for text in texts:
            # Check cache first
            cached = embedding_cache.get(text)
            if cached is not None:
                results.append(cached)
                continue
            
            # Get from API if not in cache
            try:
                embedding = client.feature_extraction([text])[0]
                embedding_cache.set(text, embedding)
                results.append(embedding)
            except Exception as e:
                logger.error(f"API error for text: {text[:50]}... Error: {e}")
                results.append([0.0] * 768)  # Default embedding size
        
        return results
    
    async def get_result(self, index: int) -> List[float]:
        """Get result by index"""
        while index >= len(self.results):
            if self.current_batch:
                await self.process_batch()
            await asyncio.sleep(0.1)
        return self.results[index]
    
    async def finish(self):
        """Process any remaining items in batch"""
        if self.current_batch:
            await self.process_batch()

class ResumeParser:
    def __init__(self):
        """Initialize the resume parser"""
        self.batch_processor = BatchProcessor()
        self._initialized = False
        
        # Common section patterns with variations
        self.section_patterns = {
            "summary": [
                r"(?i)^(?:professional\s+)?summary",
                r"(?i)^(?:career\s+)?objective",
                r"(?i)^profile",
                r"(?i)^about(?:\s+me)?"
            ],
            "education": [
                r"(?i)^education(?:al)?(?:\s+background)?",
                r"(?i)^academic",
                r"(?i)^qualification"
            ],
            "experience": [
                r"(?i)^(?:work\s+)?experience",
                r"(?i)^employment(?:\s+history)?"
            ],
            "skills": [
                r"(?i)^(?:technical\s+)?skills(?:\s+&\s+abilities)?",
                r"(?i)^(?:core\s+)?competencies",
                r"(?i)^technologies"
            ],
            "projects": [
                r"(?i)^projects",
                r"(?i)^personal\s+projects",
                r"(?i)^academic\s+projects"
            ],
            "certifications": [
                r"(?i)^certifications",
                r"(?i)^certificates",
                r"(?i)^credentials"
            ],
            "languages": [
                r"(?i)^languages",
                r"(?i)^spoken\s+languages"
            ],
            "leadership": [
                r"(?i)^leadership",
                r"(?i)^activities",
                r"(?i)^extracurricular"
            ]
        }
        
        # Add date parsing patterns
        self.date_patterns = [
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:\d{2}|\d{4})',  # Month Year
            r'\d{2}/\d{2}(?:/\d{2}|\d{4})?',  # MM/DD/YY or MM/DD/YYYY
            r'\d{4}(?:\s*[-–]\s*(?:\d{4}|Present|Current|Now))?',  # YYYY - YYYY or YYYY - Present
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\s*[-–]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}',  # Month Year - Month Year
        ]
        self.date_pattern = re.compile('|'.join(f'({p})' for p in self.date_patterns), re.IGNORECASE)
        
        # Education degree patterns
        self.degree_patterns = [
            r"(?:B\.?(?:Tech|E|Sc)|Bachelor['']?s?)(?:\s+(?:of|in|degree))?\s+(?:in\s+)?([^,\n]+)",
            r"(?:M\.?(?:Tech|E|Sc|BA)|Master['']?s?)(?:\s+(?:of|in|degree))?\s+(?:in\s+)?([^,\n]+)",
            r"(?:Ph\.?D|Doctorate)(?:\s+(?:in|of))?\s+([^,\n]+)",
            r"Diploma\s+(?:in|of)?\s+([^,\n]+)"
        ]
        self.degree_pattern = re.compile('|'.join(self.degree_patterns), re.IGNORECASE)

    async def initialize(self):
        """Initialize the resume parser and required resources"""
        if not self._initialized:
            # Initialize the batch processor
            await self.batch_processor.finish()
            # Initialize any other required resources
            self._initialized = True

    async def cleanup(self):
        """Clean up resources when shutting down"""
        try:
            # Ensure batch processor finishes any pending work
            if hasattr(self, 'batch_processor'):
                await self.batch_processor.finish()
            
            logger.info("Resume parser resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during parser cleanup: {e}")

    def _extract_sections(self, text: str) -> Dict[str, List[str]]:
        """Extract sections from resume text"""
        lines = text.split('\n')
        sections = {}
        current_section = "header"
        sections[current_section] = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            is_section_header = False
            for section_name, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        # This line is a section header
                        current_section = section_name
                        sections[current_section] = []
                        is_section_header = True
                        break
                if is_section_header:
                    break
                    
            if not is_section_header:
                # Add line to current section
                sections[current_section].append(line)
        
        # Extract contact information from header
        if "header" in sections:
            header_text = '\n'.join(sections["header"])
            sections["contact_info"] = self._extract_contact_info(header_text)
            
        # Extract summary if not properly categorized
        if "summary" not in sections and "header" in sections:
            # Look for summary-like paragraph in header
            header_lines = sections["header"]
            for i, line in enumerate(header_lines):
                if len(line.split()) > 10 and "." in line:  # Likely a summary sentence
                    sections["summary"] = [line]
                    break
                    
        return sections
        
    def _parse_education(self, content: List[str]) -> List[Dict[str, str]]:
        """Parse education entries into structured data"""
        education_entries = []
        current_entry = {}
        
        for line in content:
            line = line.strip()
            if not line:
                if current_entry:
                    education_entries.append(current_entry)
                    current_entry = {}
                continue
            
            # Try to extract degree
            degree_match = self.degree_pattern.search(line)
            if degree_match:
                if current_entry:
                    education_entries.append(current_entry)
                current_entry = {"degree": degree_match.group(0)}
                # Extract institution after the degree
                parts = line.split(',')
                if len(parts) > 1:
                    current_entry["institution"] = parts[1].strip()
                continue
            
            # Look for dates
            date_match = self.date_pattern.search(line)
            if date_match and current_entry:
                current_entry["date"] = date_match.group(0)
            
            # Look for GPA/grades
            grade_match = re.search(r'(?:GPA|Grade|CGPA|Score)[:.\s]*([0-9.]+%?)', line, re.IGNORECASE)
            if grade_match and current_entry:
                current_entry["grade"] = grade_match.group(1)
            
            # Additional details
            if current_entry and "details" not in current_entry:
                current_entry["details"] = []
            if current_entry and line.startswith(('•', '-')):
                current_entry["details"].append(line)
        
        # Add last entry if exists
        if current_entry:
            education_entries.append(current_entry)
        
        return education_entries

    def _parse_experience(self, content: List[str]) -> List[Dict[str, str]]:
        """Parse experience entries into structured data"""
        experience_entries = []
        current_entry = {}
        current_bullets = []
        
        for line in content:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new position (contains both company and date)
            date_match = self.date_pattern.search(line)
            if date_match and not line.startswith(('•', '-')):
                # Save previous entry if exists
                if current_entry:
                    current_entry["description"] = current_bullets
                    experience_entries.append(current_entry)
                
                # Start new entry
                current_entry = {"date": date_match.group(0)}
                current_bullets = []
                
                # Extract position and company
                position_part = line[:date_match.start()].strip()
                if ',' in position_part:
                    position, company = [p.strip() for p in position_part.split(',', 1)]
                    current_entry["position"] = position
                    current_entry["company"] = company
                else:
                    current_entry["position"] = position_part
            
            # Add bullet points to current entry
            elif line.startswith(('•', '-')):
                current_bullets.append(line)
            # Other lines might be continuation of company/position
            elif current_entry and not current_bullets:
                current_entry["company"] = line
        
        # Add last entry
        if current_entry:
            current_entry["description"] = current_bullets
            experience_entries.append(current_entry)
        
        return experience_entries

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information using improved regex patterns"""
        contact_info = {}
        
        # Find name in first few lines
        first_lines = [line.strip() for line in text.split('\n')[:3]]
        for line in first_lines:
            words = line.split()
            if 2 <= len(words) <= 3 and all(word[0].isupper() for word in words):
                contact_info["name"] = line
                break
        
        # Improved patterns for contact details
        patterns = {
            "email": r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
            "phone": r'(?:\+\d{1,3}[-.)]\s*)?(?:\d{3}[-.)]\s*)\d{3}[-.)]\s*\d{4}',
            "linkedin": r'(?:linkedin\.com/in/|in/)([\w-]+)',
            "github": r'(?:github\.com/|github:[\s/]*)(\w+)',
            "website": r'https?://(?:www\.)?[\w\.-]+\.\w+/?[\w\.-]*/?',
            "social": r'(?:^|\s)@([\w\.-]+)'
        }
        
        # Search in first few lines with proper extraction
        header_text = '\n'.join(first_lines)
        for field, pattern in patterns.items():
            matches = re.finditer(pattern, header_text, re.IGNORECASE)
            for match in matches:
                if field in ("linkedin", "github", "social"):
                    # Extract just the username for social profiles
                    contact_info[field] = match.group(1)
                else:
                    contact_info[field] = match.group(0)
        
        return contact_info

    def _parse_projects(self, content: List[str]) -> List[Dict[str, str]]:
        """Parse project entries into structured data"""
        projects = []
        current_project = {}
        current_bullets = []
        
        for line in content:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a project title (usually has GitHub or link)
            if not line.startswith(('•', '-')) and ('github' in line.lower() or 'http' in line.lower()):
                # Save previous project
                if current_project:
                    current_project["description"] = current_bullets
                    projects.append(current_project)
                
                current_project = {"title": line.split('GitHub')[0].strip()}
                current_bullets = []
                
                # Extract GitHub link if present
                github_match = re.search(r'github\.com/[\w-]+/[\w-]+', line, re.IGNORECASE)
                if github_match:
                    current_project["link"] = github_match.group(0)
            
            # New project without link
            elif not line.startswith(('•', '-')) and len(current_bullets) == 0:
                if current_project:
                    current_project["description"] = current_bullets
                    projects.append(current_project)
                current_project = {"title": line}
                current_bullets = []
            
            # Add bullet points
            elif line.startswith(('•', '-')):
                current_bullets.append(line)
        
        # Add last project
        if current_project:
            current_project["description"] = current_bullets
            projects.append(current_project)
        
        return projects

    def _extract_skills(self, content: List[str]) -> Dict[str, List[Dict]]:
        """Extract skills from the skills section"""
        # Join all content lines
        text = " ".join(content)
        
        # Find skills using skills loader
        skill_matches = skills_loader.find_matching_skills(text)
        
        # Categorize skills
        categorized_skills = skills_loader.categorize_skills(skill_matches)
        
        return categorized_skills

    async def parse_resume(self, text: str) -> Dict:
        """Parse resume text into structured sections"""
        # Initialize parsed data dictionary
        parsed_data = {}
        
        # Extract sections
        sections = self._extract_sections(text)
        
        # Process each section with structured parsing
        for section_name, content in sections.items():
            if section_name == "skills" and content:
                parsed_data["skills"] = self._extract_skills(content)
            elif section_name == "education" and content:
                parsed_data["education"] = self._parse_education(content)
            elif section_name == "experience" and content:
                parsed_data["experience"] = self._parse_experience(content)
            elif section_name == "projects" and content:
                parsed_data["projects"] = self._parse_projects(content)
            else:
                parsed_data[section_name] = content
        
        return parsed_data

# Initialize parser instance
parser = ResumeParser()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown events"""
    # Initialize parser and loader on startup
    await parser.initialize()
    await skills_loader.initialize()
    yield
    # Cleanup on shutdown
    await parser.cleanup()

# FastAPI app with lifespan
app = FastAPI(
    title="Resume Parser API",
    description="API for parsing resumes and extracting structured data",
    version="2.0.0",
    lifespan=lifespan
)

# API Endpoints
@app.post("/upload")
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Upload and process a resume file"""
    start_time = time.time()
    resume_id = str(uuid.uuid4())
    
    try:
        # Create output directories if they don't exist
        for directory in [OUTPUT_DIR, OUTPUT_DIR / "text", OUTPUT_DIR / "json"]:
            directory.mkdir(exist_ok=True, parents=True)
        
        content = await file.read()
        
        # Extract text based on file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext == '.pdf':
            extracted_text = await extract_text_from_pdf(content)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            extracted_text = await extract_text_from_image(content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF or image files."
            )
        
        # Save extracted text
        text_path = OUTPUT_DIR / "text" / f"{resume_id}.txt"
        async with aiofiles.open(text_path, "w", encoding="utf-8") as f:
            await f.write(extracted_text)
        
        # Parse resume
        parsed_data = await parser.parse_resume(extracted_text)
        
        # Save parsed data
        json_path = OUTPUT_DIR / "json" / f"{resume_id}.json"
        async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(parsed_data, indent=2))
        
        # Update metrics in background
        background_tasks.add_task(
            evaluation.update_metrics,
            resume_id,
            start_time,
            parsed_data
        )
        
        return {
            "resume_id": resume_id,
            "message": "Resume processed successfully",
            "text_file": str(text_path),
            "json_file": str(json_path),
            "parsed_data": parsed_data
        }
        
    except Exception as e:
        logger.error(f"Error processing resume: {e}", exc_info=True)
        # Update metrics with error
        background_tasks.add_task(
            evaluation.update_metrics,
            resume_id,
            start_time,
            None,
            str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get parser performance metrics"""
    return evaluation.get_summary()

@app.post("/evaluate/{resume_id}")
async def evaluate_resume(
    resume_id: str,
    manual_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate parser performance against manual annotations"""
    # Save ground truth
    evaluation.save_ground_truth(resume_id, None, manual_data)
    # Run evaluation
    return evaluation.evaluate_against_ground_truth(resume_id)

# PDF Processing Functions
async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using multiple methods"""
    text = ""
    
    try:
        # First try normal PDF text extraction
        async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            await temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name
        
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If we got meaningful text, return it
        if len(text.strip()) > 100:  # Arbitrary threshold
            return text
        
        # If text extraction didn't yield good results, try OCR
        images = convert_from_bytes(pdf_bytes)
        ocr_texts = []
        
        # Process pages in parallel
        with ThreadPoolExecutor() as executor:
            ocr_texts = list(executor.map(
                lambda img: pytesseract.image_to_string(
                    img,
                    config='--psm 1 --oem 3'
                ),
                images
            ))
        
        text = "\n".join(ocr_texts)
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error processing PDF: {str(e)}"
        )
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_pdf_path)
        except:
            pass
    
    return text

async def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OCR with preprocessing"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Basic preprocessing
        # 1. Resize if too large
        max_dimension = 2000
        ratio = min(max_dimension / dim for dim in image.size)
        if ratio < 1:
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        # 2. Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Perform OCR with improved settings
        text = pytesseract.image_to_string(
            image,
            config='--psm 1 --oem 3'
        )
        
        return text
    
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,  # Enable auto-reload for development
        workers=4  # Number of worker processes
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())