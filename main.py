"""
MediExtract FastAPI Backend with Google Cloud Vision API and OpenRouter Embeddings
Medical Report Processing, RAG System, and Doctor Consultation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import os
import json
import tempfile
import shutil
from datetime import datetime
import logging
from dotenv import load_dotenv
import base64
import numpy as np

load_dotenv()

# Google Cloud Vision - Using REST API
import requests as http_requests

# Core imports
import qdrant_client
from groq import Groq
from qdrant_client.models import Distance, VectorParams, PointStruct

# LlamaIndex imports
from llama_index.core.schema import Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core.embeddings import BaseEmbedding

# Web scraping
import requests
from bs4 import BeautifulSoup
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION
# ================================

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    COLLECTION_NAME = "medical_reports_db"
    UPLOAD_DIR = "temp_uploads"
    EMBEDDING_MODEL = "baai/bge-large-en-v1.5"
    EMBEDDING_DIMENSION = 1024  # BGE-large dimension
    
    NORMAL_RANGES = {
        'hemoglobin': {'specialty': 'Hematologist'},
        'glucose': {'specialty': 'Endocrinologist'},
        'cholesterol': {'specialty': 'Cardiologist'},
        'tsh': {'specialty': 'Endocrinologist'},
        'creatinine': {'specialty': 'Nephrologist'},
        'wbc': {'specialty': 'Hematologist'},
        'platelet': {'specialty': 'Hematologist'},
        'alt': {'specialty': 'Hepatologist'},
        'ast': {'specialty': 'Hepatologist'},
    }
    
    @classmethod
    def validate(cls):
        missing = []
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not cls.QDRANT_URL:
            missing.append("QDRANT_URL")
        if not cls.QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")
        if not cls.GOOGLE_VISION_API_KEY:
            missing.append("GOOGLE_VISION_API_KEY")
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

# ================================
# OPENROUTER EMBEDDING CLASS
# ================================

class OpenRouterEmbedding(BaseEmbedding):
    """Custom embedding class using OpenRouter API"""
    
    # Configure Pydantic to allow extra fields
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "baai/bge-large-en-v1.5",
        **kwargs
    ):
        # Call parent init first
        super().__init__(**kwargs)
        
        # Now we can safely set our custom attributes
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "input": text
            }
            
            response = http_requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                return embedding
            else:
                raise Exception("No embedding returned from API")
                
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text"""
        return self._get_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for document text"""
        return self._get_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
            time.sleep(0.1)  # Rate limiting
        return embeddings
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of get_query_embedding"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of get_text_embedding"""
        return self._get_text_embedding(text)

# ================================
# FASTAPI APP
# ================================

app = FastAPI(
    title="MediExtract API with Google Vision & OpenRouter",
    description="Medical Report Processing with Google Cloud Vision API and OpenRouter Embeddings",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# PYDANTIC MODELS
# ================================

class QueryRequest(BaseModel):
    query: str
    patient_name: Optional[str] = None

class DoctorSearchRequest(BaseModel):
    city: str
    state: str
    specialty: str

class DoctorInfo(BaseModel):
    name: str
    specialty: str
    hospital: Optional[str] = None
    experience: Optional[str] = None
    rating: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    profile_url: Optional[str] = None

class DoctorSearchResponse(BaseModel):
    success: bool
    doctors: List[DoctorInfo]
    message: Optional[str] = None

class AbnormalTest(BaseModel):
    testName: str
    value: str
    normalRange: str
    specialty: str

class QueryResponse(BaseModel):
    response: str = ""
    success: bool = True
    is_comparison: bool = False
    table_data: Optional[Dict[str, Any]] = None
    abnormal_tests: Optional[List[AbnormalTest]] = None
    patient_name: Optional[str] = None

class ProcessingResult(BaseModel):
    success: bool
    image_filename: str
    extracted_text: Optional[str] = None
    structured_json: Optional[dict] = None
    error: Optional[str] = None

class DatabaseStatus(BaseModel):
    exists: bool
    count: Optional[int] = None

# ================================
# DOCTOR FINDER
# ================================

class DoctorFinder:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_doctors(self, city: str, state: str, specialty: str) -> List[Dict[str, Any]]:
        doctors = []
        
        try:
            doctors = self._search_practo(city, state, specialty)
        except Exception as e:
            logger.error(f"Practo search failed: {e}")
        
        if not doctors:
            doctors = self._generate_doctor_profiles(city, state, specialty)
        
        return doctors[:5]
    
    def _search_practo(self, city: str, state: str, specialty: str) -> List[Dict]:
        doctors = []
        try:
            city_slug = city.lower().replace(' ', '-')
            specialty_slug = specialty.lower().replace(' ', '-')
            url = f"https://www.practo.com/{city_slug}/{specialty_slug}"
            
            time.sleep(random.uniform(1, 2))
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                doctor_cards = soup.find_all('div', class_='info-section')[:5]
                
                for card in doctor_cards:
                    try:
                        name_elem = card.find('h2', class_='doctor-name')
                        if name_elem:
                            name = name_elem.get_text(strip=True)
                            profile_link = name_elem.find_parent('a')
                            profile_url = f"https://www.practo.com{profile_link.get('href')}" if profile_link else url
                            
                            hospital_elem = card.find('span', class_='doctor-location')
                            hospital = hospital_elem.get_text(strip=True) if hospital_elem else f'{city}, {state}'
                            
                            rating_elem = card.find('span', class_='star-rating')
                            rating = rating_elem.get_text(strip=True) if rating_elem else '4.5/5'
                            
                            exp_elem = card.find('div', class_='exp-text')
                            experience = exp_elem.get_text(strip=True) if exp_elem else '10+ years'
                            
                            doctors.append({
                                'name': name,
                                'specialty': specialty,
                                'hospital': hospital,
                                'rating': rating,
                                'experience': experience,
                                'profile_url': profile_url,
                                'phone': None,
                                'email': None
                            })
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Practo error: {e}")
        
        return doctors
    
    def _generate_doctor_profiles(self, city: str, state: str, specialty: str) -> List[Dict]:
        doctors = []
        try:
            groq_client = Groq(api_key=Config.GROQ_API_KEY)
            
            prompt = f"""Generate 5 realistic doctor profiles for {specialty} in {city}, {state}, India.
Return as JSON array:
[
  {{
    "name": "Dr. [Full Name]",
    "hospital": "[Hospital Name], {city}",
    "experience": "[Number]+ years",
    "rating": "[4.0-5.0]/5"
  }}
]"""

            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a medical directory assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=512,
            )
            
            result_text = response.choices[0].message.content.strip()
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            
            generated = json.loads(result_text.strip())
            
            for doc in generated:
                doctors.append({
                    'name': doc.get('name', 'Dr. Unknown'),
                    'specialty': specialty,
                    'hospital': doc.get('hospital', f'{city}, {state}'),
                    'experience': doc.get('experience', '10+ years'),
                    'rating': doc.get('rating', '4.5/5'),
                    'phone': None,
                    'email': None,
                    'profile_url': f"https://www.practo.com/{city.lower()}/{specialty.lower()}"
                })
        except Exception as e:
            logger.error(f"Generation error: {e}")
        
        return doctors

# ================================
# MEDICAL OCR WITH GOOGLE VISION (REST API)
# ================================

class MedicalReportOCR:
    def __init__(self):
        self.api_key = Config.GOOGLE_VISION_API_KEY
        self.vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        self.groq_client = None
        self._init_components()
    
    def _init_components(self):
        try:
            self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
            logger.info("Components initialized with Google Vision API key")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Google Vision REST API with API key"""
        try:
            # Read and encode image to base64
            with open(image_path, 'rb') as image_file:
                image_content = image_file.read()
            
            encoded_image = base64.b64encode(image_content).decode('utf-8')
            
            # Prepare the request payload
            payload = {
                "requests": [
                    {
                        "image": {
                            "content": encoded_image
                        },
                        "features": [
                            {
                                "type": "TEXT_DETECTION"
                            }
                        ]
                    }
                ]
            }
            
            # Make the API request
            response = http_requests.post(
                self.vision_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                raise Exception(f"Vision API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Check for errors in response
            if 'responses' in result and len(result['responses']) > 0:
                response_data = result['responses'][0]
                
                if 'error' in response_data:
                    raise Exception(f"Vision API error: {response_data['error']}")
                
                if 'textAnnotations' in response_data and len(response_data['textAnnotations']) > 0:
                    full_text = response_data['textAnnotations'][0]['description']
                    logger.info(f"Extracted {len(full_text)} characters")
                    return full_text
                else:
                    logger.warning("No text found in image")
                    return ""
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def generate_json_with_groq(self, extracted_text: str, image_filename: str):
        if not extracted_text or len(extracted_text.strip()) < 10:
            return {'success': False, 'error': 'Insufficient text extracted'}
        
        max_length = 4000
        if len(extracted_text) > max_length:
            extracted_text = extracted_text[:max_length]
        
        prompt = f"""Extract medical report information from this text and format as JSON:

TEXT: {extracted_text}

Return JSON with these fields (use null if not found):
{{
  "hospital_info": {{
    "hospital_name": "string or null",
    "address": "string or null"
  }},
  "patient_info": {{
    "name": "string or null",
    "age": "string or null",
    "gender": "string or null"
  }},
  "doctor_info": {{
    "referring_doctor": "string or null"
  }},
  "report_info": {{
    "report_type": "string or null",
    "report_date": "string or null"
  }},
  "test_results": [
    {{
      "test_name": "string",
      "result_value": "string",
      "reference_range": "string or null",
      "unit": "string or null"
    }}
  ]
}}

Return only valid JSON."""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Extract medical data and return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=1024,
            )
            
            json_text = response.choices[0].message.content.strip()
            
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0]
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0]
            
            json_text = json_text.strip()
            
            try:
                parsed_json = json.loads(json_text)
            except json.JSONDecodeError:
                parsed_json = {
                    "hospital_info": {"hospital_name": None, "address": None},
                    "patient_info": {"name": None, "age": None, "gender": None},
                    "doctor_info": {"referring_doctor": None},
                    "report_info": {"report_type": "Medical Report", "report_date": None},
                    "test_results": []
                }
            
            parsed_json['_metadata'] = {
                'source_image': image_filename,
                'extraction_method': 'google_vision_rest_api',
                'processing_timestamp': datetime.now().isoformat(),
                'model_used': 'llama-3.1-8b-instant'
            }
            
            return {'success': True, 'json_data': parsed_json}
            
        except Exception as e:
            logger.error(f"Groq processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_image(self, image_path: str):
        image_filename = os.path.basename(image_path)
        
        try:
            extracted_text = self.extract_text(image_path)
            
            if not extracted_text.strip():
                return {
                    'success': False,
                    'error': 'No text found in image',
                    'image_filename': image_filename
                }
            
            groq_result = self.generate_json_with_groq(extracted_text, image_filename)
            
            if groq_result['success']:
                return {
                    'success': True,
                    'image_filename': image_filename,
                    'extracted_text': extracted_text,
                    'structured_json': groq_result['json_data']
                }
            else:
                return {
                    'success': False,
                    'error': groq_result['error'],
                    'image_filename': image_filename,
                    'extracted_text': extracted_text[:500]
                }
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_filename': image_filename
            }

# ================================
# RAG SYSTEM WITH OPENROUTER EMBEDDINGS
# ================================

class RAGSystem:
    def __init__(self):
        self.client = None
        self.query_engine = None
        self.embed_model = None
        self.llm = None
        self.groq_client = None
        self._init_components()
    
    def _init_components(self):
        try:
            self.client = qdrant_client.QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY
            )
            
            # Initialize OpenRouter embeddings
            self.embed_model = OpenRouterEmbedding(
                api_key=Config.OPENROUTER_API_KEY,
                model_name=Config.EMBEDDING_MODEL
            )
            
            self.llm = GroqLLM(
                model="llama-3.3-70b-versatile",
                api_key=Config.GROQ_API_KEY,
                temperature=0.1,
                max_tokens=1024
            )
            
            self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
            
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            
            logger.info("RAG system initialized with OpenRouter embeddings")
            
        except Exception as e:
            logger.error(f"RAG initialization error: {e}")
            raise
    
    def detect_abnormal_values(self, context: str) -> List[Dict]:
        abnormal_tests = []
        
        try:
            prompt = f"""Analyze this medical data and identify abnormal test results:

{context}

Return as JSON array:
[
  {{
    "test_name": "Test Name",
    "value": "patient value",
    "normal_range": "normal range",
    "specialty": "recommended specialist"
  }}
]

Only abnormal values. If none, return []."""

            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a medical analyst. Identify abnormal results."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=512,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            
            try:
                abnormal_data = json.loads(result_text.strip())
                
                for item in abnormal_data:
                    test_name_lower = item.get('test_name', '').lower()
                    specialty = item.get('specialty', 'General Physician')
                    
                    for known_test, info in Config.NORMAL_RANGES.items():
                        if known_test in test_name_lower:
                            specialty = info['specialty']
                            break
                    
                    abnormal_tests.append({
                        'testName': item.get('test_name', ''),
                        'value': item.get('value', ''),
                        'normalRange': item.get('normal_range', ''),
                        'specialty': specialty
                    })
            
            except json.JSONDecodeError:
                pass
        
        except Exception as e:
            logger.error(f"Abnormal detection error: {e}")
        
        return abnormal_tests
    
    def create_documents_from_reports(self, processed_reports: List[dict]):
        documents = []
        
        for report in processed_reports:
            if not report.get('success'):
                continue
            
            try:
                json_data = report['structured_json']
                text_parts = []
                
                hospital_info = json_data.get('hospital_info', {})
                if hospital_info.get('hospital_name'):
                    text_parts.append(f"Hospital: {hospital_info['hospital_name']}")
                
                patient_info = json_data.get('patient_info', {})
                if patient_info.get('name'):
                    text_parts.append(f"Patient: {patient_info['name']}")
                if patient_info.get('age'):
                    text_parts.append(f"Age: {patient_info['age']}")
                if patient_info.get('gender'):
                    text_parts.append(f"Gender: {patient_info['gender']}")
                
                report_info = json_data.get('report_info', {})
                if report_info.get('report_type'):
                    text_parts.append(f"Report Type: {report_info['report_type']}")
                if report_info.get('report_date'):
                    text_parts.append(f"Report Date: {report_info['report_date']}")
                
                test_results = json_data.get('test_results', [])
                for test in test_results:
                    if isinstance(test, dict) and test.get('test_name'):
                        test_text = f"Test: {test['test_name']}"
                        if test.get('result_value'):
                            test_text += f" Result: {test['result_value']}"
                        if test.get('reference_range'):
                            test_text += f" Reference: {test['reference_range']}"
                        text_parts.append(test_text)
                
                if 'extracted_text' in report:
                    text_parts.append(f"Original Text: {report['extracted_text']}")
                
                text_content = "\n".join(text_parts)
                
                document = Document(
                    text=text_content,
                    metadata={
                        'source_image': report['image_filename'],
                        'patient_name': patient_info.get('name', 'Unknown'),
                        'hospital_name': hospital_info.get('hospital_name', 'Unknown'),
                        'report_type': report_info.get('report_type', 'Medical Report'),
                        'report_date': report_info.get('report_date', 'Unknown')
                    }
                )
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Document creation error: {e}")
                continue
        
        return documents
    
    def setup_database(self, processed_reports: List[dict]):
        try:
            documents = self.create_documents_from_reports(processed_reports)
            
            if not documents:
                return False, "No valid documents"
            
            try:
                self.client.delete_collection(Config.COLLECTION_NAME)
            except:
                pass
            
            # Create collection with proper vector configuration
            self.client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=Config.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=Config.COLLECTION_NAME
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Index documents with OpenRouter embeddings
            logger.info(f"Indexing {len(documents)} documents with OpenRouter embeddings...")
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=False
            )
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            self._init_query_engine()
            
            return True, f"Successfully indexed {len(documents)} reports"
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            return False, str(e)
    
    def _init_query_engine(self):
        try:
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=Config.COLLECTION_NAME
            )
            
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )
            
            template = """Context from medical reports:
---------------------
{context_str}
---------------------

Answer questions about the medical reports based on the context above.

Instructions:
1. Be specific and cite actual values
2. Include test name, value, unit, reference range
3. If information is not available, state clearly
4. Use bullet points for multiple results

Question: {query_str}

Answer:"""
            
            qa_prompt = PromptTemplate(template)
            
            # Query engine without reranking
            self.query_engine = index.as_query_engine(
                llm=self.llm,
                similarity_top_k=10
            )
            self.query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
            
        except Exception as e:
            logger.error(f"Query engine error: {e}")
            raise
    
    def query(self, query_text: str, patient_name: Optional[str] = None):
        try:
            if self.query_engine is None:
                self._init_query_engine()
            
            enhanced_query = f"For patient {patient_name}: {query_text}" if patient_name else query_text
            response = self.query_engine.query(enhanced_query)
            
            return str(response), patient_name
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise
    
    def generate_comparison_table(self, query_text: str, patient_name: Optional[str] = None):
        try:
            context, detected_patient = self.query(query_text, patient_name)
            
            prompt = f"""Create a comparison table in markdown format:

Medical Data:
{context}

Query: {query_text}

Format:
| Test Parameter | Report 1 (Date) | Report 2 (Date) |
| --- | --- | --- |
| Test Name | Value1 | Value2 |"""

            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Create clean comparison tables."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1024,
            )
            
            table_text = response.choices[0].message.content.strip()
            table_data = self._parse_table(table_text)
            abnormal_tests = self.detect_abnormal_values(context)
            
            return {
                'success': True,
                'response': '',
                'table_data': table_data,
                'is_comparison': True,
                'abnormal_tests': abnormal_tests,
                'patient_name': detected_patient
            }
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return {
                'success': False,
                'response': str(e),
                'table_data': None,
                'is_comparison': False
            }
    
    def _parse_table(self, text: str):
        try:
            lines = [line.strip() for line in text.split('\n') if '|' in line]
            
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                rows = []
                
                for line in lines[1:]:
                    if all(c in '-|: ' for c in line):
                        continue
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if len(cells) == len(headers):
                        rows.append(cells)
                
                if rows:
                    return {'headers': headers, 'rows': rows}
        except Exception as e:
            logger.error(f"Table parsing error: {e}")
        
        return None
    
    def get_database_status(self):
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if Config.COLLECTION_NAME in collection_names:
                collection_info = self.client.get_collection(Config.COLLECTION_NAME)
                return {'exists': True, 'count': collection_info.points_count}
            else:
                return {'exists': False, 'count': 0}
                
        except Exception as e:
            logger.error(f"Database status error: {e}")
            return {'exists': False, 'count': 0}

# ================================
# GLOBAL INSTANCES
# ================================

ocr_processor = None
rag_system = None
doctor_finder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_processor, rag_system, doctor_finder
    
    try:
        Config.validate()
        ocr_processor = MedicalReportOCR()
        rag_system = RAGSystem()
        doctor_finder = DoctorFinder()
        logger.info("All components initialized with OpenRouter embeddings")
        yield
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

app.router.lifespan_context = lifespan

# ================================
# API ENDPOINTS
# ================================

@app.get("/")
async def root():
    return {
        "message": "MediExtract API with Google Vision & OpenRouter",
        "version": "4.0.0",
        "status": "running",
        "embedding_provider": "OpenRouter (baai/bge-large-en-v1.5)"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ocr_ready": ocr_processor is not None,
        "rag_ready": rag_system is not None,
        "embedding_model": Config.EMBEDDING_MODEL
    }

@app.get("/api/database/status", response_model=DatabaseStatus)
async def get_database_status():
    try:
        return rag_system.get_database_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-reports")
async def process_reports(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_reports = []
    
    for file in files:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=Config.UPLOAD_DIR) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
            
            result = ocr_processor.process_image(temp_path)
            result['original_filename'] = file.filename
            processed_reports.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            processed_reports.append({
                'success': False,
                'error': str(e),
                'image_filename': file.filename
            })
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    successful_reports = [r for r in processed_reports if r.get('success')]
    
    if successful_reports:
        rag_system.setup_database(processed_reports)
    
    return {
        "success": len(successful_reports) > 0,
        "total_count": len(processed_reports),
        "successful_count": len(successful_reports),
        "failed_count": len(processed_reports) - len(successful_reports),
        "results": processed_reports
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_reports(request: QueryRequest):
    try:
        db_status = rag_system.get_database_status()
        if not db_status['exists']:
            raise HTTPException(status_code=400, detail="No data available. Upload reports first.")
        
        comparison_keywords = ['compare', 'comparison', 'tabular', 'table', 'versus', 'vs']
        is_comparison = any(kw in request.query.lower() for kw in comparison_keywords)
        
        if is_comparison:
            result = rag_system.generate_comparison_table(request.query, request.patient_name)
            return QueryResponse(**result)
        else:
            response, patient = rag_system.query(request.query, request.patient_name)
            abnormal_tests = rag_system.detect_abnormal_values(response)
            
            return QueryResponse(
                response=response,
                success=True,
                abnormal_tests=abnormal_tests if abnormal_tests else None,
                patient_name=patient
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/find-doctors", response_model=DoctorSearchResponse)
async def find_doctors(request: DoctorSearchRequest):
    try:
        if not all([request.city, request.state, request.specialty]):
            raise HTTPException(status_code=400, detail="City, state, and specialty required")
        
        doctors = doctor_finder.search_doctors(request.city, request.state, request.specialty)
        
        if not doctors:
            return DoctorSearchResponse(
                success=False,
                doctors=[],
                message=f"No doctors found for {request.specialty} in {request.city}"
            )
        
        doctor_list = [DoctorInfo(**doc) for doc in doctors]
        
        return DoctorSearchResponse(
            success=True,
            doctors=doctor_list,
            message=f"Found {len(doctor_list)} specialists"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
