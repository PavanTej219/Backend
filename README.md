# MedAnalyze Backend

A FastAPI-based medical report processing system with OCR, RAG (Retrieval-Augmented Generation), and doctor consultation features.

## Features

- 🔍 **Medical Report OCR**: Extract text from medical report images using Google Cloud Vision API
- 🧠 **AI-Powered Analysis**: Process and structure medical data using Google Gemini API
- 📊 **RAG System**: Query and compare medical reports using vector embeddings (OpenRouter)
- 👨‍⚕️ **Doctor Finder**: Search for specialists based on abnormal test results
- 📈 **Comparison Tables**: Generate side-by-side comparisons of test results
- 🎯 **Abnormal Detection**: Automatically identify out-of-range test values

## Tech Stack

- **FastAPI**: Modern Python web framework
- **Google Cloud Vision API**: OCR for medical reports
- **Google Gemini API**: LLM for medical data extraction and analysis
- **OpenRouter**: Embedding generation (baai/bge-large-en-v1.5)
- **Qdrant**: Vector database for semantic search
- **LlamaIndex**: RAG framework
- **BeautifulSoup**: Web scraping for doctor information

## Prerequisites

- Python 3.8+
- API Keys:
  - Google Cloud Vision API key
  - Google Gemini API key
  - OpenRouter API key
  - Qdrant Cloud account (URL + API key)

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd mediextract-backend
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
# Google APIs
GOOGLE_VISION_API_KEY=your_google_vision_api_key
GEMINI_API_KEY=your_gemini_api_key

# OpenRouter (for embeddings)
OPENROUTER_API_KEY=your_openrouter_api_key

# Qdrant Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

## Getting API Keys

### Google Cloud Vision API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable the **Cloud Vision API**
4. Create credentials → API Key
5. Copy the API key to `.env`

### Google Gemini API

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key to `.env`

**Free Tier**: 60 requests per minute

### OpenRouter API

1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up/Login
3. Go to Keys section
4. Create a new API key
5. Add credits ($5 minimum recommended)

### Qdrant Cloud

1. Visit [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a free account
3. Create a new cluster (1GB free tier available)
4. Copy the cluster URL and API key

## Running the Server

### Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, visit:

- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### Health Check

```http
GET /api/health
```

Returns server status and component readiness.

### Process Medical Reports

```http
POST /api/process-reports
Content-Type: multipart/form-data

files: [File, File, ...]
```

Upload one or more medical report images for OCR and processing.

**Response:**
```json
{
  "success": true,
  "total_count": 2,
  "successful_count": 2,
  "failed_count": 0,
  "results": [
    {
      "success": true,
      "image_filename": "report.jpg",
      "extracted_text": "...",
      "structured_json": {
        "patient_info": {...},
        "test_results": [...]
      }
    }
  ]
}
```

### Query Reports

```http
POST /api/query
Content-Type: application/json

{
  "query": "What is my hemoglobin level?",
  "patient_name": "John Doe"
}
```

**Response:**
```json
{
  "response": "Your hemoglobin level is 12.5 g/dL...",
  "success": true,
  "is_comparison": false,
  "abnormal_tests": [
    {
      "testName": "Hemoglobin",
      "value": "12.5 g/dL",
      "normalRange": "13.5-17.5 g/dL",
      "specialty": "Hematologist"
    }
  ],
  "patient_name": "John Doe"
}
```

### Compare Reports

```http
POST /api/query
Content-Type: application/json

{
  "query": "Compare my blood sugar levels",
  "patient_name": "John Doe"
}
```

Returns a comparison table with historical data.

### Find Doctors

```http
POST /api/find-doctors
Content-Type: application/json

{
  "city": "Mumbai",
  "state": "Maharashtra",
  "specialty": "Cardiologist"
}
```

**Response:**
```json
{
  "success": true,
  "doctors": [
    {
      "name": "Dr. Rajesh Kumar",
      "specialty": "Cardiologist",
      "hospital": "Apollo Hospital, Mumbai",
      "experience": "15+ years",
      "rating": "4.8/5",
      "profile_url": "https://www.practo.com/..."
    }
  ],
  "message": "Found 5 specialists"
}
```

### Database Status

```http
GET /api/database/status
```

Check if reports are indexed in the vector database.

## Project Structure

```
mediextract-backend/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── .env.example          # Example environment file
├── temp_uploads/         # Temporary file storage
└── README.md            # This file
```

## Configuration

Edit the `Config` class in `main.py` to customize:

```python
class Config:
    COLLECTION_NAME = "medical_reports_db"
    UPLOAD_DIR = "temp_uploads"
    EMBEDDING_MODEL = "baai/bge-large-en-v1.5"
    EMBEDDING_DIMENSION = 1024
    
    NORMAL_RANGES = {
        'hemoglobin': {'specialty': 'Hematologist'},
        'glucose': {'specialty': 'Endocrinologist'},
        # Add more test ranges...
    }
```

## Usage Examples

### Using cURL

```bash
# Upload reports
curl -X POST "http://localhost:8000/api/process-reports" \
  -F "files=@report1.jpg" \
  -F "files=@report2.jpg"

# Query reports
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my cholesterol levels?"}'

# Find doctors
curl -X POST "http://localhost:8000/api/find-doctors" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Delhi",
    "state": "Delhi",
    "specialty": "Cardiologist"
  }'
```

### Using Python Requests

```python
import requests

# Process reports
files = [
    ('files', open('report1.jpg', 'rb')),
    ('files', open('report2.jpg', 'rb'))
]
response = requests.post(
    'http://localhost:8000/api/process-reports',
    files=files
)
print(response.json())

# Query
response = requests.post(
    'http://localhost:8000/api/query',
    json={
        'query': 'Show my test results',
        'patient_name': 'John Doe'
    }
)
print(response.json())
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (missing parameters, no data)
- `500`: Internal Server Error

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Logging

Logs are output to console with INFO level by default. To change:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Rate Limiting**: OpenRouter and Gemini have rate limits
- **Concurrent Requests**: Use `--workers` flag for production
- **File Size**: Large images may take longer to process
- **Embeddings**: First query after upload may be slower

## Troubleshooting

### "Missing required environment variables"
- Check `.env` file exists
- Verify all API keys are set
- Restart the server after adding keys

### "No text found in image"
- Ensure image quality is good
- Check image orientation
- Try different image formats (JPG, PNG)

### "OpenRouter API error"
- Verify API key is correct
- Check account has credits
- Ensure rate limits not exceeded

### "Qdrant connection failed"
- Verify Qdrant URL is correct
- Check API key is valid
- Ensure cluster is running

## Dependencies

Main packages (see `requirements.txt` for full list):

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
google-cloud-vision==3.4.4
google-generativeai==0.3.2
llama-index==0.9.14
qdrant-client==1.7.0
beautifulsoup4==4.12.2
requests==2.31.0
python-dotenv==1.0.0
```

## Security Notes

- Never commit `.env` file to version control
- Use environment variables for all sensitive data
- Implement authentication for production use
- Validate and sanitize all file uploads
- Use HTTPS in production

## Future Enhancements

- [ ] Add user authentication
- [ ] Implement caching layer
- [ ] Add support for PDF reports
- [ ] Multi-language support
- [ ] Report generation (PDF/CSV export)
- [ ] Real-time notifications
- [ ] Advanced analytics dashboard

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Open an issue on GitHub
- Email: support@mediextract.com

## Acknowledgments

- Google Cloud Vision API for OCR
- Google Gemini for AI capabilities
- OpenRouter for embeddings
- Qdrant for vector storage
- LlamaIndex for RAG framework

---

**Made with ❤️ for better healthcare**
