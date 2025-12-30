# FutureX Finance - Backend

AI-powered financial services platform - Backend API built with FastAPI.

## Tech Stack
- FastAPI
- Python 3.11+
- PyOD (Fraud Detection)
- OpenCV + Tesseract (KYC/OCR)
- scikit-learn

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

## Environment Variables

Create `.env`:

```bash
SECRET_KEY=your-secret-key
CORS_ORIGINS=http://localhost:3000
PORT=8000
```

## API Endpoints

- `/api/v1/kyc/verify` - KYC verification
- `/api/v1/fraud/analyze` - Fraud detection
- `/api/v1/documents/parse` - Document parsing
- `/api/v1/chatbot/chat` - AI chatbot
- `/api/v1/insights/analyze` - Customer insights
- `/api/v1/compliance/alerts` - Compliance monitoring

## Deployment

Deploy to Railway:
```bash
railway up
```

## License
MIT
