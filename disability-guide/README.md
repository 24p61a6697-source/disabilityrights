# ♿ SAHAY — Disability Rights & Accessibility Guide
### Government of India | Ministry of Social Justice & Empowerment

> A full-stack RAG-powered chatbot providing disability rights information, government scheme navigation, assistive technology recommendations, and accessibility guidance.

---

## 🏗️ Technology Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **Language** | Python 3.9 | Backend |
| **Frontend** | React 18 | SPA |
| **RAG Framework** | LangChain-compatible Custom Pipeline | Multi-query + RRF |
| **Embedding Model** | `all-MiniLM-L6-v2` | 384 dimensions, HuggingFace |
| **Vector Database** | **FAISS** (IndexFlatIP) | Cosine similarity |
| **LLM** | **Ollama (llama2)** | Local, privacy-first |
| **PDF Processing** | PyPDF2 / pypdf | |
| **Auth** | JWT (python-jose) + bcrypt | |
| **Database** | SQLite (SQLAlchemy) | Users, chat history |
| **API** | FastAPI + Uvicorn | |

---

## 📁 Project Structure

```
disability-rights-guide/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   └── config.py          # Settings & configuration
│   │   ├── models/
│   │   │   └── database.py        # SQLAlchemy models
│   │   ├── services/
│   │   │   └── auth_service.py    # JWT authentication
│   │   ├── rag/
│   │   │   ├── ingestion.py       # PDF → chunks → FAISS
│   │   │   ├── retrieval.py       # Multi-query RAG + RRF
│   │   │   └── local_vectorstore.py  # Numpy vector store
│   │   ├── data/
│   │   │   └── docs/              # PDF documents (RPWD Act, WHO data)
│   │   └── main.py                # FastAPI application
│   ├── faiss_index/               # Built at startup
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── contexts/
│   │   │   └── AuthContext.js     # Auth + API state management
│   │   ├── i18n/
│   │   │   └── languages.js       # 23 Indian language support
│   │   ├── pages/
│   │   │   ├── LanguageSelect.js  # Step 1: Language selection
│   │   │   ├── AuthPage.js        # Step 2: Login / Register
│   │   │   └── ChatPage.js        # Main chatbot interface
│   │   ├── App.js
│   │   └── index.js
│   ├── public/
│   └── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- [Ollama](https://ollama.ai) (for LLM)

### 1. Install & Start Ollama
```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama2 model
ollama pull llama2

# Start Ollama server
ollama serve
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
SECRET_KEY=your-secret-key-here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
EOF

# Start backend (FAISS index auto-builds on first run)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Fast Start on Windows (No venv activation needed)
```powershell
# From repository root
cd disability-guide/backend
..\\..\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001

# Reload mode for development (watches only backend app folder)
..\\..\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload --reload-dir app
```

### One-Click Launch Scripts (Windows)
```powershell
# Start backend only
powershell -ExecutionPolicy Bypass -File .\start-backend.ps1

# Start backend with reload
powershell -ExecutionPolicy Bypass -File .\start-backend.ps1 -Reload

# Start backend + frontend (opens two terminals)
powershell -ExecutionPolicy Bypass -File .\start-all.ps1

# Validate commands without starting services
powershell -ExecutionPolicy Bypass -File .\start-backend.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File .\start-all.ps1 -DryRun
```

NPM shortcuts from repository root:
```powershell
npm run quick:backend
npm run quick:backend:reload
npm run quick:start
```

### 3. Frontend Setup
```bash
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start
```

### 4. Docker Compose (All-in-one)
```bash
docker-compose up --build
# App: http://localhost:3000
# API: http://localhost:8000/api/docs
```

---

## 🧠 RAG Architecture

```
User Query
    │
    ▼
Query Expansion (Multi-Query)
    │ ─── generates 4 query variations
    ▼
FAISS Vector Search (all-MiniLM-L6-v2, 384-dim)
    │ ─── top-K per query variation
    ▼
Reciprocal Rank Fusion (RRF, k=60)
    │ ─── merges & re-ranks across variations
    ▼
Context Formation
    │ ─── top-5 chunks with source metadata
    ▼
Answer Generation (Ollama llama2 / Extractive fallback)
    │
    ▼
Response with Sources & Citations
```

### Knowledge Base Sources
- RPWD Act 2016 (Act 49 of 2016)
- WHO World Report on Disability (2011, 2022)
- Government Scheme Data (ADIP, UDID, NHFDC, etc.)
- Census of India 2011 Disability Data
- UNCRPD Framework
- Accessibility Standards (IS 9954:2020, WCAG 2.0)
- Disability_WHO_RPWD_Thesis_Reference.pdf (uploaded)

---

## 🌐 Languages Supported (23 Indian Languages)

| Code | Language | Script |
|------|----------|--------|
| en | English | Latin |
| hi | हिंदी Hindi | Devanagari |
| ta | தமிழ் Tamil | Tamil |
| te | తెలుగు Telugu | Telugu |
| bn | বাংলা Bengali | Bengali |
| mr | मराठी Marathi | Devanagari |
| gu | ગુજરાતી Gujarati | Gujarati |
| kn | ಕನ್ನಡ Kannada | Kannada |
| ml | മലയാളം Malayalam | Malayalam |
| pa | ਪੰਜਾਬੀ Punjabi | Gurmukhi |
| or | ଓଡ଼ିଆ Odia | Odia |
| as | অসমীয়া Assamese | Bengali |
| ur | اردو Urdu | Perso-Arabic |
| + 10 more | ... | ... |

---

## ♿ Accessibility Features

- **Screen Reader Compatible**: NVDA, JAWS, VoiceOver
- **ARIA Labels**: All interactive elements labeled
- **Keyboard Navigation**: Full tab/enter/space support
- **Skip Links**: Skip to main content link
- **High Contrast**: Forced-colors media query support
- **Focus Indicators**: Visible focus rings (GOI orange)
- **WCAG 2.0 AA**: Color contrast ≥ 4.5:1
- **Semantic HTML**: Proper roles, landmarks, headings

---

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Login |
| GET | `/api/auth/me` | Current user |
| POST | `/api/chat` | Chat (authenticated) |
| POST | `/api/chat/guest` | Chat (guest) |
| GET | `/api/schemes` | Government schemes list |
| GET | `/api/rights` | Rights under RPWD Act |
| GET | `/api/status` | System health check |

Interactive docs: `http://localhost:8000/api/docs`

---

## 🏛️ Government Portals Referenced

- [DEPwD Portal](https://disabilityaffairs.gov.in)
- [UDID Card](https://www.swavlambancard.gov.in)
- [Chief Commissioner](https://ccdisabilities.nic.in)
- [National Trust](https://thenationaltrust.gov.in)
- [NHFDC](https://nhfdc.nic.in)
- [ALIMCO](https://www.alimco.in)
- [Accessible India](https://accessibleindia.gov.in)

---

## 📦 Dependencies

### Backend (Python 3.9)
```
fastapi==0.104.1          # Web framework
uvicorn                   # ASGI server
sentence-transformers     # all-MiniLM-L6-v2 embeddings
faiss-cpu==1.7.4          # Vector database
pypdf==3.17.4             # PDF processing
python-jose               # JWT authentication
passlib[bcrypt]           # Password hashing
sqlalchemy==2.0.23        # ORM
pydantic-settings         # Config management
```

### Frontend (Node 18)
```
react 18.2                # UI framework
react-router-dom 6.18     # Routing
axios 1.6                 # HTTP client
react-markdown 9          # Markdown rendering
lucide-react              # Icons
```

---

## 🔒 Security

- JWT tokens (7-day expiry)
- bcrypt password hashing
- SQLite with parameterized queries
- CORS configured for frontend origin
- No PII in vector store
- Aadhaar optional, stored encrypted

---

*Built for the Ministry of Social Justice & Empowerment, Government of India*  
*Compliant with RPWD Act 2016 | WCAG 2.0 AA | GIGW Guidelines*
