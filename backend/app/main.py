try:
    from fastapi import FastAPI, HTTPException, Depends, Body, status  # type: ignore
    from contextlib import asynccontextmanager
except Exception:
    # Minimal stubs used when fastapi is not installed or the editor cannot resolve the import.
    # At runtime ensure fastapi is installed for full functionality.
    from contextlib import asynccontextmanager
    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    def Depends(x=None):
        return x

    Body = lambda *args, **kwargs: None

    class _Status:
        HTTP_201_CREATED = 201
        HTTP_401_UNAUTHORIZED = 401
    status = _Status()

    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass
        def add_middleware(self, *args, **kwargs):
            pass
        def on_event(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
        def get(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
        def post(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
        def put(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
        def delete(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
# fastapi.middleware.cors may not be available in some editor environments; use a fallback stub to avoid import errors
try:
    # Use importlib to attempt a runtime import of the CORS middleware to avoid static analyzer import errors.
    from importlib import import_module
    cors_mod = import_module("fastapi.middleware.cors")
    CORSMiddleware = cors_mod.CORSMiddleware
except Exception:
    # Minimal stub used when fastapi is not installed or the editor cannot resolve the import.
    # This preserves runtime behavior and prevents import errors; at runtime ensure fastapi is installed.
    class CORSMiddleware:
        def __init__(self, app, **kwargs):
            self.app = app
        def __call__(self, scope):
            async def asgi(receive, send):
                return await self.app(scope, receive, send)
            return asgi
# fastapi.responses may not be available in some editor environments; use a fallback stub to avoid import errors
try:
    # Use importlib to attempt a runtime import of JSONResponse to avoid static analyzer import errors.
    from importlib import import_module
    resp_mod = import_module("fastapi.responses")
    JSONResponse = resp_mod.JSONResponse
except Exception:
    # Minimal stub used when fastapi/starlette is not installed or the editor cannot resolve the import.
    # At runtime ensure fastapi is installed for full functionality.
    class JSONResponse:
        def __init__(self, content, status_code: int = 200):
            self.content = content
            self.status_code = status_code
        def __call__(self):
            # When running without FastAPI, returning the raw content is sufficient for editors/tests.
            return self.content
try:
    # Use importlib to attempt a runtime import of pydantic to avoid static analyzer import errors.
    from importlib import import_module
    pyd_mod = import_module("pydantic")
    BaseModel = pyd_mod.BaseModel
    EmailStr = pyd_mod.EmailStr
    Field = pyd_mod.Field
except Exception:
    # Minimal stubs used when pydantic is not installed or the editor cannot resolve the import.
    # At runtime ensure pydantic is installed for full functionality.
    class BaseModel:
        def __init__(self, *args, **kwargs):
            pass
        def dict(self):
            return self.__dict__

    class EmailStr(str):
        pass

    def Field(*args, **kwargs):
        return None
from typing import Optional, List, Dict, Any
from datetime import timedelta
import uuid
import logging
import os
import sys
import time
import threading
import io
import base64
from pathlib import Path

# Make imports work even when running `python main.py` from backend/app
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.models.database import create_tables, get_db, User, ChatMessage, ChatSession, IndexedDocument
from app.services.auth_service import (
    authenticate_user, create_user, create_access_token,
    get_user_by_email, get_user_by_mobile, require_auth, get_current_user
)
try:
    # Attempt a runtime import of SQLAlchemy's Session to avoid static analyzer errors when
    # SQLAlchemy isn't installed in the editor environment.
    from importlib import import_module
    sa_mod = import_module("sqlalchemy.orm")
    Session = sa_mod.Session
except Exception:
    # Lightweight fallback stub used for editing / testing when SQLAlchemy is not available.
    # At runtime ensure SQLAlchemy is installed for full functionality.
    class Session:
        def __init__(self, *args, **kwargs):
            pass
# Suppress noisy library logs globally
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence HuggingFace Hub unauthenticated warning
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ─── FastAPI App Setup ────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app):
    create_tables()
    logger.info("Database tables created")
    if settings.RAG_WARMUP_ON_STARTUP:
        initialize_rag_pipeline(background=True)
    yield

app = FastAPI(
    title="Disability Rights & Accessibility Guide API",
    description="Government of India - Disability Rights Information System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = None
rag_init_in_progress = False
rag_init_error: Optional[str] = None
rag_lock = threading.Lock()

# ─── Pydantic Schemas ─────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=200)
    email: EmailStr
    mobile: str = Field(..., min_length=10, max_length=15)
    password: str = Field(..., min_length=6)
    disability_type: Optional[str] = None
    preferred_language: Optional[str] = "en"
    state: Optional[str] = None
    aadhaar_number: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    language: Optional[str] = "en"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    docs_retrieved: int

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    language: Optional[str] = "en"

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    disability_type: Optional[str] = None
    preferred_language: Optional[str] = None
    state: Optional[str] = None


@app.post("/api/tts")
async def synthesize_tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    lang_code = (req.language or "en").lower().split("-")[0]
    lang_map = {
        "en": "en",
        "hi": "hi",
        "ta": "ta",
        "te": "te",
        "kn": "kn",
        "ml": "ml",
    }
    tts_lang = lang_map.get(lang_code, "en")

    try:
        from gtts import gTTS
    except Exception:
        raise HTTPException(status_code=500, detail="TTS dependency not installed")

    try:
        fp = io.BytesIO()
        gTTS(text=text, lang=tts_lang).write_to_fp(fp)
        audio_base64 = base64.b64encode(fp.getvalue()).decode("ascii")
        return {
            "audio_base64": audio_base64,
            "mime_type": "audio/mpeg",
            "language": tts_lang,
        }
    except Exception as e:
        logger.error("TTS generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate speech audio")

# ─── Auth Routes ──────────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_email(db, req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if get_user_by_mobile(db, req.mobile):
        raise HTTPException(status_code=400, detail="Mobile number already registered")

    user = create_user(
        db, full_name=req.full_name, email=req.email, mobile=req.mobile,
        password=req.password, disability_type=req.disability_type,
        preferred_language=req.preferred_language, state=req.state,
        aadhaar_number=req.aadhaar_number
    )
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id, "full_name": user.full_name, "email": user.email,
            "mobile": user.mobile, "disability_type": user.disability_type,
            "preferred_language": user.preferred_language, "state": user.state
        }
    }

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id, "full_name": user.full_name, "email": user.email,
            "mobile": user.mobile, "disability_type": user.disability_type,
            "preferred_language": user.preferred_language, "state": user.state
        }
    }

@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(require_auth)):
    return {
        "id": current_user.id, "full_name": current_user.full_name,
        "email": current_user.email, "mobile": current_user.mobile,
        "disability_type": current_user.disability_type,
        "preferred_language": current_user.preferred_language,
        "state": current_user.state
    }

@app.put("/api/auth/profile")
async def update_profile(
    update: UserProfileUpdate,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    if update.full_name: current_user.full_name = update.full_name
    if update.disability_type: current_user.disability_type = update.disability_type
    if update.preferred_language: current_user.preferred_language = update.preferred_language
    if update.state: current_user.state = update.state
    db.commit()
    db.refresh(current_user)
    return {"message": "Profile updated", "user": {
        "id": current_user.id, "full_name": current_user.full_name,
        "preferred_language": current_user.preferred_language
    }}

# ─── Chat / RAG Routes ────────────────────────────────────────────────────────────

CHAT_TRANSLATIONS = {
    "service_unavailable": {
        "en": "I apologize, but the AI service is currently not available. Please try again later or contact support for assistance with disability rights information.",
        "hi": "मुझे क्षम करें, लेकिन एआई सेवा वर्तमान में उपलब्ध नहीं है। कृपया बाद में पुनः प्रयास करें या सहायता के लिए संपर्क करें।",
        "ta": "மன்னிக்கவும், இப்போது AI சேவை கிடைக்கவில்லை. தயவுசெய்து பின்னர் முயன்றால் அல்லது உதவிக்காக தொடர்பு கொள்ளவும்.",
        "te": "క్షమించండి, ఇప్పుడే AI సేవ అందుబాటులో లేదు. దయచేసి తరువాత ప్రయత్నించండి లేదా సహాయానికి సంప్రదించండి.",
        "kn": "ಕ್ಷಮಿಸಿ, AI ಸೇವೆ ಈಗ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ನಂತರ ಪ್ರಯತ್ನಿಸಿ ಅಥವಾ ಸಹಾಯಕ್ಕಾಗಿ ಸಂಪರ್ಕಿಸಿ.",
        "ml": "ക്ഷമിക്കണം, ഇപ്പോൾ AI സേവനം ലഭ്യമല്ല. ദയവായി പിന്നീട് ശ്രമിക്കുക അല്ലെങ്കിൽ സഹായത്തിനായി ബന്ധപ്പെടുക.",
        "bn": "দুঃখিত, বর্তমানে AI পরিষেবা উপলব্ধ নয়। দয়া করে পরে চেষ্টা করুন অথবা সাহায্যের জন্য যোগাযোগ করুন।"
    },
    "rights": {
        "en": "In India, disability rights are protected under the Rights of Persons with Disabilities Act, 2016 (RPWD Act). Key rights include equality, access to public spaces, education and employment, healthcare, and social security.",
        "hi": "भारत में, दिव्यांग अधिकार 2016 के अधिकारों के अधिनियम (RPWD एक्ट) के तहत संरक्षित हैं। मुख्य अधिकारों में समानता, सार्वजनिक स्थानों तक पहुंच, शिक्षा और रोजगार, स्वास्थ्य सेवाएं, और सामाजिक सुरक्षा शामिल हैं।",
        "ta": "இந்தியாவில், மாற்றுத் திறனாளர்களின் உரிமைகள் 2016 ஆம் ஆண்டு வெளியான RPWD சட்டத்தின் கீழ் பாதுகாக்கப்படுகின்றன. முக்கிய உரிமைகள் சமத்துவம், பொதுமக்கள் இடங்களுக்கு அணுகுதல், கல்வி மற்றும் வேலைவாய்ப்பு, சுகாதார சேவைகள் மற்றும் சமூக பாதுகாப்பைக் கொண்டுள்ளன.",
        "te": "భారతదేశంలో, వికారుల హక్కులు 2016 యొక్క RPWD చట్టం క్రింద రక్షించబడ్డాయి. ప్రధాన హక్కులలో సమానత్వం, పౌర స్థలాలకు ప్రవేశం, విద్య మరియు ఉద్యోగం, ఆరోగ్య సంరక్షణ, మరియు సామాజిక భద్రత ఉన్నాయి.",
        "kn": "ಭಾರತದಲ್ಲಿ, ಅಂಗವಿಕಲರ ಹಕ್ಕುಗಳು 2016 ರ RPWD ಕಾಯಿದೆ ಅಡಿಯಲ್ಲಿ ರಕ್ಷಿಸಲ್ಪಟ್ಟಿವೆ. ಪ್ರಮುಖ ಹಕ್ಕುಗಳಲ್ಲಿ ಸಮಾನತೆ, ಸಾರ್ವಜನಿಕ ಸ್ಥಳಗಳಿಗೆ ಪ್ರವೇಶ, ಶಿಕ್ಷಣ ಮತ್ತು ಉದ್ಯೋಗ, ಆರೋಗ್ಯ ಸೇವೆಗಳು ಮತ್ತು ಸಾಮಾಜಿಕ ಭದ್ರತೆ ಸೇರಿವೆ.",
        "ml": "ഇന്ത്യയിൽ, വികലാംഗാവകാശങ്ങൾ 2016ലെ RPWD ആക്ടിന്റെ കീഴിൽ സംരക്ഷിതമാണ്. പ്രധാന അവകാശങ്ങളിൽ സമത്വം, പൊതു സ്ഥലങ്ങളിൽ പ്രവേശനം, വിദ്യാഭ്യാസം மற்றும் തൊഴിൽ, ആരോഗ്യപരിപാലനം, സാമൂഹ്യസുരക്ഷ എന്നിവയുണ്ട്.",
        "bn": "ভারতে, প্রতিবন্ধী অধিকারগুলি 2016 সালের RPWD আইন অনুযায়ী সুরক্ষিত। প্রধান অধিকারগুলির মধ্যে সমতা, জনসাধারণের স্থানে প্রবেশাধিকার, শিক্ষা ও কর্মসংস্থান, স্বাস্থ্যসেবা, এবং সামাজিক সুরক্ষা রয়েছে।"
    },
    "schemes": {
        "en": "Major disability welfare schemes in India include Accessible India Campaign, DDRS, National Fellowship for Persons with Disabilities, assistive device support, scholarships, and pension schemes.",
        "hi": "भारत में प्रमुख विकलांग कल्याण योजनाओं में सुगम भारत अभियान, DDRS, दिव्यांगों के लिए राष्ट्रीय छात्रवृत्ति, सहायक उपकरण सहायता, छात्रवृत्ति और पेंशन योजनाएं शामिल हैं।",
        "ta": "இந்தியாவில் முக்கிய மாற்றுத்திறனாளி நலப் திட்டங்களில் சுகம்யா பாரத், DDRS, தேசிய அடைவை மற்றும் உதவி சாதன உதவி, கல்வி விசா மற்றும் ஓய்வூதியத்திட்டங்கள் அடங்கும்.",
        "te": "భారతదేశంలో ప్రధాన వికలాంగ సంక్షేమ పథకాల్లో సుగమ్య భారత్ ప్రచారం, DDRS, నేషనల్ ఫెలోషిప్, సహాయక పరికరాల సాయం, స్కాలర్షిప్‌లు, పెన్షన్ పథకాలు ఉన్నాయి.",
        "kn": "ಭಾರತದಲ್ಲಿ ಪ್ರಮುಖ ಅಂಗವಿಕಲ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳಲ್ಲಿ ಸುಗಮ ಭಾರತ ಅಭಿಯಾನ, DDRS, ರಾಷ್ಟ್ರೀಯ ಫೆಲೋಶಿಪ್, ಸಹಾಯಕ ಸಾಧನಗಳ ಸಹಾಯ, ವಿದ್ಯಾರ್ಥಿವೇತನ ಮತ್ತು ಪಿಂಚಣಿ ಯೋಜನೆಗಳು ಸೇರಿವೆ.",
        "ml": "ഇന്ത്യയിലെ പ്രധാന വൈകല്യമുള്ളവരുടെ ക്ഷേമ പദ്ധതികളിൽ സുഖമാ് ഇന്ത്യ, DDRS, ദേശീയ ഫെലോഷിപ്പ്, സഹായ ഉപകരണ സഹായം, വിദ്യാർത്ഥി ബഹുമതി, പെൻഷൻ പദ്ധതികൾ എന്നിവയുണ്ട്.",
        "bn": "ভারতে প্রধান প্রতিবন্ধী কল্যাণ প্রকল্পগুলির মধ্যে রয়েছে সুলভ ভারত অভিযাত্রা, DDRS, জাতীয় ফেলোশিপ, সহায়ক যন্ত্রপাতি সহায়তা, বৃত্তি এবং পেনশন প্রকল্প।"
    },
    "assistive": {
        "en": "Assistive technology includes tools like screen readers (NVDA/JAWS), hearing aids, wheelchairs, prosthetics, Braille kits, communication devices, and mobility aids. In India, support is available through ADIP Scheme, ALIMCO, and national institutes. Eligibility and device type depend on disability assessment and income criteria.",
        "hi": "सहायक तकनीक में स्क्रीन रीडर (NVDA/JAWS), हियरिंग एड, व्हीलचेयर, कृत्रिम अंग, ब्रेल किट, संचार उपकरण और चलने-फिरने की सहायक सामग्री शामिल हैं। भारत में ADIP योजना, ALIMCO और राष्ट्रीय संस्थानों के माध्यम से सहायता उपलब्ध है। पात्रता और उपकरण का प्रकार विकलांगता आकलन और आय मानदंड पर निर्भर करता है।",
        "ta": "உதவி தொழில்நுட்பத்தில் ஸ்கிரீன் ரீடர்கள் (NVDA/JAWS), கேட்கும் கருவிகள், சக்கர நாற்காலிகள், செயற்கை உறுப்புகள், பிரெயில் கருவிகள், தொடர்பு சாதனங்கள் மற்றும் இயக்க உதவிகள் அடங்கும். இந்தியாவில் ADIP திட்டம், ALIMCO மற்றும் தேசிய நிறுவனங்கள் மூலம் உதவி கிடைக்கிறது. தகுதி மற்றும் கருவி வகை மாற்றுத்திறன் மதிப்பீடு மற்றும் வருமான நிபந்தனைகளைப் பொறுத்தது.",
        "te": "సహాయక సాంకేతికతలో స్క్రీన్ రీడర్లు (NVDA/JAWS), హియరింగ్ ఎయిడ్స్, వీల్‌చెయర్లు, కృత్రిమ అవయవాలు, బ్రెయిల్ కిట్లు, కమ్యూనికేషన్ పరికరాలు, మొబిలిటీ సహాయక పరికరాలు ఉంటాయి. భారతదేశంలో ADIP పథకం, ALIMCO మరియు జాతీయ సంస్థల ద్వారా సహాయం లభిస్తుంది. అర్హత మరియు పరికరం రకం వికలాంగత అంచనా మరియు ఆదాయ ప్రమాణాలపై ఆధారపడతాయి.",
        "kn": "ಸಹಾಯಕ ತಂತ್ರಜ್ಞಾನದಲ್ಲಿ ಸ್ಕ್ರೀನ್ ರೀಡರ್‌ಗಳು (NVDA/JAWS), ಹೇರಿಂಗ್ ಏಡ್ಸ್, ವೀಲ್ಚೇರ್‌ಗಳು, ಕೃತಕ ಅಂಗಗಳು, ಬ್ರೈಲ್ ಕಿಟ್‌ಗಳು, ಸಂವಹನ ಸಾಧನಗಳು ಮತ್ತು ಚಲನೆ ಸಹಾಯಕ ಸಾಧನಗಳು ಸೇರಿವೆ. ಭಾರತದಲ್ಲಿ ADIP ಯೋಜನೆ, ALIMCO ಮತ್ತು ರಾಷ್ಟ್ರೀಯ ಸಂಸ್ಥೆಗಳ ಮೂಲಕ ಸಹಾಯ ಲಭ್ಯವಿದೆ. ಅರ್ಹತೆ ಮತ್ತು ಸಾಧನದ ವಿಧವು ಅಂಗವೈಕಲ್ಯ ಮೌಲ್ಯಮಾಪನ ಹಾಗೂ ಆದಾಯ ಮಾನದಂಡಗಳ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿದೆ.",
        "ml": "സഹായ സാങ്കേതികവിദ്യയിൽ സ്ക്രീൻ റീഡറുകൾ (NVDA/JAWS), കേൾവി ഉപകരണങ്ങൾ, വീൽചെയറുകൾ, കൃത്രിമ അവയവങ്ങൾ, ബ്രെയിൽ കിറ്റുകൾ, ആശയവിനിമയ ഉപകരണങ്ങൾ, ചലന സഹായങ്ങൾ എന്നിവ ഉൾപ്പെടുന്നു. ഇന്ത്യയിൽ ADIP പദ്ധതി, ALIMCO, ദേശീയ സ്ഥാപനങ്ങൾ എന്നിവ വഴി സഹായം ലഭ്യമാണ്. യോഗ്യതയും ഉപകരണ തരംവും വൈകല്യ നിർണയവും വരുമാന മാനദണ്ഡവും ആശ്രയിച്ചിരിക്കും.",
        "bn": "সহায়ক প্রযুক্তির মধ্যে স্ক্রিন রিডার (NVDA/JAWS), হিয়ারিং এইড, হুইলচেয়ার, কৃত্রিম অঙ্গ, ব্রেইল কিট, যোগাযোগ ডিভাইস এবং চলাচল সহায়ক অন্তর্ভুক্ত। ভারতে ADIP স্কিম, ALIMCO এবং জাতীয় প্রতিষ্ঠানের মাধ্যমে সহায়তা পাওয়া যায়। যোগ্যতা ও ডিভাইসের ধরন প্রতিবন্ধিতা মূল্যায়ন এবং আয়ের মানদণ্ডের উপর নির্ভর করে।"
    },
    "job_reservation": {
        "en": "Under Section 34 of the RPWD Act, 2016, at least 4% reservation is provided in government establishments for persons with benchmark disabilities. This includes identified posts across disability categories. You can apply through regular recruitment notifications and claim applicable relaxation/support under RPWD rules.",
        "hi": "RPWD अधिनियम, 2016 की धारा 34 के तहत सरकारी संस्थानों में बेंचमार्क दिव्यांग व्यक्तियों के लिए कम से कम 4% आरक्षण प्रदान किया गया है। यह विभिन्न दिव्यांगता श्रेणियों के लिए चिन्हित पदों पर लागू होता है। आप सामान्य भर्ती विज्ञापनों के माध्यम से आवेदन कर सकते हैं और RPWD नियमों के अनुसार छूट/सुविधाएं प्राप्त कर सकते हैं।",
        "ta": "RPWD சட்டம், 2016 இன் பிரிவு 34ன் கீழ் அரசு அமைப்புகளில் benchmark மாற்றுத்திறனாளிகளுக்கு குறைந்தபட்சம் 4% இடஒதுக்கீடு வழங்கப்பட்டுள்ளது. இது மாற்றுத்திறனின் பல பிரிவுகளுக்கான அடையாளப்படுத்தப்பட்ட பணியிடங்களில் பொருந்தும். சாதாரண ஆட்சேர்ப்பு அறிவிப்புகள் மூலம் விண்ணப்பிக்கலாம்; RPWD விதிகளின்படி தளர்வுகள்/உதவிகளை பெறலாம்.",
        "te": "RPWD చట్టం, 2016 లోని సెక్షన్ 34 ప్రకారం ప్రభుత్వ సంస్థల్లో benchmark వికలాంగుల కోసం కనీసం 4% రిజర్వేషన్ ఉంది. ఇది వివిధ వికలాంగత వర్గాలకు గుర్తించిన పోస్టులకు వర్తిస్తుంది. సాధారణ నియామక ప్రకటనల ద్వారా దరఖాస్తు చేయవచ్చు మరియు RPWD నిబంధనల ప్రకారం వర్తించే సడలింపులు/సహాయం పొందవచ్చు.",
        "kn": "RPWD ಕಾಯಿದೆ, 2016 ರ ಕಲಂ 34 ಅನ್ವಯ, ಸರ್ಕಾರದ ಸಂಸ್ಥೆಗಳಲ್ಲಿ benchmark ಅಂಗವಿಕಲ ವ್ಯಕ್ತಿಗಳಿಗೆ ಕನಿಷ್ಠ 4% ಮೀಸಲಾತಿ ನೀಡಲಾಗಿದೆ. ಇದು ವಿವಿಧ ಅಂಗವೈಕಲ್ಯ ವರ್ಗಗಳಿಗೆ ಗುರುತಿಸಲಾದ ಹುದ್ದೆಗಳಿಗೆ ಅನ್ವಯಿಸುತ್ತದೆ. ಸಾಮಾನ್ಯ ನೇಮಕಾತಿ ಅಧಿಸೂಚನೆಗಳ ಮೂಲಕ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ, RPWD ನಿಯಮಗಳಡಿ ದೊರೆಯುವ ಸಡಿಲಿಕೆ/ಸಹಾಯವನ್ನು ಪಡೆಯಬಹುದು.",
        "ml": "RPWD നിയമം, 2016 ലെ വകുപ്പ് 34 പ്രകാരം സർക്കാർ സ്ഥാപനങ്ങളിൽ benchmark വൈകല്യമുള്ള വ്യക്തികൾക്ക് കുറഞ്ഞത് 4% സംവരണം നൽകിയിട്ടുണ്ട്. ഇത് വിവിധ വൈകല്യ വിഭാഗങ്ങൾക്കുള്ള തിരിച്ചറിഞ്ഞ ഒഴിവുകൾക്ക് ബാധകമാണ്. സാധാരണ നിയമന വിജ്ഞാപനങ്ങൾ വഴി അപേക്ഷിക്കാം; RPWD ചട്ടങ്ങൾ പ്രകാരം ലഭ്യമായ ഇളവുകൾ/സഹായങ്ങൾ ആവശ്യപ്പെടാം.",
        "bn": "RPWD আইন, 2016 এর ধারা 34 অনুযায়ী সরকারি প্রতিষ্ঠানে benchmark প্রতিবন্ধী ব্যক্তিদের জন্য ন্যূনতম 4% সংরক্ষণ রয়েছে। এটি বিভিন্ন প্রতিবন্ধিতা শ্রেণির জন্য নির্ধারিত পদে প্রযোজ্য। নিয়মিত নিয়োগ বিজ্ঞপ্তির মাধ্যমে আবেদন করতে পারেন এবং RPWD নিয়ম অনুযায়ী প্রযোজ্য ছাড়/সহায়তা দাবি করতে পারেন।"
    },
    "sensory": {
        "en": "Sensory disabilities include impairments in vision, hearing, touch, taste, or smell. In India these are covered under the RPWD Act and may qualify for access benefits, assistive devices, education accommodations, and disability certification.",
        "hi": "द्रवण विकलांगताओं में दृष्टि, श्रवण, स्पर्श, स्‍वाद या गंध में अक्षमता शामिल हैं। भारत में इन्हें RPWD अधिनियम के अंतर्गत कवर किया जाता है और ये पहुंच सुविधाएँ, सहायक उपकरण, शिक्षा में समायोजन और विकलांगता प्रमाणन के लिए पात्र हो सकते हैं।",
        "ta": "உணர்வுத் திறன் குறைகளில் பார்க்கும், கேட்கும், தொடுதல், சுவை அல்லது மணத்தை உணருவதில் புலபின்மை அடங்கும். இந்தியாவில் இதற்கு RPWD சட்டத்தின் கீழ் பாதுகாப்பு உள்ளது மற்றும் அணுகல் நலன்கள், உதவி சாதனங்கள், கல்வி ஒத்துழைப்பு மற்றும் மாற்றுத்திறனுக் சான்றிதழுக்கு தகுதி பெற்றிருக்கலாம்.",
        "te": "సెన్సరీ వికలాంగతలు దృష్టి, వినికిడి, అనుభూతి, రుచిఅనే వాసనలో లోపాలను కలిగి ఉంటాయి. భారతదేశంలో ఇవి RPWD చట్టం క్రింద కవర్ చేయబడతాయి మరియు ప్రాప్తి ప్రయోజనాలు, సహాయక పరికరాలు, విద్యా సౌకర్యాలు మరియు వికలాంగత సర్టిఫికేషన్ కోసం అర్హత పొందవచ్చు.",
        "kn": "ಸಂವೇದಿ ಅಂಗವಿಕಲತೆಗಳಲ್ಲಿ ದೃಷ್ಟಿ, ಕೇಳುವಿಕೆ, ಸ್ಪರ್ಶ, ರುಚಿ ಅಥವಾ ವಾಸನೆಯಲ್ಲಿರುವ ಅಶಕ್ತಿ ಸೇರಿವೆ. ಭಾರತದಲ್ಲಿ ಇವುಗಳನ್ನು RPWD ಕಾಯಿದೆಯಡಿ ಒಳಗೊಂಡಿರುತ್ತವೆ ಮತ್ತು ಪ್ರವೇಶ ಪ್ರಯೋಜನಗಳು, ಸಹಾಯಕ ಸಾಧನಗಳು, ಶಿಕ್ಷಣ ಸೌಲಭ್ಯಗಳು ಮತ್ತು ಅಂಗವಿಕಲತೆ ಪ್ರಮಾಣೀಕರಣಕ್ಕಾಗಿ ಅರ್ಹರಾಗಬಹುದು.",
        "ml": "സെൻസറി വൈകല്യങ്ങളിൽ ദൃഷ്ടി, കേൾവ്, സ്പർശം, രുചി അല്ലെങ്കിൽ ഗന്ധത്തിൽ തകരാറുകൾ ഉൾപ്പെടുന്നു. ഭാരതത്തിൽ ഇത് RPWD ആക്ടിന്റെ കീഴിൽ ഉൾപ്പെടുന്നു, പ്രാപ്യതാ ആനുകൂല്യങ്ങൾ, സഹായ ഉപകരണങ്ങൾ, വിദ്യാഭ്യാസ സൗകര്യങ്ങൾ, വൈകല്യമാന്യപത്രക്കാർിക്ക് തർഹത എന്നിവക്ക് അർഹത നൽകാം.",
        "bn": "সেন্সরি প্রতিবন্ধীতায় দৃষ্টি, শ্রবণ, স্পর্শ, স্বাদ বা গন্ধে সমস্যা অন্তর্ভুক্ত থাকে। ভারতে এগুলি RPWD অ্যাক্টের আওতায় আসে এবং অ্যাক্সেস সুবিধা, সহায়ক ডিভাইস, শিক্ষাগত সুযোগ এবং প্রতিবন্ধী সনদ পেতে যোগ্য হতে পারে।"
    },
    "general": {
        "en": "I understand you're asking about '{question}'. For comprehensive disability rights information in India, please visit the Department of Empowerment of Persons with Disabilities website or contact your local disability welfare office.",
        "hi": "मैं समझता हूँ कि आप '{question}' के बारे में पूछ रहे हैं। भारत में व्यापक दिव्यांग अधिकार जानकारी के लिए कृपया दिव्यांगजन सशक्तिकरण विभाग की वेबसाइट देखें या अपने स्थानीय विकलांगता कल्याण कार्यालय से संपर्क करें।",
        "ta": "நான் நீங்கள் '{question}' பற்றி கேட்கிறீர்கள் என்று புரிந்துகொள்கிறேன். இந்தியாவில் விரிவான மாற்றுத்திறனாளி உரிமைகள் தகவலுக்கு, மாற்றுத்திறனாளி நலத்துறையின் இணையதளத்தை பார்வையிடவும் அல்லது உங்கள் உள்ளூர் நல அலுவலரிடம் தொடர்பு கொள்க.",
        "te": "నేను మీరు '{question}' గురించి అడుగుతున్నారని అర్థం చేసుకుంటున్నాను. భారతదేశంలో సమగ్ర వికలాంగ హక్కుల సమాచారం కోసం, వికలాంగుల సాధికారత విభాగం వెబ్‌సైట్‌ను సందర్శించండి లేదా మీ స్థానిక సంక్షేమ కార్యాలయాన్ని సంప్రదించండి.",
        "kn": "ನಾನು ನೀವು '{question}' ಬಗ್ಗೆ ಕೇಳುತ್ತಿರುವಿರಿ ಎಂದು ನಾನು ಅರ್ಥಮಾಡಿಕೊಂಡಿದ್ದೇನೆ. ಭಾರತದಲ್ಲಿ ಸಂಪೂರ್ಣ ಅಂಗವಿಕಲರ ಹಕ್ಕುಗಳ ಮಾಹಿತಿಗಾಗಿ, ಅಂಗವಿಕಲರ ಸಶಕ್ತೀಕರಣ ಇಲಾಖೆಯ ವೆಬ್‌ಸೈಟ್ ಅಥವಾ ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕಲ್ಯಾಣ ಕಚೇರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "ml": "ഞാൻ നിങ്ങൾ '{question}' എന്നതിനെക്കുറിച്ച് ചോദിക്കുന്നുണ്ടെന്ന് ഞാൻ മനസിലാക്കുന്നു. ഇന്ത്യയിലെ സമഗ്രമായ വൈകല്യാവകാശ വിവരങ്ങൾക്ക്, ദയവായി പ്രാദേശിക ക്ഷേമ ഓഫിസിനോ ദിവ്യാംഗസഹായ വകുപ്പ് വെബ്സൈറ്റിനോ സമീപിക്കുക.",
        "bn": "আমি বুঝতে পারছি আপনি '{question}' সম্পর্কে জানতে চাইছেন। ভারতের ব্যাপক প্রতিবন্ধী অধিকার তথ্যের জন্য, অনুগ্রহ করে প্রতিবন্ধী ক্ষমতায়ন দফতের ওয়েবসাইট দেখুন বা আপনার স্থানীয় কল্যাণ অফিসে যোগাযোগ করুন।"
    }
}

def translate_chat_answer(key: str, language: str = "en", **kwargs) -> str:
    lang = (language or "en").lower()
    if lang not in CHAT_TRANSLATIONS.get(key, {}):
        lang = "en"
    template = CHAT_TRANSLATIONS.get(key, {}).get(lang, CHAT_TRANSLATIONS.get(key, {}).get("en", ""))
    return template.format(**kwargs)


def _initialize_rag_pipeline_sync():
    global rag_pipeline, rag_init_in_progress, rag_init_error
    started_at = time.perf_counter()
    try:
        from app.rag.retrieval import DisabilityRAGPipeline
        from app.rag.ingestion import run_ingestion

        index_path = settings.FAISS_INDEX_PATH
        meta_file = Path(index_path) / "metadata.json"
        if not meta_file.exists():
            logger.info("Building FAISS index from documents...")
            docs_dir = Path(__file__).parent / "data" / "docs"
            run_ingestion(docs_dir=str(docs_dir), index_path=index_path)

        pipeline = DisabilityRAGPipeline(
            index_path=index_path,
            ollama_url=settings.OLLAMA_BASE_URL,
            ollama_model=settings.OLLAMA_MODEL
        )
        pipeline.initialize()
        rag_pipeline = pipeline
        rag_init_error = None
        logger.info("RAG pipeline initialized successfully in %.2fs", time.perf_counter() - started_at)
    except Exception as e:
        rag_pipeline = None
        rag_init_error = str(e)
        logger.error(f"RAG initialization error: {e}")
    finally:
        rag_init_in_progress = False


def initialize_rag_pipeline(background: bool = False):
    global rag_pipeline, rag_init_in_progress
    with rag_lock:
        if rag_pipeline is not None or rag_init_in_progress:
            return
        rag_init_in_progress = True

    if background:
        thread = threading.Thread(target=_initialize_rag_pipeline_sync, daemon=True)
        thread.start()
        logger.info("RAG warmup started in background")
        return

    _initialize_rag_pipeline_sync()


_pdf_chunks_cache = []


def load_pdf_chunks(docs_dir: str = None):
    global _pdf_chunks_cache
    if _pdf_chunks_cache:
        return _pdf_chunks_cache

    try:
        from app.rag.ingestion import DisabilityRAGIngestion
        docs_path = Path(docs_dir or Path(__file__).parent / "data" / "docs")
        if not docs_path.exists():
            return []

        loader = DisabilityRAGIngestion(index_path=settings.FAISS_INDEX_PATH)
        for pdf_file in docs_path.glob("*.pdf"):
            pdf_chunks = loader.load_pdf_document(str(pdf_file))
            _pdf_chunks_cache.extend(pdf_chunks)
    except Exception as e:
        logger.warning(f"Failed to load PDF docs for local search: {e}")
    return _pdf_chunks_cache


def search_pdf_docs(question: str, top_n: int = 3):
    try:
        import re
        chunks = load_pdf_chunks()
        query_tokens = set(re.findall(r"\w+", (question or "").lower()))
        if not query_tokens or not chunks:
            return []

        scored = []
        for chunk in chunks:
            text_content = chunk.get("text", "").lower()
            chunk_tokens = set(re.findall(r"\w+", text_content))
            score = len(query_tokens & chunk_tokens)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_n]]
    except Exception as e:
        logger.warning(f"PDF document search failed: {e}")
        return []


def local_doc_search(question: str, top_n: int = 3):
    try:
        import re
        from app.rag.ingestion import DISABILITY_KNOWLEDGE_BASE

        query_tokens = set(re.findall(r"\w+", (question or "").lower()))
        if not query_tokens:
            return []

        scored = []
        for doc in DISABILITY_KNOWLEDGE_BASE:
            text_content = f"{doc.get('text', '')} {doc.get('source', '')} {doc.get('chapter', '')}".lower()
            doc_tokens = set(re.findall(r"\w+", text_content))
            score = len(query_tokens & doc_tokens)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]
    except Exception as e:
        logger.warning(f"Local knowledge base lookup failed: {e}")
        return []


def detect_intent(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["job", "employment", "reservation", "quota", "recruitment"]):
        return "job_reservation"
    if any(k in q for k in ["assistive", "technology", "screen reader", "wheelchair", "hearing aid", "prosthetic", "braille"]):
        return "assistive"
    if any(k in q for k in ["sensory", "vision", "hearing", "blind", "deaf"]):
        return "sensory"
    if any(k in q for k in ["scheme", "benefit", "pension", "udid", "adip"]):
        return "schemes"
    if any(k in q for k in ["rights", "rpwd", "discrimination", "legal"]):
        return "rights"
    return "general"


def local_doc_search_semantic(question: str, top_n: int = 3):
    """Intent-aware lexical semantic fallback over built-in knowledge base."""
    try:
        import re
        from app.rag.ingestion import DISABILITY_KNOWLEDGE_BASE

        q = (question or "").lower()
        intent = detect_intent(q)

        query_variants = [q]
        if intent == "job_reservation":
            query_variants.extend([
                "section 34 reservation benchmark disability",
                "government jobs reservation persons with benchmark disabilities",
                "employment rights rpwd act"
            ])
        elif intent == "assistive":
            query_variants.extend([
                "assistive technology adip scheme alimco",
                "screen reader hearing aid wheelchair braille",
                "rehabilitation devices disability support"
            ])
        elif intent == "rights":
            query_variants.extend(["rpwd rights section", "non discrimination disability rights"])
        elif intent == "schemes":
            query_variants.extend(["government disability schemes india", "udid adip pension benefits"])

        query_tokens = set()
        for variant in query_variants:
            query_tokens.update(re.findall(r"\w+", variant))

        category_boost = {
            "job_reservation": {"employment": 5, "legal": 3, "rpwd_act": 2},
            "assistive": {"assistive_tech": 6, "schemes": 3, "resources": 2},
            "rights": {"rights": 5, "rpwd_act": 3, "legal": 3},
            "schemes": {"schemes": 5, "benefits": 3, "resources": 2},
            "sensory": {"disability_definitions": 5, "benefits": 2},
            "general": {},
        }

        scored = []
        for doc in DISABILITY_KNOWLEDGE_BASE:
            text_content = f"{doc.get('text', '')} {doc.get('source', '')} {doc.get('chapter', '')}".lower()
            doc_tokens = set(re.findall(r"\w+", text_content))
            overlap = len(query_tokens & doc_tokens)
            if overlap <= 0:
                continue

            cat = doc.get("category", "general")
            boost = category_boost.get(intent, {}).get(cat, 0)

            phrase_bonus = 0
            if intent == "job_reservation" and ("section 34" in text_content or "reservation" in text_content):
                phrase_bonus = 4
            if intent == "assistive" and ("assistive technology" in text_content or "adip" in text_content or "alimco" in text_content):
                phrase_bonus = 4

            score = overlap + boost + phrase_bonus
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]
    except Exception as e:
        logger.warning(f"Semantic local lookup failed: {e}")
        return []


def format_local_sources(docs):
    sources = []
    seen = set()
    for doc in docs:
        source = doc.get("metadata", {}).get("source") if isinstance(doc.get("metadata"), dict) else doc.get("source", "Reference")
        chapter = doc.get("metadata", {}).get("chapter", "") if isinstance(doc.get("metadata"), dict) else doc.get("chapter", "")
        category = doc.get("metadata", {}).get("category", "general") if isinstance(doc.get("metadata"), dict) else doc.get("category", "general")
        key = f"{source}:{chapter}"
        if key not in seen:
            sources.append({"source": source, "chapter": chapter, "category": category})
            seen.add(key)
    return sources


def fallback_chat_answer(question: str, language: str = "en"):
    def localize_answer(text: str) -> str:
        if not text or not language or language.lower() == "en":
            return text
        try:
            from app.rag.translation import translate_text_with_google
            translated = translate_text_with_google(text, language)
            if translated:
                return translated
        except Exception as e:
            logger.warning(f"Fallback answer localization failed: {e}")
        return text

    normalized_question = question
    if settings.TRANSLATION_ENABLED and language and language.lower() != "en":
        try:
            from app.rag.translation import translate_text_to_english
            translated_q = translate_text_to_english(question, language)
            if translated_q:
                normalized_question = translated_q
        except Exception as e:
            logger.debug(f"Fallback question normalization skipped: {e}")

    # Fast path: built-in knowledge base is in-memory and much faster than PDF parsing.
    local_docs = local_doc_search_semantic(normalized_question, top_n=3)
    if not local_docs:
        local_docs = local_doc_search(normalized_question, top_n=3)
        
    if local_docs:
        snippets = [doc["text"].strip() for doc in local_docs if doc.get("text")]
        answer = "\n\n".join(snippets[:2])
        if answer:
            return localize_answer(answer), format_local_sources(local_docs), len(local_docs)

    if settings.ENABLE_PDF_FALLBACK:
        pdf_docs = search_pdf_docs(normalized_question, top_n=3)
        if pdf_docs:
            snippets = [doc.get("text", "").strip() for doc in pdf_docs if doc.get("text")]
            answer = "\n\n".join(snippets[:2])
            if answer:
                return localize_answer(answer), format_local_sources(pdf_docs), len(pdf_docs)

    intent = detect_intent(normalized_question)
    if intent == "job_reservation":
        answer = translate_chat_answer("job_reservation", language)
    elif intent == "assistive":
        answer = translate_chat_answer("assistive", language)
    elif intent == "sensory":
        answer = translate_chat_answer("sensory", language)
    elif intent == "rights":
        answer = translate_chat_answer("rights", language)
    elif intent == "schemes":
        answer = translate_chat_answer("schemes", language)
    else:
        answer = translate_chat_answer("general", language, question=question[:50])
    return answer, [], 0


def query_rag_or_fallback(question: str, session_id: str, language: str = "en"):
    if rag_pipeline is None:
        initialize_rag_pipeline(background=True)

    if rag_pipeline is not None:
        try:
            response = rag_pipeline.query(
                question=question,
                session_id=session_id,
                language=language
            )
            return (
                response.get("answer", ""),
                response.get("sources", []),
                response.get("docs_retrieved", 0)
            )
        except Exception as e:
            logger.error(f"RAG query failed: {e}")

    logger.warning("Falling back to local chat response without RAG.")
    return fallback_chat_answer(question, language)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    session_id = req.session_id or str(uuid.uuid4())
    language = req.language or current_user.preferred_language or "en"
    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()

    try:
        answer, sources, docs_retrieved = query_rag_or_fallback(
            req.question,
            session_id,
            language
        )

        # Save chat message to database
        chat_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not chat_session:
            chat_session = ChatSession(
                session_id=session_id,
                user_id=current_user.id,
                language=language
            )
            db.add(chat_session)
            db.commit()

        # Save user message
        user_message = ChatMessage(
            session_id=session_id,
            user_id=current_user.id,
            role="user",
            content=req.question,
            language=language
        )
        db.add(user_message)

        # Save assistant message
        assistant_message = ChatMessage(
            session_id=session_id,
            user_id=current_user.id,
            role="assistant",
            content=answer,
            sources=str(sources) if sources else None,
            language=language
        )
        db.add(assistant_message)
        db.commit()

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "chat_completed request_id=%s user_id=%s session_id=%s lang=%s docs=%s elapsed_ms=%s",
            request_id,
            current_user.id,
            session_id,
            language,
            docs_retrieved,
            elapsed_ms,
        )

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "docs_retrieved": docs_retrieved
        }

    except Exception as e:
        logger.exception("Chat error request_id=%s session_id=%s", request_id, session_id)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to process your request. Please try again.",
                "request_id": request_id,
                "error": str(e),
            },
        )

@app.post("/api/chat/guest", response_model=ChatResponse)
async def chat_guest(req: ChatRequest, db: Session = Depends(get_db)):
    session_id = req.session_id or str(uuid.uuid4())
    language = req.language or "en"
    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()

    try:
        answer, sources, docs_retrieved = query_rag_or_fallback(
            req.question,
            session_id,
            language
        )

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "guest_chat_completed request_id=%s session_id=%s lang=%s docs=%s elapsed_ms=%s",
            request_id,
            session_id,
            language,
            docs_retrieved,
            elapsed_ms,
        )

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "docs_retrieved": docs_retrieved
        }

    except Exception as e:
        logger.exception("Guest chat error request_id=%s session_id=%s", request_id, session_id)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to process your request. Please try again.",
                "request_id": request_id,
                "error": str(e),
            },
        )


@app.get("/api/chat/history/{session_id}")
async def get_history(
    session_id: str,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db)
):
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id,
        ChatMessage.user_id == current_user.id
    ).order_by(ChatMessage.created_at).all()
    return [
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in messages
    ]

# ─── Info Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/schemes")
async def get_schemes():
    schemes = [
        {"id": "adip", "name": "ADIP Scheme", "desc": "Free assistive devices for PwDs", "ministry": "DEPwD", "url": "https://disabilityaffairs.gov.in"},
        {"id": "udid", "name": "UDID Card", "desc": "Unique Disability Identity Card", "ministry": "DEPwD", "url": "https://www.swavlambancard.gov.in"},
        {"id": "nhfdc", "name": "NHFDC Loans", "desc": "Loans for self-employment", "ministry": "DEPwD", "url": "https://nhfdc.nic.in"},
        {"id": "igndps", "name": "Disability Pension", "desc": "Monthly pension for BPL PwDs", "ministry": "Rural Dev.", "url": "https://nsap.nic.in"},
        {"id": "national_trust", "name": "National Trust", "desc": "Autism/CP/ID/Multiple Disability", "ministry": "MSJE", "url": "https://thenationaltrust.gov.in"},
        {"id": "accessible_india", "name": "Sugamya Bharat", "desc": "Accessible India Campaign", "ministry": "DEPwD", "url": "https://accessibleindia.gov.in"},
        {"id": "sipda", "name": "SIPDA", "desc": "State implementation of PwD Act", "ministry": "DEPwD", "url": "https://disabilityaffairs.gov.in"},
        {"id": "ddrs", "name": "DDRS", "desc": "NGO rehabilitation funding", "ministry": "DEPwD", "url": "https://disabilityaffairs.gov.in"},
    ]
    return {"schemes": schemes}

@app.get("/api/rights")
async def get_rights():
    rights = [
        {"title": "Right to Non-Discrimination", "section": "Section 3, RPWD Act 2016", "desc": "No discrimination on grounds of disability in any sphere of life."},
        {"title": "Employment Reservation", "section": "Section 34, RPWD Act 2016", "desc": "Minimum 4% reservation in all government establishments."},
        {"title": "Education Reservation", "section": "Section 32, RPWD Act 2016", "desc": "5% seats reserved in higher educational institutions."},
        {"title": "Exam Accommodations", "section": "Section 17, RPWD Act 2016", "desc": "Extra time, scribe, and assistive devices in examinations."},
        {"title": "Right to Legal Capacity", "section": "Section 14, RPWD Act 2016", "desc": "Full legal capacity equal to others in all aspects of life."},
        {"title": "Accessible Infrastructure", "section": "Sections 40-46, RPWD Act 2016", "desc": "All public buildings, transport, and websites must be accessible."},
        {"title": "Healthcare Rights", "section": "Section 25, RPWD Act 2016", "desc": "Free health care in government hospitals, access to medicines."},
        {"title": "Reproductive Rights", "section": "Section 10, RPWD Act 2016", "desc": "Right to decide family size and retain fertility."},
    ]
    return {"rights": rights}
    
@app.get("/api/assistive")
async def get_assistive():
    assistive = [
        {"id": "hearing_aids", "name": "Digital Hearing Aids", "desc": "Programmable digital hearing aids for hearing impairment.", "category": "Hearing"},
        {"id": "smart_cane", "name": "Smart Cane", "desc": "Electronic travel aid for person with visual impairment.", "category": "Vision"},
        {"id": "wheelchair_electric", "name": "Motorized Wheelchair", "desc": "Battery operated wheelchairs for locomotor disability.", "category": "Mobility"},
        {"id": "braille_display", "name": "Refreshable Braille Display", "desc": "Display for reading digital text in braille.", "category": "Vision"},
        {"id": "speech_software", "name": "Screen Reading Software", "desc": "Software like JAWS/NVDA for computer access.", "category": "Digital"},
    ]
    return {"assistive": assistive}

@app.get("/api/accessibility")
async def get_accessibility():
    accessibility = [
        {"title": "GIGW Guidelines", "desc": "Guidelines for Indian Government Websites for digital accessibility."},
        {"title": "Harmonized Guidelines 2021", "desc": "Standards for barrier-free environment in buildings and public spaces."},
        {"title": "AIS 153 Standards", "desc": "Automotive Industry Standards for accessible public transport."},
        {"title": "Sugamya Pustakalaya", "desc": "Online library for people with print disabilities."},
    ]
    return {"accessibility": accessibility}

@app.get("/api/employment")
async def get_employment():
    employment = [
        {"title": "Government Job Reservation", "desc": "4% reservation in government establishments as per RPWD Act Section 34."},
        {"title": "NHFDC Skill Training", "desc": "Skill development programs for PwDs through National Handicapped Finance and Development Corporation."},
        {"title": "Private Sector Incentives", "desc": "Government incentives for employers in the private sector for employing PwDs."},
        {"title": "Special Employment Exchanges", "desc": "Dedicated cells to assist PwDs in finding suitable job opportunities."},
    ]
    return {"employment": employment}

@app.get("/api/education")
async def get_education():
    education = [
        {"title": "Inclusive Education (Samagra Shiksha)", "desc": "Support for education of children with disabilities in regular schools."},
        {"title": "Higher Education Reservation", "desc": "5% reservation in all government and government-aided higher education institutions."},
        {"title": "Scholarship for Students with Disabilities", "desc": "Pre-matric, Post-matric, and Top Class Education scholarships."},
        {"title": "National Fellowship", "desc": "Financial assistance for pursuing M.Phil and Ph.D. degrees."},
    ]
    return {"education": education}

@app.get("/api/status")
async def status_check():
    return {
        "status": "online",
        "rag_ready": rag_pipeline is not None,
        "rag_initializing": rag_init_in_progress,
        "rag_init_error": rag_init_error,
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }

@app.get("/")
async def root():
    return {"message": "Disability Rights & Accessibility Guide API", "version": "1.0.0"}

if __name__ == "__main__":
    try:
        import uvicorn
        reload_enabled = os.environ.get("UVICORN_RELOAD", "0") == "1"
        port = int(os.environ.get("PORT", "8001"))
        uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=reload_enabled)
    except ImportError:
        raise RuntimeError("uvicorn is required to run the backend server. Install it with `pip install uvicorn`.")
