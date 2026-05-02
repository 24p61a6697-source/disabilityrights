"""
RAG Retrieval Pipeline
- Multi-query retrieval (generates query variations)
- Reciprocal Rank Fusion (RRF) for result merging
- FAISS vector search with sentence-transformers
- History-aware answer generation
"""

import json
import logging
import os
import math
import re
import sys
import warnings

# ─── Suppress Unwanted Logs & Warnings ───────────────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_DISABLE"] = "true"
os.environ["DISABLE_TQDM"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, module="tqdm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from app.core.config import settings

# Import translation utilities
from app.rag.translation import (
    translate_text_with_google, 
    translate_text_to_english,
    LANGUAGE_NAMES, 
    language_code_to_google_code
)

# Silence specialized library logs and prevent unnecessary Hub checks
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
_ollama_available = True
_ollama_fallback_logged = False

GROK_MODEL_FALLBACKS = [
    "grok",
    "grok-1.0",
    "grok-1.0-mini",
    "grok-1.0-large",
    "grok-2.0",
    "grok-2.0-mini",
    "grok-2.0-large"
]


def _is_groq_key(api_key: str) -> bool:
    return isinstance(api_key, str) and api_key.startswith("gsk_")


def _get_groq_api_base(api_key: str) -> str:
    if _is_groq_key(api_key):
        return "https://api.groq.com/openai/v1"
    return "https://api.x.ai/v1"


def _get_grok_model_candidates(model: Optional[str] = None, api_key: str = None) -> List[str]:
    candidates = []
    if model:
        candidates.append(model)

    if _is_groq_key(api_key):
        # Groq uses OpenAI-compatible model names and a single configured model.
        if not candidates:
            candidates.append("openai/gpt-oss-20b")
        return candidates

    if settings.GROK_MODEL and settings.GROK_MODEL not in candidates:
        candidates.append(settings.GROK_MODEL)
    for fallback in GROK_MODEL_FALLBACKS:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ─── Category-based query expansion ─────────────────────────────────────────────

QUERY_EXPANSIONS = {
    "rights": ["RPWD Act 2016 rights", "legal entitlements persons with disabilities", "Section 3 RPWD Act"],
    "scheme": ["government schemes for disability India", "ADIP scheme eligibility", "DDRS benefits"],
    "job": ["Section 34 RPWD Act reservation", "4% reservation government jobs", "employment for benchmark disability"],
    "education": ["Section 32 education reservation", "5% seats higher education", "right to free education disability"],
    "blindness": ["RPWD Act blindness", "definition of blindness", "visual impairment blindness"],
    "vision": ["blindness and low vision", "visual impairment RPWD Act", "vision disability definition"],
    "sensory": ["sensory disability blind vision", "definitions of sensory impairment", "visual impairment disability"],
    "technology": ["assistive devices ADIP scheme", "screen readers hearing aids wheelchairs", "ALIMCO assistive technology"],
    "certificate": ["UDID card application process", "how to get disability certificate", "benchmark disability 40%"],
    "pension": ["Indira Gandhi National Disability Pension Scheme", "disability pension eligibility India", "state pension for PwD"],
    "accessibility": ["Sugamya Bharat Campaign", "accessible infrastructure standards", "web accessibility GIGW"],
    "health": ["Section 25 healthcare rights", "free health care for PwD", "disability rehabilitation services"],
    "complaint": ["file complaint Chief Commissioner for Persons with Disabilities", "Grievance redressal RPWD Act", "State Commissioner PwD"],
}

def expand_query(query: str) -> List[str]:
    """Generate query variations for multi-query retrieval"""
    q_lower = query.lower()
    variations = [query]
    for keyword, expansions in QUERY_EXPANSIONS.items():
        if keyword in q_lower:
            variations.extend(expansions[:2])
            break
    if len(variations) == 1:
        variations.append(f"India disability {query}")
        variations.append(f"RPWD Act {query}")
    return list(dict.fromkeys(variations))[:4]


def reciprocal_rank_fusion(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion to merge multiple ranked result lists"""
    rrf_scores = defaultdict(float)
    doc_store = {}
    for ranked_list in ranked_lists:
        for position, doc in enumerate(ranked_list, 1):
            doc_id = doc["id"]
            rrf_scores[doc_id] += 1.0 / (k + position)
            doc_store[doc_id] = doc
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [
        {**doc_store[did], "rrf_score": rrf_scores[did]}
        for did in sorted_ids
    ]


class DisabilityRAGRetriever:
    def __init__(self, index_path: str = "./faiss_index"):
        self.index_path = Path(index_path)
        self.model = None
        self.faiss_index = None
        self.metadata: List[Dict] = []
        self._loaded = False
        self._metadata_mtime = None

    def _metadata_changed(self) -> bool:
        meta_path = self.index_path / "metadata.json"
        if not meta_path.exists():
            return False
        try:
            mtime = meta_path.stat().st_mtime
            return self._metadata_mtime is None or mtime != self._metadata_mtime
        except Exception:
            return False

    def load(self):
        if self._loaded and not self._metadata_changed():
            return
        if self._loaded:
            logger.info("FAISS metadata changed on disk; reloading retriever")

        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            # Use local_files_only if the model is already downloaded to speed up startup
            try:
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            # Warm up embedding inference so first user query is not penalized.
            self.model.encode(["warmup"], normalize_embeddings=True)
        except Exception as e:
            logger.warning(f"Embedding model unavailable; falling back to keyword search: {e}")
            self.model = None

        meta_path = self.index_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        try:
            import faiss
            idx_path = self.index_path / "index.faiss"
            if idx_path.exists():
                self.faiss_index = faiss.read_index(str(idx_path))
                logger.info(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")
        except ImportError:
            logger.warning("FAISS not available, will use numpy cosine search")
            emb_path = self.index_path / "embeddings.npy"
            if emb_path.exists():
                self.embeddings_np = np.load(str(emb_path))

        self._loaded = True
        meta_path = self.index_path / "metadata.json"
        if meta_path.exists():
            try:
                self._metadata_mtime = meta_path.stat().st_mtime
            except Exception:
                self._metadata_mtime = None

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def _search_with_vector(self, qvec: np.ndarray, k: int = 10) -> List[Dict]:
        """Search using a pre-computed query vector"""
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(qvec.reshape(1, -1), k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    item = self.metadata[idx]
                    results.append({
                        "id": item["id"],
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "score": float(score)
                    })
            return results
        elif hasattr(self, 'embeddings_np'):
            dots = self.embeddings_np @ qvec
            top_k = np.argsort(-dots)[:k]
            return [
                {
                    "id": self.metadata[i]["id"],
                    "text": self.metadata[i]["text"],
                    "metadata": self.metadata[i]["metadata"],
                    "score": float(dots[i])
                }
                for i in top_k if i < len(self.metadata)
            ]
        return []

    def _search_keyword(self, query: str, k: int = 10) -> List[Dict]:
        """Fallback keyword search when embedding search is unavailable."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for i, item in enumerate(self.metadata):
            text_tokens = set(re.findall(r"\w+", item["text"].lower()))
            overlap = len(query_tokens & text_tokens)
            scored.append((overlap, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for overlap, idx in scored[:k]:
            if overlap <= 0:
                break
            item = self.metadata[idx]
            results.append({
                "id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "score": float(overlap)
            })
        return results

    def _search_single(self, query: str, k: int = 10) -> List[Dict]:
        """Search FAISS index for a single query"""
        if self.model is None:
            return self._search_keyword(query, k)
        qvec = self._embed_query(query)
        return self._search_with_vector(qvec, k)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Multi-query retrieval with RRF - Optimized"""
        if not self._loaded:
            self.load()
        
        query_variations = expand_query(query)
        if len(query_variations) > 1:
            if self.model is None:
                all_results = [self._search_keyword(q, k=k * 2) for q in query_variations]
            else:
                qvecs = self.model.encode(query_variations, normalize_embeddings=True).astype(np.float32)
                all_results = []
                for qvec in qvecs:
                    all_results.append(self._search_with_vector(qvec, k=k * 2))
            fused = reciprocal_rank_fusion(all_results, k=60)
        else:
            fused = self._search_single(query, k=k * 2)
            
        intent = self._detect_intent(query)
        filtered = self._filter_results_by_intent(fused, intent, k)
        return filtered[:k]

    def format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents as context string"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            source = meta.get("source", "Reference")
            chapter = meta.get("chapter", "")
            header = f"[Source {i}: {source}"
            if chapter:
                header += f" - {chapter}"
            header += "]"
            context_parts.append(f"{header}\n{doc['text']}")
        return "\n\n---\n\n".join(context_parts)

    def _detect_intent(self, query: str) -> str:
        q = (query or "").lower()
        if any(k in q for k in ["locomotor", "mobility", "movement", "limb", "walking", "paralysis", "amputation"]):
            return "locomotor"
        if any(k in q for k in ["sensory", "vision", "blind", "deaf", "hearing"]):
            return "sensory"
        if any(k in q for k in ["job", "employment", "reservation", "quota", "recruitment"]):
            return "job_reservation"
        if any(k in q for k in ["assistive", "technology", "screen reader", "wheelchair", "hearing aid", "prosthetic", "braille"]):
            return "assistive"
        if any(k in q for k in ["scheme", "benefit", "pension", "udid", "adip"]):
            return "schemes"
        if any(k in q for k in ["rights", "rpwd", "discrimination", "legal", "section"]):
            return "rights"
        return "general"

    def _filter_results_by_intent(self, docs: List[Dict], intent: str, k: int) -> List[Dict]:
        if intent == "general" or not docs:
            return docs

        intent_keywords = {
            "locomotor": ["locomotor", "mobility", "movement", "wheelchair", "paralysis", "amputation", "muscle", "rehabilitation", "walking", "physical disability"],
            "sensory": ["vision", "blindness", "deaf", "hearing", "low vision", "visual impairment", "sensory"],
            "job_reservation": ["reservation", "quota", "employment", "jobs", "section 34", "govt jobs", "benchmar"],
            "assistive": ["assistive", "adip", "alimco", "screen reader", "hearing aid", "wheelchair", "prosthetic", "mobility aid"],
            "schemes": ["scheme", "benefit", "pension", "udid", "national fellowship", "ddrs"],
            "rights": ["rights", "rpwd act", "discrimination", "legal", "section 3", "access", "non-discrimination"]
        }
        keywords = intent_keywords.get(intent, [])
        if not keywords:
            return docs

        filtered = []
        for doc in docs:
            text_content = " ".join([
                str(doc.get("text", "")),
                str(doc.get("metadata", {}).get("source", "")),
                str(doc.get("metadata", {}).get("chapter", "")),
                str(doc.get("metadata", {}).get("category", ""))
            ]).lower()
            if any(kw in text_content for kw in keywords):
                filtered.append(doc)

        if filtered:
            return filtered[:k]
        return docs


# ─── Answer Generation ────────────────────────────────────────────────────────────

DISABILITY_SYSTEM_PROMPT = """You are SAHAY - a highly advanced AI research assistant specializing in Disability Rights, Government Schemes, and Accessibility in India. 
Your goal is to provide accurate, comprehensive, and empathetic answers that feel like they come from an expert legal and social consultant.

Guidelines:
1. **Synthesize & Summarize**: Do not just repeat snippets. Analyze the provided context and provide a structured, synthesized answer.
2. **Citation**: Always cite the specific source once at the start of your answer when referencing the same law section, scheme, or document. Do not repeat the source after every sentence.
3. **Actionable Advice**: Provide step-by-step guidance on how to apply for schemes or where to file complaints.
4. **Simple Language**: Use everyday words, short sentences, and explain things in a clear sequence. When possible, present processes as numbered steps so people can follow them easily.
5. **Formatting**: Use clear headings, bullet points, and bold text for readability.
6. **Rich Content**: If the context contains tables or image references (e.g., Markdown tables or image URLs), INCLUDE them in your response to help the user understand better. Use Markdown format for tables.
6. **Tone**: Professional, authoritative yet deeply empathetic and accessible.
7. **Multilingual**: If the user asks in a regional language, respond in that language while maintaining technical accuracy.
8. **Nuance**: Acknowledge when information might be specific to certain states or categories of disability.

Think like a combination of a legal expert and a compassionate guide."""

LANGUAGE_PROMPTS = {
    "hi": "कृपया हिंदी में उत्तर दें।",
    "ta": "தமிழில் பதிலளிக்கவும்.",
    "te": "తెలుగులో సమాధానం ఇవ్వండి.",
    "bn": "বাংলায় উত্তর দিন।",
    "mr": "मराठीत उत्तर द्या.",
    "gu": "ગુજરાતીમાં જવાબ આપો.",
    "kn": "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
    "ml": "മലയാളത്തിൽ ഉത്തരം നൽകുക.",
    "pa": "ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ۔",
    "or": "ଓଡ଼ିଆରେ ଉତ୍ତର ଦିଅ।",
    "as": "অসমীয়াত উত্তৰ দিয়ক।",
    "ur": "اردو میں جواب دیں۔",
    "en": ""
}

LOCAL_FALLBACK_TRANSLATIONS = {
    "general": {
        "hi": "मैंने दस्तावेज़ों में संबंधित जानकारी पायी है। कृपया विस्तृत जानकारी के लिए नीचे दिए गए स्रोतों को देखें।",
        "ta": "நான் ஆவணங்களில் தொடர்புடைய தகவலை கண்டேன். விரிவான தகவலுக்கு கீழே உள்ள மூலங்களை பாருங்கள்.",
        "te": "నేను డాక్యుమెంట్లలో సంబంధిత సమాచారాన్ని కనుగొన్నాను. పూర్తి సమాచారం కోసం దిగువన ఉన్న మూలాలను చూడండి.",
        "bn": "আমি নথিগুলিতে প্রাসঙ্গিক তথ্য পেয়েছি। বিস্তারিত তথ্যের জন্য নিচের সূত্রগুলি দেখুন।",
        "kn": "ನಾನು ದಾಖಲೆಗಳಲ್ಲಿ ಸಂಬಂಧಿತ ಮಾಹಿತಿಯನ್ನು ಕಂಡುಹಿಡಿದಿದ್ದೇನೆ. ವಿವರವಾದ ಮಾಹಿತಿಗಾಗಿ ಕೆಳಗಿನ ಮೂಲಗಳನ್ನು ನೋಡಿ.",
        "ml": "ഞാൻ ദസ്താവേജുകളിൽ ബന്ധപ്പെട്ട വിവരം കണ്ടെത്തി. വിശദമായ വിവരങ്ങൾക്ക് ചുവടെയുള്ള ഉറവുകൾ കാണുക.",
        "ur": "میں نے دستاویزات میں متعلقہ معلومات حاصل کی ہیں۔ تفصیلی معلومات کے لیے نیچے دی گئی ذرائع دیکھیں۔",
    },
    "rights": {
        "hi": "भारत में विकलांग अधिकार RPWD अधिनियम 2016 के तहत संरक्षित हैं। प्रमुख अधिकारों में समानता, सार्वजनिक स्थानों तक पहुंच, शिक्षा, रोजगार, स्वास्थ्य देखभाल और सामाजिक सुरक्षा शामिल हैं।",
        "ta": "இந்தியாவில் மாற்றுத்திறனாளி உரிமைகள் RPWD சட்டம் 2016 இன் கீழ் பாதுகாக்கப்பட்டுள்ளன. முக்கிய உரிமைகளில் சமத்துவம், பொதுமக்கள் இடங்களுக்கு அணுகல், கல்வி, வேலைவாய்ப்பு, சுகாதார பராமரிப்பு மற்றும் சமூக பாதுகாப்பு உள்ளன.",
        "te": "భారతదేశంలో వికలాంగుల హక్కులు RPWD చట్టం 2016 కింద రక్షించబడ్డాయి. ముఖ్య హక్కులలో సమానత్వం, ప్రజా ప్రదేశాలకు ప్రవేశం, విద్య, ఉద్యోగం, ఆరోగ్య సంరక్షణ, మరియు సామాజిక భద్రత ఉన్నాయి.",
        "bn": "ভারতে প্রতিবন্ধী অধিকারগুলি RPWD আইন 2016 এর আওতায় সুরক্ষিত। প্রধান অধিকারগুলির মধ্যে সমতা, জনসাধারণের স্থানে প্রবেশাধিকার, শিক্ষা, কর্মসংস্থান, স্বাস্থ্যসেবা এবং সামাজিক নিরাপত্তা রয়েছে।",
        "kn": "ಭಾರತದಲ್ಲಿ ಅಂಗವಿಕಲರ ಹಕ್ಕುಗಳು RPWD ಕಾಯಿದೆ 2016 ಅಡಿಯಲ್ಲಿ ರಕ್ಷಿಸಲ್ಪಟ್ಟಿವೆ. ಪ್ರಮುಖ ಹಕ್ಕುಗಳಲ್ಲಿ ಸಮಾನತೆ, ಸಾರ್ವಜನಿಕ ಸ್ಥಳಗಳಿಗೆ ಪ್ರವೇಶ, ಶಿಕ್ಷಣ, ಉದ್ಯೋಗ, ಆರೋಗ್ಯಸೇವೆ ಮತ್ತು ಸಾಮಾಜಿಕ ಭದ್ರತೆ ಸೇರಿವೆ.",
        "ml": "ഇന്ത്യയിൽ വൈകല്യാവകാശങ്ങൾ RPWD നിയമം 2016 പ്രകാരം സംരക്ഷിക്കപ്പെടുന്നു. പ്രധാന അവകാശങ്ങളിൽ സമത്വം, പൊതുസ്ഥലങ്ങളിൽ പ്രവേശനം, വിദ്യാഭ്യാസം, തൊഴിൽ, ആരോഗ്യ പരിപാലനം, സാമൂഹ്യ സുരക്ഷ എന്നിവയുണ്ട്.",
        "ur": "بھارت میں معذوری کے حقوق RPWD ایکٹ 2016 کے تحت محفوظ ہیں۔ اہم حقوق میں مساوات، عوامی مقامات تک رسائی، تعلیم، ملازمت، صحت کی دیکھ بھال، اور سماجی تحفظ شامل ہیں۔",
    },
    "schemes": {
        "hi": "भारत में प्रमुख विकलांग कल्याण योजनाओं में सुलभ भारत अभियान, DDRS, राष्ट्रीय छात्रवृत्ति, सहायक उपकरण सहायता, छात्रवृत्ति और पेंशन योजनाएं शामिल हैं।",
        "ta": "இந்தியாவில் முக்கிய மாற்றுத்திறனாளி நலத்திட்டங்களில் சுகம்யா பாரத், DDRS, தேசிய சிறப்ப грант போன்றவை உள்ளன.",
        "te": "భారతదేశంలో ప్రధాన వికలాంగ సంక్షేమ పథకాల్లో సుగమ్య భారత్, DDRS, జాతీయ ఫెలోషిప్ మరియు సహాయక పరికరాలు, విద్యాసహాయాలు, పెన్షన్ పథకాలు ఉన్నాయి.",
        "bn": "ভারতে প্রধান প্রতিবন্ধী কল্যাণ প্রকল্পগুলির মধ্যে সুলভ ভারত অভিযাত্রা, DDRS, জাতীয় ফেলোশিপ, সহায়ক ডিভাইস, বৃত্তি এবং পেনশন প্রকল্প রয়েছে।",
        "kn": "ಭಾರತದಲ್ಲಿ ಪ್ರಮುಖ ಅಂಗವಿಕಲ ಕಲ್ಯಾಣ ಯೋಜನೆಗಳಲ್ಲಿ ಸುಗಮ ಭಾರತ, DDRS, ರಾಷ್ಟ್ರೀಯ ಫೆಲೋಶಿಪ್ ಮತ್ತು ಸಹಾಯಕ ಸಾಧನಗಳು, ವಿದ್ಯಾರ್ಥಿವೇತನ ಮತ್ತು ಪಿಂಚಣಿ ಯೋಜನೆಗಳು ಸೇರಿವೆ.",
        "ml": "ഇന്ത്യയിലെ പ്രധാന വൈകല്യൻ ക്ഷേമ പദ്ധതികളിൽ സുകമ്യാ ഭാരത്, DDRS, ദേശീയ ഫെലോഷിപ്പ്, സഹായ ഉപകരണങ്ങൾ, വിദ്യാർത്ഥി സഹായങ്ങൾ, പെൻഷൻ പദ്ധതികൾ എന്നിവയുണ്ട്.",
        "ur": "بھارت میں اہم معذوری فلاح و بہبود اسکیموں میں سُگَمیا بھارت، DDRS، قومی فیلوشپ، معاون آلات، وظائف، اور پنشن اسکیمیں شامل ہیں۔",
    },
    "blindness": {
        "hi": "अंधत्व एक दृश्य障碍 है जिसमें दिखने की क्षमता आंशिक या पूर्ण रूप से घट जाती है। भारत में यह RPWD अधिनियम 2016 के अंतर्गत मान्यता प्राप्त विकलांगता है और इसके लिए सहायता उपकरण, शिक्षा समायोजन, और सरकारी योजनाएं उपलब्ध हैं।",
        "ta": "கண்ணின்மை என்பது ஒரு பார்வை குறைபாடு ஆகும், இதில் பார்வை சற்றும் அல்லது முழுமையாக குறைகிறது. இந்தியாவில் இது RPWD சட்டம் 2016 இன் கீழ் குறிக்கப்பட்டிருக்கிறது மற்றும் உதவி சாதனங்கள், கல்வி ஒத்துழைப்பு, மற்றும் அரசு திட்டங்கள் கிடைக்கின்றன.",
        "te": "కళ్ళు చూపని పరిస్థితి ఒక సెన్సరీ వికలాంగత, ఇందులో చూపటి సామర్థ్యం భాగంగా లేదా పూర్తిగా తగ్గిపోతుంది. భారతదేశంలో ఇది RPWD చట్టం 2016 కింద గుర్తింపు పొందిన వికలాంగతగా ఉంది మరియు సహాయక పరికరాలు, విద్యా సౌకర్యాలు, ప్రభుత్వం పథకాలు అందుబాటులో ఉంటాయి.",
        "bn": "অন্ধত্ব একটি দৃষ্টিশক্তি ব্যর্থতা, যেখানে দৃষ্টি আংশিক বা সম্পূর্ণভাবে হারিয়ে যায়। ভারতীয় আইনে এটি RPWD আইন 2016 এর অধীনে স্বীকৃত একটি প্রতিবন্ধিতা এবং এর জন্য সহায়তা সরঞ্জাম, শিক্ষা ব্যবস্থাপনা, ও সরকারি প্রকল্প রয়েছে।",
        "kn": "ಅಂಧತ್ವವು ಒಂದು ದೃಷ್ಟಿ ಅಂಗವಿಕಲತೆ, ಇದರಲ್ಲಿ ხედುವ ಶಕ್ತಿ ಭಾಗಶಃ ಅಥವಾ ಪೂರ್ಣವಾಗಿ ಕಡಿಮೆಯಾಗುತ್ತದೆ. ಭಾರತದಲ್ಲಿ ಇದು RPWD ಕಾಯಿದೆ 2016 ಅಡಿಯಲ್ಲಿ ಮಾನ್ಯವಾದ ಅಂಗವಿಕಲತೆ ಮತ್ತು ಇದರಿಗಾಗಿ ಸಹಾಯಕ ಸಾಧನಗಳು, ಶಿಕ್ಷಣ ಸೌಲಭ್ಯಗಳು, ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಲಭ್ಯವಿವೆ.",
        "ml": "കാണാതായിരിക്കുക ഒരു ദൃഷ്ടി വൈകല്യമാണു, ഇതിൽ ദൃഷ്ടി ഭാഗികമായോ പൂര്‍ണമായോ കുറയുന്നു. ഇന്ത്യയിൽ ഇത് RPWD നിയമം 2016 പ്രകാരം അംഗീകരിച്ചിട്ടുള്ള വൈകല്യമാണ്, സഹായ ഉപകരണങ്ങൾ, വിദ്യാഭ്യാസ സൗകര്യങ്ങൾ, സർക്കാർ പദ്ധതികൾ ലഭ്യമാണ്.",
        "ur": "بینائی ایک حسی معذوری ہے جس میں دیکھنے کی صلاحیت جزوی یا مکمل طور پر ختم ہو جاتی ہے۔ بھارت میں یہ RPWD ایکٹ 2016 کے تحت تسلیم شدہ معذوری ہے اور اس کے لیے معاون آلات، تعلیمی سہولیات، اور سرکاری اسکیمیں دستیاب ہیں۔",
    }
}


def translate_local_fallback(key: str, language: str = "en") -> str:
    if language == "en":
        return None
    lang = language.lower()
    return LOCAL_FALLBACK_TRANSLATIONS.get(key, {}).get(lang)


def generate_answer_local(query: str, context: str, history: List[Dict] = None, language: str = "en") -> str:
    """Enhanced Local extractive answer generation with better synthesis"""
    history = history or []
    
    scoring_query = query
    if language and language.lower() != "en":
        translated_query = translate_text_to_english(query, language)
        if translated_query:
            scoring_query = translated_query

    # Split context into paragraphs/documents first
    docs = context.split("\n\n---\n\n")
    all_sents = []
    for doc in docs:
        # Extract title if present [Source X: Title]
        match = re.search(r"\[Source \d+: (.*?)\]", doc)
        source_title = match.group(1) if match else ""
        
        # Clean text
        text = re.sub(r"\[Source \d+:.*?\]", "", doc).strip()
        sents = re.split(r'(?<=[.!?])\s+', text)
        for s in sents:
            if len(s.strip()) > 15:
                all_sents.append({"text": s.strip(), "source": source_title})

    q_tokens = set(re.findall(r"\w+", scoring_query.lower()))
    
    # Score sentences
    for item in all_sents:
        toks = set(re.findall(r"\w+", item["text"].lower()))
        overlap = len(q_tokens & toks)
        # Bonus for shorter, punchier sentences if they have overlap
        item["score"] = overlap * 10 - (len(item["text"]) / 100)
    
    all_sents.sort(key=lambda x: x["score"], reverse=True)
    
    selected = []
    seen_text = set()
    for item in all_sents:
        if item["score"] <= 0:
            continue
        # Avoid near-duplicates
        if any(len(set(item["text"].lower().split()) & set(prev.lower().split())) / max(len(item["text"].split()), 1) > 0.7 for prev in seen_text):
            continue
            
        selected.append(item)
        seen_text.add(item["text"])
        if len(selected) >= 4:
            break
    
    if not selected:
        answer = (
            "I found relevant sources, but none of the retrieved information directly answered your question. "
            "Please try a more specific query or review the sources below for detailed guidance."
        )
    else:
        sources = [item["source"] for item in selected if item.get("source")]
        unique_sources = []
        for src in sources:
            if src not in unique_sources:
                unique_sources.append(src)

        answer_texts = [item["text"] for item in selected]
        answer_body = " ".join(answer_texts)

        if unique_sources:
            if len(unique_sources) == 1:
                answer = f"According to {unique_sources[0]}, {answer_body}"
            else:
                source_list = "; ".join(unique_sources)
                answer = f"Sources: {source_list}. {answer_body}"
        else:
            answer = answer_body

    # Try to translate the answer to target language
    if language and language.lower() != "en":
        translated = translate_text_with_google(answer, language)
        if translated:
            return translated
            
    return answer


def generate_answer_openai(query: str, context: str, history: List[Dict] = None,
                          language: str = "en", api_key: str = None) -> str:
    """Generate answer using OpenAI API"""
    import requests
    
    if not api_key:
        return None

    history = history or []
    lang_instruction = LANGUAGE_PROMPTS.get(language, "")
    lang_name = LANGUAGE_NAMES.get(language, language)

    messages = [
        {"role": "system", "content": DISABILITY_SYSTEM_PROMPT}
    ]

    # Add history
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})

    lang_addition = f"\n[Respond in {lang_name}]" if language and language != "en" else ""

    user_content = f"""Use the following context to answer the user's question.
Only use the context and do not invent facts.
If the context does not contain the answer, say you don't know the answer instead of guessing.

Context:
{context}

Question: {query}
{lang_instruction}{lang_addition}"""
    
    messages.append({"role": "user", "content": user_content})

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo", # Default to a fast model
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 800
            },
            timeout=15
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            logger.warning(f"OpenAI API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")

    return None


def generate_answer_ollama(query: str, context: str, history: List[Dict] = None,
                            language: str = "en", model: str = "llama2",
                            base_url: str = "http://localhost:11434") -> str:
    """Generate answer using Ollama LLM"""
    import requests
    from requests.exceptions import RequestException

    global _ollama_available, _ollama_fallback_logged
    if not base_url or not base_url.strip() or not _ollama_available:
        return None

    history = history or []
    
    lang_instruction = LANGUAGE_PROMPTS.get(language, "")
    lang_name = LANGUAGE_NAMES.get(language, language)
    
    history_text = ""
    if history:
        recent = history[-6:]
        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        ])
    
    # Add language specification to prompt
    lang_addition = ""
    if language and language != "en":
        lang_addition = f"\n[Current user language: {lang_name}. Respond in {lang_name} if possible.]"
    
    prompt = f"""{DISABILITY_SYSTEM_PROMPT}

{f'Conversation History:{chr(10)}{history_text}{chr(10)}' if history_text else ''}

Context from Knowledge Base:
{context}

User Question: {query}
{lang_instruction}{lang_addition}

Only use the provided context to answer. If the answer is not contained in the provided context, say that you don't know rather than hallucinating. Provide a concise and accurate response with citations to the context where possible. Use simple words and short sentences, and explain any process in a clear sequence or numbered steps when appropriate."""

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300}
            },
            timeout=max(1, settings.OLLAMA_TIMEOUT_SECONDS)
        )
        if resp.status_code == 200:
            answer = resp.json().get("response", "").strip()
            if answer:
                if settings.TRANSLATION_ENABLED and language and language.lower() != "en":
                    translated = translate_text_with_google(answer, language)
                    if translated:
                        logger.info(f"Ollama answer translated to {language}")
                        return translated
                    logger.warning(f"Could not translate Ollama answer to {language}; returning original answer")
                logger.info(f"Ollama generated answer in language: {language}")
                return answer
        else:
            logger.warning(
                f"Ollama returned {resp.status_code}; response body: {resp.text[:200]}"
            )
    except RequestException as e:
        # Ollama is optional; log this once and fall back to local generation.
        if not _ollama_fallback_logged:
            logger.info(f"Ollama unavailable; using local answer generation fallback: {e}")
            _ollama_fallback_logged = True
        _ollama_available = False
    except Exception as e:
        if not _ollama_fallback_logged:
            logger.info(f"Ollama error; using local answer generation fallback: {e}")
            _ollama_fallback_logged = True
        _ollama_available = False
    
    return None


def _get_grok_model_candidates(model: Optional[str] = None, api_key: str = None) -> List[str]:
    candidates = []
    if model:
        candidates.append(model)

    if _is_groq_key(api_key):
        if not candidates:
            candidates.append("openai/gpt-oss-20b")
        return candidates

    if settings.GROK_MODEL and settings.GROK_MODEL not in candidates:
        candidates.append(settings.GROK_MODEL)
    for fallback in GROK_MODEL_FALLBACKS:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _discover_grok_models(api_key: str) -> List[str]:
    import requests
    if _is_groq_key(api_key):
        endpoints = [
            "https://api.groq.com/openai/v1/models"
        ]
    else:
        endpoints = [
            "https://api.x.ai/v1/models",
            "https://api.x.ai/v1/chat/models"
        ]
    discovered = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for endpoint in endpoints:
        try:
            resp = requests.get(endpoint, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue

            body = resp.json()
            models = []
            if isinstance(body, dict):
                if "data" in body and isinstance(body["data"], list):
                    models = [item.get("id") or item.get("name") for item in body["data"] if isinstance(item, dict)]
                elif "models" in body and isinstance(body["models"], list):
                    models = [item.get("id") or item.get("name") for item in body["models"] if isinstance(item, dict)]
            elif isinstance(body, list):
                models = [item.get("id") or item.get("name") for item in body if isinstance(item, dict)]

            for name in models:
                if name and "grok" in name.lower() and name not in discovered:
                    discovered.append(name)

            if discovered:
                logger.info(f"Discovered Grok models: {discovered}")
                return discovered
        except Exception as e:
            logger.warning(f"Failed to discover Grok models from {endpoint}: {e}")
    return discovered


def generate_answer_grok(query: str, context: str, history: List[Dict] = None,
                         language: str = "en", api_key: str = None,
                         model: str = None) -> str:
    """Generate answer using Grok/Groq API."""
    import requests
    from requests.exceptions import RequestException

    if not api_key:
        return None

    models = _get_grok_model_candidates(model, api_key)
    base_url = _get_groq_api_base(api_key)
    endpoint = f"{base_url}/chat/completions"

    history = history or []
    lang_instruction = LANGUAGE_PROMPTS.get(language, "")
    lang_name = LANGUAGE_NAMES.get(language, language)

    messages = [
        {"role": "system", "content": DISABILITY_SYSTEM_PROMPT}
    ]

    # Add history
    if history:
        for m in history[-6:]:
            messages.append({"role": m["role"], "content": m["content"]})

    # Add language specification to prompt
    lang_addition = ""
    if language and language != "en":
        lang_addition = f"\n[Current user language: {lang_name}. Respond in {lang_name} if possible.]"

    user_content = f"""Context from Knowledge Base:
{context}

User Question: {query}
{lang_instruction}{lang_addition}

Only use the information in the context above. If the answer is not contained in the context, say you don't know rather than inventing a response. Please provide a comprehensive, helpful answer based on the context. Use simple words and short sentences, and explain any process in a clear sequence or numbered steps when appropriate."""
    
    messages.append({"role": "user", "content": user_content})

    for candidate in models:
        try:
            resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": candidate,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=15
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
                if answer:
                    logger.info(f"Grok generated answer with model {candidate} in language: {language}")
                    return answer
            else:
                body = resp.text or ""
                if resp.status_code == 400 and "Model not found" in body:
                    if _is_groq_key(api_key):
                        logger.warning(f"Groq model {candidate} not found; stopping retries")
                        break
                    logger.warning(f"Grok model {candidate} not found; trying next fallback")
                    continue
                logger.warning(f"Grok API returned {resp.status_code} for model {candidate}: {body}")
                break
        except RequestException as e:
            logger.error(f"Grok API request error for model {candidate}: {e}")
            break
        except Exception as e:
            logger.error(f"Grok API error for model {candidate}: {e}")
            break

    discovered = _discover_grok_models(api_key)
    for candidate in discovered:
        if candidate in models:
            continue
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": candidate,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 1000
                },
                timeout=15
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
                if answer:
                    logger.info(f"Grok generated answer with discovered model {candidate} in language: {language}")
                    return answer
            else:
                body = resp.text or ""
                if resp.status_code == 400 and "Model not found" in body:
                    if _is_groq_key(api_key):
                        logger.warning(f"Groq model {candidate} not valid; stopping retries")
                        break
                    logger.warning(f"Discovered Grok model {candidate} not valid; trying next")
                    continue
                logger.warning(f"Grok API returned {resp.status_code} for discovered model {candidate}: {body}")
                break
        except RequestException as e:
            logger.error(f"Grok API request error for discovered model {candidate}: {e}")
            break
        except Exception as e:
            logger.error(f"Grok API error for discovered model {candidate}: {e}")
            break

    return None


class DisabilityRAGPipeline:
    """Complete RAG pipeline: retrieve + generate"""

    def __init__(self, index_path: str = "./faiss_index",
                 ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "llama2"):
        self.retriever = DisabilityRAGRetriever(index_path=index_path)
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.chat_histories: Dict[str, List[Dict]] = {}

    def initialize(self):
        self.retriever.load()
        self._probe_ollama()
        logger.info("RAG Pipeline initialized")

    def _probe_ollama(self):
        """Mark Ollama unavailable early to avoid repeated request-time delays."""
        import requests

        global _ollama_available
        if not self.ollama_url:
            _ollama_available = False
            return

        try:
            resp = requests.get(f"{self.ollama_url}/api/models", timeout=1)
            if resp.status_code != 200:
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=1)
            _ollama_available = resp.status_code == 200
            if not _ollama_available:
                logger.info("Ollama probe returned status %s; local generation will be used.", resp.status_code)
        except Exception:
            _ollama_available = False
            logger.info("Ollama not reachable at startup; local generation will be used.")

    def _condense_question(self, question: str, history: List[Dict]) -> str:
        """Use history to condense a follow-up question into a standalone query."""
        if not history:
            return question

        # Limit history for condensation
        recent_history = history[-4:]
        history_str = "\n".join([f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in recent_history])

        prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question about disability rights and services in India.
If the question is already standalone, return it as is. Do NOT answer the question.

Conversation History:
{history_str}

Follow-up Question: {question}
Standalone Question:"""

        try:
            # We use Grok only for question condensation when available.
            condensed = None
            if settings.GROK_API_KEY:
                import requests
                candidates = _get_grok_model_candidates(settings.GROK_MODEL)
                for candidate in candidates:
                    resp = requests.post(
                        f"{_get_groq_api_base(settings.GROK_API_KEY)}/chat/completions",
                        headers={"Authorization": f"Bearer {settings.GROK_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": candidate,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                            "max_tokens": 100
                        },
                        timeout=10
                    )
                    if resp.status_code == 200:
                        condensed = resp.json()["choices"][0]["message"]["content"].strip()
                        break
                    body = resp.text or ""
                    if resp.status_code == 400 and "Model not found" in body:
                        logger.warning(f"Grok model {candidate} not found during condensation; trying next fallback")
                        continue
                    logger.warning(f"Grok condensation returned {resp.status_code} for model {candidate}: {body}")
                    break

                if not condensed:
                    discovered = _discover_grok_models(settings.GROK_API_KEY)
                    for candidate in discovered:
                        if candidate in candidates:
                            continue
                        resp = requests.post(
                            f"{_get_groq_api_base(settings.GROK_API_KEY)}/chat/completions",
                            headers={"Authorization": f"Bearer {settings.GROK_API_KEY}", "Content-Type": "application/json"},
                            json={
                                "model": candidate,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0,
                                "max_tokens": 100
                            },
                            timeout=10
                        )
                        if resp.status_code == 200:
                            condensed = resp.json()["choices"][0]["message"]["content"].strip()
                            break
                        body = resp.text or ""
                        if resp.status_code == 400 and "Model not found" in body:
                            logger.warning(f"Discovered Grok model {candidate} not found during condensation; trying next")
                            continue
                        logger.warning(f"Grok condensation returned {resp.status_code} for discovered model {candidate}: {body}")
                        break

            if condensed:
                logger.info(f"Condensed question: '{question}' -> '{condensed}'")
                return condensed
        except Exception as e:
            logger.warning(f"Question condensation failed: {e}")

        return question

    def query(self, question: str, session_id: str = "default",
              language: str = "en", k: int = None) -> Dict[str, Any]:
        """Full RAG query: retrieve + generate"""
        history = self.chat_histories.get(session_id, [])
        k = k or settings.TOP_K_RESULTS

        # 1. Translate current question to English for retrieval if needed
        english_question = question
        if settings.TRANSLATION_ENABLED and language and language.lower() != "en":
            translated_question = translate_text_to_english(question, language)
            if translated_question:
                english_question = translated_question
        
        # 2. Condense the question using history to handle multi-turn queries
        retrieval_question = self._condense_question(english_question, history)

        # Retrieve relevant documents
        docs = self.retriever.retrieve(retrieval_question, k=k)
        if not docs and retrieval_question != question:
            docs = self.retriever.retrieve(question, k=k)
        context = self.retriever.format_context(docs)

        if not context or not context.strip():
            answer = generate_answer_local(retrieval_question, context, history, language)
        else:
            # Generate answer using Grok only, then fall back to local generation.
            answer = None
            if settings.GROK_API_KEY:
                answer = generate_answer_grok(
                    retrieval_question, context, history, language,
                    settings.GROK_API_KEY,
                    settings.GROK_MODEL
                )

            if not answer:
                answer = generate_answer_local(retrieval_question, context, history, language)
        
        # Build sources list
        sources = []
        seen = set()
        for doc in docs:
            meta = doc.get("metadata", {})
            src = meta.get("source", "Reference")
            cat = meta.get("category", "general")
            chapter = meta.get("chapter", "")
            key = f"{src}:{chapter}"
            if key not in seen:
                sources.append({"source": src, "chapter": chapter, "category": cat})
                seen.add(key)
        
        # Update history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        self.chat_histories[session_id] = history[-20:]  # Keep last 20 messages
        
        return {
            "answer": answer,
            "sources": sources,
            "docs_retrieved": len(docs),
            "session_id": session_id
        }


# Singleton pipeline instance
_pipeline_instance: Optional[DisabilityRAGPipeline] = None

def get_pipeline(index_path: str = "./faiss_index") -> DisabilityRAGPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        from app.core.config import settings
        _pipeline_instance = DisabilityRAGPipeline(
            index_path=index_path,
            ollama_url=settings.OLLAMA_BASE_URL,
            ollama_model=settings.OLLAMA_MODEL
        )
        _pipeline_instance.initialize()
    return _pipeline_instance
