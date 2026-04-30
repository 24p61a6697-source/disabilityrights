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
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
_ollama_available = True
_ollama_fallback_logged = False

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ─── Category-based query expansion ─────────────────────────────────────────────

QUERY_EXPANSIONS = {
    "rights": ["legal rights disability India", "RPWD Act entitlements", "disability discrimination"],
    "scheme": ["government schemes disability", "benefits disabled persons India", "welfare programs"],
    "job": ["employment reservation disability", "4% reservation government jobs", "disability quota"],
    "education": ["education disability India", "5% reservation college", "exam accommodation disability"],
    "technology": ["assistive technology disability", "screen reader", "wheelchair"],
    "certificate": ["disability certificate India", "UDID card", "benchmark disability"],
    "pension": ["disability pension India", "IGNDPS", "financial assistance"],
    "accessibility": ["accessible building India", "ramp standard", "WCAG accessibility"],
    "health": ["healthcare disability India", "rehabilitation services", "NPCB NPPCD"],
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

    def load(self):
        if self._loaded:
            return
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
            logger.error(f"Failed to load embedding model: {e}")
            raise

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

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def _search_single(self, query: str, k: int = 10) -> List[Dict]:
        """Search FAISS index for a single query"""
        qvec = self._embed_query(query)
        
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

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Multi-query retrieval with RRF"""
        if not self._loaded:
            self.load()
        query_variations = expand_query(query)
        all_results = []
        for q in query_variations:
            results = self._search_single(q, k=k * 2)
            all_results.append(results)
        fused = reciprocal_rank_fusion(all_results, k=60)
        return fused[:k]

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


# ─── Answer Generation ────────────────────────────────────────────────────────────

DISABILITY_SYSTEM_PROMPT = """You are SAHAY - an expert AI assistant on Disability Rights, Government Schemes, and Accessibility in India.
You help persons with disabilities understand:
- Their rights under RPWD Act 2016
- Government schemes and how to apply
- Assistive technology recommendations
- Accessibility standards
- Education and employment accommodations

Guidelines:
- Always cite the specific law, scheme, or source (e.g., "Under Section 34 of RPWD Act 2016...")
- Provide practical, actionable information
- Mention relevant government portals and contact numbers
- Be empathetic, clear, and accessible
- If asked in a regional language context, acknowledge it
- For medical conditions, always recommend consulting a certified doctor
- Provide step-by-step guidance when explaining processes

Always structure your answers clearly with relevant headings when appropriate."""

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
    """Local extractive answer generation (when Ollama is unavailable)"""
    history = history or []
    
    scoring_query = query
    if language and language.lower() != "en":
        translated_query = translate_text_to_english(query, language)
        if translated_query:
            scoring_query = translated_query

    # Extract relevant sentences from context
    sents = re.split(r'(?<=[.!?])\s+', context)
    q_tokens = set(re.findall(r"\w+", scoring_query.lower()))
    
    # Score sentences by relevance to query
    scores = []
    for i, sent in enumerate(sents):
        if len(sent.strip()) < 20:
            continue
        toks = set(re.findall(r"\w+", sent.lower()))
        overlap = len(q_tokens & toks)
        scores.append((i, overlap, len(sent)))
    
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    selected = []
    for i, score, _ in scores[:5]:
        if score > 0:
            selected.append(sents[i].strip())
    
    if not selected:
        selected = [s.strip() for s in sents[:3] if len(s.strip()) > 20]
    
    answer = " ".join(selected[:4])
    
    if not answer:
        answer = "I found relevant information in the disability rights database. Please refer to the sources above for detailed information."
    
    # Try to translate the answer to target language
    if language and language.lower() != "en":
        translated = translate_text_with_google(answer, language)
        if translated:
            logger.info(f"Answer successfully translated to {language}")
            return translated
        else:
            # If translation fails, log it but return English answer instead of adding notice
            lang_name = LANGUAGE_NAMES.get(language, language)
            logger.warning(f"Translation to {lang_name} could not be completed. Returning English answer.")
            # Return just the English answer without the notice - user will understand it's from the database
            return answer
    
    return answer


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

Please provide a comprehensive, helpful answer based on the context above:"""

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


def generate_answer_grok(query: str, context: str, history: List[Dict] = None,
                         language: str = "en", api_key: str = None) -> str:
    """Generate answer using Grok API (X.AI)"""
    import requests
    from requests.exceptions import RequestException

    if not api_key:
        return None

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

Please provide a comprehensive, helpful answer based on the context above:"""
    
    messages.append({"role": "user", "content": user_content})

    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-beta", # Or appropriate grok model
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000
            },
            timeout=15
        )
        if resp.status_code == 200:
            answer = resp.json()["choices"][0]["message"]["content"].strip()
            if answer:
                logger.info(f"Grok generated answer in language: {language}")
                return answer
        else:
            logger.warning(f"Grok API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Grok API error: {e}")

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
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=1)
            _ollama_available = resp.status_code == 200
            if not _ollama_available:
                logger.info("Ollama probe returned status %s; local generation will be used.", resp.status_code)
        except Exception:
            _ollama_available = False
            logger.info("Ollama not reachable at startup; local generation will be used.")

    def query(self, question: str, session_id: str = "default",
              language: str = "en", k: int = None) -> Dict[str, Any]:
        """Full RAG query: retrieve + generate"""
        history = self.chat_histories.get(session_id, [])
        k = k or settings.TOP_K_RESULTS

        retrieval_question = question
        if settings.TRANSLATION_ENABLED and language and language.lower() != "en":
            translated_question = translate_text_to_english(question, language)
            if translated_question:
                retrieval_question = translated_question
        
        # Retrieve relevant documents
        docs = self.retriever.retrieve(retrieval_question, k=k)
        if not docs and retrieval_question != question:
            docs = self.retriever.retrieve(question, k=k)
        context = self.retriever.format_context(docs)

        if not context or not context.strip():
            answer = generate_answer_local(retrieval_question, context, history, language)
        else:
            # Generate answer - try Ollama first, then Grok, fall back to extractive
            answer = generate_answer_ollama(
                retrieval_question, context, history, language,
                self.ollama_model, self.ollama_url
            )
        
            if not answer and settings.GROK_API_KEY:
                answer = generate_answer_grok(
                    retrieval_question, context, history, language,
                    settings.GROK_API_KEY
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
