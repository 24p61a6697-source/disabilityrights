# FIXED + OPTIMIZED RAG INGESTION

import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512

# Disability Knowledge Base - Multilingual Content
DISABILITY_KNOWLEDGE_BASE = [
    {
        "text": "In India, disability rights are protected under the Rights of Persons with Disabilities Act, 2016 (RPWD Act). Key rights include equality, access to public spaces, education and employment, healthcare, and social security.",
        "source": "RPWD Act 2016",
        "chapter": "Rights Overview",
        "language": "en"
    },
    {
        "text": "भारत में, दिव्यांग अधिकार 2016 के अधिकारों के अधिनियम (RPWD एक्ट) के तहत संरक्षित हैं। मुख्य अधिकारों में समानता, सार्वजनिक स्थानों तक पहुंच, शिक्षा और रोजगार, स्वास्थ्य सेवाएं, और सामाजिक सुरक्षा शामिल हैं।",
        "source": "RPWD अधिनियम 2016",
        "chapter": "अधिकार अवलोकन",
        "language": "hi"
    },
    {
        "text": "இந்தியாவில், மாற்றுத் திறனாளர்களின் உரிமைகள் 2016 ஆம் ஆண்டு வெளியான RPWD சட்டத்தின் கீழ் பாதுகாக்கப்படுகின்றன. முக்கிய உரிமைகள் சமத்துவம், பொதுமக்கள் இடங்களுக்கு அணுகுதல், கல்வி மற்றும் வேலைவாய்ப்பு, சுகாதார சேவைகள் மற்றும் சமூக பாதுகாப்பைக் கொண்டுள்ளன.",
        "source": "RPWD சட்டம் 2016",
        "chapter": "உரிமைகள் கண்ணோட்டம்",
        "language": "ta"
    },
    {
        "text": "భారతదేశంలో, వికారుల హక్కులు 2016 యొక్క RPWD చట్టం క్రింద రక్షించబడ్డాయి. ప్రధాన హక్కులలో సమానత్వం, పౌర స్థలాలకు ప్రవేశం, విద్య మరియు ఉద్యోగం, ఆరోగ్య సంరక్షణ, మరియు సామాజిక భద్రత ఉన్నాయి.",
        "source": "RPWD చట్టం 2016",
        "chapter": "హక్కుల అవలోకనం",
        "language": "te"
    },
    {
        "text": "ಭಾರತದಲ್ಲಿ, ಅಂಗವಿಕಲರ ಹಕ್ಕುಗಳು 2016 ರ RPWD ಕಾಯಿದೆ ಅಡಿಯಲ್ಲಿ ರಕ್ಷಿಸಲ್ಪಟ್ಟಿವೆ. ಪ್ರಮುಖ ಹಕ್ಕುಗಳಲ್ಲಿ ಸಮಾನತೆ, ಸಾರ್ವಜನಿಕ ಸ್ಥಳಗಳಿಗೆ ಪ್ರವೇಶ, ಶಿಕ್ಷಣ ಮತ್ತು ಉದ್ಯೋಗ, ಆರೋಗ್ಯಸೇವೆಗಳು ಮತ್ತು ಸಾಮಾಜಿಕ ಭದ್ರತೆ ಸೇರಿವೆ.",
        "source": "RPWD ಕಾಯಿದೆ 2016",
        "chapter": "ಹಕ್ಕುಗಳ ಅವಲೋಕನ",
        "language": "kn"
    },
    {
        "text": "ഇന്ത്യയിൽ, വികലാംഗാവകാശങ്ങൾ 2016ലെ RPWD ആക്ടിന്റെ കീഴിൽ സംരക്ഷിതമാണ്. പ്രധാന അവകാശങ്ങളിൽ സമത്വം, പൊതു സ്ഥലങ്ങളിൽ പ്രവേശനം, വിദ്യാഭ്യാസം മറ്റും തൊഴിൽ, ആരോഗ്യപരിപാലനം, സാമൂഹ്യസുരക്ഷ എന്നിവയുണ്ട്.",
        "source": "RPWD നിയമം 2016",
        "chapter": "അവകാശങ്ങളുടെ അവലോകനം",
        "language": "ml"
    },
    {
        "text": "Assistive technology includes tools like screen readers (NVDA/JAWS), hearing aids, wheelchairs, prosthetics, Braille kits, communication devices, and mobility aids. In India, support is available through ADIP Scheme, ALIMCO, and national institutes. Eligibility and device type depend on disability assessment and income criteria.",
        "source": "ADIP Scheme",
        "chapter": "Assistive Technology",
        "language": "en"
    },
    {
        "text": "सहायक तकनीक में स्क्रीन रीडर (NVDA/JAWS), हियरिंग एड, व्हीलचेयर, कृत्रिम अंग, ब्रेल किट, संचार उपकरण और चलने-फिरने की सहायक सामग्री शामिल हैं। भारत में ADIP योजना, ALIMCO और राष्ट्रीय संस्थानों के माध्यम से सहायता उपलब्ध है। पात्रता और उपकरण का प्रकार विकलांगता आकलन और आय मानदंड पर निर्भर करता है।",
        "source": "ADIP योजना",
        "chapter": "सहायक तकनीक",
        "language": "hi"
    },
    {
        "text": "உதவி தொழில்நுட்பத்தில் ஸ்கிரீன் ரீடர்கள் (NVDA/JAWS), கேட்கும் கருவிகள், சக்கர நாற்காலிகள், செயற்கை உறுப்புகள், பிரெயில் கருவிகள், தொடர்பு சாதனங்கள் மற்றும் இயக்க உதவிகள் அடங்கும். இந்தியாவில் ADIP திட்டம், ALIMCO மற்றும் தேசிய நிறுவனங்கள் மூலம் உதவி கிடைக்கிறது. தகுதி மற்றும் கருவி வகை மாற்றுத்திறன் மதிப்பீடு மற்றும் வருமான நிபந்தனைகளைப் பொறுத்தது.",
        "source": "ADIP திட்டம்",
        "chapter": "உதவி தொழில்நுட்பம்",
        "language": "ta"
    },
    {
        "text": "సహాయక సాంకేతికతలో స్క్రీన్ రీడర్లు (NVDA/JAWS), హియరింగ్ ఎయిడ్స్, వీల్‌చెయర్లు, కృత్రిమ అవయవాలు, బ్రెయిల్ కిట్లు, కమ్యూనికేషన్ పరికరాలు, మొబిలిటీ సహాయక పరికరాలు ఉంటాయి. భారతదేశంలో ADIP పథకం, ALIMCO మరియు జాతీయ సంస్థల ద్వారా సహాయం లభిస్తుంది. అర్హత మరియు పరికరం రకం వికలాంగత అంచనా మరియు ఆదాయ ప్రమాణాలపై ఆధారపడతాయి.",
        "source": "ADIP పథకం",
        "chapter": "సహాయక సాంకేతికత",
        "language": "te"
    },
    {
        "text": "ಸಹಾಯಕ ತಂತ್ರಜ್ಞಾನದಲ್ಲಿ ಸ್ಕ್ರೀನ್ ರೀಡರ್‌ಗಳು (NVDA/JAWS), ಹೇರಿಂಗ್ ಏಡ್ಸ್, ವೀಲ್ಚೇರ್‌ಗಳು, ಕೃತಕ ಅಂಗಗಳು, ಬ್ರೈಲ್ ಕಿಟ್‌ಗಳು, ಸಂವಹನ ಸಾಧನಗಳು ಮತ್ತು ಚಲನೆ ಸಹಾಯಕ ಸಾಧನಗಳು ಸೇರಿವೆ. ಭಾರತದಲ್ಲಿ ADIP ಯೋಜನೆ, ALIMCO ಮತ್ತು ರಾಷ್ಟ್ರೀಯ ಸಂಸ್ಥೆಗಳ ಮೂಲಕ ಸಹಾಯ ಲಭ್ಯವಿದೆ. ಅರ್ಹತೆ ಮತ್ತು ಸಾಧನದ ವಿಧವು ಅಂಗವೈಕಲ್ಯ ಮೌಲ್ಯಮಾಪನ ಹಾಗೂ ಆದಾಯ ಮಾನದಂಡಗಳ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿದೆ.",
        "source": "ADIP ಯೋಜನೆ",
        "chapter": "ಸಹಾಯಕ ತಂತ್ರಜ್ಞಾನ",
        "language": "kn"
    },
    {
        "text": "സഹായ സാങ്കേതികവിദ്യയിൽ സ്ക്രീൻ റീഡറുകൾ (NVDA/JAWS), കേൾവി ഉപകരണങ്ങൾ, വീൽചെയറുകൾ, കൃത്രിമ അവയവങ്ങൾ, ബ്രെയിൽ കിറ്റുകൾ, ആശയവിനിമയ ഉപകരണങ്ങൾ, ചലന സഹായങ്ങൾ എന്നിവ ഉൾപ്പെടുന്നു. ഇന്ത്യയിൽ ADIP പദ്ധതി, ALIMCO, ദേശീയ സ്ഥാപനങ്ങൾ എന്നിവ വഴി സഹായം ലഭ്യമാണ്. യോഗ്യതയും ഉപകരണ തരംവും വൈകല്യ നിർണയവും വരുമാന മാനദണ്ഡവും ആശ്രയിച്ചിരിക്കും.",
        "source": "ADIP പദ്ധതി",
        "chapter": "സഹായ സാങ്കേതികവിദ്യ",
        "language": "ml"
    },
    {
        "text": "Under Section 34 of the RPWD Act, 2016, at least 4% reservation is provided in government establishments for persons with benchmark disabilities. This includes identified posts across disability categories. You can apply through regular recruitment notifications and claim applicable relaxation/support under RPWD rules.",
        "source": "RPWD Act Section 34",
        "chapter": "Employment Reservation",
        "language": "en"
    },
    {
        "text": "RPWD अधिनियम, 2016 की धारा 34 के तहत सरकारी संस्थानों में बेंचमार्क दिव्यांग व्यक्तियों के लिए कम से कम 4% आरक्षण प्रदान किया गया है। यह विभिन्न दिव्यांगता श्रेणियों के लिए चिन्हित पदों पर लागू होता है। आप सामान्य भर्ती विज्ञापनों के माध्यम से आवेदन कर सकते हैं और RPWD नियमों के अनुसार छूट/सुविधाएं प्राप्त कर सकते हैं।",
        "source": "RPWD अधिनियम धारा 34",
        "chapter": "रोजगार आरक्षण",
        "language": "hi"
    },
    {
        "text": "RPWD சட்டம், 2016 இன் பிரிவு 34ன் கீழ் அரசு அமைப்புகளில் benchmark மாற்றுத்திறனாளிகளுக்கு குறைந்தபட்சம் 4% இடஒதுக்கீடு வழங்கப்பட்டுள்ளது. இது மாற்றுத்திறனின் பல பிரிவுகளுக்கான அடையாளப்படுத்தப்பட்ட பணியிடங்களில் பொருந்தும். சாதாரண ஆட்சேர்ப்பு அறிவிப்புகள் மூலம் விண்ணப்பிக்கலாம்; RPWD விதிகளின்படி தளர்வுகள்/உதவிகளை பெறலாம்.",
        "source": "RPWD சட்டம் பிரிவு 34",
        "chapter": "வேலைவாய்ப்பு இடஒதுக்கீடு",
        "language": "ta"
    },
    {
        "text": "RPWD చట్టం, 2016 లోని సెక్షన్ 34 ప్రకారం ప్రభుత్వ సంస్థల్లో benchmark వికలాంగుల కోసం కనీసం 4% రిజర్వేషన్ ఉంది. ఇది వివిధ వికలాంగత వర్గాలకు గుర్తించిన పోస్టులకు వర్తిస్తుంది. సాధారణ నియామక ప్రకటనల ద్వారా దరఖాస్తు చేయవచ్చు మరియు RPWD నిబంధనల ప్రకారం వర్తించే సడలింపులు/సహాయం పొందవచ్చు.",
        "source": "RPWD చట్టం సెక్షన్ 34",
        "chapter": "ఉద్యోగ రిజర్వేషన్",
        "language": "te"
    },
    {
        "text": "RPWD ಕಾಯಿದೆ, 2016 ರ ಕಲಂ 34 ಅನ್ವಯ, ಸರ್ಕಾರದ ಸಂಸ್ಥೆಗಳಲ್ಲಿ benchmark ಅಂಗವಿಕಲ ವ್ಯಕ್ತಿಗಳಿಗೆ ಕನಿಷ್ಠ 4% ಮೀಸಲಾತಿ ನೀಡಲಾಗಿದೆ. ಇದು ವಿವಿಧ ಅಂಗವೈಕಲ್ಯ ವರ್ಗಗಳಿಗೆ ಗುರುತಿಸಲಾದ ಹುದ್ದೆಗಳಿಗೆ ಅನ್ವಯಿಸುತ್ತದೆ. ಸಾಮಾನ್ಯ ನೇಮಕಾತಿ ಅಧಿಸೂಚನೆಗಳ ಮೂಲಕ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ, RPWD ನಿಯಮಗಳಡಿ ದೊರೆಯುವ ಸಡಿಲಿಕೆ/ಸಹಾಯವನ್ನು ಪಡೆಯಬಹುದು.",
        "source": "RPWD ಕಾಯಿದೆ ಕಲಂ 34",
        "chapter": "ಉದ್ಯೋಗ ಮೀಸಲಾತಿ",
        "language": "kn"
    },
    {
        "text": "RPWD നിയമം, 2016 ലെ വകുപ്പ് 34 പ്രകാരം സർക്കാർ സ്ഥാപനങ്ങളിൽ benchmark വൈകല്യമുള്ള വ്യക്തികൾക്ക് കുറഞ്ഞത് 4% സംവരണം നൽകിയിട്ടുണ്ട്. ഇത് വിവിധ വൈകല്യ വിഭാഗങ്ങൾക്കുള്ള തിരിച്ചറിഞ്ഞ ഒഴിവുകൾക്ക് ബാധകമാണ്. സാധാരണ നിയമന വിജ്ഞാപനങ്ങൾ വഴി അപേക്ഷിക്കാം; RPWD ചട്ടങ്ങൾ പ്രകാരം ലഭ്യമായ ഇളവുകൾ/സഹായങ്ങൾ ആവശ്യപ്പെടാം.",
        "source": "RPWD നിയമം വകുപ്പ് 34",
        "chapter": "ഉദ്യോഗ സംവരണം",
        "language": "ml"
    },
    {
        "text": "Sensory disabilities include impairments in vision, hearing, touch, taste, or smell. In India these are covered under the RPWD Act and may qualify for access benefits, assistive devices, education accommodations, and disability certification.",
        "source": "RPWD Act Sensory Disabilities",
        "chapter": "Disability Definitions",
        "language": "en"
    },
    {
        "text": "द्रवण विकलांगताओं में दृष्टि, श्रवण, स्पर्श, स्‍वाद या गंध में अक्षमता शामिल हैं। भारत में इन्हें RPWD अधिनियम के अंतर्गत कवर किया जाता है और ये पहुंच सुविधाएँ, सहायक उपकरण, शिक्षा में समायोजन और विकलांगता प्रमाणन के लिए पात्र हो सकते हैं।",
        "source": "RPWD अधिनियम द्रवण विकलांगताएं",
        "chapter": "विकलांगता परिभाषाएं",
        "language": "hi"
    },
    {
        "text": "உணர்வுத் திறன் குறைகளில் பார்க்கும், கேட்கும், தொடுதல், சுவை அல்லது மணத்தை உணருவதில் புலபின்மை அடங்கும். இந்தியாவில் இதற்கு RPWD சட்டத்தின் கீழ் பாதுகாப்பு உள்ளது மற்றும் அணுகல் நலன்கள், உதவி சாதனங்கள், கல்வி ஒத்துழைப்பு மற்றும் மாற்றுத்திறனுக் சான்றிதழுக்கு தகுதி பெற்றிருக்கலாம்.",
        "source": "RPWD சட்டம் உணர்வுத் திறன் குறைகள்",
        "chapter": "மாற்றுத்திறன் வரையறைகள்",
        "language": "ta"
    },
    {
        "text": "సెన్సరీ వికలాంగతలు దృష్టి, వినికిడి, అనుభూతి, రుచిఅనే వాసనలో లోపాలను కలిగి ఉంటాయి. భారతదేశంలో ఇవి RPWD చట్టం క్రింద కవర్ చేయబడతాయి మరియు ప్రాప్తి ప్రయోజనాలు, సహాయక పరికరాలు, విద్యా సౌకర్యాలు మరియు వికలాంగత సర్టిఫికేషన్ కోసం అర్హత పొందవచ్చు.",
        "source": "RPWD చట్టం సెన్సరీ వికలాంగతలు",
        "chapter": "వికలాంగత నిర్వచనాలు",
        "language": "te"
    },
    {
        "text": "ಸಂವೇದಿ ಅಂಗವಿಕಲತೆಗಳಲ್ಲಿ ದೃಷ್ಟಿ, ಕೇಳುವಿಕೆ, ಸ್ಪರ್ಶ, ರುಚಿ ಅಥವಾ ವಾಸನೆಯಲ್ಲಿರುವ ಅಶಕ್ತಿ ಸೇರಿವೆ. ಭಾರತದಲ್ಲಿ ಇವುಗಳನ್ನು RPWD ಕಾಯಿದೆಯಡಿ ಒಳಗೊಂಡಿರುತ್ತವೆ ಮತ್ತು ಪ್ರವೇಶ ಪ್ರಯೋಜನಗಳು, ಸಹಾಯಕ ಸಾಧನಗಳು, ಶಿಕ್ಷಣ ಸೌಲಭ್ಯಗಳು ಮತ್ತು ಅಂಗವೈಕಲತೆ ಪ್ರಮಾಣೀಕರಣಕ್ಕಾಗಿ ಅರ್ಹರಾಗಬಹುದು.",
        "source": "RPWD ಕಾಯಿದೆ ಸಂವೇದಿ ಅಂಗವಿಕಲತೆಗಳು",
        "chapter": "ಅಂಗವಿಕಲತೆ ನಿರ್ವಚನೆಗಳು",
        "language": "kn"
    },
    {
        "text": "സെൻസറി വൈകല്യങ്ങളിൽ ദൃഷ്ടി, കേൾവ്, സ്പർശം, രുചി അല്ലെങ്കിൽ ഗന്ധത്തിൽ തകരാറുകൾ ഉൾപ്പെടുന്നു. ഭാരതത്തിൽ ഇത് RPWD ആക്ടിന്റെ കീഴിൽ ഉൾപ്പെടുന്നു, പ്രാപ്യതാ ആനുകൂല്യങ്ങൾ, സഹായ ഉപകരണങ്ങൾ, വിദ്യാഭ്യാസ സൗകര്യങ്ങൾ, വൈകല്യമാന്യപത്രക്കാർിക്ക് തർഹത എന്നിവക്ക് അർഹത നൽകാം.",
        "source": "RPWD നിയമം സെൻസറി വൈകല്യങ്ങൾ",
        "chapter": "വൈകല്യ നിർവചനങ്ങൾ",
        "language": "ml"
    }
]

class DisabilityRAGIngestion:
    def __init__(self, index_path: str = "./faiss_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SentenceTransformer] = None
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

        self.batch_size = 32  # 🔥 control memory

    # ---------------- MODEL ---------------- #

    def load_embedding_model(self):
        if self.model:
            return
        logger.info(f"Loading model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ---------------- SAFE CHUNKING ---------------- #

    def _safe_chunk(self, text: str, source: str):
        if not text or len(text.strip()) < 50:
            return []

        chunks = []
        words = text.split()

        for i in range(0, len(words), CHUNK_SIZE):
            part = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": part,
                "metadata": {"source": source}
            })

        return chunks

    # ---------------- PDF ---------------- #

    def load_pdf_document(self, pdf_path: str):
        chunks = []

        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)

            for page in reader.pages:
                text = page.extract_text() or ""
                chunks.extend(self._safe_chunk(text, Path(pdf_path).name))

        except Exception as e:
            logger.error(f"PDF failed: {e}")

        return chunks

    # ---------------- INGEST ---------------- #

    def ingest_all(self, docs_dir=None, include_knowledge_base=True):
        self.load_embedding_model()

        all_chunks = []

        # 1. knowledge base
        if include_knowledge_base:
            for item in DISABILITY_KNOWLEDGE_BASE:
                if not item.get("text"):
                    continue
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": item["text"],
                    "metadata": item
                })

        # 2. PDFs
        if docs_dir:
            docs_path = Path(docs_dir)
            if docs_path.exists():
                for pdf in docs_path.glob("*.pdf"):
                    all_chunks.extend(self.load_pdf_document(str(pdf)))

        if not all_chunks:
            raise ValueError("No data to index")

        self.chunks = all_chunks

        # ---------------- EMBEDDINGS ---------------- #

        texts = [c["text"] for c in all_chunks]

        embeddings_list = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            emb = self.model.encode(
                batch,
                normalize_embeddings=True
            )
            embeddings_list.append(emb)

        self.embeddings = np.vstack(embeddings_list)

        logger.info(f"Embeddings ready: {self.embeddings.shape}")

        self._build_faiss_index()
        self._save_metadata()

        return len(all_chunks)

    # ---------------- FAISS ---------------- #

    def _build_faiss_index(self):
        try:
            import faiss

            dim = self.embeddings.shape[1]

            index = faiss.IndexFlatIP(dim)
            index.add(self.embeddings.astype(np.float32))

            faiss.write_index(index, str(self.index_path / "index.faiss"))

        except Exception as e:
            logger.error(f"FAISS failed: {e}")
            raise RuntimeError("FAISS is required for production")

    # ---------------- METADATA ---------------- #

    def _save_metadata(self):
        metadata = [
            {"id": c["id"], "text": c["text"], "metadata": c["metadata"]}
            for c in self.chunks
        ]

        with open(self.index_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)