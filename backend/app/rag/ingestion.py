# FIXED + OPTIMIZED RAG INGESTION

import json
import logging
import os
import uuid
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "true"
os.environ["DISABLE_TQDM"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning, module="tqdm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512

# Disability Knowledge Base - Multilingual Content
DISABILITY_KNOWLEDGE_BASE = [
    {
        "text": "Disability is a complex and multidimensional phenomenon that reflects the interaction between features of a person's body and the physical and social environment in which they live. It is not merely a medical condition or an individual deficit, but rather a dynamic interaction between health conditions and contextual factors – both personal and environmental. The World Health Organization (WHO) defines disability as: 'Disability is part of the human condition. Almost everyone will be temporarily or permanently impaired at some point in life, and those who survive to old age will experience increasing difficulties in functioning.'",
        "source": "WHO World Report on Disability, 2011",
        "chapter": "Introduction to Disability",
        "category": "definition",
        "language": "en"
    },
    {
        "text": "WHO and World Bank data consistently show that disability and poverty are closely linked in a self-reinforcing cycle. Disability can cause poverty through reduced employment, increased medical costs, reduced productivity, and social exclusion. Conversely, poverty can cause disability through poor nutrition, exposure to disease, unsafe living conditions, and limited access to healthcare. Key findings include: employment rates of PwDs in OECD countries are about 44% versus 75% for non-disabled people; people with disabilities often earn 30–40% less; households with a disabled member spend more on healthcare and support; 69% of PwDs in India live in rural areas with low service access; and the extra costs of disability deepen poverty.",
        "source": "WHO / World Bank / Disability_WHO_RPWD_Thesis_Reference__1_.pdf",
        "chapter": "Disability and Poverty",
        "category": "poverty",
        "language": "en"
    },
    {
        "text": "In India, disability rights are protected under the Rights of Persons with Disabilities Act, 2016 (RPWD Act). Key rights include equality, access to public spaces, education and employment, healthcare, and social security.",
        "source": "RPWD Act 2016",
        "chapter": "Rights Overview",
        "category": "rights",
        "language": "en"
    },
    {
        "text": "Blindness is a visual impairment in which a person has very low vision or no usable vision even after best correction. Under the RPWD Act 2016, blindness is recognized as a disability and may qualify for disability certification, assistive devices, education accommodations, and government benefits.",
        "source": "RPWD Act 2016",
        "chapter": "Blindness Definition",
        "category": "definitions",
        "language": "en"
    },
    {
        "text": "Locomotor disability refers to impairments that affect movement, muscle strength, or mobility. Under the RPWD Act 2016, locomotor disability is recognized as a benchmark disability and may qualify individuals for disability certification, assistive mobility devices, education accommodations, and government benefits.",
        "source": "RPWD Act 2016",
        "chapter": "Locomotor Disability",
        "category": "definitions",
        "language": "en"
    },
    {
        "text": "Visual disabilities include blindness and low vision. In India, visual disability is covered under the RPWD Act 2016 and persons with visual impairment can access disability certification, assistive technology, education support, and reservations in employment and education.",
        "source": "RPWD Act Visual Disabilities",
        "chapter": "Visual Disability",
        "category": "definitions",
        "language": "en"
    },
    {
        "text": "Persons with disabilities can file complaints regarding discrimination or violation of rights with the Chief Commissioner for Persons with Disabilities (CCPD) at the National level or State Commissioners at the State level. You can file a complaint online through the CCPD website (ccpdisabilities.nic.in) or by writing to the Commissioner. The Commissioners have powers of a civil court to hear grievances.",
        "source": "RPWD Act Section 75",
        "chapter": "Grievance Redressal",
        "category": "complaint",
        "language": "en"
    },
    {
        "text": "Under Section 32 of the RPWD Act 2016, all government and government-aided higher education institutions must reserve at least 5% seats for persons with benchmark disabilities. Furthermore, children with disabilities aged 6 to 18 years have the right to free education in a neighborhood school or a special school of their choice, as per Section 16 of the Act.",
        "source": "RPWD Act Section 32",
        "chapter": "Education Rights",
        "category": "education",
        "language": "en"
    },
    {
        "text": "Assistive technology includes tools like screen readers (NVDA/JAWS), hearing aids, wheelchairs, prosthetics, Braille kits, communication devices, and mobility aids. In India, support is available through ADIP Scheme, ALIMCO, and national institutes. Eligibility and device type depend on disability assessment and income criteria.",
        "source": "ADIP Scheme",
        "chapter": "Assistive Technology",
        "category": "assistive",
        "language": "en"
    },
    {
        "text": "Under Section 34 of the RPWD Act, 2016, at least 4% reservation is provided in government establishments for persons with benchmark disabilities. This includes identified posts across disability categories. You can apply through regular recruitment notifications and claim applicable relaxation/support under RPWD rules.",
        "source": "RPWD Act Section 34",
        "chapter": "Employment Reservation",
        "category": "job_reservation",
        "language": "en"
    },
    {
        "text": "The UDID (Unique Disability ID) card is a single document for identification and verification of persons with disabilities for availing various government benefits and schemes. It is valid across India and eliminates the need for multiple certificates for different purposes.",
        "source": "UDID Project",
        "chapter": "Disability Identity",
        "category": "schemes",
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
        "text": "Reservation Percentages for Benchmark Disabilities in Government Jobs (Section 34 of RPWD Act 2016):\n\n| Category of Disability | Reservation Percentage |\n| :--- | :--- |\n| (a) Blindness and low vision | 1% |\n| (b) Deaf and hard of hearing | 1% |\n| (c) Locomotor disability including cerebral palsy, leprosy cured, dwarfism, acid attack victims and muscular dystrophy | 1% |\n| (d) Autism, intellectual disability, specific learning disability and mental illness; (e) Multiple disabilities from amongst persons under clauses (a) to (d) including deaf-blindness | 1% |\n| **Total Reservation** | **4%** |",
        "source": "RPWD Act 2016 Section 34",
        "chapter": "Employment Reservation Table",
        "category": "job_reservation",
        "language": "en"
    },
    {
        "text": "The UDID card contains vital information for persons with disabilities. Below is a visual representation of the UDID Card features:\n![UDID Card Features](https://www.swavlambancard.gov.in/images/udid-card-sample.png)\nThe card includes the name, photo, disability type, percentage of disability, and a unique 18-digit enrollment number.",
        "source": "UDID Portal",
        "chapter": "UDID Card Visual Guide",
        "category": "schemes",
        "language": "en"
    },
    {
        "text": "Assistive Devices available under ADIP Scheme:\n\n| Disability Category | Type of Assistive Device |\n| :--- | :--- |\n| Visual Impairment | Braille kits, Smart Canes, Screen Reading Software |\n| Hearing Impairment | Digital Hearing Aids, TDD/TTY devices |\n| Locomotor Disability | Tricycles, Wheelchairs, Crutches, Artificial Limbs |\n| Leprosy Cured | ADL Kits, Protective Footwear |",
        "source": "ADIP Scheme Guidelines",
        "chapter": "Assistive Devices Table",
        "category": "assistive",
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
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading model: {EMBEDDING_MODEL_NAME}")
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.warning(f"Unable to load embedding model for ingestion: {e}")
            self.model = None

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