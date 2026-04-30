"""
RAG Ingestion Pipeline
- Loads PDF documents (RPWD Act, WHO Disability data) 
- Splits into chunks using RecursiveCharacterTextSplitter
- Embeds using all-MiniLM-L6-v2 (384 dim)
- Stores in FAISS vector database
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ─── Disability Knowledge Base ─────────────────────────────────────────────────

DISABILITY_KNOWLEDGE_BASE = [
    # RPWD Act - Rights & Definitions
    {
        "text": "The Rights of Persons with Disabilities Act, 2016 (RPWD Act) is India's landmark disability legislation enacted on December 16, 2016. It received Presidential assent on December 27, 2016 and came into force on April 19, 2017 (Act 49 of 2016). It replaces the Persons with Disabilities Act of 1995.",
        "category": "rpwd_act", "source": "RPWD Act 2016", "chapter": "Overview"
    },
    {
        "text": "The RPWD Act 2016 recognizes 21 specified disabilities, expanded from 7 under the 1995 Act. These include: Blindness, Low Vision, Leprosy Cured, Hearing Impairment (Deaf), Hard of Hearing, Speech & Language Disability, Locomotor Disability, Dwarfism, Intellectual Disability, Specific Learning Disabilities, Autism Spectrum Disorder, Mental Illness, Cerebral Palsy, Muscular Dystrophy, Chronic Neurological Conditions, Multiple Sclerosis, Haemophilia, Thalassemia, Sickle Cell Disease, Acid Attack Victims, and Parkinson's Disease.",
        "category": "rpwd_act", "source": "RPWD Act 2016", "chapter": "Chapter 10 - 21 Disabilities"
    },
    {
        "text": "Under RPWD Act 2016, Benchmark Disability means a person with not less than 40% of a specified disability as certified by a certifying authority. Only persons with benchmark disability are entitled to reservations in government jobs and higher education.",
        "category": "rpwd_act", "source": "RPWD Act 2016", "chapter": "Definitions"
    },
    {
        "text": "Employment Rights under RPWD Act 2016: Minimum 4% reservation in government vacancies for persons with benchmark disabilities (increased from 3% in 1995 Act). Breakdown: 1% each for persons with blindness/low vision; deaf/hard of hearing; locomotor disability/cerebral palsy/leprosy cured/dwarfism/acid attack victims; and 1% for persons with intellectual/mental illness/multiple disabilities.",
        "category": "employment", "source": "RPWD Act 2016", "chapter": "Section 34 - Reservation"
    },
    {
        "text": "Higher Education Reservation under RPWD Act 2016: Minimum 5% seats reserved for persons with benchmark disabilities in all government educational institutions and institutions receiving government aid. Students also receive scholarships and fee waivers.",
        "category": "education", "source": "RPWD Act 2016", "chapter": "Section 32 - Higher Education"
    },

    # Definitions of Disabilities - Visual Impairment
    {
        "text": "Blindness (Visual Impairment Definition under RPWD Act 2016): Blindness is complete absence of sight. A person is considered legally blind if their vision is 6/60 (20/200) or worse in the better eye with correction, or their visual field is restricted to 20 degrees or less. Blindness can be congenital (from birth) or acquired due to injury, disease (cataract, glaucoma, retinitis pigmentosa, diabetic retinopathy), or aging. Common causes in India include cataracts, corneal scarring, retinal diseases, and optic nerve damage.",
        "category": "disability_definitions", "source": "RPWD Act 2016 + WHO ICD-10", "chapter": "Blindness Definition"
    },
    {
        "text": "Low Vision (Visual Impairment Definition under RPWD Act 2016): Low vision means reduced visual acuity that cannot be corrected by glasses or surgery but preserves some functional vision. Defined as vision worse than 6/18 but not less than 6/60 (or 20/60 to 20/200) in the better eye with correction, or visual field restricted from 20-40 degrees. Persons with low vision can use magnification devices, large print materials, and modified lighting to read and navigate. Causes include macular degeneration, diabetic retinopathy, cataracts, and refractive errors.",
        "category": "disability_definitions", "source": "RPWD Act 2016 + WHO", "chapter": "Low Vision Definition"
    },
    {
        "text": "Hearing Impairment (Deaf and Hard of Hearing Definition under RPWD Act 2016): Deafness is substantial/permanent loss of hearing resulting in inability to hear and understand sound and speech. Hard of Hearing refers to partial/fluctuating hearing loss not corrected by hearing aids, leading to difficulty in communication. Deaf and hard of hearing are separate disabilities recognized under RPWD Act. Hearing loss measured in decibels (dB): Normal (0-20dB), Mild (20-40dB), Moderate (40-60dB), Severe (60-80dB), Profound (>80dB). Causes include congenital absence, infections (meningitis, otitis media), aging, noise exposure, ototoxic drugs.",
        "category": "disability_definitions", "source": "RPWD Act 2016 + WHO ICD-10", "chapter": "Hearing Impairment Definition"
    },
    {
        "text": "Speech and Language Disability (Definition under RPWD Act 2016): Permanent disability of speech and/or language affecting communication ability regardless of intelligibility. Includes: Dysarthria (speech motor control problems), Apraxia (speech planning disorder), Voice disorders (pitch/quality abnormalities), Fluency disorders (stammering/stuttering), Receptive/Expressive language disorders. Can result from neurological damage (stroke, cerebral palsy, Parkinson's), structural abnormalities (cleft palate), hearing loss, intellectual disability, or autism. Affects social participation, education, and employment.",
        "category": "disability_definitions", "source": "RPWD Act 2016", "chapter": "Speech & Language Definition"
    },

    # Accessibility Provisions 
    {
        "text": "Accessibility provisions under RPWD Act (Sections 40-46): All government buildings must be accessible within a prescribed timeframe. Accessible roads, transport and infrastructure required. All public websites must comply with accessibility standards (WCAG). Public and private service providers must comply with accessibility standards.",
        "category": "accessibility", "source": "RPWD Act 2016", "chapter": "Sections 40-46"
    },
    {
        "text": "Rights under RPWD Act: Right to Equality and Non-Discrimination (Section 3). No discrimination on grounds of disability. Right to Life with Dignity. Protection from torture, cruel or inhuman treatment. Right to Community Inclusion. Reproductive Rights (right to decide family size, retain fertility). Right to Legal Capacity (Section 14). Right to Vote in accessible polling stations. Right to Own Property.",
        "category": "rights", "source": "RPWD Act 2016", "chapter": "Chapter 2 - Rights"
    },
    {
        "text": "Education Rights under RPWD Act (Sections 16-18): Right to free and compulsory education in neighborhood schools. 5% reservation in higher education. Accommodation in examinations: extra time (minimum 20 minutes per hour), scribe facility, use of assistive devices. Barrier-free accessible school buildings. Appointment of Special Educators. Home-based education for those unable to attend school.",
        "category": "education", "source": "RPWD Act 2016", "chapter": "Sections 16-18"
    },

    # Government Schemes
    {
        "text": "Accessible India Campaign (Sugamya Bharat Abhiyan): Launched in 2015 by Government of India. Aims to make built environment, transportation systems, and ICT ecosystem accessible for persons with disabilities. Targets: 50% government buildings fully accessible, all airports and railway stations accessible, all government websites WCAG 2.0 compliant.",
        "category": "schemes", "source": "Government of India", "chapter": "Accessibility Schemes"
    },
    {
        "text": "ADIP Scheme (Assistance to Disabled Persons for Purchase/Fitting of Aids and Appliances): Free/subsidized assistive devices to PwDs below income threshold (Rs. 20,000/month). Devices include: wheelchairs, hearing aids, white canes, Braille kits, tricycles, prosthetic limbs, orthotic appliances, DAISY players for the blind, communication aids.",
        "category": "schemes", "source": "DEPwD, Ministry of Social Justice", "chapter": "ADIP Scheme"
    },
    {
        "text": "SIPDA (Scheme for Implementation of Persons with Disabilities Act): Provides financial assistance to state governments and NGOs for implementing RPWD Act. Funds accessible infrastructure, awareness programs, and disability-related services.",
        "category": "schemes", "source": "DEPwD", "chapter": "SIPDA"
    },
    {
        "text": "DDRS (Deendayal Disabled Rehabilitation Scheme): Financial support to NGOs providing rehabilitation services for persons with disabilities. Covers special schools, vocational training centers, early intervention centers, and halfway homes.",
        "category": "schemes", "source": "DEPwD", "chapter": "DDRS"
    },
    {
        "text": "NHFDC (National Handicapped Finance and Development Corporation): Provides loans at concessional rates for self-employment and entrepreneurship by persons with disabilities. Loan amounts up to Rs. 30 lakh. Interest rates starting from 5% per annum.",
        "category": "schemes", "source": "DEPwD", "chapter": "NHFDC"
    },
    {
        "text": "Indira Gandhi National Disability Pension Scheme (IGNDPS): Monthly pension for below poverty line persons with disabilities. Amount: Rs. 300-500 per month (state governments may add additional amounts). Eligibility: BPL, age 18-79, 80% or more disability.",
        "category": "schemes", "source": "Ministry of Rural Development", "chapter": "Pension Schemes"
    },
    {
        "text": "UDID (Unique Disability Identity Card / Swavalamban Card): National database of persons with disabilities. Provides standardized disability certificate. Apply at: www.swavlambancard.gov.in. Required for accessing government reservations and benefits. Contains embedded chip with disability information.",
        "category": "schemes", "source": "DEPwD", "chapter": "UDID System"
    },
    {
        "text": "National Programme for Control of Blindness (NPCB): Free cataract surgeries, free glasses to school children with low vision, free treatment for corneal blindness. Vision centers at PHC level. Free IOL implants. Pre and post operative care. Objectives: eliminate avoidable blindness, reduce visual impairment burden, provide rehabilitation services. Website: npcb.nic.in",
        "category": "schemes", "source": "Ministry of Health", "chapter": "Health Schemes"
    },
    {
        "text": "Special Benefits for Persons with Visual Impairment (Blindness/Low Vision) under RPWD Act 2016: Free Braille kits and Braille learning materials via ADIP scheme. Free white canes and orientation & mobility training. Priority in employment (1% quota in government jobs for blind/low vision). Free assistive devices including screen readers, DAISY players, magnification software. Tax exemptions on reading materials and books. Exemption from property/entertainment taxes in several states.",
        "category": "benefits", "source": "RPWD Act 2016 + State Governments", "chapter": "Visual Impairment Benefits"
    },
    {
        "text": "Education and Employment Support for Persons with Visual Impairment: NIRD provides training in Braille, orientation & mobility, daily living skills. IIT training programs for IT skills for blind persons. NIEDB (National Institute for Empowerment of Blind) at Dehradun offers rehabilitation training. Government jobs: 4% quota (1% for blind/low vision). 20 minutes extra time per hour in exams. Use of screen reader or scribe in examinations. Audio books and Braille materials provided free.",
        "category": "education", "source": "NIEDB + NIRD + Government", "chapter": "VI Education Support"
    },
    {
        "text": "Free Assistive Technology and Rehabilitation for Visual Impairment: National Institute for the Visually Handicapped (NIVH), Dehradun provides free training in: Braille reading/writing, typing, computer skills with screen readers, orientation and mobility, rehabilitation services, low vision aids training. Free screen reader software: NVDA (open source), JAWS (subsidized for students). DAISY players for audio books. Free training provided across India through state centers.",
        "category": "assistive_tech", "source": "NIVH / DeEmpowerment", "chapter": "VI Assistive Tech"
    },
    {
        "text": "NIEDB (National Institute for Empowerment of Blind/Low Vision Persons): Autonomous organization under Ministry of Social Justice working since 1989. Offers rehabilitation, vocational training, IT training, Braille education. Conducts research on visual impairment. Provides free counseling and guidance. Centers in Delhi, Kolkata, and regional programs. 100% scholarship for courses. Website: niedb.org. Email: niedb@niedb.org. Phone: 011-41591234",
        "category": "organizations", "source": "DEPwD", "chapter": "NIEDB Services"
    },
    {
        "text": "NPPCD (National Programme for Prevention and Control of Deafness): Free hearing aids through government hospitals and audiological and speech therapy centers. Newborn hearing screening. School hearing screening. Advanced hearing aids for severe/profound hearing loss.",
        "category": "schemes", "source": "Ministry of Health", "chapter": "Health Schemes"
    },
    {
        "text": "National Trust Act 1999: For welfare of persons with Autism, Cerebral Palsy, Intellectual Disability and Multiple Disabilities. Key schemes: SAMARTH (care homes), GHARAUNDA (group homes for adults), NIRAMAYA (health insurance up to Rs. 1 lakh), PRERNA (early intervention), SAMBHAV (assistive device camps), VIKAAS (day care). Apply at: nationaltrust.gov.in",
        "category": "schemes", "source": "National Trust, MSJE", "chapter": "National Trust Schemes"
    },
    {
        "text": "Samagra Shiksha Inclusive Education Component: Individualized Education Plans (IEPs) for children with disabilities. Resource rooms with therapy equipment. Home-based education for severely disabled children. Braille kits, hearing aids, and assistive devices. Orientation and mobility training. Rs. 3500 per child per year for disability-related needs.",
        "category": "education", "source": "Ministry of Education", "chapter": "Samagra Shiksha"
    },

    # Assistive Technology
    {
        "text": "Assistive Technology for Visual Impairment: Screen readers (NVDA - free, JAWS - paid, Voiceover for Apple). Braille displays and Braille printers. DAISY players for audio books. Magnification software (ZoomText, Windows Magnifier). White cane (provided free under ADIP scheme). Optical Character Recognition (OCR) software. Available through ADIP scheme and National Institute for Visually Handicapped, Dehradun.",
        "category": "assistive_tech", "source": "WHO / NIEPVD", "chapter": "Visual AT"
    },
    {
        "text": "Assistive Technology for Hearing Impairment: Hearing aids (BTE, ITE, RIC types - free under ADIP/NPPCD). Cochlear implants (government schemes for children). FM systems for classroom use. Video relay services and captioning. Sign language interpretation. Vibrating alert systems. TTY/TDD devices. AYJNIHH Mumbai provides free hearing aids.",
        "category": "assistive_tech", "source": "WHO / AYJNIHH", "chapter": "Hearing AT"
    },
    {
        "text": "Assistive Technology for Locomotor Disability: Wheelchairs (manual and motorized - free under ADIP). Prosthetic limbs (free at ALIMCO centers). Orthotic appliances (calipers, splints). Crutches and walking frames. Stair lifts and ramps. Modified vehicles (tax exemption available). Tricycles with hand pedals. Available through ALIMCO (Artificial Limbs Manufacturing Corporation of India).",
        "category": "assistive_tech", "source": "ALIMCO / DEPwD", "chapter": "Locomotor AT"
    },
    {
        "text": "Assistive Technology for Intellectual and Learning Disabilities: Augmentative and Alternative Communication (AAC) devices. Symbol-based communication boards (low-tech). Speech generating devices (high-tech). Special education software (Boardmaker, LetMeTalk). Modified keyboards and mice. Text-to-speech software. Reading pens. Graphic organizers and visual schedules.",
        "category": "assistive_tech", "source": "WHO / NIEPID", "chapter": "Intellectual/Learning AT"
    },
    {
        "text": "ALIMCO (Artificial Limbs Manufacturing Corporation of India): Government enterprise manufacturing and distributing assistive devices. Products: prosthetic limbs, orthotic appliances, wheelchairs, hearing aids, Braille kits, tricycles. Devices provided free/subsidized through ADIP scheme. Centers across India. Website: www.alimco.in",
        "category": "assistive_tech", "source": "ALIMCO", "chapter": "ALIMCO"
    },

    # Accessibility in Public Spaces
    {
        "text": "Accessibility Standards for Buildings (IS 9954:2020 / Harmonized Guidelines 2016): Ramps: minimum width 1.5m, slope 1:12. Accessible parking: minimum 2% of total spaces. Accessible toilets mandatory in all public buildings. Door width minimum 900mm. Lifts with Braille buttons and audio announcements. Tactile pathways from entrance to key facilities. Handrails on both sides of stairs.",
        "category": "accessibility", "source": "BIS / MoHUA", "chapter": "Building Accessibility"
    },
    {
        "text": "Digital Accessibility Standards: All central government websites must comply with GIGW (Guidelines for Indian Government Websites) and WCAG 2.0 Level AA. Key requirements: alt text for images, keyboard navigation, sufficient color contrast (4.5:1 ratio), captions for videos, screen reader compatibility. NIC provides audit services.",
        "category": "accessibility", "source": "GIGW / NIC", "chapter": "Digital Accessibility"
    },
    {
        "text": "Transport Accessibility in India: Indian Railways: accessible coaches in all trains, accessible toilets, reserved seats near entrance. Priority boarding for PwDs. Buses: CRPF mandates low-floor buses. Metro: all stations must have lifts, tactile tiles, audio announcements. Aviation: DGCA mandates accessible airports, wheelchair assistance free of charge, no extra fare for mobility aids.",
        "category": "accessibility", "source": "MoRTH / Railways / DGCA", "chapter": "Transport Accessibility"
    },

    # WHO Global Data
    {
        "text": "WHO Global Disability Statistics (2023): 1.3 billion people (16% of world population) live with significant disability - world's largest minority group. 80% live in low and middle income countries. 46% of people aged 60+ have disability. 240 million children with disabilities worldwide. Persons with disabilities are 3x more likely to be denied healthcare.",
        "category": "who_data", "source": "WHO 2023", "chapter": "Global Statistics"
    },
    {
        "text": "India Disability Statistics (Census 2011): 26.8 million persons with disabilities (2.21% of population - likely underestimate). WHO estimates actual prevalence 10-20%. Distribution: Locomotor 20.3%, Hearing 18.9%, Vision 18.8%, Other 17.9%, Multiple 8.3%, Speech 7.5%, Intellectual 5.6%, Mental illness 2.7%. 69% in rural areas. Literacy rate 54.5% vs national 74%.",
        "category": "statistics", "source": "Census of India 2011", "chapter": "India Data"
    },

    # Educational Accommodations
    {
        "text": "Examination Accommodations for Students with Disabilities in India: CBSE/UGC/UPSC provisions: Extra time (20 minutes per hour for benchmark disability). Scribe (writer) facility. Use of assistive devices (calculator for dyscalculia, magnifiers for low vision). Question papers in accessible formats (large print, Braille). Separate examination room. Compensatory time for those using scribes.",
        "category": "education", "source": "CBSE / UGC / UPSC", "chapter": "Exam Accommodations"
    },
    {
        "text": "University Disability Cells (Equal Opportunity Cells): All universities must have Equal Opportunity Cells under RPWD Act. Functions: identify and enroll students with disabilities, ensure reasonable accommodations, resolve grievances, liaise with departments. Contact the Disability/Equal Opportunity Cell of your university for: fee concessions, accessible hostels, assistive devices, exam accommodations.",
        "category": "education", "source": "UGC Guidelines 2012", "chapter": "University Accommodations"
    },

    # Grievance and Legal
    {
        "text": "Grievance Redressal under RPWD Act: Chief Commissioner for Persons with Disabilities (national level) - File complaint at ccdisabilities.nic.in. State Commissioner for Persons with Disabilities (state level). Liaison Officers in every government establishment. High Courts/Supreme Court via writ petitions. National Human Rights Commission. Limitation: 1 year from date of grievance.",
        "category": "legal", "source": "RPWD Act 2016 Sections 60-73", "chapter": "Grievance Redressal"
    },
    {
        "text": "Legal Provisions for Employment Discrimination under RPWD Act: Sections 20-24. Equal Opportunity Policy mandatory for all establishments. Employer cannot discriminate in recruitment, promotion, or service conditions on grounds of disability. Penalty for contravention: fine up to Rs. 10,000 (first offence), Rs. 50,000-5 lakh (subsequent). File complaint with Liaison Officer or State/Chief Commissioner.",
        "category": "legal", "source": "RPWD Act 2016", "chapter": "Employment Legal"
    },
    {
        "text": "National Institutes for Disability Services in India: NIEPMD Chennai (Multiple Disabilities), SVNIRTAR Odisha (Locomotor), NILD Kolkata (Locomotor), NIHH Mumbai (Hearing), AYJNIHH Mumbai (Speech & Hearing), NIEPVD Dehradun (Visual), NIMHR Secunderabad (Intellectual), NIEPID Secunderabad (Intellectual Disability). All provide free/subsidized rehabilitation, training, and assistive devices.",
        "category": "resources", "source": "DEPwD", "chapter": "National Institutes"
    },
    {
        "text": "Disability Certificate Process in India: Step 1: Visit government hospital with disability specialist. Step 2: Medical Board assesses disability type and percentage. Step 3: Certificate issued with UDID number. Step 4: Register on swavlambancard.gov.in for digital UDID card. Required documents: identity proof, address proof, medical reports. Processing time: 30-60 days. Valid for lifetime (some may require renewal).",
        "category": "schemes", "source": "DEPwD / State Governments", "chapter": "Disability Certificate"
    },
    {
        "text": "Tax Benefits for Persons with Disabilities in India: Income Tax Act Section 80U: Deduction of Rs. 75,000 (40-80% disability) or Rs. 1,25,000 (severe disability 80%+) for persons with disability. Section 80DD: Deduction up to Rs. 1,25,000 for dependent with severe disability. GST exemption on assistive devices (wheelchairs, hearing aids, Braille watches). Customs duty exemption on imported assistive devices.",
        "category": "schemes", "source": "Income Tax Act / GST", "chapter": "Tax Benefits"
    },
    {
        "text": "State Government Disability Schemes vary across India. Common benefits include: disability pension (Rs. 500-3000/month by state), free bus travel, free train travel (50-75% concession), free/subsidized housing under state schemes, subsidized loans, priority in BPL ration cards. Contact State Commissioner for Persons with Disabilities or State Welfare Department for state-specific schemes.",
        "category": "schemes", "source": "State Governments", "chapter": "State Schemes"
    },
    {
        "text": "National Mission on Sickle Cell Anaemia (2023): Launched to address high burden among tribal populations. Free screening, counseling, and treatment. Targets 7 crore people in high-prevalence states. States covered: Madhya Pradesh, Rajasthan, Gujarat, Maharashtra, Chhattisgarh, Jharkhand, Odisha, Uttarakhand, Uttar Pradesh, Tamil Nadu, Andhra Pradesh, Karnataka, Assam, West Bengal.",
        "category": "schemes", "source": "Ministry of Health 2023", "chapter": "Sickle Cell Mission"
    },
]

# ─── Embedding & Indexing ────────────────────────────────────────────────────────

class DisabilityRAGIngestion:
    def __init__(self, index_path: str = "./faiss_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.model: Optional[SentenceTransformer] = None
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def load_embedding_model(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")

    def load_pdf_document(self, pdf_path: str) -> List[Dict]:
        """Load and chunk PDF document"""
        chunks = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                full_text += f"\n\n{page_text}"
            chunks = self._chunk_text(full_text, source=Path(pdf_path).name)
            logger.info(f"Loaded PDF: {pdf_path} -> {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_path}: {e}")
        return chunks

    def _chunk_text(self, text: str, source: str = "document") -> List[Dict]:
        """Recursive character text splitting"""
        chunks = []
        separators = ["\n\n", "\n", ". ", " ", ""]
        
        def split_by_separators(text: str, seps: List[str], chunk_size: int, overlap: int) -> List[str]:
            if not seps or len(text) <= chunk_size:
                return [text] if text.strip() else []
            sep = seps[0]
            parts = text.split(sep)
            result = []
            current = ""
            for part in parts:
                if len(current) + len(part) + len(sep) <= chunk_size:
                    current += (sep if current else "") + part
                else:
                    if current:
                        result.append(current)
                    if len(part) > chunk_size:
                        sub = split_by_separators(part, seps[1:], chunk_size, overlap)
                        result.extend(sub)
                        current = ""
                    else:
                        current = part
            if current:
                result.append(current)
            return result

        raw_chunks = split_by_separators(text, separators, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text.strip(),
                    "metadata": {
                        "source": source,
                        "chunk_index": i,
                        "category": "rpwd_pdf",
                        "chapter": "RPWD/WHO Reference"
                    }
                })
        return chunks

    def ingest_all(self, docs_dir: str = None, include_knowledge_base: bool = True):
        """Main ingestion pipeline"""
        if self.model is None:
            self.load_embedding_model()

        all_chunks = []

        # 1. Ingest built-in knowledge base
        if include_knowledge_base:
            for item in DISABILITY_KNOWLEDGE_BASE:
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": item["text"],
                    "metadata": {
                        "source": item["source"],
                        "category": item["category"],
                        "chapter": item.get("chapter", "")
                    }
                })
            logger.info(f"Loaded {len(DISABILITY_KNOWLEDGE_BASE)} knowledge base entries")

        # 2. Ingest PDF documents
        if docs_dir:
            docs_path = Path(docs_dir)
            if docs_path.exists():
                for pdf_file in docs_path.glob("*.pdf"):
                    pdf_chunks = self.load_pdf_document(str(pdf_file))
                    all_chunks.extend(pdf_chunks)

        self.chunks = all_chunks
        logger.info(f"Total chunks to index: {len(all_chunks)}")

        # 3. Generate embeddings
        texts = [c["text"] for c in all_chunks]
        logger.info("Generating embeddings...")
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        logger.info(f"Generated embeddings shape: {self.embeddings.shape}")

        # 4. Build FAISS index
        self._build_faiss_index()

        # 5. Save metadata
        self._save_metadata()

        logger.info("Ingestion complete!")
        return len(all_chunks)

    def _build_faiss_index(self):
        """Build and save FAISS index"""
        try:
            import faiss
            dim = self.embeddings.shape[1]
            # Use IndexFlatIP for cosine similarity (with normalized vectors)
            index = faiss.IndexFlatIP(dim)
            index.add(self.embeddings.astype(np.float32))
            faiss.write_index(index, str(self.index_path / "index.faiss"))
            logger.info(f"FAISS index built with {index.ntotal} vectors (dim={dim})")
        except ImportError:
            logger.warning("FAISS not available, saving numpy arrays as fallback")
            np.save(str(self.index_path / "embeddings.npy"), self.embeddings)

    def _save_metadata(self):
        """Save chunk metadata to JSON"""
        metadata = [
            {"id": c["id"], "text": c["text"], "metadata": c["metadata"]}
            for c in self.chunks
        ]
        with open(self.index_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata for {len(metadata)} chunks")


def run_ingestion(docs_dir: str = None, index_path: str = "./faiss_index"):
    """Entry point for ingestion"""
    logging.basicConfig(level=logging.INFO)
    pipeline = DisabilityRAGIngestion(index_path=index_path)
    count = pipeline.ingest_all(docs_dir=docs_dir, include_knowledge_base=True)
    return count


if __name__ == "__main__":
    docs_path = Path(__file__).parent.parent / "data" / "docs"
    idx_path = Path(__file__).parent.parent.parent / "faiss_index"
    run_ingestion(docs_dir=str(docs_path), index_path=str(idx_path))
