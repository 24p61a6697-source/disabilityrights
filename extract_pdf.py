from pypdf import PdfReader
import os

pdf_path = r"c:\Users\kumma\OneDrive\Desktop\Downloads\disability-rights-guide\backend\app\data\docs\Disability_WHO_RPWD_Thesis_Reference__1_.pdf"
if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    exit(1)

reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(f"Extracted {len(text)} characters.")
print("Preview:")
print(text[:2000])

with open("extracted_pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
