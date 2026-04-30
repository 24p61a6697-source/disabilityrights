import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

try:
    print("Testing imports...")
    from app.rag.retrieval import DisabilityRAGPipeline
    from app.rag.ingestion import run_ingestion
    print("Imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
