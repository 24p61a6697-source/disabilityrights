import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional


class LocalVectorStore:
    """A very small local vector store using numpy arrays and JSON metadata.

    Data layout under directory `path`:
      - vectors.npy : float32 array shape (N, D)
      - ids.json : list of ids
      - metadata.json : list of metadata dicts
    """

    def __init__(self, path: str = "db/local_vectorstore"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.vectors_file = self.path / "vectors.npy"
        self.ids_file = self.path / "ids.json"
        self.meta_file = self.path / "metadata.json"
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if self.vectors_file.exists():
            self._vectors = np.load(self.vectors_file)
            self._ids = json.loads(self.ids_file.read_text())
            self._meta = json.loads(self.meta_file.read_text())
            self._dim = int(self._vectors.shape[1])
        else:
            self._vectors = np.zeros((0, 0), dtype=np.float32)
            self._ids = []
            self._meta = []
            self._dim = None
        self._loaded = True

    def save(self):
        if self._vectors is None:
            return
        np.save(self.vectors_file, self._vectors)
        self.ids_file.write_text(json.dumps(self._ids))
        self.meta_file.write_text(json.dumps(self._meta))

    def upsert(self, items: List[Dict[str, Any]]):
        """Upsert items: each item is {"id": str, "vector": list[float], "metadata": dict}
        """
        self.load()
        for it in items:
            vid = it["id"]
            vec = np.array(it["vector"], dtype=np.float32)
            meta = it.get("metadata", {})
            if self._dim is None:
                self._dim = vec.shape[0]
            if vec.shape[0] != self._dim:
                raise ValueError("Vector dimension mismatch")

            if vid in self._ids:
                idx = self._ids.index(vid)
                self._vectors[idx] = vec
                self._meta[idx] = meta
            else:
                if self._vectors.size == 0:
                    self._vectors = vec.reshape(1, -1)
                else:
                    self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])
                self._ids.append(vid)
                self._meta.append(meta)

        self.save()

    def search(self, query_vector: List[float], k: int = 5):
        """Return top-k nearest items by cosine similarity as list of dicts {id,score,metadata}"""
        self.load()
        if self._vectors is None or len(self._ids) == 0:
            return []
        q = np.array(query_vector, dtype=np.float32)
        if q.shape[0] != self._dim:
            # try to normalize/resize by truncation or padding
            if q.shape[0] > self._dim:
                q = q[: self._dim]
            else:
                q = np.pad(q, (0, self._dim - q.shape[0]))

        # cosine similarity
        dots = self._vectors @ q
        norms = np.linalg.norm(self._vectors, axis=1) * (np.linalg.norm(q) + 1e-12)
        sims = dots / (norms + 1e-12)
        topk_idx = np.argsort(-sims)[:k]
        results = []
        for idx in topk_idx:
            results.append({
                "id": self._ids[idx],
                "score": float(sims[idx]),
                "metadata": self._meta[idx],
            })
        return results
