def search(self, query_vector: List[float], k: int = 5, threshold: float = 0.65):
    self.load()

    if self._vectors is None or len(self._ids) == 0:
        return []

    q = np.array(query_vector, dtype=np.float32)

    if q.shape[0] != self._dim:
        raise ValueError("Query vector dimension mismatch")

    # normalize query
    q = q / (np.linalg.norm(q) + 1e-12)

    sims = self._vectors @ q

    # sort by similarity
    sorted_idx = np.argsort(-sims)

    results = []
    for i in sorted_idx:
        score = float(sims[i])

        # 🚨 HARD FILTER
        if score < threshold:
            continue

        meta = self._meta[i]

        # 🚨 VALIDATE METADATA
        if not meta or "text" not in meta:
            continue

        results.append({
            "id": self._ids[i],
            "score": score,
            "metadata": meta,
        })

        if len(results) >= k:
            break

    return results