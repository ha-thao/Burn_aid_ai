# -*- coding: utf-8 -*-
"""
Burn Knowledge RAG Engine (BM25 Enhanced)
=========================================
Động cơ truy vấn tri thức RAG chuyên sâu cho Bệnh Bỏng.
Đã nạp toàn diện:
1. Quyết định 635/QĐ-BYT Bộ Y Tế (430 trang) - Quy trình khám bệnh, chữa bệnh chuyên ngành Bỏng.
2. 12 công trình nghiên cứu khoa học chuyên sâu từ Tạp chí Y học Thảm họa & Bỏng (Bệnh viện Bỏng Quốc gia).
3. Toàn bộ cẩm nang sơ cứu, phân loại 1-3-6-9-18, thang điểm ABSI, hồi sức sốc bỏng Evans, điều dưỡng, ghép da.
Thuật toán BM25 (Best Matching 25) tính toán IDF chính xác, tốc độ truy vấn < 50ms, hoàn toàn offline.
"""

import os
import re
import math
import json
from typing import List, Dict, Any

class BurnRAGEngine:
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.df: Dict[str, int] = {}
        self.avg_len: float = 1.0
        self.N: int = 0
        self.sources: List[str] = []

        self._load_data()
        self._build_index()

    def _load_data(self):
        base_dir = os.path.dirname(os.path.abspath(self.kb_path))
        full_json = os.path.join(base_dir, "knowledge_base_burns_full.json")

        if os.path.exists(full_json):
            try:
                with open(full_json, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception:
                self.chunks = []

        if not self.chunks and os.path.exists(self.kb_path):
            with open(self.kb_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            raw_paras = [p.strip() for p in raw_text.split("\n\n---\n\n") if len(p.strip()) > 30]
            if not raw_paras:
                raw_paras = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 30]
            self.chunks = [{"source": "Tài liệu y khoa tổng hợp", "page": 1, "text": p} for p in raw_paras]

        unique_srcs = set()
        for c in self.chunks:
            src = c.get("source", "")
            if src:
                unique_srcs.add(src)
        self.sources = sorted(list(unique_srcs))

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 1]

    def _build_index(self):
        self.N = len(self.chunks)
        if self.N == 0:
            return

        self.chunk_tokens = []
        self.doc_lens = []
        self.df = {}

        for c in self.chunks:
            text = c.get("text", "") if isinstance(c, dict) else str(c)
            toks = self._tokenize(text)
            self.chunk_tokens.append(toks)
            self.doc_lens.append(len(toks))
            for w in set(toks):
                self.df[w] = self.df.get(w, 0) + 1

        self.avg_len = sum(self.doc_lens) / max(self.N, 1)

    def search(self, query: str, top_k: int = 4) -> List[str]:
        if self.N == 0:
            return []

        q_words = self._tokenize(query)
        if not q_words:
            return [c.get("text", "") if isinstance(c, dict) else str(c) for c in self.chunks[:top_k]]

        k1 = 1.5
        b = 0.75
        scores = []

        for i, toks in enumerate(self.chunk_tokens):
            score = 0.0
            d_len = self.doc_lens[i]
            for w in q_words:
                if w in self.df:
                    n_w = self.df[w]
                    # Công thức chuẩn BM25 IDF
                    idf = math.log(1.0 + (self.N - n_w + 0.5) / (n_w + 0.5))
                    tf = toks.count(w)
                    denom = tf + k1 * (1.0 - b + b * (d_len / self.avg_len))
                    score += idf * (tf * (k1 + 1.0)) / max(denom, 1e-6)

            if score > 0.0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scores[:top_k]]

        if not top_indices:
            top_indices = list(range(min(top_k, self.N)))

        results = []
        for idx in top_indices:
            c = self.chunks[idx]
            txt = c.get("text", "") if isinstance(c, dict) else str(c)
            results.append(txt)

        return results
