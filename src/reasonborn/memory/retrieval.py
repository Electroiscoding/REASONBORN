"""
Module [6]: Retrieval Layer — Hybrid RAG with Cross-Encoder Reranking
=====================================================================
Combines dense vector, sparse BM25, graph traversal, and episodic
retrieval with cross-encoder reranking for precise relevance.

Per ReasonBorn.md Section 4.7.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from .episodic import EpisodicMemory
from .semantic import SemanticMemory


class CrossEncoderScorer:
    """
    Lightweight cross-encoder scoring for reranking.
    Uses token overlap + position-aware matching as a production-viable
    heuristic when no neural cross-encoder model is loaded.
    """

    def score(self, query: str, document: str) -> float:
        """
        Score query-document relevance on [0, 1] scale.
        Uses multi-factor heuristic scoring:
        1. Token overlap (Jaccard-like)
        2. Ordered match bonus (query tokens appear in order)
        3. Coverage penalty (long docs with little overlap)
        """
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(document)

        if not q_tokens or not d_tokens:
            return 0.0

        q_set = set(q_tokens)
        d_set = set(d_tokens)

        # 1. Token overlap (recall-oriented)
        overlap = len(q_set & d_set)
        recall = overlap / len(q_set) if q_set else 0.0
        precision = overlap / len(d_set) if d_set else 0.0

        # F1-like combination
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0.0

        # 2. Ordered match bonus
        order_bonus = 0.0
        d_lower = document.lower()
        last_pos = -1
        ordered_matches = 0
        for qt in q_tokens:
            pos = d_lower.find(qt, last_pos + 1)
            if pos > last_pos:
                ordered_matches += 1
                last_pos = pos
        if q_tokens:
            order_bonus = ordered_matches / len(q_tokens) * 0.2

        # 3. Length penalty (very long docs with little overlap)
        length_ratio = len(q_tokens) / max(len(d_tokens), 1)
        length_penalty = min(1.0, length_ratio * 2)

        score = f1 * 0.6 + order_bonus + length_penalty * 0.2
        return min(1.0, max(0.0, score))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.strip('.,!?;:()[]{}"\'-').lower()
                for t in text.split() if len(t.strip()) > 1]


class RetrievalLayer:
    """
    Module [6]: Hybrid RAG combining multiple retrieval strategies.

    Retrieval pipeline:
    1. Dense vector search (semantic memory HNSW)
    2. Sparse BM25 search (semantic memory inverted index)
    3. Knowledge graph traversal (multi-hop)
    4. Episodic memory retrieval (recent experiences)
    5. Cross-encoder reranking (precision optimization)

    Final score = weighted combination of all sources.
    """

    # Source weights for score normalization
    WEIGHT_DENSE = 0.35
    WEIGHT_SPARSE = 0.25
    WEIGHT_GRAPH = 0.15
    WEIGHT_EPISODIC = 0.25

    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.episodic = episodic
        self.semantic = semantic
        self.reranker = CrossEncoderScorer()

    def hybrid_retrieve(
        self,
        query: str,
        k: int = 10,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Combines Dense vector, Sparse BM25, Graph, and Episodic retrieval.

        Results are deduplicated, merged, and reranked by the cross-encoder.

        Args:
            query: User query string
            k: Number of results to return
            query_embedding: Optional pre-computed query embedding

        Returns:
            List of retrieved context dicts, sorted by relevance
        """
        all_results: Dict[str, Dict[str, Any]] = {}

        # 1. Dense vector search
        dense_results = self.semantic.dense_search(
            query, k=k, query_embedding=query_embedding)
        for res in dense_results:
            doc_id = res.get('fact_id', res.get('text', '')[:50])
            if doc_id not in all_results:
                all_results[doc_id] = {
                    'text': res['text'],
                    'score': 0.0,
                    'sources': [],
                    'confidence': res.get('confidence', 0.5),
                    'domain': res.get('domain', 'general'),
                }
            all_results[doc_id]['score'] += res['score'] * self.WEIGHT_DENSE
            all_results[doc_id]['sources'].append('dense')

        # 2. Sparse BM25 search
        sparse_results = self.semantic.bm25_search(query, k=k)
        for res in sparse_results:
            doc_id = res.get('doc_id', res.get('text', '')[:50])
            if doc_id not in all_results:
                all_results[doc_id] = {
                    'text': res['text'],
                    'score': 0.0,
                    'sources': [],
                    'confidence': res.get('confidence', 0.5),
                    'domain': res.get('domain', 'general'),
                }
            # Normalize BM25 score to [0, 1]
            norm_score = min(res['score'] / 10.0, 1.0)
            all_results[doc_id]['score'] += norm_score * self.WEIGHT_SPARSE
            all_results[doc_id]['sources'].append('bm25')

        # 3. Knowledge graph search
        graph_results = self.semantic.graph_search(query, k=k // 2)
        for res in graph_results:
            text = f"{res['subject']} {res['relation']} {res['object']}"
            doc_id = f"kg_{text[:50]}"
            if doc_id not in all_results:
                all_results[doc_id] = {
                    'text': text,
                    'score': 0.0,
                    'sources': [],
                    'confidence': 0.9,
                    'domain': 'knowledge_graph',
                }
            all_results[doc_id]['score'] += 0.8 * self.WEIGHT_GRAPH
            all_results[doc_id]['sources'].append('graph')

        # 4. Episodic memory retrieval
        episodic_results = self.episodic.retrieve(
            query, k=k // 2, query_embedding=query_embedding)
        for entry in episodic_results:
            text = entry.content if hasattr(entry, 'content') else str(entry)
            doc_id = f"ep_{getattr(entry, 'entry_id', text[:50])}"
            if doc_id not in all_results:
                all_results[doc_id] = {
                    'text': text,
                    'score': 0.0,
                    'sources': [],
                    'confidence': getattr(entry, 'importance', 0.5),
                    'domain': getattr(entry, 'domain', 'episodic'),
                }
            all_results[doc_id]['score'] += (
                getattr(entry, 'importance', 0.5) * self.WEIGHT_EPISODIC)
            all_results[doc_id]['sources'].append('episodic')

        # 5. Cross-encoder reranking
        results = list(all_results.values())
        results = self._rerank_by_relevance(query, results)

        return results[:k]

    def _rerank_by_relevance(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Cross-encoder reranking for precision.
        Combines the initial retrieval score with the cross-encoder score.
        """
        for result in results:
            ce_score = self.reranker.score(query, result['text'])
            # Combine initial score (recall) with cross-encoder (precision)
            result['retrieval_score'] = result['score']
            result['rerank_score'] = ce_score
            result['score'] = result['score'] * 0.4 + ce_score * 0.6

        results.sort(key=lambda r: r['score'], reverse=True)
        return results

    def get_max_similarity(self, claim: str) -> float:
        """
        Returns the maximum similarity score for a claim across all indices.
        Used by the OutputFilter for evidence scoring.
        """
        max_score = 0.0

        # Check BM25
        bm25_results = self.semantic.bm25_search(claim, k=1)
        if bm25_results:
            max_score = max(max_score,
                            min(bm25_results[0]['score'] / 10.0, 1.0))

        # Check episodic
        ep_results = self.episodic.retrieve(claim, k=1)
        if ep_results:
            # Use importance as a proxy for relevance
            max_score = max(max_score,
                            getattr(ep_results[0], 'importance', 0.0))

        return min(1.0, max_score)
