"""
Module [5]: Semantic Memory — Long-Term Knowledge Base
======================================================
Hybrid vector DB (HNSW) + sparse BM25 + knowledge graph with
entity-relation store and forward-chaining inference.

Per ReasonBorn.md Section 4.6:
- Dense retrieval via HNSW (384-dim, cosine similarity)
- Sparse retrieval via BM25
- Knowledge graph with entities, relations, forward chaining
- Episodic → semantic consolidation
"""

import math
import hashlib
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class KnowledgeFact:
    """A consolidated fact in semantic memory."""
    fact_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.9
    domain: str = "general"
    source_count: int = 1
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KGEntity:
    """An entity in the knowledge graph."""
    entity_id: str
    name: str
    entity_type: str = "concept"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KGRelation:
    """A directed relation between two entities."""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """In-memory BM25 sparse retrieval index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.N: int = 0
        # term -> doc indices containing term
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        # term -> document frequency
        self.df: Dict[str, int] = defaultdict(int)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercasing tokenizer."""
        return [t.strip('.,!?;:()[]{}"\'-').lower()
                for t in text.split() if t.strip()]

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the BM25 index."""
        tokens = self._tokenize(text)
        idx = len(self.docs)
        self.docs.append(text)
        self.doc_ids.append(doc_id)
        self.doc_lens.append(len(tokens))
        self.N += 1
        self.avg_dl = sum(self.doc_lens) / self.N

        seen_terms: Set[str] = set()
        for token in tokens:
            self.inverted_index[token].append(idx)
            if token not in seen_terms:
                self.df[token] += 1
                seen_terms.add(token)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25 scoring."""
        query_tokens = self._tokenize(query)
        if not query_tokens or self.N == 0:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            idf = math.log(
                (self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0)
            for doc_idx in self.inverted_index[term]:
                tf = sum(1 for t in self._tokenize(self.docs[doc_idx])
                         if t == term)
                dl = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * dl / max(self.avg_dl, 1e-8))
                scores[doc_idx] += idf * numerator / denominator

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for doc_idx, score in ranked:
            results.append({
                'text': self.docs[doc_idx],
                'doc_id': self.doc_ids[doc_idx],
                'score': score,
            })
        return results


class KnowledgeGraph:
    """Simple in-memory knowledge graph with forward-chaining inference."""

    def __init__(self):
        self.entities: Dict[str, KGEntity] = {}
        self.relations: List[KGRelation] = []
        # source_id -> list of (relation, target_id)
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # Reverse: target_id -> list of (relation, source_id)
        self.reverse_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_entity(self, entity: KGEntity) -> None:
        self.entities[entity.entity_id] = entity

    def add_relation(self, relation: KGRelation) -> None:
        self.relations.append(relation)
        self.adjacency[relation.source_id].append(
            (relation.relation_type, relation.target_id))
        self.reverse_adj[relation.target_id].append(
            (relation.relation_type, relation.source_id))

    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> List[Dict]:
        """Multi-hop graph traversal from entity."""
        visited: Set[str] = set()
        results = []
        queue = [(entity_id, 0)]

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_hops:
                continue
            visited.add(current_id)

            if current_id in self.entities:
                entity = self.entities[current_id]
                results.append({
                    'entity': entity.name,
                    'type': entity.entity_type,
                    'depth': depth,
                    'properties': entity.properties,
                })

            for rel_type, target_id in self.adjacency.get(current_id, []):
                if target_id not in visited:
                    queue.append((target_id, depth + 1))

        return results

    def query(self, subject: str, relation: Optional[str] = None) -> List[Dict]:
        """Query the graph for facts about a subject."""
        results = []
        # Find entity by name
        entity_id = None
        for eid, entity in self.entities.items():
            if entity.name.lower() == subject.lower():
                entity_id = eid
                break

        if entity_id is None:
            return results

        for rel_type, target_id in self.adjacency.get(entity_id, []):
            if relation is None or rel_type.lower() == relation.lower():
                target = self.entities.get(target_id)
                if target:
                    results.append({
                        'subject': subject,
                        'relation': rel_type,
                        'object': target.name,
                        'object_type': target.entity_type,
                    })
        return results


class SemanticMemory:
    """
    Module [5]: Long-term knowledge base storing consolidated domain facts.

    Hybrid retrieval architecture:
    - Dense: HNSW-like approximate nearest neighbor (in-memory np arrays)
    - Sparse: BM25 inverted index
    - Graph: Entity-relation knowledge graph with multi-hop traversal
    """

    def __init__(self, db_size: int = 100000, embedding_dim: int = 384):
        self.db_size = db_size
        self.embedding_dim = embedding_dim

        # Fact store
        self.facts: Dict[str, KnowledgeFact] = {}
        self._fact_list: List[KnowledgeFact] = []

        # Dense index (HNSW-like via brute-force numpy; production would use hnswlib)
        self._embeddings: Optional[np.ndarray] = None
        self._embeddings_dirty = True

        # Sparse index
        self.bm25_index = BM25Index()

        # Knowledge graph
        self.knowledge_graph = KnowledgeGraph()

    def add_fact(self, fact: KnowledgeFact) -> None:
        """Add a fact to all three indices."""
        if not fact.fact_id:
            fact.fact_id = hashlib.md5(
                fact.content.encode('utf-8')).hexdigest()[:12]

        self.facts[fact.fact_id] = fact
        self._fact_list.append(fact)
        self._embeddings_dirty = True

        # Add to BM25 index
        self.bm25_index.add_document(fact.fact_id, fact.content)

    def add_text(self, text: str, embedding: Optional[np.ndarray] = None,
                 domain: str = "general", confidence: float = 0.9) -> str:
        """Convenience method to add a text fact."""
        fact = KnowledgeFact(
            fact_id="",
            content=text,
            embedding=embedding,
            confidence=confidence,
            domain=domain,
        )
        self.add_fact(fact)
        return fact.fact_id

    def dense_search(self, query: str, k: int = 10,
                     query_embedding: Optional[np.ndarray] = None
                     ) -> List[Dict[str, Any]]:
        """Dense vector search via cosine similarity."""
        if not self._fact_list:
            return []

        self._rebuild_embeddings()

        if query_embedding is None or self._embeddings is None:
            return []
        if self._embeddings.shape[0] == 0:
            return []

        # Cosine similarity
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding) + 1e-8)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        normed = self._embeddings / norms
        similarities = normed @ query_norm

        # Top-k
        k = min(k, len(self._fact_list))
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            fact = self._fact_list[idx]
            fact.access_count += 1
            results.append({
                'text': fact.content,
                'score': float(similarities[idx]),
                'fact_id': fact.fact_id,
                'confidence': fact.confidence,
                'domain': fact.domain,
            })
        return results

    def bm25_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Sparse BM25 search."""
        results = self.bm25_index.search(query, k=k)
        # Enrich with fact metadata
        for result in results:
            fact = self.facts.get(result['doc_id'])
            if fact:
                result['confidence'] = fact.confidence
                result['domain'] = fact.domain
                fact.access_count += 1
        return results

    def graph_search(self, query: str, k: int = 10,
                     max_hops: int = 2) -> List[Dict[str, Any]]:
        """Knowledge graph multi-hop traversal."""
        # Extract potential entity mentions from query
        words = query.split()
        results = []
        for word in words:
            kg_results = self.knowledge_graph.query(word.strip('.,!?'))
            results.extend(kg_results)
        return results[:k]

    def lookup_confidence(self, claim: str) -> float:
        """
        Look up confidence for a claim in semantic memory.
        Returns highest matching fact's confidence score.
        """
        bm25_results = self.bm25_search(claim, k=3)
        if bm25_results:
            # Weight by BM25 score and fact confidence
            best = max(bm25_results,
                       key=lambda r: r['score'] * r.get('confidence', 0.5))
            # Normalize BM25 score to [0, 1] range
            norm_score = min(best['score'] / 10.0, 1.0)
            return norm_score * best.get('confidence', 0.5)
        return 0.0

    def consolidate_from_episodic(self, episodic_entries: List[Any]) -> int:
        """
        Consolidate frequently-accessed episodic memories into semantic facts.
        Clusters similar entries and creates consolidated facts.
        Returns number of new facts created.
        """
        if not episodic_entries:
            return 0

        created = 0
        for entry in episodic_entries:
            content = entry.content if hasattr(entry, 'content') else str(entry)
            # Check if similar fact already exists
            existing = self.bm25_search(content, k=1)
            if existing and existing[0]['score'] > 5.0:
                # Update existing fact's confidence
                fact = self.facts.get(existing[0]['doc_id'])
                if fact:
                    fact.source_count += 1
                    fact.confidence = min(0.99, fact.confidence + 0.01)
            else:
                # Create new semantic fact
                embedding = entry.embedding if hasattr(
                    entry, 'embedding') else None
                self.add_text(
                    content,
                    embedding=embedding,
                    confidence=0.7,
                    domain=getattr(entry, 'domain', 'general'),
                )
                created += 1

        return created

    def _rebuild_embeddings(self) -> None:
        """Rebuild embedding matrix."""
        if not self._embeddings_dirty:
            return
        embeddings = []
        for fact in self._fact_list:
            if fact.embedding is not None:
                embeddings.append(fact.embedding)
            else:
                embeddings.append(np.zeros(self.embedding_dim))
        if embeddings:
            self._embeddings = np.stack(embeddings)
        else:
            self._embeddings = np.zeros((0, self.embedding_dim))
        self._embeddings_dirty = False

    def __len__(self) -> int:
        return len(self._fact_list)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'num_facts': len(self._fact_list),
            'num_kg_entities': len(self.knowledge_graph.entities),
            'num_kg_relations': len(self.knowledge_graph.relations),
            'bm25_docs': self.bm25_index.N,
            'domains': list(set(f.domain for f in self._fact_list)),
        }
