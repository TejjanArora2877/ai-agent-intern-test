"""Generic conflict detection engine for identifying contradictory active official policies."""

import re
from typing import List, Optional, Tuple, Dict, Set
from src.models.schemas import RetrievedChunk
from src.rag.extractor import split_sentences


class ConflictDetector:
    """
    Generic conflict detector identifying opposing polarities/directives
    across distinct active official knowledge-base documents without hardcoded filenames.
    """

    # Generic semantic modality polarities for care/cleaning/temperature maintenance
    POLARITY_PAIRS = [
        # Care/Cleaning/Maintenance: Hand-wash / Manual / Restrictive vs Dishwasher / Machine Washable / Permissive
        (
            {"hand-washed", "hand-wash", "hand wash", "spot-clean", "spot clean", "do not machine wash", "never place in", "top rack only", "hand wash only", "body must be hand-washed"},
            {"dishwasher safe", "dishwasher", "machine washable", "machine wash", "all components are dishwasher safe", "completely dishwasher safe", "all components"}
        ),
        # Temperature/Heat: Low/No Heat vs High Heat
        (
            {"cool water", "no heat", "air-dry completely", "do not microwave", "avoid heat"},
            {"high heat", "tumble dry", "hot water", "microwave safe"}
        ),
    ]

    @classmethod
    def detect_conflict(
        cls, chunks: List[RetrievedChunk], query: str = ""
    ) -> Tuple[bool, List[RetrievedChunk], Optional[str]]:
        """
        Evaluate retrieved active official chunks to detect opposing instructions on the same subject.
        
        Returns:
            (is_conflict, [conflicting_chunks], explanation_text)
        """
        if not chunks:
            return False, [], None

        # Only evaluate conflicts when query relates to care, cleaning, temperature, or maintenance
        if query:
            is_care_query = any(w in query.lower() for w in ["wash", "clean", "dishwasher", "machine", "care", "dry", "microwave", "temperature", "heat", "hand-wash"])
            if not is_care_query:
                return False, [], None

        # Group chunks by document ID
        docs_map: Dict[str, List[RetrievedChunk]] = {}
        for c in chunks:
            if c.metadata.status == "active" and c.metadata.policy_authority == "official":
                doc_id = c.metadata.document_id
                if doc_id not in docs_map:
                    docs_map[doc_id] = []
                docs_map[doc_id].append(c)

        # Must have at least two distinct active official documents to have a conflict
        if len(docs_map) < 2:
            return False, [], None

        doc_ids = list(docs_map.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                d1_chunks = docs_map[doc_ids[i]]
                d2_chunks = docs_map[doc_ids[j]]

                for c1 in d1_chunks:
                    for c2 in d2_chunks:
                        conflict_found, stmt1, stmt2 = cls._check_chunk_pair_conflict(c1, c2)
                        if conflict_found:
                            explanation = (
                                f"Current official sources conflict regarding this topic: "
                                f"The {c1.title} ({c1.file_name} > {c1.heading}) states that '{stmt1}', "
                                f"whereas the {c2.title} ({c2.file_name} > {c2.heading}) states that '{stmt2}'. "
                                f"As safest interim guidance, we recommend following the more conservative care instructions. "
                                f"I am recommending a human confirmation to resolve this discrepancy."
                            )
                            return True, [c1, c2], explanation

        return False, [], None

    @classmethod
    def _check_chunk_pair_conflict(
        cls, c1: RetrievedChunk, c2: RetrievedChunk
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if two chunks share a common subject noun and contain opposing polarity statements."""
        text1 = f"{c1.heading} {c1.content}".lower()
        text2 = f"{c2.heading} {c2.content}".lower()

        # Extract non-stopword tokens to check subject overlap
        tokens1 = set(re.findall(r"\b[a-z0-9]{3,}\b", text1))
        tokens2 = set(re.findall(r"\b[a-z0-9]{3,}\b", text2))
        
        shared_subject_tokens = tokens1.intersection(tokens2)
        # Filter out common generic terms
        generic_terms = {
            "the", "and", "for", "with", "item", "items", "product", "products",
            "policy", "customer", "customers", "orders", "order", "days", "day",
            "standard", "agent", "support", "information", "status", "date",
            "available", "review", "details", "current", "official", "when", "that",
            "this", "will", "from", "have", "been", "must", "time"
        }
        shared_subject_tokens = {t for t in shared_subject_tokens if t not in generic_terms}

        if not shared_subject_tokens:
            return False, None, None

        # Check against polarity pairs
        for set_a, set_b in cls.POLARITY_PAIRS:
            match_1a = any(term in text1 for term in set_a)
            match_2b = any(term in text2 for term in set_b)
            
            match_1b = any(term in text1 for term in set_b)
            match_2a = any(term in text2 for term in set_a)

            if (match_1a and match_2b) or (match_1b and match_2a):
                s1_candidates = [s for s in split_sentences(c1.content) if any(t in s.lower() for t in (set_a if match_1a else set_b))]
                s2_candidates = [s for s in split_sentences(c2.content) if any(t in s.lower() for t in (set_b if match_1a else set_a))]
                
                stmt1 = s1_candidates[0] if s1_candidates else c1.content.splitlines()[0]
                stmt2 = s2_candidates[0] if s2_candidates else c2.content.splitlines()[0]
                return True, stmt1, stmt2

        return False, None, None
