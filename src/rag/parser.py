"""Markdown and YAML frontmatter parser for knowledge-base documents."""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml

from src.models.schemas import DocumentMetadata, RetrievedChunk


class KnowledgeBaseParser:
    """Parses Markdown documents with YAML frontmatter into section-level chunks."""

    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    @classmethod
    def parse_file(cls, filepath: Path) -> List[RetrievedChunk]:
        """Parse a single markdown file into section-level chunks with metadata."""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        metadata, body = cls._extract_frontmatter(text, filepath.name)
        chunks = cls._chunk_by_headings(body, metadata, filepath.name)
        return chunks

    @classmethod
    def _extract_frontmatter(cls, text: str, default_filename: str) -> Tuple[DocumentMetadata, str]:
        """Extract YAML frontmatter and return (DocumentMetadata, body_text)."""
        match = cls.FRONTMATTER_PATTERN.match(text)
        if not match:
            # Fallback if no frontmatter
            meta = DocumentMetadata(
                document_id=default_filename,
                title=default_filename,
                status="active",
                policy_authority="official",
                customer_answering=True,
            )
            return meta, text

        yaml_content = match.group(1)
        body = text[match.end():]
        raw_meta = yaml.safe_load(yaml_content) or {}

        # Normalize metadata fields
        eff_date = raw_meta.get("effective_date")
        last_rev = raw_meta.get("last_reviewed")
        supersedes = raw_meta.get("supersedes")
        superseded_by = raw_meta.get("superseded_by")

        meta = DocumentMetadata(
            document_id=str(raw_meta.get("document_id", default_filename)),
            title=str(raw_meta.get("title", default_filename)),
            status=str(raw_meta.get("status", "active")).lower(),
            policy_authority=str(raw_meta.get("policy_authority", "official")).lower(),
            customer_answering=bool(raw_meta.get("customer_answering", True)),
            effective_date=str(eff_date) if eff_date is not None else None,
            last_reviewed=str(last_rev) if last_rev is not None else None,
            audience=str(raw_meta.get("audience", "customer")),
            supersedes=str(supersedes) if supersedes is not None else None,
            superseded_by=str(superseded_by) if superseded_by is not None else None,
        )
        return meta, body

    @classmethod
    def _chunk_by_headings(
        cls, body: str, metadata: DocumentMetadata, filename: str
    ) -> List[RetrievedChunk]:
        """Split document body into section chunks based on Markdown headings."""
        lines = body.splitlines()
        chunks: List[RetrievedChunk] = []

        current_heading = metadata.title
        current_hierarchy = [metadata.title]
        current_lines: List[str] = []
        section_idx = 0

        for line in lines:
            heading_match = cls.HEADING_PATTERN.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # If we have accumulated content for previous heading, store chunk
                content = "\n".join(current_lines).strip()
                if content:
                    chunk_id = f"{metadata.document_id}#{section_idx}"
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=chunk_id,
                            file_name=filename,
                            title=metadata.title,
                            heading=current_heading,
                            heading_hierarchy=list(current_hierarchy),
                            content=content,
                            metadata=metadata,
                        )
                    )
                    section_idx += 1

                current_lines = []
                if level == 1:
                    current_heading = heading_text
                    current_hierarchy = [heading_text]
                elif level == 2:
                    current_heading = heading_text
                    current_hierarchy = [current_hierarchy[0] if current_hierarchy else metadata.title, heading_text]
                else:
                    current_heading = heading_text
                    current_hierarchy.append(heading_text)
            else:
                current_lines.append(line)

        # Append trailing section
        content = "\n".join(current_lines).strip()
        if content:
            chunk_id = f"{metadata.document_id}#{section_idx}"
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    file_name=filename,
                    title=metadata.title,
                    heading=current_heading,
                    heading_hierarchy=list(current_hierarchy),
                    content=content,
                    metadata=metadata,
                )
            )

        return chunks
