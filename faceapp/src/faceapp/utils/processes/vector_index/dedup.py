from __future__ import annotations

from collections import defaultdict
from typing import Any, Union

from faceapp._base.base import ProcessOutput
from faceapp._base.indexer import Indexer


class DeduplicatingIndexer(Indexer):
    """Indexer decorator that removes duplicate records before persistence."""

    def __init__(self, indexer: Indexer, enabled: bool = True):
        self.indexer = indexer
        self.enabled = enabled

    async def load(
        self,
        embeddings: list,
        metadata: list,
        *args,
        **kwargs,
    ) -> ProcessOutput:
        if not self.enabled:
            return await self.indexer.load(
                embeddings=embeddings,
                metadata=metadata,
                *args,
                **kwargs,
            )

        filtered_embeddings, filtered_metadata = self._filter_duplicates(
            embeddings=embeddings,
            metadata=metadata,
        )
        return await self.indexer.load(
            embeddings=filtered_embeddings,
            metadata=filtered_metadata,
            *args,
            **kwargs,
        )

    async def search(
        self,
        *args,
        **kwargs,
    ) -> Union[dict[str, Any], list[Any]]:
        return await self.indexer.search(*args, **kwargs)

    def existing_dedupe_hashes(
        self,
        index_name: str,
        dedupe_hashes: set[str],
    ) -> set[str]:
        return self.indexer.existing_dedupe_hashes(index_name, dedupe_hashes)

    def _filter_duplicates(
        self,
        embeddings: list,
        metadata: list,
    ) -> tuple[list, list]:
        assert len(embeddings) == len(metadata)
        grouped_indices = defaultdict(list)
        for index, payload in enumerate(metadata):
            if not isinstance(payload, dict):
                continue
            grouped_indices[payload.get("index_name")].append(index)

        keep_flags = [True] * len(metadata)
        for index_name, indices in grouped_indices.items():
            if not isinstance(index_name, str) or not index_name:
                continue

            dedupe_hashes: set[str] = set()
            for idx in indices:
                payload = metadata[idx]
                if not isinstance(payload, dict):
                    continue
                dedupe_hash = payload.get("dedupe_hash")
                if isinstance(dedupe_hash, str) and dedupe_hash:
                    dedupe_hashes.add(dedupe_hash)
            existing_hashes = self.indexer.existing_dedupe_hashes(
                index_name,
                dedupe_hashes,
            )
            seen_hashes = set(existing_hashes)

            for idx in indices:
                payload = metadata[idx]
                dedupe_hash = None
                if isinstance(payload, dict):
                    dedupe_hash = payload.get("dedupe_hash")

                if not isinstance(dedupe_hash, str) or not dedupe_hash:
                    continue

                if dedupe_hash in seen_hashes:
                    keep_flags[idx] = False
                    continue
                seen_hashes.add(dedupe_hash)

        filtered_embeddings = []
        filtered_metadata = []
        for embedding, payload, keep in zip(
            embeddings,
            metadata,
            keep_flags,
        ):
            if not keep:
                continue
            filtered_embeddings.append(embedding)
            filtered_metadata.append(payload)
        return filtered_embeddings, filtered_metadata
