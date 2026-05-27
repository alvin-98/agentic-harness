"""
Artifact store for the agent.

Artifacts are large binary blobs (>4KB) that are stored separately from memory.
Each artifact gets an ID in the format "art:<sha256-prefix>" and can be retrieved
by that ID.

The artifact store persists to disk in an artifacts/ directory alongside the memory CSV.
"""

import hashlib
import os
import json
from datetime import datetime
from pathlib import Path

from .schemas import Artifact

PARENT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PARENT_DIR / "artifacts"
ARTIFACTS_INDEX = ARTIFACTS_DIR / "index.json"

ARTIFACT_THRESHOLD_BYTES = 4096  # 4 KB


class ArtifactStore:
    def __init__(self):
        self._ensure_dir_exists()

    def _ensure_dir_exists(self):
        """Create the artifacts directory if it doesn't exist."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if not ARTIFACTS_INDEX.exists():
            self._save_index({})

    def _load_index(self) -> dict[str, dict]:
        """Load the artifact index."""
        if not ARTIFACTS_INDEX.exists():
            return {}
        with open(ARTIFACTS_INDEX, "r") as f:
            return json.load(f)

    def _save_index(self, index: dict[str, dict]):
        """Save the artifact index."""
        with open(ARTIFACTS_INDEX, "w") as f:
            json.dump(index, f, indent=2)

    def _artifact_path(self, artifact_id: str) -> Path:
        """Get the file path for an artifact."""
        safe_id = artifact_id.replace(":", "_").replace("/", "_")
        return ARTIFACTS_DIR / safe_id

    def exists(self, artifact_id: str) -> bool:
        """Check if an artifact exists."""
        if not artifact_id:
            return False
        index = self._load_index()
        return artifact_id in index and self._artifact_path(artifact_id).exists()

    def get_bytes(self, artifact_id: str) -> bytes:
        """Retrieve raw bytes for an artifact."""
        if not self.exists(artifact_id):
            raise KeyError(f"Artifact not found: {artifact_id}")
        
        path = self._artifact_path(artifact_id)
        with open(path, "rb") as f:
            return f.read()

    def get(self, artifact_id: str) -> Artifact:
        """Get artifact metadata."""
        if not self.exists(artifact_id):
            raise KeyError(f"Artifact not found: {artifact_id}")
        
        index = self._load_index()
        meta = index[artifact_id]
        return Artifact(
            id=artifact_id,
            content_type=meta["content_type"],
            size_bytes=meta["size_bytes"],
            source=meta["source"],
            descriptor=meta["descriptor"],
        )

    def put(
        self,
        data: bytes,
        source: str,
        content_type: str = "application/octet-stream",
        descriptor: str = "",
    ) -> str:
        """
        Store an artifact and return its ID.
        
        Args:
            data: Raw bytes to store.
            source: Source of the artifact (e.g., tool name, URL).
            content_type: MIME type of the content.
            descriptor: Human-readable description.
        
        Returns:
            The artifact ID in format "art:<sha256-prefix>".
        """
        sha = hashlib.sha256(data).hexdigest()[:12]
        artifact_id = f"art:{sha}"
        
        path = self._artifact_path(artifact_id)
        with open(path, "wb") as f:
            f.write(data)
        
        index = self._load_index()
        index[artifact_id] = {
            "content_type": content_type,
            "size_bytes": len(data),
            "source": source,
            "descriptor": descriptor,
            "created_at": datetime.now().isoformat(),
        }
        self._save_index(index)
        
        return artifact_id

    def delete(self, artifact_id: str):
        """Delete an artifact."""
        if not self.exists(artifact_id):
            return
        
        path = self._artifact_path(artifact_id)
        if path.exists():
            path.unlink()
        
        index = self._load_index()
        if artifact_id in index:
            del index[artifact_id]
            self._save_index(index)

    def reset(self):
        """Delete all artifacts."""
        index = self._load_index()
        for artifact_id in list(index.keys()):
            self.delete(artifact_id)
        self._save_index({})

    def list_all(self) -> list[Artifact]:
        """List all artifacts."""
        index = self._load_index()
        artifacts = []
        for artifact_id, meta in index.items():
            if self._artifact_path(artifact_id).exists():
                artifacts.append(Artifact(
                    id=artifact_id,
                    content_type=meta["content_type"],
                    size_bytes=meta["size_bytes"],
                    source=meta["source"],
                    descriptor=meta["descriptor"],
                ))
        return artifacts
