"""Embedding-based post-Curator playbook hygiene for AppWorld ACE.

This module is deliberately distinct from Curator MERGE candidate discovery:
the analyzer performs a separate, automatic deduplication pass after a valid
Curator update.  It is useful as an ablation against evidence-aware MERGE.
"""

from __future__ import annotations

from typing import Any

from .playbook import format_playbook_line, parse_playbook_line


class BulletpointAnalyzer:
    """Cluster semantically duplicate bullets and consolidate each cluster."""

    def __init__(
        self,
        model: Any,
        sentence_transformer: Any,
        threshold: float = 0.90,
        clustering: str = "pairwise",
        dbscan_eps: float = 0.12,
        dbscan_min_samples: int = 2,
    ) -> None:
        self.model = model
        self.sentence_transformer = sentence_transformer
        self.threshold = threshold
        self.clustering = clustering
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def _groups(self, bullets: list[dict[str, Any]]) -> list[list[int]]:
        if len(bullets) < 2 or self.sentence_transformer is None:
            return []
        import numpy as np

        vectors = np.asarray(
            self.sentence_transformer.encode(
                [bullet["content"] for bullet in bullets], normalize_embeddings=True
            ),
            dtype=np.float32,
        )
        if self.clustering == "dbscan":
            from sklearn.cluster import DBSCAN

            labels = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric="cosine",
            ).fit_predict(vectors)
            return [
                np.where(labels == label)[0].tolist()
                for label in sorted(set(labels))
                if label >= 0 and int((labels == label).sum()) >= self.dbscan_min_samples
            ]
        if self.clustering != "pairwise":
            raise ValueError(f"Unknown BulletpointAnalyzer clustering mode: {self.clustering}")

        similarities = vectors @ vectors.T
        groups: list[list[int]] = []
        consumed: set[int] = set()
        for index in range(len(bullets)):
            if index in consumed:
                continue
            group = [index] + [
                other for other in range(index + 1, len(bullets))
                if similarities[index, other] >= self.threshold and other not in consumed
            ]
            if len(group) > 1:
                groups.append(group)
                consumed.update(group)
        return groups

    def _merge(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Ask the Curator model for one conservative replacement rule."""
        primary = group[0]
        helpful = sum(item["helpful"] for item in group)
        harmful = sum(item["harmful"] for item in group)
        source_text = "\n".join(
            f"- [{item['id']}] {item['content']}" for item in group
        )
        prompt = f"""Consolidate semantically duplicate AppWorld playbook rules.

Rules:
{source_text}

Write one concise rule that retains only information shared or safely
compatible across the inputs. Do not invent APIs, conditions, or facts.
Return only the replacement rule text, without an ID, counters, Markdown, or
explanation."""
        try:
            response = self.model.generate(messages=[{"role": "user", "content": prompt}])
            content = str(response.get("content", "")).strip()
            if content:
                return {
                    "id": primary["id"],
                    "helpful": helpful,
                    "harmful": harmful,
                    "content": content,
                }
        except Exception as exc:  # Hygiene must never fail a task.
            print(f"[BulletpointAnalyzer] Merge call failed; keeping first bullet: {exc}")
        return {
            "id": primary["id"],
            "helpful": helpful,
            "harmful": harmful,
            "content": primary["content"],
        }

    def analyze(self, playbook: str) -> tuple[str, dict[str, Any]]:
        """Return cleaned playbook and structured statistics for lifecycle logs."""
        lines = playbook.splitlines()
        bullet_positions: list[tuple[int, dict[str, Any]]] = []
        for position, line in enumerate(lines):
            parsed = parse_playbook_line(line)
            if parsed and parsed["content"]:
                bullet_positions.append((position, parsed))
        bullets = [item for _, item in bullet_positions]
        groups = self._groups(bullets)
        stats: dict[str, Any] = {
            "clustering": self.clustering,
            "threshold": self.threshold,
            "dbscan_eps": self.dbscan_eps if self.clustering == "dbscan" else None,
            "dbscan_min_samples": self.dbscan_min_samples if self.clustering == "dbscan" else None,
            "input_bullet_count": len(bullets),
            "cluster_count": len(groups),
            "clusters": [[bullets[index]["id"] for index in group] for group in groups],
        }
        if not groups:
            stats.update({"output_bullet_count": len(bullets), "removed_or_merged_count": 0})
            return playbook, stats

        replacement_by_position: dict[int, str] = {}
        removed_positions: set[int] = set()
        for group in groups:
            merged = self._merge([bullets[index] for index in group])
            first_position = bullet_positions[group[0]][0]
            replacement_by_position[first_position] = format_playbook_line(
                merged["id"], merged["helpful"], merged["harmful"], merged["content"]
            )
            removed_positions.update(bullet_positions[index][0] for index in group[1:])

        output = [
            replacement_by_position.get(position, line)
            for position, line in enumerate(lines)
            if position not in removed_positions
        ]
        stats.update({
            "output_bullet_count": len(bullets) - len(removed_positions),
            "removed_or_merged_count": len(removed_positions),
        })
        return "\n".join(output), stats
