import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np

## Statistics
@dataclass
class Stat:
    # Specificity
    specificity_average: float | None = None
    specificity_range: list[float] = field(default_factory=list)
    # Entailment
    entail_1_average: float | None = None
    entail_2_average: float | None = None
    entail_1_range: list[float] = field(default_factory=list)
    entail_2_range: list[float] = field(default_factory=list)
    # Faithfulness
    faithfulness_average: float | None = None
    faithfulness_range: list[float] = field(default_factory=list)
    # Informativeness
    informativeness_average: float | None = None
    informative_range: list[float] = field(default_factory=list)
    # Specificity-faithfulness
    specificity_faithfulness_average: float | None = None
    specificity_faithfulness_range: list[float] = field(default_factory=list)
    # Specificity-informativeness
    specificity_informativeness_average: float | None = None
    specificity_informativeness_range: list[float] = field(default_factory=list)

## Node
@dataclass
class Node:
    id: int
    parent: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    node_type: str | None = None
    prompt: str | None = None
    capability: str | None = None
    # Matching information
    match: int | None = None
    score: float | None = None
    score2: float | None = None
    # Summary information
    mainIdea: str | None = None
    summary: str | None = None
    summary_blocked: bool | None = False
    # Similarity
    text_similarity: float | None = None
    idea_similarity: float | None = None
    # Matching status
    unmatched: bool | None = None
    # Evaluation
    evaluation: str | None = None
    evalScore: int | None = None
    entail_1: int | None = None
    entail_2: int | None = None
    # Tree construction
    alive: bool = True
    # Third-node information
    third: int | None = None
    third_score: int | None = None
    # Specificity
    evaluation_specificity: str | None = None
    specificity: float | None = None
    # Faithfulness / Informativeness
    faithfulness1: float | None = None
    faithfulness2: float | None = None
    informativeness1: float | None = None
    informativeness2: float | None = None
    evaluation_faithfulness: str | None = None
    evaluation_informativeness: str | None = None
    specificity_faithfulness: float | None = None
    specificity_informativeness: float | None = None
    evaluation_specificity_faithfulness: str | None = None
    evaluation_specificity_informativeness: str | None = None
    # WildBench scores
    wb_score_mistral: int | None = None
    wb_score_llama: int | None = None
    wb_scores: dict = field(default_factory=dict)
    cluster_size: int | None = None
    level_num: int | None = None

## Level
@dataclass
class Level:
    level: int
    stat: Stat = field(default_factory=Stat)
    nodes: list[Node] = field(default_factory=list)

## Tree
@dataclass
class Tree:
    levels: list[Level]
    next_id: int = 0
    config: dict = field(default_factory=dict)
    # Generate a new node ID
    def new_id(self) -> int:
        _id = self.next_id
        self.next_id += 1
        return _id

    # Load tree from JSON
    @classmethod
    def load(cls, path: str | Path) -> "Tree":
        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ValueError("Tree.load expects a *.json file")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload["levels"]
        lvls = []
        for lvl in raw:
            nodes = []
            for n in lvl["nodes"]:
                node = Node(
                    # Basic information
                    id=n.get("id", n.get("row")),
                    parent=n.get("parent", []),
                    children=n.get("children", []),
                    node_type=n.get("node_type"),
                    # Content
                    prompt=n.get("prompt"),
                    capability=n.get("capability"),
                    # Matching
                    match=n.get("match"),
                    score=n.get("score"),
                    score2=n.get("score2"),
                    # Summary
                    mainIdea=n.get("mainIdea"),
                    summary=n.get("summary"),
                    summary_blocked=n.get(
                        "summary_blocked",
                        False
                    ),
                    # Similarity
                    text_similarity=n.get("text_similarity"),
                    idea_similarity=n.get("idea_similarity"),
                    # Matching status
                    unmatched=n.get("unmatched", False),
                    # Evaluation
                    evalution=n.get("evaluation"),
                    evalScore=n.get("evalScore"),
                    entail_1=n.get("entail_1"),
                    entail_2=n.get("entail_2"),
                    # Tree construction
                    alive=n.get("alive", True),
                    # Third node
                    third=n.get("third"),
                    third_score=n.get("third_score"),
                    # Specificity
                    specificity=n.get("specificity"),
                    evaluation_specificity=n.get("evaluation_specificity"),
                    # Faithfulness / Informativeness
                    faithfulness1=n.get("faithfulness1"),
                    faithfulness2=n.get("faithfulness2"),
                    informativeness1=n.get("informativeness1"),
                    informativeness2=n.get("informativeness2"),
                    evaluation_faithfulness=n.get(
                        "evaluation_faithfulness"
                    ),
                    evaluation_informativeness=n.get(
                        "evaluation_informativeness"
                    ),
                    specificity_faithfulness=n.get(
                        "specificity_faithfulness"
                    ),
                    specificity_informativeness=n.get(
                        "evaluation_specificity_informativeness"
                    ),
                    ## WildBench
                    wb_score_mistral=n.get(
                        "wb_score_mistral"
                    ),
                    wb_score_llama=n.get(
                        "wb_score_llama"
                    ),
                    wb_scores=n.get(
                        "wb_scores",
                        {}
                    ),
                    cluster_size=n.get(
                        "cluster_size"
                    ),
                    level_num=n.get(
                        "level_num"
                    ),
                )
                nodes.append(node)
            level = Level(
                level=lvl["level"],
                stat=Stat(**lvl.get("stat", {})),
                nodes=nodes
            )
            lvls.append(level)
        # Calculate next available node ID
        max_id = max(
            (
                n.id
                for lvl in lvls
                for n in lvl.nodes
            ),
            default=-1
        )
        return cls(
            levels=lvls,
            next_id = max_id + 1,
            config=payload.get("config", {})
        )

    # Dump tree to JSON
    def dump(self, path: str | Path):
        path = Path(path)
        payload = {}
        if self.config:
            payload["config"] = self.config
        payload["levels"] = [
            {
                "level": lvl.level,
                "stat": asdict(lvl.stat),
                "nodes": [
                    asdict(node)
                    for node in lvl.nodes
                ],
            }
            for lvl in self.levels
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,
                default=Tree._np_encoder
            )

    # NumPy JSON encoder
    @staticmethod
    def _np_encoder(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(
            f"{repr(obj)} is not JSON serializabl"
        )



