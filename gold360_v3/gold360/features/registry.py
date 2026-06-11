from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FeatureDefinition:
    name: str
    group: str
    description: str
    formula: str
    dtype: str
    category: str
    dependencies: List[str] = field(default_factory=list)
    version: str = "v001"
    tags: List[str] = field(default_factory=list)
    economic_rationale: Optional[str] = None


class FeatureRegistry:
    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}
        self._computers: Dict[str, Callable] = {}

    def register(self, definition: FeatureDefinition, computer: Optional[Callable] = None):
        self._features[definition.name] = definition
        if computer:
            self._computers[definition.name] = computer

    def get(self, name: str) -> FeatureDefinition:
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not registered")
        return self._features[name]

    def list_features(self, group: Optional[str] = None) -> List[FeatureDefinition]:
        if group:
            return [f for f in self._features.values() if f.group == group]
        return list(self._features.values())

    def list_groups(self) -> List[str]:
        return sorted(set(f.group for f in self._features.values()))

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        return {name: f.dependencies for name, f in self._features.items()}

    def compute(self, name: str, df: Any, **kwargs) -> Any:
        if name not in self._computers:
            raise KeyError(f"No computer registered for feature '{name}'")
        return self._computers[name](df, **kwargs)

    def compute_group(self, group: str, df: Any, **kwargs) -> Dict[str, Any]:
        results = {}
        for feat in self.list_features(group=group):
            if feat.name in self._computers:
                results[feat.name] = self.compute(feat.name, df, **kwargs)
        return results

    def registry_report(self) -> str:
        lines = ["FEATURE REGISTRY", "=" * 60]
        for group in self.list_groups():
            lines.append(f"\n[{group}]")
            for feat in self.list_features(group=group):
                lines.append(
                    f"  {feat.name}: ({feat.dtype}) {feat.description}"
                )
        return "\n".join(lines)
