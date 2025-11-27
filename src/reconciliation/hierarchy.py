"""Hierarchy definition and management for hierarchical reconciliation.

This module provides the HierarchyDefinition class that loads hierarchy
configurations from YAML, constructs computational graphs using NetworkX,
generates aggregation matrices, and validates hierarchy structure.

The hierarchy represents the Brazilian electric grid structure:
- National level (1 node): Total Brazil demand
- Subsystem level (4 nodes): SE, S, NE, N
- Area level (17 nodes): Individual consumption areas
- Loss components (4 nodes): Calculated losses

Example:
    ```python
    from src.reconciliation.hierarchy import HierarchyDefinition

    # Load from YAML
    hierarchy = HierarchyDefinition("config/hierarchy_brazil.yaml")

    # Or build programmatically
    hierarchy = HierarchyDefinition()
    hierarchy.add_node("National", level=0, node_type="aggregate")
    hierarchy.add_node("SE", level=1, node_type="aggregate", parent="National")
    hierarchy.build_graph()
    hierarchy.build_aggregation_matrix()

    # Validate
    errors = hierarchy.validate_structure()
    if errors:
        print(f"Validation errors: {errors}")

    # Use for reconciliation
    S = hierarchy.aggregation_matrix
    y_aggregated = S @ y_bottom
    ```
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HierarchyNode:
    """Represents a single node in the forecasting hierarchy.

    Each node corresponds to a time series in the hierarchical forecasting
    system. Nodes can be aggregate (sum of children), bottom-level (leaf nodes
    that are directly forecasted), or calculated (derived from other series).

    Attributes:
        name: Unique identifier for the node (e.g., "BR_National", "SE", "Area_1").
        level: Hierarchy level (0=National, 1=Subsystem, 2=Area).
            Lower levels are higher in the hierarchy.
        node_type: Type of node:
            - "aggregate": Sum of children (e.g., National, Subsystems)
            - "bottom": Leaf node that is directly forecasted (e.g., Areas)
            - "calculated": Derived from other series (e.g., Losses)
        children: List of child node names.
        parent: Parent node name (None for root).
        aggregation_weights: Weights for aggregation. Default is 1.0 for each
            child, meaning simple sum. Can be customized for weighted aggregation.
        metadata: Additional node properties (e.g., region type, calculation method).

    Example:
        >>> node = HierarchyNode(
        ...     name="SE",
        ...     level=1,
        ...     node_type="aggregate",
        ...     children=["Area_1", "Area_2", "Area_3"],
        ...     parent="National"
        ... )
        >>> print(node)
        HierarchyNode(name='SE', level=1, type='aggregate')
    """

    name: str
    level: int = 0
    node_type: str = "aggregate"
    children: list[str] = field(default_factory=list)
    parent: str | None = None
    aggregation_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate node after initialization."""
        valid_types = {"aggregate", "bottom", "calculated"}
        if self.node_type not in valid_types:
            msg = f"node_type must be one of {valid_types}, got {self.node_type!r}"
            raise ValueError(msg)

        if self.level < 0:
            msg = f"level must be non-negative, got {self.level}"
            raise ValueError(msg)

    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children).

        Returns:
            True if node has no children.
        """
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if node is root (no parent).

        Returns:
            True if node has no parent.
        """
        return self.parent is None

    def get_aggregation_weight(self, child_name: str) -> float:
        """Get aggregation weight for a child node.

        Args:
            child_name: Name of the child node.

        Returns:
            Aggregation weight (default 1.0).
        """
        return self.aggregation_weights.get(child_name, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary.

        Returns:
            Dictionary representation of the node.
        """
        return {
            "name": self.name,
            "level": self.level,
            "node_type": self.node_type,
            "children": self.children.copy(),
            "parent": self.parent,
            "aggregation_weights": self.aggregation_weights.copy(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchyNode":
        """Create node from dictionary.

        Args:
            data: Dictionary with node attributes.

        Returns:
            HierarchyNode instance.
        """
        return cls(
            name=data["name"],
            level=data.get("level", 0),
            node_type=data.get("node_type", "aggregate"),
            children=data.get("children", []),
            parent=data.get("parent"),
            aggregation_weights=data.get("aggregation_weights", {}),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation.
        """
        return f"HierarchyNode(name={self.name!r}, level={self.level}, type={self.node_type!r})"


class HierarchyDefinition:
    """Manages hierarchical structure for forecasting reconciliation.

    The hierarchy represents relationships between forecast series in the
    Brazilian electric grid:
    - National level (root): Total Brazil demand
    - Subsystem level (4 nodes): SE (Southeast/Midwest), S (South),
      NE (Northeast), N (North)
    - Area level (17 nodes): Individual consumption areas
    - Loss components (4 nodes): System losses (calculated, not forecasted)

    The aggregation matrix S defines how bottom-level series aggregate to
    higher levels:
        y_aggregated = S @ y_bottom

    Example YAML configuration:
        ```yaml
        hierarchy:
          national:
            name: "BR_National"
            children: ["SE", "S", "NE", "N"]

          subsystems:
            SE:
              name: "SE"
              children: ["Area_SP", "Area_RJ", "Area_MG", "Area_ES", "Area_GO"]
            S:
              name: "S"
              children: ["Area_PR", "Area_SC", "Area_RS"]
            # ... etc

          areas:
            Area_SP:
              name: "Area_SP"
              type: "consumption"
            # ... etc
        ```

    Attributes:
        nodes: Dictionary mapping node names to HierarchyNode instances.
        graph: NetworkX DiGraph representing the hierarchy structure.
        aggregation_matrix: Numpy array for aggregation relationships.
        config: Raw configuration dictionary from YAML.

    Example:
        >>> hierarchy = HierarchyDefinition("config/hierarchy_brazil.yaml")
        >>> hierarchy.validate_structure()
        []  # No errors
        >>> S = hierarchy.aggregation_matrix
        >>> print(f"Matrix shape: {S.shape}")
        Matrix shape: (26, 17)  # 26 series, 17 bottom-level
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize hierarchy definition.

        Args:
            config_path: Path to YAML configuration file. If provided, the
                hierarchy is loaded and built automatically.

        Example:
            >>> # Load from file
            >>> hierarchy = HierarchyDefinition("config/hierarchy.yaml")

            >>> # Build programmatically
            >>> hierarchy = HierarchyDefinition()
            >>> hierarchy.add_node("National", level=0, node_type="aggregate")
        """
        self.nodes: dict[str, HierarchyNode] = {}
        self.graph: nx.DiGraph | None = None
        self.aggregation_matrix: np.ndarray | None = None
        self.config: dict[str, Any] = {}

        if config_path:
            self.load_from_yaml(config_path)
            self.build_graph()
            self.build_aggregation_matrix()

    @property
    def n_nodes(self) -> int:
        """Get total number of nodes.

        Returns:
            Number of nodes in the hierarchy.
        """
        return len(self.nodes)

    @property
    def n_levels(self) -> int:
        """Get number of hierarchy levels.

        Returns:
            Number of distinct levels in the hierarchy.
        """
        if not self.nodes:
            return 0
        return len(set(node.level for node in self.nodes.values()))

    @property
    def root_node(self) -> HierarchyNode | None:
        """Get the root node of the hierarchy.

        Returns:
            Root node (node with no parent), or None if hierarchy is empty.
        """
        for node in self.nodes.values():
            if node.is_root:
                return node
        return None

    def load_from_yaml(self, config_path: str | Path) -> None:
        """Load hierarchy definition from YAML file.

        Parses the YAML configuration and creates HierarchyNode instances
        for each node in the hierarchy.

        Args:
            config_path: Path to YAML configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML is invalid.
            ValueError: If hierarchy structure is invalid.

        Example:
            >>> hierarchy = HierarchyDefinition()
            >>> hierarchy.load_from_yaml("config/hierarchy_brazil.yaml")
            >>> print(f"Loaded {len(hierarchy.nodes)} nodes")
        """
        config_path = Path(config_path)
        logger.info("Loading hierarchy from %s", config_path)

        if not config_path.exists():
            msg = f"Hierarchy config not found: {config_path}"
            raise FileNotFoundError(msg)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        if "hierarchy" not in self.config:
            msg = "YAML must contain 'hierarchy' section"
            raise ValueError(msg)

        # Parse hierarchy structure
        self._parse_hierarchy(self.config["hierarchy"])

        # Assign levels based on graph structure
        self._assign_levels()

        logger.info("Loaded %d nodes from hierarchy", len(self.nodes))

    def _parse_hierarchy(self, hierarchy_config: dict[str, Any]) -> None:
        """Parse hierarchy configuration and create nodes.

        Args:
            hierarchy_config: Hierarchy section from YAML.
        """
        # Parse national level (root)
        if "national" in hierarchy_config:
            national_config = hierarchy_config["national"]
            national_node = HierarchyNode(
                name=national_config["name"],
                level=0,
                node_type="aggregate",
                children=national_config.get("children", []),
            )
            self.nodes[national_node.name] = national_node

        # Parse subsystems
        if "subsystems" in hierarchy_config:
            for sub_id, sub_config in hierarchy_config["subsystems"].items():
                sub_node = HierarchyNode(
                    name=sub_config["name"],
                    level=1,
                    node_type="aggregate",
                    children=sub_config.get("children", []),
                    metadata={"subsystem_id": sub_id},
                )
                self.nodes[sub_node.name] = sub_node

        # Parse areas (bottom level)
        if "areas" in hierarchy_config:
            for area_id, area_config in hierarchy_config["areas"].items():
                area_node = HierarchyNode(
                    name=area_config["name"],
                    level=2,
                    node_type="bottom",
                    children=[],
                    metadata={
                        "area_id": area_id,
                        "type": area_config.get("type", "consumption"),
                    },
                )
                self.nodes[area_node.name] = area_node

        # Parse losses (calculated nodes)
        if "losses" in hierarchy_config:
            for loss_id, loss_config in hierarchy_config["losses"].items():
                loss_node = HierarchyNode(
                    name=loss_config["name"],
                    level=loss_config.get("level", 2),
                    node_type="calculated",
                    children=[],
                    metadata={
                        "loss_id": loss_id,
                        "calculation": loss_config.get("calculation", "difference"),
                    },
                )
                self.nodes[loss_node.name] = loss_node

        # Map parent relationships based on children lists
        self._map_parent_relationships()

    def _map_parent_relationships(self) -> None:
        """Map parent references based on children lists."""
        for node_name, node in self.nodes.items():
            for child_name in node.children:
                if child_name in self.nodes:
                    self.nodes[child_name].parent = node_name
                else:
                    logger.warning(
                        "Child %r of node %r not found in hierarchy",
                        child_name,
                        node_name,
                    )

    def _assign_levels(self) -> None:
        """Assign hierarchy levels based on depth from root.

        Uses BFS from root node(s) to assign levels to all nodes.

        Raises:
            ValueError: If no root node is found.
        """
        # Find root nodes (nodes with no parent)
        root_nodes = [name for name, node in self.nodes.items() if node.parent is None]

        if not root_nodes:
            msg = "No root node found in hierarchy"
            raise ValueError(msg)

        if len(root_nodes) > 1:
            logger.warning("Multiple root nodes found: %s", root_nodes)

        # BFS to assign levels
        for root in root_nodes:
            self._assign_level_recursive(root, 0)

    def _assign_level_recursive(self, node_name: str, level: int) -> None:
        """Recursively assign levels to nodes using DFS.

        Args:
            node_name: Name of current node.
            level: Level to assign.
        """
        if node_name not in self.nodes:
            return

        node = self.nodes[node_name]
        node.level = level

        for child_name in node.children:
            self._assign_level_recursive(child_name, level + 1)

    def add_node(
        self,
        name: str,
        level: int = 0,
        node_type: str = "aggregate",
        children: list[str] | None = None,
        parent: str | None = None,
        aggregation_weights: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HierarchyNode:
        """Add a node to the hierarchy programmatically.

        Args:
            name: Unique node identifier.
            level: Hierarchy level (0=root).
            node_type: Node type ("aggregate", "bottom", "calculated").
            children: List of child node names.
            parent: Parent node name.
            aggregation_weights: Weights for aggregation.
            metadata: Additional node properties.

        Returns:
            Created HierarchyNode instance.

        Raises:
            ValueError: If node with same name already exists.

        Example:
            >>> hierarchy = HierarchyDefinition()
            >>> hierarchy.add_node("National", level=0, node_type="aggregate")
            >>> hierarchy.add_node("SE", level=1, parent="National")
        """
        if name in self.nodes:
            msg = f"Node {name!r} already exists in hierarchy"
            raise ValueError(msg)

        node = HierarchyNode(
            name=name,
            level=level,
            node_type=node_type,
            children=children or [],
            parent=parent,
            aggregation_weights=aggregation_weights or {},
            metadata=metadata or {},
        )

        self.nodes[name] = node

        # Update parent's children list
        if parent and parent in self.nodes:
            if name not in self.nodes[parent].children:
                self.nodes[parent].children.append(name)

        logger.debug("Added node %s at level %d", name, level)

        return node

    def build_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph from hierarchy.

        Creates a directed graph where edges point from parent to child.
        This is useful for graph traversal and validation.

        Returns:
            NetworkX DiGraph representing the hierarchy.

        Example:
            >>> hierarchy.build_graph()
            >>> print(hierarchy.graph.number_of_nodes())
            26
        """
        logger.info("Building hierarchy graph")

        self.graph = nx.DiGraph()

        # Add nodes with attributes
        for node_name, node in self.nodes.items():
            self.graph.add_node(
                node_name,
                level=node.level,
                node_type=node.node_type,
                **node.metadata,
            )

        # Add edges (parent → child)
        for node_name, node in self.nodes.items():
            for child_name in node.children:
                if child_name in self.nodes:
                    weight = node.get_aggregation_weight(child_name)
                    self.graph.add_edge(node_name, child_name, weight=weight)

        logger.info(
            "Graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        return self.graph

    def build_aggregation_matrix(self) -> np.ndarray:
        """Construct aggregation matrix S.

        The aggregation matrix S maps bottom-level series to all series:
            y_all = S @ y_bottom

        Each row corresponds to a series (sorted by level, then name).
        Each column corresponds to a bottom-level series.

        For aggregate nodes, the row contains 1s in columns corresponding
        to their bottom-level descendants.

        Returns:
            Aggregation matrix S of shape (n_total, n_bottom).

        Example:
            >>> S = hierarchy.build_aggregation_matrix()
            >>> print(f"Shape: {S.shape}")
            Shape: (26, 17)

            >>> # Aggregate predictions
            >>> y_all = S @ y_bottom
        """
        logger.info("Building aggregation matrix")

        # Get bottom-level nodes (sorted for consistency)
        bottom_nodes = sorted(self.get_bottom_level_nodes())
        n_bottom = len(bottom_nodes)

        # Get all nodes sorted by level then name
        all_nodes = sorted(
            self.nodes.keys(),
            key=lambda x: (self.nodes[x].level, x),
        )
        n_total = len(all_nodes)

        # Create index mappings
        bottom_idx = {name: i for i, name in enumerate(bottom_nodes)}
        node_idx = {name: i for i, name in enumerate(all_nodes)}

        # Initialize matrix with zeros
        S = np.zeros((n_total, n_bottom))

        # Fill matrix
        for node_name in all_nodes:
            node = self.nodes[node_name]
            row_idx = node_idx[node_name]

            if node.node_type == "bottom":
                # Identity for bottom-level nodes
                if node_name in bottom_idx:
                    S[row_idx, bottom_idx[node_name]] = 1.0
            elif node.node_type == "aggregate":
                # Sum of bottom-level descendants
                descendants = self._get_bottom_descendants(node_name)
                for desc_name in descendants:
                    if desc_name in bottom_idx:
                        weight = node.get_aggregation_weight(desc_name)
                        S[row_idx, bottom_idx[desc_name]] = weight
            # "calculated" nodes are handled separately (not in aggregation)

        self.aggregation_matrix = S

        logger.info("Aggregation matrix shape: %s", S.shape)

        return S

    def _get_bottom_descendants(self, node_name: str) -> list[str]:
        """Get all bottom-level descendants of a node.

        Recursively finds all leaf nodes (bottom-level) that are
        descendants of the given node.

        Args:
            node_name: Name of the node.

        Returns:
            List of bottom-level descendant node names.
        """
        if node_name not in self.nodes:
            return []

        node = self.nodes[node_name]

        if node.node_type == "bottom":
            return [node_name]

        descendants = []
        for child_name in node.children:
            descendants.extend(self._get_bottom_descendants(child_name))

        return descendants

    def validate_structure(self) -> list[str]:
        """Validate hierarchy structure for consistency.

        Performs various validation checks:
        - No cycles in the graph
        - Single root node (or warns for multiple)
        - All children exist as nodes
        - No orphan nodes (except root)
        - Graph is connected

        Returns:
            List of validation error messages (empty if valid).

        Example:
            >>> errors = hierarchy.validate_structure()
            >>> if errors:
            ...     print(f"Validation failed: {errors}")
            ... else:
            ...     print("Hierarchy is valid")
        """
        errors: list[str] = []

        # Need graph for validation
        if self.graph is None:
            self.build_graph()

        # Check for cycles
        if self.graph is not None and not nx.is_directed_acyclic_graph(self.graph):
            errors.append("Hierarchy contains cycles")

        # Check for root nodes
        if self.graph is not None:
            root_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
            if len(root_nodes) == 0:
                errors.append("No root node found")
            elif len(root_nodes) > 1:
                errors.append(f"Multiple root nodes: {root_nodes}")

        # Check for orphan nodes (non-root nodes without parent)
        for node_name, node in self.nodes.items():
            if node.parent is None and node.level > 0:
                errors.append(f"Orphan node: {node_name} (level {node.level} but no parent)")

        # Check that all children exist
        for node_name, node in self.nodes.items():
            for child_name in node.children:
                if child_name not in self.nodes:
                    errors.append(f"Missing child: {child_name} (parent: {node_name})")

        # Check connectivity
        if self.graph is not None and len(self.graph) > 0:
            undirected = self.graph.to_undirected()
            if not nx.is_connected(undirected):
                n_components = nx.number_connected_components(undirected)
                errors.append(f"Graph has {n_components} disconnected components")

        if errors:
            logger.warning("Validation errors: %s", errors)
        else:
            logger.info("Hierarchy structure is valid")

        return errors

    def get_node(self, name: str) -> HierarchyNode | None:
        """Get node by name.

        Args:
            name: Node name to look up.

        Returns:
            HierarchyNode instance, or None if not found.

        Example:
            >>> node = hierarchy.get_node("SE")
            >>> print(node.children)
            ['Area_SP', 'Area_RJ', ...]
        """
        return self.nodes.get(name)

    def get_nodes_at_level(self, level: int) -> list[HierarchyNode]:
        """Get all nodes at specified level.

        Args:
            level: Hierarchy level (0=National, 1=Subsystem, 2=Area).

        Returns:
            List of nodes at the specified level.

        Example:
            >>> subsystems = hierarchy.get_nodes_at_level(1)
            >>> print([n.name for n in subsystems])
            ['SE', 'S', 'NE', 'N']
        """
        return [node for node in self.nodes.values() if node.level == level]

    def get_bottom_level_nodes(self) -> list[str]:
        """Get all bottom-level (leaf) node names.

        Bottom-level nodes are nodes with node_type="bottom" that are
        directly forecasted and form the basis for aggregation.

        Returns:
            List of bottom-level node names.

        Example:
            >>> areas = hierarchy.get_bottom_level_nodes()
            >>> print(f"Number of areas: {len(areas)}")
            Number of areas: 17
        """
        return [name for name, node in self.nodes.items() if node.node_type == "bottom"]

    def get_children(self, node_name: str) -> list[str]:
        """Get children of a node.

        Args:
            node_name: Name of the parent node.

        Returns:
            List of child node names (empty if not found or no children).

        Example:
            >>> children = hierarchy.get_children("SE")
            >>> print(children)
            ['Area_SP', 'Area_RJ', 'Area_MG', 'Area_ES', 'Area_GO']
        """
        node = self.nodes.get(node_name)
        return node.children.copy() if node else []

    def get_parent(self, node_name: str) -> str | None:
        """Get parent of a node.

        Args:
            node_name: Name of the child node.

        Returns:
            Parent node name, or None if not found or is root.

        Example:
            >>> parent = hierarchy.get_parent("Area_SP")
            >>> print(parent)
            SE
        """
        node = self.nodes.get(node_name)
        return node.parent if node else None

    def get_aggregation_path(self, node_name: str) -> list[str]:
        """Get path from node to root.

        Traverses parent relationships to find the aggregation path
        from a node up to the root.

        Args:
            node_name: Starting node name.

        Returns:
            List of node names from node to root.

        Example:
            >>> path = hierarchy.get_aggregation_path("Area_SP")
            >>> print(path)
            ['Area_SP', 'SE', 'BR_National']
        """
        path = []
        current = node_name

        while current is not None:
            path.append(current)
            current = self.get_parent(current)

        return path

    def get_node_names_sorted(self) -> list[str]:
        """Get all node names sorted by level and name.

        Returns:
            Sorted list of node names (used for matrix indexing).
        """
        return sorted(
            self.nodes.keys(),
            key=lambda x: (self.nodes[x].level, x),
        )

    def export_to_yaml(self, output_path: str | Path) -> None:
        """Export hierarchy to YAML file.

        Serializes the hierarchy structure to a YAML file that can
        be loaded again later.

        Args:
            output_path: Path to output YAML file.

        Example:
            >>> hierarchy.export_to_yaml("config/hierarchy_export.yaml")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reconstruct hierarchy config
        hierarchy_config: dict[str, Any] = {
            "hierarchy": {
                "national": None,
                "subsystems": {},
                "areas": {},
                "losses": {},
            }
        }

        for node_name, node in self.nodes.items():
            node_data = {
                "name": node.name,
                "children": node.children,
            }

            if node.level == 0 and node.node_type == "aggregate":
                hierarchy_config["hierarchy"]["national"] = node_data
            elif node.level == 1 and node.node_type == "aggregate":
                sub_id = node.metadata.get("subsystem_id", node.name)
                hierarchy_config["hierarchy"]["subsystems"][sub_id] = node_data
            elif node.node_type == "bottom":
                area_id = node.metadata.get("area_id", node.name)
                hierarchy_config["hierarchy"]["areas"][area_id] = {
                    "name": node.name,
                    "type": node.metadata.get("type", "consumption"),
                }
            elif node.node_type == "calculated":
                loss_id = node.metadata.get("loss_id", node.name)
                hierarchy_config["hierarchy"]["losses"][loss_id] = {
                    "name": node.name,
                    "level": node.level,
                    "calculation": node.metadata.get("calculation", "difference"),
                }

        with open(output_path, "w") as f:
            yaml.dump(hierarchy_config, f, default_flow_style=False, sort_keys=False)

        logger.info("Exported hierarchy to %s", output_path)

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the hierarchy.
        """
        n_bottom = len(self.get_bottom_level_nodes())
        return f"HierarchyDefinition(nodes={self.n_nodes}, levels={self.n_levels}, bottom={n_bottom})"
