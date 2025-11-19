# PC-048-06A: Hierarchy Definition System

**Ticket ID:** PC-048-06A  
**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**User Story:** US-1  
**Story Points:** 8  
**Priority:** Critical  
**Assignee:** TBD  
**Status:** 📝 To Do

---

## 📋 Description

Implement `HierarchyDefinition` class that loads hierarchy configurations from YAML, constructs computational graphs using NetworkX, generates aggregation matrices, and validates hierarchy structure. Supports multi-level hierarchies (National → Subsystems → Areas) with special handling for losses and interconnections in Brazilian electric load forecasting system.

**As a** system architect  
**I want** a flexible hierarchy definition system that converts YAML configurations to computational graphs  
**So that** I can define and validate complex hierarchical structures for electric load forecasting

---

## ✅ Acceptance Criteria

- [ ] `HierarchyDefinition` class loads hierarchy from YAML configuration
- [ ] Supports multi-level hierarchies (National → Subsystem → Area)
- [ ] Generates computational graph with aggregation relationships
- [ ] Validates hierarchy completeness and consistency
- [ ] Handles special cases like losses and interconnections
- [ ] Constructs aggregation matrix S for reconciliation
- [ ] NetworkX graph for efficient hierarchy traversal
- [ ] YAML schema validation
- [ ] Support for 4 subsystems, 17 areas, 4 losses, 1 national level
- [ ] Integration with reconciliation framework

---

## 🔧 Implementation Tasks

### 1. Create Hierarchy Module
- [ ] Create `src/models/reconciliation/hierarchy.py`
- [ ] Import NetworkX for graph operations
- [ ] Import PyYAML for configuration loading
- [ ] Import numpy for matrix operations
- [ ] Add module docstrings

### 2. Implement HierarchyNode Dataclass
- [ ] Create `HierarchyNode` dataclass
- [ ] Define node attributes: name, level, type
- [ ] Define relationships: children, parent
- [ ] Define aggregation_weights dictionary
- [ ] Add node metadata storage

### 3. Implement HierarchyDefinition Class
- [ ] Create `HierarchyDefinition` class
- [ ] Initialize with config_path parameter
- [ ] Define node storage dictionary
- [ ] Define graph attribute (NetworkX DiGraph)
- [ ] Define aggregation_matrix attribute

### 4. Implement YAML Schema Definition
- [ ] Create `_define_yaml_schema()` method
- [ ] Define hierarchy section schema
- [ ] Define node properties: name, children, type, level
- [ ] Define constraints section schema
- [ ] Return schema dictionary

### 5. Implement YAML Loading
- [ ] Create `load_from_yaml()` method
- [ ] Open and parse YAML file
- [ ] Validate YAML structure against schema
- [ ] Extract hierarchy definition
- [ ] Extract constraints configuration
- [ ] Handle YAML parsing errors

### 6. Implement Node Creation
- [ ] Create `_create_node()` method
- [ ] Parse node configuration from YAML
- [ ] Create HierarchyNode instance
- [ ] Assign node level based on position
- [ ] Store node in nodes dictionary
- [ ] Return created node

### 7. Implement Hierarchy Graph Construction
- [ ] Create `build_graph()` method
- [ ] Initialize NetworkX DiGraph
- [ ] Add nodes from hierarchy definition
- [ ] Add edges for parent-child relationships
- [ ] Set edge weights for aggregation
- [ ] Store graph structure

### 8. Implement Parent-Child Relationship Mapping
- [ ] Create `_map_relationships()` method
- [ ] Traverse YAML hierarchy structure
- [ ] Link children to parents
- [ ] Set parent references in nodes
- [ ] Validate bidirectional relationships
- [ ] Handle orphan nodes

### 9. Implement Aggregation Matrix Construction
- [ ] Create `build_aggregation_matrix()` method
- [ ] Determine matrix dimensions (bottom-level series)
- [ ] Create matrix S: y_bottom = S * y_aggregated
- [ ] Map aggregation relationships to matrix entries
- [ ] Handle weighted aggregations
- [ ] Return sparse or dense matrix

### 10. Implement Hierarchy Validation
- [ ] Create `validate_structure()` method
- [ ] Check for cycles in graph
- [ ] Verify complete coverage (all nodes reachable)
- [ ] Validate proper aggregation relationships
- [ ] Check for missing nodes
- [ ] Return list of validation errors

### 11. Implement Level Assignment
- [ ] Create `_assign_levels()` method
- [ ] Perform topological sort
- [ ] Assign level based on depth from root
- [ ] Level 0: National, Level 1: Subsystems, Level 2: Areas
- [ ] Store level in each node
- [ ] Validate level consistency

### 12. Implement Node Lookup Methods
- [ ] Create `get_node()` method for name lookup
- [ ] Create `get_nodes_at_level()` method
- [ ] Create `get_children()` method
- [ ] Create `get_parent()` method
- [ ] Create `get_bottom_level_nodes()` method
- [ ] Handle missing node errors

### 13. Implement Aggregation Path Finding
- [ ] Create `get_aggregation_path()` method
- [ ] Find path from bottom node to top
- [ ] Traverse parent relationships
- [ ] Return ordered list of nodes
- [ ] Handle disconnected nodes

### 14. Implement Special Cases Handling
- [ ] Create `_handle_losses()` method
- [ ] Identify loss nodes from YAML
- [ ] Mark as calculated (not forecasted)
- [ ] Create special aggregation rules
- [ ] Integrate with energy balance

### 15. Implement Hierarchy Visualization
- [ ] Create `visualize_hierarchy()` method
- [ ] Generate graph visualization with matplotlib
- [ ] Color nodes by level
- [ ] Show aggregation relationships
- [ ] Save to file or display
- [ ] Optional: interactive visualization

### 16. Implement Hierarchy Export
- [ ] Create `export_to_yaml()` method
- [ ] Serialize hierarchy structure
- [ ] Preserve node metadata
- [ ] Write to YAML file
- [ ] Validate round-trip (load → export → load)

### 17. Handle Edge Cases
- [ ] Handle empty hierarchy
- [ ] Handle single-node hierarchy
- [ ] Handle disconnected components
- [ ] Handle missing children lists
- [ ] Validate node name uniqueness

### 18. Write Comprehensive Tests
- [ ] Create `tests/models/reconciliation/test_hierarchy.py`
- [ ] Test YAML loading
- [ ] Test graph construction
- [ ] Test aggregation matrix generation
- [ ] Test validation logic
- [ ] Test node lookup methods
- [ ] Test with real Brazilian hierarchy (4 subsystems, 17 areas)

### 19. Write Integration Tests
- [ ] Test with complete hierarchy configuration
- [ ] Test with partial hierarchies
- [ ] Test error handling for invalid YAML
- [ ] Test performance with large hierarchies
- [ ] Test serialization round-trip

### 20. Create Example YAML Configurations
- [ ] Create `config/hierarchy_brazil.yaml`
- [ ] Define National level
- [ ] Define 4 Subsystems (SE, S, NE, N)
- [ ] Define 17 Areas
- [ ] Define 4 Loss components
- [ ] Add constraints configuration

---

## 💻 Implementation Details

### HierarchyDefinition Implementation

```python
"""Hierarchy definition and management for hierarchical reconciliation."""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import networkx as nx
import numpy as np
import yaml
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HierarchyNode:
    """
    Represents a single node in the forecasting hierarchy.
    
    Attributes:
        name: Unique identifier for the node
        level: Hierarchy level (0=National, 1=Subsystem, 2=Area)
        node_type: Type of node ('aggregate', 'bottom', 'calculated')
        children: List of child node names
        parent: Parent node name (None for root)
        aggregation_weights: Weights for aggregation (default 1.0 for simple sum)
        metadata: Additional node properties
    """
    
    name: str
    level: int = 0
    node_type: str = 'aggregate'  # 'aggregate', 'bottom', 'calculated'
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    aggregation_weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchyDefinition:
    """
    Manages hierarchical structure for forecasting reconciliation.
    
    The hierarchy represents relationships between forecast series:
    - National level (root)
    - Subsystem level (4 subsystems)
    - Area level (17 areas)
    - Loss components (4 losses)
    
    Aggregation Matrix S:
        y_bottom = S * y_aggregated
        
    For example:
        Area_1 + Area_2 + ... + Area_5 = Subsystem_SE
        Subsystem_SE + Subsystem_S + ... = National
    
    Example YAML:
        hierarchy:
          national:
            name: "BR_National"
            children: ["SE", "S", "NE", "N"]
          subsystems:
            SE:
              name: "SE_Subsystem"
              children: ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"]
    
    Example:
        >>> hierarchy = HierarchyDefinition("config/hierarchy_brazil.yaml")
        >>> hierarchy.validate_structure()
        >>> S = hierarchy.build_aggregation_matrix()
        >>> print(f"Aggregation matrix shape: {S.shape}")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize hierarchy definition.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.nodes: Dict[str, HierarchyNode] = {}
        self.graph: Optional[nx.DiGraph] = None
        self.aggregation_matrix: Optional[np.ndarray] = None
        self.config: Dict[str, Any] = {}
        
        if config_path:
            self.load_from_yaml(config_path)
            self.build_graph()
            self.build_aggregation_matrix()
    
    def load_from_yaml(self, config_path: str) -> None:
        """
        Load hierarchy definition from YAML file.
        
        Args:
            config_path: Path to YAML configuration
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If hierarchy structure is invalid
        """
        logger.info(f"Loading hierarchy from {config_path}")
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Hierarchy config not found: {config_path}")
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if 'hierarchy' not in self.config:
            raise ValueError("YAML must contain 'hierarchy' section")
        
        # Parse hierarchy structure
        self._parse_hierarchy(self.config['hierarchy'])
        
        # Assign levels
        self._assign_levels()
        
        logger.info(f"Loaded {len(self.nodes)} nodes from hierarchy")
    
    def _parse_hierarchy(self, hierarchy_config: Dict[str, Any]) -> None:
        """
        Parse hierarchy configuration and create nodes.
        
        Args:
            hierarchy_config: Hierarchy section from YAML
        """
        # Parse national level
        if 'national' in hierarchy_config:
            national_config = hierarchy_config['national']
            national_node = HierarchyNode(
                name=national_config['name'],
                level=0,
                node_type='aggregate',
                children=national_config.get('children', [])
            )
            self.nodes[national_node.name] = national_node
        
        # Parse subsystems
        if 'subsystems' in hierarchy_config:
            for sub_id, sub_config in hierarchy_config['subsystems'].items():
                sub_node = HierarchyNode(
                    name=sub_config['name'],
                    level=1,
                    node_type='aggregate',
                    children=sub_config.get('children', [])
                )
                self.nodes[sub_node.name] = sub_node
        
        # Parse areas (bottom level)
        if 'areas' in hierarchy_config:
            for area_id, area_config in hierarchy_config['areas'].items():
                area_node = HierarchyNode(
                    name=area_config['name'],
                    level=2,
                    node_type='bottom',
                    children=[],
                    metadata={'type': area_config.get('type', 'consumption')}
                )
                self.nodes[area_node.name] = area_node
        
        # Parse losses (calculated nodes)
        if 'losses' in hierarchy_config:
            for loss_id, loss_config in hierarchy_config['losses'].items():
                loss_node = HierarchyNode(
                    name=loss_config['name'],
                    level=loss_config.get('level', 2),
                    node_type='calculated',
                    children=[],
                    metadata={'calculation': loss_config.get('calculation', 'difference')}
                )
                self.nodes[loss_node.name] = loss_node
        
        # Map parent relationships
        self._map_parent_relationships()
    
    def _map_parent_relationships(self) -> None:
        """Map parent references based on children lists."""
        for node_name, node in self.nodes.items():
            for child_name in node.children:
                if child_name in self.nodes:
                    self.nodes[child_name].parent = node_name
                else:
                    logger.warning(f"Child '{child_name}' not found in hierarchy")
    
    def _assign_levels(self) -> None:
        """Assign hierarchy levels based on depth from root."""
        # Find root (node with no parent)
        root_nodes = [name for name, node in self.nodes.items() if node.parent is None]
        
        if len(root_nodes) == 0:
            raise ValueError("No root node found in hierarchy")
        if len(root_nodes) > 1:
            logger.warning(f"Multiple root nodes found: {root_nodes}")
        
        # BFS to assign levels
        for root in root_nodes:
            self._assign_level_recursive(root, 0)
    
    def _assign_level_recursive(self, node_name: str, level: int) -> None:
        """Recursively assign levels to nodes."""
        if node_name not in self.nodes:
            return
        
        node = self.nodes[node_name]
        node.level = level
        
        for child_name in node.children:
            self._assign_level_recursive(child_name, level + 1)
    
    def build_graph(self) -> nx.DiGraph:
        """
        Build NetworkX directed graph from hierarchy.
        
        Returns:
            NetworkX DiGraph representing hierarchy
        """
        logger.info("Building hierarchy graph")
        
        self.graph = nx.DiGraph()
        
        # Add nodes
        for node_name, node in self.nodes.items():
            self.graph.add_node(
                node_name,
                level=node.level,
                node_type=node.node_type
            )
        
        # Add edges (parent → child)
        for node_name, node in self.nodes.items():
            for child_name in node.children:
                if child_name in self.nodes:
                    weight = node.aggregation_weights.get(child_name, 1.0)
                    self.graph.add_edge(node_name, child_name, weight=weight)
        
        logger.info(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def build_aggregation_matrix(self) -> np.ndarray:
        """
        Construct aggregation matrix S.
        
        The aggregation matrix S maps bottom-level series to aggregated series:
            y_aggregated = S @ y_bottom
        
        Returns:
            Aggregation matrix S
        """
        logger.info("Building aggregation matrix")
        
        # Get bottom-level nodes
        bottom_nodes = self.get_bottom_level_nodes()
        n_bottom = len(bottom_nodes)
        
        # Get all nodes sorted by level
        all_nodes = sorted(self.nodes.keys(), 
                          key=lambda x: (self.nodes[x].level, x))
        n_total = len(all_nodes)
        
        # Create mapping
        bottom_idx = {name: i for i, name in enumerate(bottom_nodes)}
        node_idx = {name: i for i, name in enumerate(all_nodes)}
        
        # Initialize matrix
        S = np.zeros((n_total, n_bottom))
        
        # Bottom level: identity
        for name in bottom_nodes:
            S[node_idx[name], bottom_idx[name]] = 1.0
        
        # Aggregate levels: sum of children
        for node_name in all_nodes:
            node = self.nodes[node_name]
            if node.node_type == 'aggregate':
                # Find all bottom-level descendants
                descendants = self._get_bottom_descendants(node_name)
                for desc_name in descendants:
                    if desc_name in bottom_idx:
                        weight = node.aggregation_weights.get(desc_name, 1.0)
                        S[node_idx[node_name], bottom_idx[desc_name]] = weight
        
        self.aggregation_matrix = S
        logger.info(f"Aggregation matrix shape: {S.shape}")
        
        return S
    
    def _get_bottom_descendants(self, node_name: str) -> List[str]:
        """Get all bottom-level descendants of a node."""
        if node_name not in self.nodes:
            return []
        
        node = self.nodes[node_name]
        
        if node.node_type == 'bottom':
            return [node_name]
        
        descendants = []
        for child_name in node.children:
            descendants.extend(self._get_bottom_descendants(child_name))
        
        return descendants
    
    def validate_structure(self) -> List[str]:
        """
        Validate hierarchy structure.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for cycles
        if self.graph and not nx.is_directed_acyclic_graph(self.graph):
            errors.append("Hierarchy contains cycles")
        
        # Check for disconnected nodes
        if self.graph:
            root_nodes = [n for n in self.graph.nodes() 
                         if self.graph.in_degree(n) == 0]
            if len(root_nodes) == 0:
                errors.append("No root node found")
            elif len(root_nodes) > 1:
                errors.append(f"Multiple root nodes: {root_nodes}")
        
        # Check for orphan nodes
        for node_name, node in self.nodes.items():
            if node.parent is None and node.level > 0:
                errors.append(f"Orphan node: {node_name}")
        
        # Check aggregation consistency
        for node_name, node in self.nodes.items():
            for child_name in node.children:
                if child_name not in self.nodes:
                    errors.append(f"Missing child: {child_name} (parent: {node_name})")
        
        if errors:
            logger.warning(f"Validation errors: {errors}")
        else:
            logger.info("Hierarchy structure valid")
        
        return errors
    
    def get_node(self, name: str) -> Optional[HierarchyNode]:
        """Get node by name."""
        return self.nodes.get(name)
    
    def get_nodes_at_level(self, level: int) -> List[HierarchyNode]:
        """Get all nodes at specified level."""
        return [node for node in self.nodes.values() if node.level == level]
    
    def get_bottom_level_nodes(self) -> List[str]:
        """Get all bottom-level (leaf) node names."""
        return [name for name, node in self.nodes.items() 
                if node.node_type == 'bottom']
    
    def get_children(self, node_name: str) -> List[str]:
        """Get children of a node."""
        node = self.nodes.get(node_name)
        return node.children if node else []
    
    def get_parent(self, node_name: str) -> Optional[str]:
        """Get parent of a node."""
        node = self.nodes.get(node_name)
        return node.parent if node else None
```

---

## 🧪 Testing & Validation

```python
"""Tests for hierarchy definition."""
import pytest
import numpy as np
from pathlib import Path

from src.models.reconciliation.hierarchy import HierarchyDefinition, HierarchyNode


@pytest.fixture
def sample_hierarchy_yaml(tmp_path):
    """Create sample hierarchy YAML."""
    yaml_content = """
hierarchy:
  national:
    name: "BR_National"
    children: ["SE", "S"]
  
  subsystems:
    SE:
      name: "SE_Subsystem"
      children: ["Area_1", "Area_2"]
    S:
      name: "S_Subsystem"
      children: ["Area_3"]
  
  areas:
    Area_1:
      name: "Area_1"
      type: "consumption"
    Area_2:
      name: "Area_2"
      type: "consumption"
    Area_3:
      name: "Area_3"
      type: "consumption"
"""
    yaml_file = tmp_path / "hierarchy.yaml"
    yaml_file.write_text(yaml_content)
    return str(yaml_file)


def test_hierarchy_loading(sample_hierarchy_yaml):
    """Test hierarchy loading from YAML."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    assert len(hierarchy.nodes) == 6  # 1 national + 2 subsystems + 3 areas
    assert "BR_National" in hierarchy.nodes
    assert "Area_1" in hierarchy.nodes


def test_graph_construction(sample_hierarchy_yaml):
    """Test graph construction."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    assert hierarchy.graph is not None
    assert hierarchy.graph.number_of_nodes() == 6
    assert hierarchy.graph.number_of_edges() == 5  # National→2, SE→2, S→1


def test_aggregation_matrix(sample_hierarchy_yaml):
    """Test aggregation matrix generation."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    S = hierarchy.aggregation_matrix
    assert S is not None
    assert S.shape[1] == 3  # 3 bottom-level areas
    
    # Check subsystem aggregation
    # SE_Subsystem = Area_1 + Area_2
    se_idx = list(hierarchy.nodes.keys()).index("SE_Subsystem")
    assert S[se_idx, 0] == 1.0  # Area_1
    assert S[se_idx, 1] == 1.0  # Area_2


def test_hierarchy_validation(sample_hierarchy_yaml):
    """Test hierarchy validation."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    errors = hierarchy.validate_structure()
    assert len(errors) == 0


def test_node_lookup(sample_hierarchy_yaml):
    """Test node lookup methods."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    # Get node
    national = hierarchy.get_node("BR_National")
    assert national is not None
    assert national.level == 0
    
    # Get children
    children = hierarchy.get_children("BR_National")
    assert len(children) == 2
    
    # Get parent
    parent = hierarchy.get_parent("Area_1")
    assert parent == "SE_Subsystem"


def test_bottom_level_nodes(sample_hierarchy_yaml):
    """Test bottom level node retrieval."""
    hierarchy = HierarchyDefinition(sample_hierarchy_yaml)
    
    bottom_nodes = hierarchy.get_bottom_level_nodes()
    assert len(bottom_nodes) == 3
    assert "Area_1" in bottom_nodes
```

---

## 📝 Technical Notes

### Aggregation Matrix Construction
- Identity for bottom-level series
- Sum relationships for aggregates
- Sparse matrix for large hierarchies

### YAML Schema
- Clear node hierarchy
- Children relationships explicit
- Support for weights and metadata

---

## 🔗 Dependencies

**Depends On:**
- Epic-04: Hierarchical model outputs
- NetworkX library
- PyYAML library

**Blocks:**
- PC-049-06A: Base Reconciler Interface
- PC-050-06A: MinT Reconciler
- All Epic-06A tickets

---

## 📊 Definition of Done

- [ ] All acceptance criteria met
- [ ] YAML loading working correctly
- [ ] Graph construction validated
- [ ] Aggregation matrix accurate
- [ ] Hierarchy validation comprehensive
- [ ] Node lookup methods functional
- [ ] Special cases (losses) handled
- [ ] Unit tests >85% coverage
- [ ] Integration tests with Brazilian hierarchy pass
- [ ] Documentation complete
- [ ] Code reviewed and approved

---

**Epic:** [Epic-06A: Core Hierarchical Reconciliation](../epics/Epic-06A.md)  
**Next:** [PC-049-06A: Base Reconciliation Interface](PC-049-06A-base-reconciler-interface.md)
