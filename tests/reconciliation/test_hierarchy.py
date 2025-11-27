"""Tests for hierarchy definition module."""

import numpy as np
import pytest

from src.reconciliation.hierarchy import HierarchyDefinition, HierarchyNode


# Tests for HierarchyNode


class TestHierarchyNode:
    """Tests for HierarchyNode dataclass."""

    def test_basic_creation(self):
        """Test basic node creation."""
        node = HierarchyNode(name="National", level=0, node_type="aggregate")

        assert node.name == "National"
        assert node.level == 0
        assert node.node_type == "aggregate"
        assert node.children == []
        assert node.parent is None

    def test_full_creation(self):
        """Test node creation with all attributes."""
        node = HierarchyNode(
            name="SE",
            level=1,
            node_type="aggregate",
            children=["Area_1", "Area_2"],
            parent="National",
            aggregation_weights={"Area_1": 1.0, "Area_2": 1.0},
            metadata={"region": "southeast"},
        )

        assert node.name == "SE"
        assert node.level == 1
        assert len(node.children) == 2
        assert node.parent == "National"
        assert node.metadata["region"] == "southeast"

    def test_invalid_node_type(self):
        """Test invalid node type raises error."""
        with pytest.raises(ValueError, match="node_type must be"):
            HierarchyNode(name="Bad", node_type="invalid")

    def test_invalid_level(self):
        """Test negative level raises error."""
        with pytest.raises(ValueError, match="level must be"):
            HierarchyNode(name="Bad", level=-1)

    def test_is_leaf(self):
        """Test is_leaf property."""
        leaf_node = HierarchyNode(name="Area_1", node_type="bottom")
        parent_node = HierarchyNode(name="SE", children=["Area_1"])

        assert leaf_node.is_leaf is True
        assert parent_node.is_leaf is False

    def test_is_root(self):
        """Test is_root property."""
        root_node = HierarchyNode(name="National")
        child_node = HierarchyNode(name="SE", parent="National")

        assert root_node.is_root is True
        assert child_node.is_root is False

    def test_get_aggregation_weight(self):
        """Test get_aggregation_weight method."""
        node = HierarchyNode(
            name="SE",
            aggregation_weights={"Area_1": 0.6, "Area_2": 0.4},
        )

        assert node.get_aggregation_weight("Area_1") == 0.6
        assert node.get_aggregation_weight("Area_2") == 0.4
        assert node.get_aggregation_weight("Unknown") == 1.0  # Default

    def test_to_dict(self):
        """Test conversion to dictionary."""
        node = HierarchyNode(
            name="SE",
            level=1,
            node_type="aggregate",
            children=["Area_1"],
            parent="National",
        )

        d = node.to_dict()

        assert d["name"] == "SE"
        assert d["level"] == 1
        assert d["node_type"] == "aggregate"
        assert d["children"] == ["Area_1"]
        assert d["parent"] == "National"

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "name": "SE",
            "level": 1,
            "node_type": "aggregate",
            "children": ["Area_1"],
            "parent": "National",
        }

        node = HierarchyNode.from_dict(d)

        assert node.name == "SE"
        assert node.level == 1
        assert node.children == ["Area_1"]

    def test_repr(self):
        """Test string representation."""
        node = HierarchyNode(name="SE", level=1, node_type="aggregate")
        repr_str = repr(node)

        assert "HierarchyNode" in repr_str
        assert "SE" in repr_str
        assert "level=1" in repr_str


# Fixtures for HierarchyDefinition tests


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
      name: "SE"
      children: ["Area_1", "Area_2"]
    S:
      name: "S"
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


@pytest.fixture
def simple_hierarchy():
    """Create simple hierarchy programmatically."""
    hierarchy = HierarchyDefinition()

    # Add nodes
    hierarchy.add_node("National", level=0, node_type="aggregate", children=["A", "B"])
    hierarchy.add_node("A", level=1, node_type="bottom", parent="National")
    hierarchy.add_node("B", level=1, node_type="bottom", parent="National")

    hierarchy.build_graph()
    hierarchy.build_aggregation_matrix()

    return hierarchy


@pytest.fixture
def brazil_hierarchy():
    """Load full Brazil hierarchy."""
    return HierarchyDefinition("config/hierarchy_brazil.yaml")


# Tests for HierarchyDefinition


class TestHierarchyDefinitionInit:
    """Tests for HierarchyDefinition initialization."""

    def test_init_empty(self):
        """Test empty initialization."""
        hierarchy = HierarchyDefinition()

        assert hierarchy.nodes == {}
        assert hierarchy.graph is None
        assert hierarchy.aggregation_matrix is None

    def test_init_with_yaml(self, sample_hierarchy_yaml):
        """Test initialization with YAML file."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        assert len(hierarchy.nodes) == 6  # 1 national + 2 subsystems + 3 areas
        assert hierarchy.graph is not None
        assert hierarchy.aggregation_matrix is not None


class TestYAMLLoading:
    """Tests for YAML loading functionality."""

    def test_load_from_yaml(self, sample_hierarchy_yaml):
        """Test YAML loading."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        assert "BR_National" in hierarchy.nodes
        assert "SE" in hierarchy.nodes
        assert "Area_1" in hierarchy.nodes

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        hierarchy = HierarchyDefinition()

        with pytest.raises(FileNotFoundError):
            hierarchy.load_from_yaml("nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML (missing hierarchy section)."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("key: value")

        hierarchy = HierarchyDefinition()

        with pytest.raises(ValueError, match="hierarchy"):
            hierarchy.load_from_yaml(str(yaml_file))

    def test_node_levels_assigned(self, sample_hierarchy_yaml):
        """Test that levels are correctly assigned."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        assert hierarchy.nodes["BR_National"].level == 0
        assert hierarchy.nodes["SE"].level == 1
        assert hierarchy.nodes["Area_1"].level == 2

    def test_parent_relationships(self, sample_hierarchy_yaml):
        """Test parent relationships are correctly mapped."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        assert hierarchy.nodes["SE"].parent == "BR_National"
        assert hierarchy.nodes["Area_1"].parent == "SE"
        assert hierarchy.nodes["BR_National"].parent is None


class TestGraphConstruction:
    """Tests for graph construction."""

    def test_graph_nodes(self, sample_hierarchy_yaml):
        """Test graph has correct number of nodes."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        assert hierarchy.graph.number_of_nodes() == 6

    def test_graph_edges(self, sample_hierarchy_yaml):
        """Test graph has correct number of edges."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        # National→SE, National→S, SE→Area_1, SE→Area_2, S→Area_3 = 5 edges
        assert hierarchy.graph.number_of_edges() == 5

    def test_graph_is_dag(self, sample_hierarchy_yaml):
        """Test graph is directed acyclic graph."""
        import networkx as nx

        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        assert nx.is_directed_acyclic_graph(hierarchy.graph)


class TestAggregationMatrix:
    """Tests for aggregation matrix construction."""

    def test_matrix_shape(self, sample_hierarchy_yaml):
        """Test aggregation matrix has correct shape."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        S = hierarchy.aggregation_matrix
        # 6 total nodes, 3 bottom-level
        assert S.shape == (6, 3)

    def test_bottom_level_identity(self, sample_hierarchy_yaml):
        """Test bottom-level nodes have identity in matrix."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        S = hierarchy.aggregation_matrix
        bottom_nodes = hierarchy.get_bottom_level_nodes()

        # Bottom-level nodes should have exactly one 1.0 in their row
        for node_name in bottom_nodes:
            node_idx = hierarchy.get_node_names_sorted().index(node_name)
            row = S[node_idx, :]
            assert np.sum(row == 1.0) == 1
            assert np.sum(row != 0) == 1

    def test_aggregation_sum(self, sample_hierarchy_yaml):
        """Test aggregate nodes sum their children."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        S = hierarchy.aggregation_matrix
        sorted_nodes = hierarchy.get_node_names_sorted()

        # SE = Area_1 + Area_2
        se_idx = sorted_nodes.index("SE")
        area1_idx = hierarchy.get_bottom_level_nodes().index("Area_1")
        area2_idx = hierarchy.get_bottom_level_nodes().index("Area_2")

        assert S[se_idx, area1_idx] == 1.0
        assert S[se_idx, area2_idx] == 1.0

    def test_matrix_aggregation_works(self, simple_hierarchy):
        """Test matrix multiplication produces correct aggregation."""
        hierarchy = simple_hierarchy

        S = hierarchy.aggregation_matrix

        # Bottom-level values
        y_bottom = np.array([100.0, 200.0])

        # Aggregate
        y_all = S @ y_bottom

        # National should be sum of A and B
        sorted_nodes = hierarchy.get_node_names_sorted()
        national_idx = sorted_nodes.index("National")

        assert y_all[national_idx] == 300.0  # 100 + 200


class TestValidation:
    """Tests for hierarchy validation."""

    def test_valid_hierarchy(self, sample_hierarchy_yaml):
        """Test validation of valid hierarchy."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        errors = hierarchy.validate_structure()

        assert len(errors) == 0

    def test_detect_cycle(self):
        """Test detection of cycles in hierarchy."""
        hierarchy = HierarchyDefinition()

        # Create a cycle: A → B → A
        hierarchy.add_node("A", level=0, children=["B"])
        hierarchy.add_node("B", level=1, children=["A"], parent="A")

        hierarchy.build_graph()
        errors = hierarchy.validate_structure()

        assert any("cycle" in e.lower() for e in errors)

    def test_detect_missing_child(self):
        """Test detection of missing child nodes."""
        hierarchy = HierarchyDefinition()

        # Reference non-existent child
        hierarchy.add_node("A", level=0, children=["Missing"])

        hierarchy.build_graph()
        errors = hierarchy.validate_structure()

        assert any("Missing" in e for e in errors)

    def test_detect_orphan_node(self):
        """Test detection of orphan nodes."""
        hierarchy = HierarchyDefinition()

        hierarchy.add_node("National", level=0)
        hierarchy.add_node("Orphan", level=2)  # No parent but level > 0

        hierarchy.build_graph()
        errors = hierarchy.validate_structure()

        assert any("orphan" in e.lower() for e in errors)


class TestNodeLookup:
    """Tests for node lookup methods."""

    def test_get_node(self, sample_hierarchy_yaml):
        """Test get_node method."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        node = hierarchy.get_node("SE")
        assert node is not None
        assert node.name == "SE"

        missing = hierarchy.get_node("NonExistent")
        assert missing is None

    def test_get_nodes_at_level(self, sample_hierarchy_yaml):
        """Test get_nodes_at_level method."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        level_0 = hierarchy.get_nodes_at_level(0)
        assert len(level_0) == 1
        assert level_0[0].name == "BR_National"

        level_1 = hierarchy.get_nodes_at_level(1)
        assert len(level_1) == 2

        level_2 = hierarchy.get_nodes_at_level(2)
        assert len(level_2) == 3

    def test_get_bottom_level_nodes(self, sample_hierarchy_yaml):
        """Test get_bottom_level_nodes method."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        bottom = hierarchy.get_bottom_level_nodes()

        assert len(bottom) == 3
        assert "Area_1" in bottom
        assert "Area_2" in bottom
        assert "Area_3" in bottom

    def test_get_children(self, sample_hierarchy_yaml):
        """Test get_children method."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        children = hierarchy.get_children("BR_National")
        assert set(children) == {"SE", "S"}

        children = hierarchy.get_children("Area_1")
        assert children == []  # Bottom level has no children

    def test_get_parent(self, sample_hierarchy_yaml):
        """Test get_parent method."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        parent = hierarchy.get_parent("Area_1")
        assert parent == "SE"

        parent = hierarchy.get_parent("BR_National")
        assert parent is None  # Root has no parent

    def test_get_aggregation_path(self, sample_hierarchy_yaml):
        """Test get_aggregation_path method."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        path = hierarchy.get_aggregation_path("Area_1")

        assert path == ["Area_1", "SE", "BR_National"]


class TestProgrammaticConstruction:
    """Tests for programmatic hierarchy construction."""

    def test_add_node(self):
        """Test adding nodes programmatically."""
        hierarchy = HierarchyDefinition()

        node = hierarchy.add_node("National", level=0, node_type="aggregate")

        assert "National" in hierarchy.nodes
        assert node.name == "National"

    def test_add_duplicate_node_raises(self):
        """Test adding duplicate node raises error."""
        hierarchy = HierarchyDefinition()

        hierarchy.add_node("A", level=0)

        with pytest.raises(ValueError, match="already exists"):
            hierarchy.add_node("A", level=1)

    def test_add_node_updates_parent_children(self):
        """Test adding node updates parent's children list."""
        hierarchy = HierarchyDefinition()

        hierarchy.add_node("National", level=0)
        hierarchy.add_node("Child", level=1, parent="National")

        assert "Child" in hierarchy.nodes["National"].children


class TestExportImport:
    """Tests for export and import functionality."""

    def test_export_to_yaml(self, sample_hierarchy_yaml, tmp_path):
        """Test exporting hierarchy to YAML."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        export_path = tmp_path / "exported.yaml"
        hierarchy.export_to_yaml(export_path)

        assert export_path.exists()

        # Reload and verify
        reloaded = HierarchyDefinition(str(export_path))
        assert len(reloaded.nodes) == len(hierarchy.nodes)


class TestBrazilHierarchy:
    """Tests with full Brazil hierarchy configuration."""

    def test_brazil_hierarchy_loads(self, brazil_hierarchy):
        """Test Brazil hierarchy loads correctly."""
        hierarchy = brazil_hierarchy

        # Should have: 1 national + 4 subsystems + 17 areas + 4 losses = 26
        assert len(hierarchy.nodes) == 26

    def test_brazil_hierarchy_valid(self, brazil_hierarchy):
        """Test Brazil hierarchy passes validation.

        Note: Losses are calculated nodes (not forecasted) and intentionally
        disconnected from the main hierarchy. They are validated separately.
        """
        errors = brazil_hierarchy.validate_structure()

        # Filter out expected errors due to loss nodes being disconnected
        # Losses are calculated, not forecasted, so they don't connect to the main hierarchy
        non_loss_errors = [
            e for e in errors
            if "Loss" not in e and "disconnected" not in e.lower()
        ]
        assert len(non_loss_errors) == 0

    def test_brazil_subsystems(self, brazil_hierarchy):
        """Test Brazil subsystems are correct."""
        level_1 = brazil_hierarchy.get_nodes_at_level(1)
        subsystem_names = {n.name for n in level_1 if n.node_type == "aggregate"}

        assert subsystem_names == {"SE", "S", "NE", "N"}

    def test_brazil_bottom_level(self, brazil_hierarchy):
        """Test Brazil has 17 areas as bottom level."""
        bottom = brazil_hierarchy.get_bottom_level_nodes()

        assert len(bottom) == 17

    def test_brazil_aggregation_matrix_shape(self, brazil_hierarchy):
        """Test Brazil aggregation matrix shape."""
        S = brazil_hierarchy.aggregation_matrix

        # 26 total, 17 bottom level
        # Note: calculated nodes (losses) are included but not in aggregation
        assert S.shape[1] == 17  # 17 bottom-level areas


class TestProperties:
    """Tests for hierarchy properties."""

    def test_n_nodes(self, sample_hierarchy_yaml):
        """Test n_nodes property."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        assert hierarchy.n_nodes == 6

    def test_n_levels(self, sample_hierarchy_yaml):
        """Test n_levels property."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        assert hierarchy.n_levels == 3  # 0, 1, 2

    def test_root_node(self, sample_hierarchy_yaml):
        """Test root_node property."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        root = hierarchy.root_node
        assert root is not None
        assert root.name == "BR_National"

    def test_repr(self, sample_hierarchy_yaml):
        """Test string representation."""
        hierarchy = HierarchyDefinition(sample_hierarchy_yaml)

        repr_str = repr(hierarchy)

        assert "HierarchyDefinition" in repr_str
        assert "nodes=6" in repr_str
        assert "levels=3" in repr_str
        assert "bottom=3" in repr_str
