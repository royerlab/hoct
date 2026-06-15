"""Tests for hoct.features.graph module."""


def test_graph_module_integration():
    """Test that graph module has correct integration with dependencies."""
    from hoct.features.graph import (
        EDGE_GT_KEY,
        REGIONPROPS,
        convert_to_2d,
        convert_to_3d,
        create_graph,
    )

    # Verify constants are accessible
    assert EDGE_GT_KEY == "edge_is_gt"
    assert len(REGIONPROPS) == 7

    # Verify main functions exist and are callable
    assert callable(create_graph)
    assert callable(convert_to_3d)
    assert callable(convert_to_2d)
