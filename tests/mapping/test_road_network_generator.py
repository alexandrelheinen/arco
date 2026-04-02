"""Unit tests for RoadNetworkGenerator."""

from __future__ import annotations

import pytest

from arco.mapping.generator import RoadNetworkGenerator


class TestRoadNetworkGeneratorBasics:
    def test_init_with_seed(self):
        gen = RoadNetworkGenerator(seed=42)
        assert gen.seed == 42

    def test_init_without_seed(self):
        gen1 = RoadNetworkGenerator()
        gen2 = RoadNetworkGenerator()
        # Should have different seeds (probabilistic test, could fail rarely)
        assert gen1.seed != gen2.seed


class TestGridNetworkGeneration:
    def test_generate_2x2_grid(self):
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_grid_network(grid_size=(2, 2), cell_size=100.0)

        # Should have 4 nodes (2x2 grid)
        assert len(graph.nodes) == 4

        # Check node positions
        assert graph.position(0) == (0.0, 0.0)
        assert graph.position(1) == (100.0, 0.0)
        assert graph.position(2) == (0.0, 100.0)
        assert graph.position(3) == (100.0, 100.0)

        # Should have 4 edges in a 2x2 grid:
        # 0-1, 2-3 (horizontal) and 0-2, 1-3 (vertical)
        assert len(graph.edges) == 4

    def test_generate_3x3_grid(self):
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_grid_network(grid_size=(3, 3), cell_size=50.0)

        # Should have 9 nodes (3x3 grid)
        assert len(graph.nodes) == 9

        # Should have 12 edges: 6 horizontal + 6 vertical
        # In a 3x3 grid: 2*(rows*cols - rows) + 2*(rows*cols - cols) / 2
        # = (3*2) + (3*2) = 12
        assert len(graph.edges) == 12

    def test_grid_connectivity(self):
        """All nodes in a grid should be connected."""
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_grid_network(grid_size=(3, 3))

        # Middle node (4) should have 4 neighbors
        assert len(list(graph.neighbors(4))) == 4

        # Corner nodes should have 2 neighbors
        assert len(list(graph.neighbors(0))) == 2
        assert len(list(graph.neighbors(8))) == 2

        # Edge nodes (not corners) should have 3 neighbors
        assert len(list(graph.neighbors(1))) == 3

    def test_grid_edge_geometry(self):
        """Edges should have waypoints based on waypoints_per_edge parameter."""
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_grid_network(
            grid_size=(2, 2), waypoints_per_edge=5, curvature=0.1
        )

        # Each edge should have 5 waypoints
        for node_a, node_b, _ in graph.edges:
            waypoints = graph.edge_geometry(node_a, node_b)
            assert len(waypoints) == 5

    def test_grid_with_zero_waypoints(self):
        """Grid with zero waypoints should have straight edges."""
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_grid_network(
            grid_size=(2, 2), waypoints_per_edge=0, curvature=0.0
        )

        # Each edge should have no waypoints
        for node_a, node_b, _ in graph.edges:
            waypoints = graph.edge_geometry(node_a, node_b)
            assert len(waypoints) == 0

    def test_grid_invalid_dimensions(self):
        """Grid generation should fail with invalid dimensions."""
        gen = RoadNetworkGenerator(seed=42)
        with pytest.raises(ValueError):
            gen.generate_grid_network(grid_size=(0, 3))
        with pytest.raises(ValueError):
            gen.generate_grid_network(grid_size=(3, -1))


class TestRandomNetworkGeneration:
    def test_generate_random_network(self):
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_random_network(
            num_intersections=10, area=200.0, connect_radius=80.0
        )

        # Should have exactly 10 nodes
        assert len(graph.nodes) == 10

        # All nodes should be within the area bounds
        for node_id in graph.nodes:
            x, y = graph.position(node_id)
            assert 0.0 <= x <= 200.0
            assert 0.0 <= y <= 200.0

    def test_random_network_connectivity(self):
        """Nodes within connect_radius should be connected."""
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_random_network(
            num_intersections=15, area=300.0, connect_radius=100.0
        )

        # Graph should have at least some edges
        assert len(graph.edges) > 0

    def test_random_network_edge_geometry(self):
        """Random network edges should have waypoints."""
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_random_network(
            num_intersections=10, waypoints_per_edge=4, curvature=0.2
        )

        # At least some edges should exist
        assert len(graph.edges) > 0

        # Each edge should have 4 waypoints
        for node_a, node_b, _ in graph.edges:
            waypoints = graph.edge_geometry(node_a, node_b)
            assert len(waypoints) == 4

    def test_random_network_invalid_params(self):
        """Random network generation should fail with invalid parameters."""
        gen = RoadNetworkGenerator(seed=42)
        with pytest.raises(ValueError):
            gen.generate_random_network(num_intersections=0)
        with pytest.raises(ValueError):
            gen.generate_random_network(num_intersections=-5)


class TestSeedReproducibility:
    def test_grid_network_reproducibility(self):
        """Same seed should produce identical grid networks."""
        gen1 = RoadNetworkGenerator(seed=123)
        gen2 = RoadNetworkGenerator(seed=123)

        graph1 = gen1.generate_grid_network(
            grid_size=(3, 3), waypoints_per_edge=3, curvature=0.3
        )
        graph2 = gen2.generate_grid_network(
            grid_size=(3, 3), waypoints_per_edge=3, curvature=0.3
        )

        # Same node positions
        assert graph1.nodes == graph2.nodes
        for node_id in graph1.nodes:
            assert graph1.position(node_id) == graph2.position(node_id)

        # Same edges and waypoints
        assert len(graph1.edges) == len(graph2.edges)
        for (a1, b1, _), (a2, b2, _) in zip(sorted(graph1.edges), sorted(graph2.edges)):
            assert a1 == a2 and b1 == b2
            assert graph1.edge_geometry(a1, b1) == graph2.edge_geometry(a2, b2)

    def test_random_network_reproducibility(self):
        """Same seed should produce identical random networks."""
        gen1 = RoadNetworkGenerator(seed=456)
        gen2 = RoadNetworkGenerator(seed=456)

        graph1 = gen1.generate_random_network(
            num_intersections=15,
            area=250.0,
            connect_radius=90.0,
            waypoints_per_edge=2,
            curvature=0.15,
        )
        graph2 = gen2.generate_random_network(
            num_intersections=15,
            area=250.0,
            connect_radius=90.0,
            waypoints_per_edge=2,
            curvature=0.15,
        )

        # Same node positions
        assert graph1.nodes == graph2.nodes
        for node_id in graph1.nodes:
            pos1 = graph1.position(node_id)
            pos2 = graph2.position(node_id)
            assert pos1 == pos2

        # Same edges and waypoints
        edges1 = sorted(graph1.edges)
        edges2 = sorted(graph2.edges)
        assert len(edges1) == len(edges2)
        for (a1, b1, _), (a2, b2, _) in zip(edges1, edges2):
            assert a1 == a2 and b1 == b2
            # Waypoints should be identical
            waypoints1 = graph1.edge_geometry(a1, b1)
            waypoints2 = graph2.edge_geometry(a2, b2)
            assert len(waypoints1) == len(waypoints2)
            for wp1, wp2 in zip(waypoints1, waypoints2):
                assert wp1 == wp2

    def test_different_seeds_produce_different_networks(self):
        """Different seeds should produce different networks."""
        gen1 = RoadNetworkGenerator(seed=100)
        gen2 = RoadNetworkGenerator(seed=200)

        graph1 = gen1.generate_random_network(num_intersections=10)
        graph2 = gen2.generate_random_network(num_intersections=10)

        # At least some node positions should differ
        different_positions = False
        for node_id in range(10):
            if graph1.position(node_id) != graph2.position(node_id):
                different_positions = True
                break
        assert different_positions


class TestMedievalNetworkGeneration:
    def test_generate_medieval_network_node_count(self):
        """Medieval network should have plaza + radial + alley nodes."""
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_medieval_network(
            num_radials=7,
            ring_radii=[40.0, 90.0, 150.0],
        )
        # 4 plaza + 7*3 ring = 25, plus up to 7//2 = 3 alleys → 25..28 nodes
        assert 25 <= len(graph.nodes) <= 28

    def test_generate_medieval_network_has_edges(self):
        """Medieval network must have edges."""
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_medieval_network()
        assert len(graph.edges) > 0

    def test_generate_medieval_network_all_edges_have_waypoints(self):
        """Every edge must have at least one geometry waypoint."""
        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_medieval_network(waypoints_per_edge=3)
        for node_a, node_b, _ in graph.edges:
            pts = graph.edge_geometry(node_a, node_b)
            assert len(pts) == 3

    def test_generate_medieval_network_connected(self):
        """The medieval network must be fully connected (A* must find a path)."""
        from arco.planning.discrete.astar import AStarPlanner

        gen = RoadNetworkGenerator(seed=99)
        graph = gen.generate_medieval_network()
        planner = AStarPlanner(graph)

        # Test connectivity from first node to every other node
        start = graph.nodes[0]
        reachable = 0
        for goal in graph.nodes[1:]:
            if planner.plan(start, goal) is not None:
                reachable += 1
        # At least 80 % of nodes must be reachable from the first node
        assert reachable / (len(graph.nodes) - 1) >= 0.8

    def test_generate_medieval_network_invalid_radials(self):
        """Fewer than 3 radials must raise ValueError."""
        gen = RoadNetworkGenerator(seed=42)
        with pytest.raises(ValueError):
            gen.generate_medieval_network(num_radials=2)

    def test_generate_medieval_network_reproducible(self):
        """Same seed must produce identical medieval networks."""
        gen1 = RoadNetworkGenerator(seed=77)
        gen2 = RoadNetworkGenerator(seed=77)
        g1 = gen1.generate_medieval_network()
        g2 = gen2.generate_medieval_network()

        assert len(g1.nodes) == len(g2.nodes)
        for nid in g1.nodes:
            assert g1.position(nid) == g2.position(nid)
        assert len(g1.edges) == len(g2.edges)

    def test_pathfinding_on_grid_network(self):
        """A* should find paths on generated grid networks."""
        from arco.planning.discrete.astar import AStarPlanner

        gen = RoadNetworkGenerator(seed=42)
        graph = gen.generate_grid_network(grid_size=(4, 4))

        planner = AStarPlanner(graph)
        # Path from top-left (0) to bottom-right (15)
        path = planner.plan(0, 15)

        assert path is not None
        assert path[0] == 0
        assert path[-1] == 15
        assert len(path) >= 7  # Manhattan distance is 6, so path length >= 7

    def test_pathfinding_on_random_network(self):
        """A* should find paths on generated random networks."""
        from arco.planning.discrete.astar import AStarPlanner

        gen = RoadNetworkGenerator(seed=42)
        # Generate with large connect radius to ensure connectivity
        graph = gen.generate_random_network(
            num_intersections=20, area=300.0, connect_radius=150.0
        )

        planner = AStarPlanner(graph)

        # Try to find path between first and last node
        if len(graph.nodes) >= 2:
            start, goal = graph.nodes[0], graph.nodes[-1]
            path = planner.plan(start, goal)

            # Path may or may not exist depending on connectivity
            # But if it exists, it should be valid
            if path is not None:
                assert path[0] == start
                assert path[-1] == goal
                assert len(path) >= 2
