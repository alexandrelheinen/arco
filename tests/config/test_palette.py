"""Unit tests for arco.config.palette — color palette utilities."""

from __future__ import annotations

import pytest

from arco.config.palette import (
    LAYER_ALPHA,
    _adjust_hsl,
    _derive,
    annotation_hex,
    annotation_rgb,
    hex_to_float,
    hex_to_rgb,
    layer_float,
    layer_hex,
    layer_rgb,
    method_base_float,
    method_base_hex,
    method_base_rgb,
    obstacle_float,
    obstacle_hex,
    obstacle_rgb,
    ui_rgb,
)

# ---------------------------------------------------------------------------
# hex_to_rgb / hex_to_float
# ---------------------------------------------------------------------------


class TestHexConverters:
    """Tests for low-level hex conversion utilities."""

    def test_hex_to_rgb_with_hash(self) -> None:
        assert hex_to_rgb("#ff0000") == (255, 0, 0)

    def test_hex_to_rgb_without_hash(self) -> None:
        assert hex_to_rgb("00ff00") == (0, 255, 0)

    def test_hex_to_rgb_black(self) -> None:
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_hex_to_rgb_white(self) -> None:
        assert hex_to_rgb("#ffffff") == (255, 255, 255)

    def test_hex_to_float_range(self) -> None:
        r, g, b = hex_to_float("#4477CC")
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_hex_to_float_black(self) -> None:
        assert hex_to_float("#000000") == (0.0, 0.0, 0.0)

    def test_hex_to_float_white(self) -> None:
        r, g, b = hex_to_float("#ffffff")
        assert abs(r - 1.0) < 1e-9
        assert abs(g - 1.0) < 1e-9
        assert abs(b - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# LAYER_ALPHA
# ---------------------------------------------------------------------------


class TestLayerAlpha:
    """Tests for the LAYER_ALPHA constant dict."""

    def test_all_layers_present(self) -> None:
        for layer in ("tree", "path", "pruned", "trajectory", "vehicle"):
            assert layer in LAYER_ALPHA

    def test_alpha_in_range(self) -> None:
        for layer, alpha in LAYER_ALPHA.items():
            assert (
                0.0 <= alpha <= 1.0
            ), f"Alpha out of range for layer {layer!r}"

    def test_vehicle_fully_opaque(self) -> None:
        assert LAYER_ALPHA["vehicle"] == 1.0

    def test_tree_most_transparent(self) -> None:
        assert LAYER_ALPHA["tree"] == min(LAYER_ALPHA.values())


# ---------------------------------------------------------------------------
# Annotation colors
# ---------------------------------------------------------------------------


class TestAnnotation:
    """Tests for annotation_hex / annotation_rgb."""

    def test_annotation_hex_light_bg_is_black(self) -> None:
        assert annotation_hex(dark_bg=False) == "#000000"

    def test_annotation_hex_dark_bg_near_white(self) -> None:
        color = annotation_hex(dark_bg=True)
        assert color.startswith("#")
        r, g, b = hex_to_rgb(color)
        # Near-white means each channel > 200
        assert r > 200 and g > 200 and b > 200

    def test_annotation_rgb_returns_tuple_of_ints(self) -> None:
        rgb = annotation_rgb()
        assert len(rgb) == 3
        assert all(isinstance(v, int) for v in rgb)

    def test_annotation_rgb_dark_bg_returns_light(self) -> None:
        r, g, b = annotation_rgb(dark_bg=True)
        assert r > 200 and g > 200 and b > 200


# ---------------------------------------------------------------------------
# Obstacle colors
# ---------------------------------------------------------------------------


class TestObstacle:
    """Tests for obstacle_hex / obstacle_rgb / obstacle_float."""

    def test_obstacle_hex_is_hex_string(self) -> None:
        h = obstacle_hex()
        assert isinstance(h, str)
        assert h.startswith("#")
        assert len(h) == 7

    def test_obstacle_rgb_in_range(self) -> None:
        r, g, b = obstacle_rgb()
        for ch in (r, g, b):
            assert 0 <= ch <= 255

    def test_obstacle_float_in_range(self) -> None:
        r, g, b = obstacle_float()
        for ch in (r, g, b):
            assert 0.0 <= ch <= 1.0

    def test_obstacle_rgb_consistency_with_hex(self) -> None:
        assert obstacle_rgb() == hex_to_rgb(obstacle_hex())

    def test_obstacle_float_consistency_with_hex(self) -> None:
        expected = hex_to_float(obstacle_hex())
        result = obstacle_float()
        for a, b in zip(result, expected):
            assert abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# Method base colors
# ---------------------------------------------------------------------------


class TestMethodBase:
    """Tests for method_base_hex / method_base_rgb / method_base_float."""

    @pytest.mark.parametrize("method", ["rrt", "sst", "astar", "dstar"])
    def test_method_base_hex_defined(self, method: str) -> None:
        h = method_base_hex(method)
        assert isinstance(h, str) and h.startswith("#")

    def test_unknown_method_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            method_base_hex("unknown_algo")

    @pytest.mark.parametrize("method", ["rrt", "sst", "astar", "dstar"])
    def test_method_base_rgb_in_range(self, method: str) -> None:
        r, g, b = method_base_rgb(method)
        for ch in (r, g, b):
            assert 0 <= ch <= 255

    @pytest.mark.parametrize("method", ["rrt", "sst", "astar", "dstar"])
    def test_method_base_float_in_range(self, method: str) -> None:
        r, g, b = method_base_float(method)
        for ch in (r, g, b):
            assert 0.0 <= ch <= 1.0


# ---------------------------------------------------------------------------
# Layer colors
# ---------------------------------------------------------------------------


class TestLayerColors:
    """Tests for layer_hex / layer_rgb / layer_float and derivation rules."""

    METHODS = ["rrt", "sst", "astar", "dstar"]
    LAYERS = ["tree", "path", "pruned", "trajectory", "vehicle"]

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("layer", LAYERS)
    def test_layer_hex_is_valid(self, method: str, layer: str) -> None:
        h = layer_hex(method, layer)
        assert isinstance(h, str) and h.startswith("#") and len(h) == 7

    @pytest.mark.parametrize("method", METHODS)
    def test_vehicle_layer_equals_base(self, method: str) -> None:
        assert layer_hex(method, "vehicle") == method_base_hex(method)

    @pytest.mark.parametrize("method", METHODS)
    def test_tree_lighter_than_path(self, method: str) -> None:
        """Tree should be lighter (higher L) than path because we lighten it."""
        import colorsys

        tree_r, tree_g, tree_b = hex_to_float(layer_hex(method, "tree"))
        path_r, path_g, path_b = hex_to_float(layer_hex(method, "path"))
        _, tree_l, _ = colorsys.rgb_to_hls(tree_r, tree_g, tree_b)
        _, path_l, _ = colorsys.rgb_to_hls(path_r, path_g, path_b)
        assert (
            tree_l >= path_l
        ), f"{method}: tree lightness {tree_l:.3f} not >= path lightness {path_l:.3f}"

    @pytest.mark.parametrize("method", METHODS)
    def test_trajectory_darker_than_path(self, method: str) -> None:
        """Trajectory layer should be darker (lower L) than path."""
        import colorsys

        traj_r, traj_g, traj_b = hex_to_float(layer_hex(method, "trajectory"))
        path_r, path_g, path_b = hex_to_float(layer_hex(method, "path"))
        _, traj_l, _ = colorsys.rgb_to_hls(traj_r, traj_g, traj_b)
        _, path_l, _ = colorsys.rgb_to_hls(path_r, path_g, path_b)
        assert traj_l <= path_l, (
            f"{method}: trajectory lightness {traj_l:.3f} not <= path "
            f"lightness {path_l:.3f}"
        )

    def test_unknown_layer_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown rendering layer"):
            layer_hex("rrt", "invalid_layer")

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("layer", LAYERS)
    def test_layer_rgb_in_range(self, method: str, layer: str) -> None:
        r, g, b = layer_rgb(method, layer)
        for ch in (r, g, b):
            assert 0 <= ch <= 255

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("layer", LAYERS)
    def test_layer_float_in_range(self, method: str, layer: str) -> None:
        r, g, b = layer_float(method, layer)
        for ch in (r, g, b):
            assert 0.0 <= ch <= 1.0


# ---------------------------------------------------------------------------
# UI colors
# ---------------------------------------------------------------------------


class TestUiRgb:
    """Tests for ui_rgb."""

    @pytest.mark.parametrize(
        "key",
        [
            "background",
            "road_dot",
            "road_sdf",
            "barrier",
            "hud_text",
            "hud_dim",
            "hud_shadow",
            "hud_winner",
            "hud_tie",
        ],
    )
    def test_ui_rgb_in_range(self, key: str) -> None:
        r, g, b = ui_rgb(key)
        for ch in (r, g, b):
            assert 0 <= ch <= 255

    def test_unknown_key_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            ui_rgb("nonexistent_ui_key")

    def test_background_is_dark(self) -> None:
        r, g, b = ui_rgb("background")
        assert r < 50 and g < 50 and b < 60

    def test_hud_text_is_light(self) -> None:
        r, g, b = ui_rgb("hud_text")
        assert r > 150 and g > 150 and b > 150


# ---------------------------------------------------------------------------
# _adjust_hsl internal helper
# ---------------------------------------------------------------------------


class TestAdjustHsl:
    """Tests for the internal _adjust_hsl function."""

    def test_identity_no_delta(self) -> None:
        original = "#4477cc"
        result = _adjust_hsl(original, 0.0, 0.0)
        assert result == original

    def test_lighten_increases_lightness(self) -> None:
        import colorsys

        original = "#4477cc"
        lightened = _adjust_hsl(original, lightness_delta=+0.2)
        _, l_orig, _ = colorsys.rgb_to_hls(*hex_to_float(original))
        _, l_new, _ = colorsys.rgb_to_hls(*hex_to_float(lightened))
        assert l_new > l_orig

    def test_darken_decreases_lightness(self) -> None:
        import colorsys

        original = "#4477cc"
        darkened = _adjust_hsl(original, lightness_delta=-0.2)
        _, l_orig, _ = colorsys.rgb_to_hls(*hex_to_float(original))
        _, l_new, _ = colorsys.rgb_to_hls(*hex_to_float(darkened))
        assert l_new < l_orig

    def test_clamps_lightness_to_zero(self) -> None:
        result = _adjust_hsl("#000000", lightness_delta=-1.0)
        r, g, b = hex_to_rgb(result)
        assert r == 0 and g == 0 and b == 0

    def test_clamps_lightness_to_one(self) -> None:
        result = _adjust_hsl("#ffffff", lightness_delta=+1.0)
        r, g, b = hex_to_rgb(result)
        assert r == 255 and g == 255 and b == 255

    def test_result_is_valid_hex(self) -> None:
        result = _adjust_hsl(
            "#4477cc", lightness_delta=+0.1, saturation_delta=-0.3
        )
        assert (
            isinstance(result, str)
            and result.startswith("#")
            and len(result) == 7
        )


# ---------------------------------------------------------------------------
# _derive internal helper
# ---------------------------------------------------------------------------


class TestDerive:
    """Tests for the internal _derive function."""

    BASE = "#4477CC"

    def test_vehicle_is_identity(self) -> None:
        assert _derive(self.BASE, "vehicle") == self.BASE

    @pytest.mark.parametrize("layer", ["tree", "path", "pruned", "trajectory"])
    def test_known_layers_return_hex(self, layer: str) -> None:
        result = _derive(self.BASE, layer)
        assert isinstance(result, str)
        assert result.startswith("#")
        assert len(result) == 7

    def test_unknown_layer_raises(self) -> None:
        with pytest.raises(ValueError):
            _derive(self.BASE, "bogus")
