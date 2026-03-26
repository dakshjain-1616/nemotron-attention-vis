"""
Test suite for Nemotron Attention Visualizer.

Covers:
  1. MockAttentionExtractor — data shape, values, reproducibility
  2. HeatmapRenderer        — JSON serialisation, HTML generation
  3. End-to-end pipeline    — visualize() produces valid output files
  4. Spec requirement       — "Explain quantum computing" → attention_map.html

Run with:
    python -m pytest tests/ -v
"""

import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_extractor():
    from nemotron_attention_v import MockAttentionExtractor
    return MockAttentionExtractor(
        model_name="nvidia/Nemotron-Cascade-2-30B-A3B",
        num_layers=4,
        num_heads=8,
        seed=42,
    )


@pytest.fixture
def sample_prompt():
    return "Explain quantum computing"


@pytest.fixture
def attention_data(mock_extractor, sample_prompt):
    mock_extractor.load()
    return mock_extractor.extract(sample_prompt)


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory for render tests."""
    return tmp_path / "outputs"


# ─── 1. MockAttentionExtractor tests ─────────────────────────────────────────

class TestMockAttentionExtractor:

    def test_load_returns_self(self, mock_extractor):
        result = mock_extractor.load()
        assert result is mock_extractor

    def test_extract_returns_dict(self, attention_data):
        assert isinstance(attention_data, dict)

    def test_required_keys_present(self, attention_data):
        required = {"model", "prompt", "tokens", "layers", "attentions",
                    "num_attention_layers", "num_heads", "seq_len", "mock"}
        assert required.issubset(set(attention_data.keys()))

    def test_mock_flag_is_true(self, attention_data):
        assert attention_data["mock"] is True

    def test_model_name_preserved(self, attention_data):
        assert attention_data["model"] == "nvidia/Nemotron-Cascade-2-30B-A3B"

    def test_prompt_preserved(self, attention_data, sample_prompt):
        assert attention_data["prompt"] == sample_prompt

    def test_token_count(self, attention_data, sample_prompt):
        # mock tokenizer: split on whitespace + BOS + EOS
        word_count = len(sample_prompt.split())
        expected_tokens = word_count + 2  # <s> + words + </s>
        assert attention_data["seq_len"] == expected_tokens
        assert len(attention_data["tokens"]) == expected_tokens

    def test_bos_token_present(self, attention_data):
        assert attention_data["tokens"][0] == "<s>"

    def test_eos_token_present(self, attention_data):
        assert attention_data["tokens"][-1] == "</s>"

    def test_num_attention_layers(self, mock_extractor, sample_prompt):
        mock_extractor.load()
        data = mock_extractor.extract(sample_prompt)
        assert data["num_attention_layers"] == mock_extractor.num_layers
        assert len(data["layers"]) == mock_extractor.num_layers

    def test_attentions_list_length(self, attention_data, mock_extractor):
        assert len(attention_data["attentions"]) == mock_extractor.num_layers

    def test_attention_matrix_shape(self, attention_data, mock_extractor):
        seq_len = attention_data["seq_len"]
        for arr in attention_data["attentions"]:
            assert arr.shape == (mock_extractor.num_heads, seq_len, seq_len), \
                f"Expected shape ({mock_extractor.num_heads}, {seq_len}, {seq_len}), got {arr.shape}"

    def test_attention_values_non_negative(self, attention_data):
        for arr in attention_data["attentions"]:
            assert np.all(arr >= 0.0), "Found negative attention weights"

    def test_attention_rows_sum_to_one(self, attention_data):
        """Each query token's attention distribution must sum to 1."""
        for layer_idx, arr in enumerate(attention_data["attentions"]):
            row_sums = arr.sum(axis=-1)  # (heads, seq)
            np.testing.assert_allclose(
                row_sums, 1.0, atol=1e-5,
                err_msg=f"Layer {layer_idx} row sums deviate from 1.0"
            )

    def test_causal_mask_respected(self, attention_data):
        """No token should attend to future tokens (upper triangle should be zero)."""
        seq_len = attention_data["seq_len"]
        for layer_idx, arr in enumerate(attention_data["attentions"]):
            for head_idx in range(arr.shape[0]):
                for i in range(seq_len):
                    future = arr[head_idx, i, i + 1:]
                    assert np.all(future == 0.0), \
                        f"Layer {layer_idx} Head {head_idx} row {i} attends to future tokens"

    def test_reproducibility_same_seed(self, sample_prompt):
        from nemotron_attention_v import MockAttentionExtractor
        ext1 = MockAttentionExtractor(num_layers=2, num_heads=4, seed=99)
        ext2 = MockAttentionExtractor(num_layers=2, num_heads=4, seed=99)
        ext1.load(); ext2.load()
        d1 = ext1.extract(sample_prompt)
        d2 = ext2.extract(sample_prompt)
        np.testing.assert_array_equal(d1["attentions"][0], d2["attentions"][0])

    def test_different_seeds_differ(self, sample_prompt):
        from nemotron_attention_v import MockAttentionExtractor
        ext1 = MockAttentionExtractor(num_layers=2, num_heads=4, seed=1)
        ext2 = MockAttentionExtractor(num_layers=2, num_heads=4, seed=2)
        ext1.load(); ext2.load()
        d1 = ext1.extract(sample_prompt)
        d2 = ext2.extract(sample_prompt)
        assert not np.array_equal(d1["attentions"][0], d2["attentions"][0])

    def test_layer_indices_are_sequential(self, attention_data, mock_extractor):
        assert attention_data["layers"] == list(range(mock_extractor.num_layers))

    def test_num_heads_field(self, attention_data, mock_extractor):
        assert attention_data["num_heads"] == mock_extractor.num_heads

    def test_different_prompt_different_tokens(self):
        from nemotron_attention_v import MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=2, num_heads=4, seed=42)
        ext.load()
        d1 = ext.extract("Hello world")
        d2 = ext.extract("Explain quantum computing in simple terms")
        assert d1["seq_len"] != d2["seq_len"]
        assert d1["tokens"] != d2["tokens"]


# ─── 2. HeatmapRenderer tests ─────────────────────────────────────────────────

class TestHeatmapRenderer:

    def test_json_output_created(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        html_path, json_path = renderer.render(attention_data)
        assert json_path.exists(), "JSON output file not created"

    def test_html_output_created(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        html_path, json_path = renderer.render(attention_data)
        assert html_path.exists(), "HTML output file not created"

    def test_html_filename(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        html_path, _ = renderer.render(attention_data, html_filename="attention_map.html")
        assert html_path.name == "attention_map.html"

    def test_json_is_valid(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        _, json_path = renderer.render(attention_data)
        with open(json_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_has_required_keys(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        _, json_path = renderer.render(attention_data)
        with open(json_path) as f:
            data = json.load(f)
        for key in ("model", "prompt", "tokens", "layers", "attentions", "stats"):
            assert key in data, f"Missing key '{key}' in JSON output"

    def test_json_tokens_match(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        _, json_path = renderer.render(attention_data)
        with open(json_path) as f:
            data = json.load(f)
        assert data["tokens"] == attention_data["tokens"]

    def test_json_attentions_shape(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        _, json_path = renderer.render(attention_data)
        with open(json_path) as f:
            data = json.load(f)
        num_heads = attention_data["num_heads"]
        seq_len   = attention_data["seq_len"]
        for layer_data in data["attentions"]:
            assert len(layer_data) == num_heads
            for head_data in layer_data:
                assert len(head_data) == seq_len
                for row in head_data:
                    assert len(row) == seq_len

    def test_html_contains_plotly(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        html_path, _ = renderer.render(attention_data)
        content = html_path.read_text()
        assert "plotly" in content.lower(), "HTML should include Plotly.js reference"

    def test_html_contains_tokens(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        html_path, _ = renderer.render(attention_data)
        content = html_path.read_text()
        # At least the BOS token should appear
        assert "<s>" in content or "&lt;s&gt;" in content

    def test_html_is_self_contained(self, attention_data, tmp_output):
        """HTML should embed all attention data (not reference external JSON files)."""
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        html_path, _ = renderer.render(attention_data)
        content = html_path.read_text()
        assert "const DATA = " in content, "Attention data should be embedded in HTML"

    def test_stats_keys(self, attention_data, tmp_output):
        from nemotron_attention_v import HeatmapRenderer
        renderer = HeatmapRenderer(output_dir=str(tmp_output))
        _, json_path = renderer.render(attention_data)
        with open(json_path) as f:
            data = json.load(f)
        stats = data["stats"]
        for key in ("mean_nonzero", "max", "min_nonzero", "entropy_mean"):
            assert key in stats

    def test_output_dir_created_if_missing(self, attention_data, tmp_path):
        from nemotron_attention_v import HeatmapRenderer
        new_dir = tmp_path / "nonexistent" / "subdir"
        renderer = HeatmapRenderer(output_dir=str(new_dir))
        renderer.render(attention_data)
        assert new_dir.exists()


# ─── 3. End-to-end pipeline tests (visualize() function) ─────────────────────

class TestVisualizePipeline:

    def test_visualize_returns_two_paths(self, tmp_output):
        from nemotron_attention_v import visualize
        result = visualize(
            prompt="Hello world",
            mock=True,
            output_dir=str(tmp_output),
            num_layers=2,
            num_heads=4,
            mock_seed=0,
        )
        assert len(result) == 2

    def test_visualize_html_exists(self, tmp_output):
        from nemotron_attention_v import visualize
        html_path, _ = visualize(
            prompt="Hello world",
            mock=True,
            output_dir=str(tmp_output),
            num_layers=2,
            num_heads=4,
        )
        assert html_path.exists()

    def test_visualize_json_exists(self, tmp_output):
        from nemotron_attention_v import visualize
        _, json_path = visualize(
            prompt="Hello world",
            mock=True,
            output_dir=str(tmp_output),
            num_layers=2,
            num_heads=4,
        )
        assert json_path.exists()

    def test_visualize_html_non_empty(self, tmp_output):
        from nemotron_attention_v import visualize
        html_path, _ = visualize(
            prompt="Test",
            mock=True,
            output_dir=str(tmp_output),
            num_layers=2,
            num_heads=4,
        )
        assert html_path.stat().st_size > 1024  # at least 1KB

    def test_visualize_json_valid(self, tmp_output):
        from nemotron_attention_v import visualize
        _, json_path = visualize(
            prompt="Test JSON validity",
            mock=True,
            output_dir=str(tmp_output),
            num_layers=2,
            num_heads=4,
        )
        with open(json_path) as f:
            data = json.load(f)
        assert "attentions" in data

    def test_custom_html_filename(self, tmp_output):
        from nemotron_attention_v import visualize
        html_path, _ = visualize(
            prompt="Test",
            mock=True,
            output_dir=str(tmp_output),
            html_filename="custom_name.html",
            num_layers=2,
            num_heads=4,
        )
        assert html_path.name == "custom_name.html"

    # ── Spec requirement: "Explain quantum computing" → attention_map.html ────

    def test_spec_quantum_computing_prompt(self, tmp_output):
        """SPEC: Input='Explain quantum computing' → outputs/attention_map.html"""
        from nemotron_attention_v import visualize
        html_path, _ = visualize(
            prompt="Explain quantum computing",
            mock=True,
            output_dir=str(tmp_output),
            html_filename="attention_map.html",
            num_layers=4,
            num_heads=8,
            mock_seed=42,
        )
        assert html_path.exists(), "attention_map.html must be created"
        assert html_path.name == "attention_map.html"
        content = html_path.read_text()
        # Heatmap interactive elements present
        assert "heatmap" in content.lower()
        assert "Explain quantum computing" in content

    def test_spec_heatmap_interactive(self, tmp_output):
        """SPEC: Heatmap must be interactive (Plotly with dropdown controls)."""
        from nemotron_attention_v import visualize
        html_path, _ = visualize(
            prompt="Explain quantum computing",
            mock=True,
            output_dir=str(tmp_output),
            html_filename="attention_map.html",
            num_layers=4,
            num_heads=8,
        )
        content = html_path.read_text()
        assert "layerSelect" in content, "Layer selector missing"
        assert "headSelect" in content, "Head selector missing"
        assert "updatePlot" in content, "Interactive update function missing"
        assert "Plotly.react" in content, "Plotly reactive rendering missing"

    def test_spec_attention_matrix_extracted(self, tmp_output):
        """SPEC: Attention matrix is extracted and stored in JSON."""
        from nemotron_attention_v import visualize
        _, json_path = visualize(
            prompt="Explain quantum computing",
            mock=True,
            output_dir=str(tmp_output),
            json_filename="attention_data.json",
            num_layers=4,
            num_heads=8,
        )
        assert json_path.name == "attention_data.json"
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["attentions"]) > 0
        # Verify it's a real matrix (not empty)
        first_layer = data["attentions"][0]
        first_head  = first_layer[0]
        assert len(first_head) > 0
        assert len(first_head[0]) > 0

    def test_env_var_mock_mode(self, tmp_output, monkeypatch):
        """MOCK_MODE env var should activate mock mode."""
        monkeypatch.setenv("MOCK_MODE", "true")
        from nemotron_attention_v import visualize
        html_path, json_path = visualize(
            prompt="env var test",
            output_dir=str(tmp_output),
            num_layers=2,
            num_heads=4,
        )
        assert html_path.exists()

    def test_env_var_output_dir(self, tmp_path, monkeypatch):
        """OUTPUT_DIR env var should set output location."""
        custom_dir = tmp_path / "env_output"
        monkeypatch.setenv("OUTPUT_DIR", str(custom_dir))
        from nemotron_attention_v import visualize
        html_path, _ = visualize(
            prompt="env test",
            mock=True,
            num_layers=2,
            num_heads=4,
        )
        assert custom_dir.exists()


# ─── 4. Attention Rollout tests ───────────────────────────────────────────────

class TestAttentionRollout:

    @pytest.fixture
    def sample_attentions(self):
        from nemotron_attention_v import MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=4, num_heads=8, seed=42)
        ext.load()
        data = ext.extract("Explain quantum computing")
        return data["attentions"]

    def test_rollout_returns_ndarray(self, sample_attentions):
        from nemotron_attention_v import compute_attention_rollout
        result = compute_attention_rollout(sample_attentions)
        assert isinstance(result, np.ndarray)

    def test_rollout_shape(self, sample_attentions):
        from nemotron_attention_v import compute_attention_rollout
        result = compute_attention_rollout(sample_attentions)
        seq_len = sample_attentions[0].shape[-1]
        assert result.shape == (seq_len, seq_len)

    def test_rollout_rows_sum_to_one(self, sample_attentions):
        from nemotron_attention_v import compute_attention_rollout
        result = compute_attention_rollout(sample_attentions)
        row_sums = result.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_rollout_non_negative(self, sample_attentions):
        from nemotron_attention_v import compute_attention_rollout
        result = compute_attention_rollout(sample_attentions)
        assert np.all(result >= 0.0)

    def test_rollout_differs_from_single_layer(self, sample_attentions):
        """Rollout should not equal just the first layer averaged over heads."""
        from nemotron_attention_v import compute_attention_rollout
        rollout = compute_attention_rollout(sample_attentions)
        single_layer_mean = sample_attentions[0].mean(axis=0)
        # They should differ because rollout chains through all layers
        assert not np.allclose(rollout, single_layer_mean, atol=1e-3)

    def test_rollout_empty_raises(self):
        from nemotron_attention_v import compute_attention_rollout
        with pytest.raises(ValueError, match="empty"):
            compute_attention_rollout([])

    def test_rollout_single_layer(self):
        """With one layer, rollout ≈ mixed (avg_attn * 0.5 + I * 0.5), row-normalised."""
        from nemotron_attention_v import compute_attention_rollout
        # Create a simple 3×3 uniform attention matrix
        heads = np.ones((2, 3, 3), dtype=np.float32) / 3
        result = compute_attention_rollout([heads])
        assert result.shape == (3, 3)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-5)

    def test_rollout_in_json_output(self, tmp_path):
        """Serialised JSON must contain rollout key."""
        from nemotron_attention_v import HeatmapRenderer, MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=2, num_heads=4, seed=0)
        ext.load()
        data = ext.extract("hello world")
        renderer = HeatmapRenderer(output_dir=str(tmp_path))
        _, json_path = renderer.render(data)
        with open(json_path) as f:
            saved = json.load(f)
        assert "rollout" in saved
        assert len(saved["rollout"]) == data["seq_len"]


# ─── 5. Token Importance tests ────────────────────────────────────────────────

class TestTokenImportance:

    @pytest.fixture
    def sample_attentions(self):
        from nemotron_attention_v import MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=4, num_heads=8, seed=7)
        ext.load()
        data = ext.extract("The capital of France is")
        return data["attentions"], data["seq_len"]

    def test_importance_returns_ndarray(self, sample_attentions):
        from nemotron_attention_v import compute_token_importance
        attentions, _ = sample_attentions
        result = compute_token_importance(attentions)
        assert isinstance(result, np.ndarray)

    def test_importance_shape(self, sample_attentions):
        from nemotron_attention_v import compute_token_importance
        attentions, seq_len = sample_attentions
        result = compute_token_importance(attentions)
        assert result.shape == (seq_len,)

    def test_importance_sums_to_one(self, sample_attentions):
        from nemotron_attention_v import compute_token_importance
        attentions, _ = sample_attentions
        result = compute_token_importance(attentions)
        assert abs(result.sum() - 1.0) < 1e-5

    def test_importance_non_negative(self, sample_attentions):
        from nemotron_attention_v import compute_token_importance
        attentions, _ = sample_attentions
        result = compute_token_importance(attentions)
        assert np.all(result >= 0.0)

    def test_importance_varies_across_tokens(self, sample_attentions):
        """Not all tokens should have equal importance."""
        from nemotron_attention_v import compute_token_importance
        attentions, _ = sample_attentions
        result = compute_token_importance(attentions)
        assert result.max() > result.min()

    def test_importance_empty_raises(self):
        from nemotron_attention_v import compute_token_importance
        with pytest.raises(ValueError, match="empty"):
            compute_token_importance([])

    def test_importance_in_json_output(self, tmp_path):
        from nemotron_attention_v import HeatmapRenderer, MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=2, num_heads=4, seed=0)
        ext.load()
        data = ext.extract("Neural networks learn by")
        renderer = HeatmapRenderer(output_dir=str(tmp_path))
        _, json_path = renderer.render(data)
        with open(json_path) as f:
            saved = json.load(f)
        assert "token_importance" in saved
        assert len(saved["token_importance"]) == data["seq_len"]
        # Sum should be ~1
        assert abs(sum(saved["token_importance"]) - 1.0) < 1e-4


# ─── 6. CSV Export tests ──────────────────────────────────────────────────────

class TestCSVExport:

    @pytest.fixture
    def attention_data(self):
        from nemotron_attention_v import MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=3, num_heads=4, seed=42)
        ext.load()
        return ext.extract("Hello world test")

    def test_csv_created(self, attention_data, tmp_path):
        from nemotron_attention_v import export_attention_csv
        csv_path = tmp_path / "stats.csv"
        result = export_attention_csv(attention_data, csv_path)
        assert result.exists()

    def test_csv_returns_path(self, attention_data, tmp_path):
        from nemotron_attention_v import export_attention_csv
        csv_path = tmp_path / "stats.csv"
        result = export_attention_csv(attention_data, csv_path)
        assert isinstance(result, Path)
        assert result == csv_path

    def test_csv_is_valid(self, attention_data, tmp_path):
        import csv as _csv
        from nemotron_attention_v import export_attention_csv
        csv_path = tmp_path / "stats.csv"
        export_attention_csv(attention_data, csv_path)
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0

    def test_csv_has_expected_columns(self, attention_data, tmp_path):
        import csv as _csv
        from nemotron_attention_v import export_attention_csv
        csv_path = tmp_path / "stats.csv"
        export_attention_csv(attention_data, csv_path)
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
        expected = {"layer_idx", "head_idx", "entropy_mean", "max_weight", "gini_coefficient"}
        assert expected.issubset(set(fieldnames))

    def test_csv_row_count(self, attention_data, tmp_path):
        import csv as _csv
        from nemotron_attention_v import export_attention_csv
        csv_path = tmp_path / "stats.csv"
        export_attention_csv(attention_data, csv_path)
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        # Should have one row per layer × head
        expected_rows = attention_data["num_attention_layers"] * attention_data["num_heads"]
        assert len(rows) == expected_rows

    def test_csv_values_are_numeric(self, attention_data, tmp_path):
        import csv as _csv
        from nemotron_attention_v import export_attention_csv
        csv_path = tmp_path / "stats.csv"
        export_attention_csv(attention_data, csv_path)
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            row = next(reader)
        float(row["entropy_mean"])
        float(row["max_weight"])
        float(row["gini_coefficient"])

    def test_csv_gini_in_range(self, attention_data, tmp_path):
        import csv as _csv
        from nemotron_attention_v import export_attention_csv
        csv_path = tmp_path / "stats.csv"
        export_attention_csv(attention_data, csv_path)
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                g = float(row["gini_coefficient"])
                assert 0.0 <= g <= 1.0, f"Gini {g} out of [0, 1]"

    def test_csv_export_via_visualize(self, tmp_path):
        from nemotron_attention_v import visualize
        html_path, json_path = visualize(
            prompt="CSV export test",
            mock=True,
            output_dir=str(tmp_path),
            num_layers=2,
            num_heads=4,
            export_csv=True,
        )
        csv_path = json_path.with_suffix(".csv")
        assert csv_path.exists()

    def test_csv_creates_parent_dirs(self, attention_data, tmp_path):
        from nemotron_attention_v import export_attention_csv
        deep_path = tmp_path / "a" / "b" / "c" / "stats.csv"
        export_attention_csv(attention_data, deep_path)
        assert deep_path.exists()


# ─── 7. Per-layer stats tests ─────────────────────────────────────────────────

class TestPerLayerStats:

    @pytest.fixture
    def rendered_json(self, tmp_path):
        from nemotron_attention_v import HeatmapRenderer, MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=4, num_heads=8, seed=42)
        ext.load()
        data = ext.extract("Explain quantum computing")
        renderer = HeatmapRenderer(output_dir=str(tmp_path))
        _, json_path = renderer.render(data)
        with open(json_path) as f:
            return json.load(f)

    def test_per_layer_stats_present(self, rendered_json):
        assert "per_layer_stats" in rendered_json

    def test_per_layer_stats_length(self, rendered_json):
        stats = rendered_json["per_layer_stats"]
        assert len(stats) == rendered_json["num_attention_layers"]

    def test_per_layer_has_required_fields(self, rendered_json):
        for s in rendered_json["per_layer_stats"]:
            for field in ("layer_idx", "entropy_mean", "max_weight", "gini", "num_heads"):
                assert field in s, f"Missing field '{field}' in per_layer_stats"

    def test_per_layer_entropy_is_float(self, rendered_json):
        for s in rendered_json["per_layer_stats"]:
            assert isinstance(s["entropy_mean"], float)

    def test_per_layer_gini_in_range(self, rendered_json):
        for s in rendered_json["per_layer_stats"]:
            g = s["gini"]
            assert 0.0 <= g <= 1.0, f"Gini {g} out of [0, 1]"

    def test_per_layer_max_weight_positive(self, rendered_json):
        for s in rendered_json["per_layer_stats"]:
            assert s["max_weight"] > 0.0

    def test_generated_at_is_iso_format(self, rendered_json):
        ts = rendered_json.get("generated_at", "")
        # Should be ISO 8601 UTC: 2025-01-01T12:00:00Z
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts), \
            f"generated_at '{ts}' is not ISO 8601 format"

    def test_html_contains_layer_stats_table(self, tmp_path):
        from nemotron_attention_v import visualize
        html_path, _ = visualize(
            prompt="stats table test",
            mock=True,
            output_dir=str(tmp_path),
            num_layers=2,
            num_heads=4,
        )
        content = html_path.read_text()
        assert "layer-stats-body" in content
        assert "per_layer_stats" in content


# ─── 8. Multi-prompt comparison tests ────────────────────────────────────────

class TestComparisons:

    def test_compare_returns_path(self, tmp_path):
        from nemotron_attention_v import compare_prompts
        result = compare_prompts(
            prompts=["Hello world", "Quantum computing"],
            mock=True,
            output_dir=str(tmp_path),
            num_layers=2,
            num_heads=4,
            mock_seed=42,
        )
        assert isinstance(result, Path)

    def test_compare_html_exists(self, tmp_path):
        from nemotron_attention_v import compare_prompts
        out = compare_prompts(
            prompts=["Prompt A", "Prompt B"],
            mock=True,
            output_dir=str(tmp_path),
            num_layers=2,
            num_heads=4,
        )
        assert out.exists()

    def test_compare_html_filename(self, tmp_path):
        from nemotron_attention_v import compare_prompts
        out = compare_prompts(
            prompts=["A"],
            mock=True,
            output_dir=str(tmp_path),
            comparison_filename="my_comparison.html",
            num_layers=2,
            num_heads=4,
        )
        assert out.name == "my_comparison.html"

    def test_compare_html_contains_prompts(self, tmp_path):
        from nemotron_attention_v import compare_prompts
        prompts = ["Neural networks", "Quantum physics"]
        out = compare_prompts(
            prompts=prompts,
            mock=True,
            output_dir=str(tmp_path),
            num_layers=2,
            num_heads=4,
        )
        content = out.read_text()
        for p in prompts:
            assert p in content, f"Prompt '{p}' not found in comparison HTML"

    def test_compare_html_is_self_contained(self, tmp_path):
        from nemotron_attention_v import compare_prompts
        out = compare_prompts(
            prompts=["Test"],
            mock=True,
            output_dir=str(tmp_path),
            num_layers=2,
            num_heads=4,
        )
        content = out.read_text()
        assert "const DATA = " in content
        assert "plotly" in content.lower()

    def test_compare_raises_on_empty_prompts(self, tmp_path):
        from nemotron_attention_v import compare_prompts
        with pytest.raises(ValueError, match="at least one"):
            compare_prompts(
                prompts=[],
                mock=True,
                output_dir=str(tmp_path),
            )

    def test_compare_single_prompt(self, tmp_path):
        from nemotron_attention_v import compare_prompts
        out = compare_prompts(
            prompts=["Single prompt"],
            mock=True,
            output_dir=str(tmp_path),
            num_layers=2,
            num_heads=4,
        )
        assert out.exists()


# ─── 9. Layer Filter tests ────────────────────────────────────────────────────

class TestLayerFilter:

    def test_layer_filter_reduces_layers(self, tmp_path):
        from nemotron_attention_v import visualize
        _, json_path = visualize(
            prompt="filter test",
            mock=True,
            output_dir=str(tmp_path),
            num_layers=8,
            num_heads=4,
            layer_filter=[0, 1, 2],
        )
        with open(json_path) as f:
            data = json.load(f)
        assert data["num_attention_layers"] == 3
        assert data["layers"] == [0, 1, 2]

    def test_layer_filter_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAYER_FILTER", "0,2")
        from nemotron_attention_v import visualize
        _, json_path = visualize(
            prompt="env layer filter",
            mock=True,
            output_dir=str(tmp_path),
            num_layers=4,
            num_heads=4,
        )
        with open(json_path) as f:
            data = json.load(f)
        assert set(data["layers"]).issubset({0, 2})

    def test_layer_filter_invalid_falls_through(self, tmp_path):
        """Filtering to non-existent layers falls back to all layers."""
        from nemotron_attention_v import visualize
        _, json_path = visualize(
            prompt="invalid filter",
            mock=True,
            output_dir=str(tmp_path),
            num_layers=4,
            num_heads=4,
            layer_filter=[99, 100],  # no layers 99 or 100 in 4-layer mock
        )
        with open(json_path) as f:
            data = json.load(f)
        # Falls back to all 4 layers when filter matches nothing
        assert data["num_attention_layers"] == 4


# ─── 10. Retry / fallback tests ──────────────────────────────────────────────

class TestRetryAndFallback:

    def test_fallback_model_env_var_read(self, monkeypatch):
        """FALLBACK_MODEL env var should be read by visualize()."""
        monkeypatch.setenv("FALLBACK_MODEL", "facebook/opt-125m")
        # Just verify it doesn't crash in mock mode (fallback only used for real models)
        from nemotron_attention_v import visualize
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            html_path, _ = visualize(
                prompt="fallback test",
                mock=True,
                output_dir=td,
                num_layers=2,
                num_heads=4,
            )
            assert html_path.exists()

    def test_invalid_prompt_raises(self):
        from nemotron_attention_v import MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=2, num_heads=4, seed=0)
        ext.load()
        with pytest.raises(ValueError, match="non-empty"):
            ext.extract("")

    def test_invalid_prompt_whitespace_raises(self):
        from nemotron_attention_v import MockAttentionExtractor
        ext = MockAttentionExtractor(num_layers=2, num_heads=4, seed=0)
        ext.load()
        with pytest.raises(ValueError, match="non-empty"):
            ext.extract("   ")

    def test_rollout_env_var_export_csv(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXPORT_CSV", "true")
        from nemotron_attention_v import visualize
        html_path, json_path = visualize(
            prompt="csv env test",
            mock=True,
            output_dir=str(tmp_path),
            num_layers=2,
            num_heads=4,
        )
        csv_path = json_path.with_suffix(".csv")
        assert csv_path.exists()

    def test_load_retries_env_var(self, monkeypatch):
        """LOAD_RETRIES env var should be accepted (tested via mock path)."""
        monkeypatch.setenv("LOAD_RETRIES", "1")
        from nemotron_attention_v import visualize
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            html_path, _ = visualize(
                prompt="retry env test",
                mock=True,
                output_dir=td,
                num_layers=2,
                num_heads=4,
            )
            assert html_path.exists()
