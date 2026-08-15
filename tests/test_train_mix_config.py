"""Tests for the training script's --mix-config support.

These tests exercise ``_load_mix_config()``, ``_build_dataset()``,
``_build_mixed_dataset()``, and ``setup_data()``'s mix-config branch
directly, without invoking the full CLI or training loop.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import yaml
from PIL import Image

from clearview.data import ImagePairDataset, MixedDataset
from clearview.scripts.train import (
    _build_dataset,
    _build_mixed_dataset,
    _load_mix_config,
    setup_data,
)


def _save_random_image(path: Path, size: int = 64) -> None:
    """Save a random RGB image to disk."""
    img = Image.fromarray(
        torch.randint(0, 256, (size, size, 3), dtype=torch.uint8).numpy()
    )
    img.save(path)


def _make_pair_source(root: Path, n: int) -> Path:
    """Build an on-disk rainy/clean pair source and return its data_dir."""
    rainy_dir = root / "input"
    clean_dir = root / "target"
    rainy_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)
    for i in range(n):
        _save_random_image(rainy_dir / f"{i:03d}.png")
        _save_random_image(clean_dir / f"{i:03d}.png")
    return root


class TestLoadMixConfig:
    """Tests for _load_mix_config()."""

    def test_loads_valid_config(self, tmp_path: Path) -> None:
        """Test that a well-formed mix-config file loads its sources list."""
        source_dir = _make_pair_source(tmp_path / "a", 2)
        config_path = tmp_path / "mix.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(source_dir),
                            "weight": 2.0,
                        }
                    ]
                }
            )
        )

        sources = _load_mix_config(str(config_path))

        assert len(sources) == 1
        assert sources[0]["weight"] == 2.0

    def test_missing_sources_key_raises(self, tmp_path: Path) -> None:
        """Test that a config with no top-level 'sources' key raises."""
        config_path = tmp_path / "mix.yaml"
        config_path.write_text(yaml.dump({"not_sources": []}))

        with pytest.raises(ValueError, match="sources"):
            _load_mix_config(str(config_path))

    def test_empty_sources_raises(self, tmp_path: Path) -> None:
        """Test that an empty sources list raises."""
        config_path = tmp_path / "mix.yaml"
        config_path.write_text(yaml.dump({"sources": []}))

        with pytest.raises(ValueError, match="sources"):
            _load_mix_config(str(config_path))

    def test_source_missing_data_dir_raises(self, tmp_path: Path) -> None:
        """Test that a source without 'data_dir' raises."""
        config_path = tmp_path / "mix.yaml"
        config_path.write_text(
            yaml.dump({"sources": [{"dataset_type": "pair", "weight": 1.0}]})
        )

        with pytest.raises(ValueError, match="data_dir"):
            _load_mix_config(str(config_path))


class TestBuildDataset:
    """Tests for the shared _build_dataset() dispatcher."""

    def test_pair_type(self, tmp_path: Path) -> None:
        """Test that dataset_type='pair' builds an ImagePairDataset."""
        source_dir = _make_pair_source(tmp_path / "a", 3)

        dataset = _build_dataset(
            "pair", source_dir, None, rainy_dir="input", clean_dir="target"
        )

        assert isinstance(dataset, ImagePairDataset)
        assert len(dataset) == 3

    def test_unknown_type_falls_back_to_pair(self, tmp_path: Path) -> None:
        """Test that an unrecognized dataset_type falls back to 'pair' behavior."""
        source_dir = _make_pair_source(tmp_path / "a", 2)

        dataset = _build_dataset(
            "totally-unknown",
            source_dir,
            None,
            rainy_dir="input",
            clean_dir="target",
        )

        assert len(dataset) == 2


class TestBuildMixedDataset:
    """Tests for _build_mixed_dataset()."""

    def test_combines_sources_with_configured_weights(self, tmp_path: Path) -> None:
        """Test that sources are combined into a MixedDataset with the
        configured per-source weights."""
        big = _make_pair_source(tmp_path / "big", 6)
        small = _make_pair_source(tmp_path / "small", 2)

        config_path = tmp_path / "mix.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(big),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                            "weight": 1.0,
                        },
                        {
                            "dataset_type": "pair",
                            "data_dir": str(small),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                            "weight": 3.0,
                        },
                    ]
                }
            )
        )

        mixed = _build_mixed_dataset(str(config_path), None)

        assert isinstance(mixed, MixedDataset)
        assert len(mixed) == 8
        assert mixed.source_weights == [1.0, 3.0]
        assert mixed.sample_weights() == [1.0] * 6 + [3.0] * 2

    def test_default_weight_is_one(self, tmp_path: Path) -> None:
        """Test that a source with no 'weight' key defaults to 1.0."""
        source_dir = _make_pair_source(tmp_path / "a", 2)
        config_path = tmp_path / "mix.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(source_dir),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                        }
                    ]
                }
            )
        )

        mixed = _build_mixed_dataset(str(config_path), None)

        assert mixed.source_weights == [1.0]

    def test_max_samples_caps_a_source(self, tmp_path: Path) -> None:
        """Test that 'max_samples' truncates a source before concatenation,
        e.g. so one large validation source can't dominate a blended metric."""
        big = _make_pair_source(tmp_path / "big", 10)
        small = _make_pair_source(tmp_path / "small", 3)

        config_path = tmp_path / "mix.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(big),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                            "max_samples": 4,
                        },
                        {
                            "dataset_type": "pair",
                            "data_dir": str(small),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                        },
                    ]
                }
            )
        )

        mixed = _build_mixed_dataset(str(config_path), None)

        assert len(mixed) == 7  # 4 (capped) + 3, not 10 + 3

    def test_max_samples_larger_than_source_is_a_noop(self, tmp_path: Path) -> None:
        """Test that a max_samples larger than the source size doesn't error
        or truncate."""
        source_dir = _make_pair_source(tmp_path / "a", 3)
        config_path = tmp_path / "mix.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(source_dir),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                            "max_samples": 100,
                        }
                    ]
                }
            )
        )

        mixed = _build_mixed_dataset(str(config_path), None)

        assert len(mixed) == 3


def _make_data_args(**overrides: Any) -> argparse.Namespace:
    """Build a minimal Namespace with the fields setup_data() reads."""
    defaults: Dict[str, Any] = {
        "data_dir": None,
        "dataset_type": "pair",
        "train_rainy": "input",
        "train_clean": "target",
        "val_rainy": "input",
        "val_clean": "target",
        "train_split": "train",
        "val_split": "val",
        "mix_config": None,
        "mix_sampler": False,
        "val_mix_config": None,
        "crop_size": 32,
        "flip_prob": 0.5,
        "no_rotation": True,
        "batch_size": 2,
        "val_batch_size": 2,
        "num_workers": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestSetupDataMixConfig:
    """Tests for setup_data()'s --mix-config branch."""

    def test_mix_config_builds_combined_train_loader(self, tmp_path: Path) -> None:
        """Test that --mix-config drives the train loader while --data-dir
        still drives the (separate) val loader."""
        source_a = _make_pair_source(tmp_path / "a", 4)
        source_b = _make_pair_source(tmp_path / "b", 4)
        val_dir = _make_pair_source(tmp_path / "val", 2)

        config_path = tmp_path / "mix.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(source_a),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                            "weight": 1.0,
                        },
                        {
                            "dataset_type": "pair",
                            "data_dir": str(source_b),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                            "weight": 1.0,
                        },
                    ]
                }
            )
        )

        args = _make_data_args(data_dir=str(val_dir), mix_config=str(config_path))
        train_loader, val_loader = setup_data(args)

        assert len(train_loader.dataset) == 8
        assert len(val_loader.dataset) == 2

    def test_mix_sampler_attaches_weighted_sampler(self, tmp_path: Path) -> None:
        """Test that --mix-sampler attaches a WeightedRandomSampler instead
        of plain shuffling."""
        source_a = _make_pair_source(tmp_path / "a", 4)
        val_dir = _make_pair_source(tmp_path / "val", 2)

        config_path = tmp_path / "mix.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(source_a),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                            "weight": 2.0,
                        }
                    ]
                }
            )
        )

        args = _make_data_args(
            data_dir=str(val_dir), mix_config=str(config_path), mix_sampler=True
        )
        train_loader, _ = setup_data(args)

        assert isinstance(train_loader.sampler, torch.utils.data.WeightedRandomSampler)

    def test_without_mix_config_behaves_as_before(self, tmp_path: Path) -> None:
        """Test that omitting --mix-config keeps the original single-source
        behavior (train and val both from --data-dir)."""
        data_dir = tmp_path / "data"
        (data_dir / "train" / "input").mkdir(parents=True)
        (data_dir / "train" / "target").mkdir(parents=True)
        (data_dir / "val" / "input").mkdir(parents=True)
        (data_dir / "val" / "target").mkdir(parents=True)
        for i in range(3):
            _save_random_image(data_dir / "train" / "input" / f"{i:03d}.png")
            _save_random_image(data_dir / "train" / "target" / f"{i:03d}.png")
        _save_random_image(data_dir / "val" / "input" / "000.png")
        _save_random_image(data_dir / "val" / "target" / "000.png")

        args = _make_data_args(
            data_dir=str(data_dir),
            train_rainy="train/input",
            train_clean="train/target",
            val_rainy="val/input",
            val_clean="val/target",
        )
        train_loader, val_loader = setup_data(args)

        assert len(train_loader.dataset) == 3
        assert len(val_loader.dataset) == 1
        assert train_loader.sampler is None or not isinstance(
            train_loader.sampler, torch.utils.data.WeightedRandomSampler
        )


class TestSetupDataValMixConfig:
    """Tests for setup_data()'s --val-mix-config branch."""

    def test_val_mix_config_builds_combined_val_loader(self, tmp_path: Path) -> None:
        """Test that --val-mix-config blends multiple validation sources,
        independently of whether --mix-config is used for training."""
        train_dir = _make_pair_source(tmp_path / "train", 3)
        val_a = _make_pair_source(tmp_path / "val_a", 5)
        val_b = _make_pair_source(tmp_path / "val_b", 2)

        val_config_path = tmp_path / "val_mix.yaml"
        val_config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(val_a),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                        },
                        {
                            "dataset_type": "pair",
                            "data_dir": str(val_b),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                        },
                    ]
                }
            )
        )

        args = _make_data_args(
            data_dir=str(train_dir),
            train_rainy="input",
            train_clean="target",
            val_mix_config=str(val_config_path),
        )
        train_loader, val_loader = setup_data(args)

        assert len(train_loader.dataset) == 3  # unaffected, from --data-dir
        assert len(val_loader.dataset) == 7  # 5 + 2, blended

    def test_val_mix_config_max_samples_balances_a_dominant_source(
        self, tmp_path: Path
    ) -> None:
        """Test the actual motivating case: a large val source capped via
        max_samples so it doesn't dominate a blended metric."""
        train_dir = _make_pair_source(tmp_path / "train", 2)
        dominant = _make_pair_source(tmp_path / "dominant", 100)
        small_a = _make_pair_source(tmp_path / "small_a", 5)
        small_b = _make_pair_source(tmp_path / "small_b", 5)

        val_config_path = tmp_path / "val_mix.yaml"
        val_config_path.write_text(
            yaml.dump(
                {
                    "sources": [
                        {
                            "dataset_type": "pair",
                            "data_dir": str(dominant),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                            "max_samples": 10,
                        },
                        {
                            "dataset_type": "pair",
                            "data_dir": str(small_a),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                        },
                        {
                            "dataset_type": "pair",
                            "data_dir": str(small_b),
                            "rainy_dir": "input",
                            "clean_dir": "target",
                        },
                    ]
                }
            )
        )

        args = _make_data_args(
            data_dir=str(train_dir),
            train_rainy="input",
            train_clean="target",
            val_mix_config=str(val_config_path),
        )
        _, val_loader = setup_data(args)

        # Without max_samples this would be 100 + 5 + 5 = 110, dominated by
        # `dominant`. Capped, it's a much more balanced 10 + 5 + 5 = 20.
        assert len(val_loader.dataset) == 20
