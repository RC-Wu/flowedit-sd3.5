from __future__ import annotations

import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_2D_CASE = REPO_ROOT / "benchmarks" / "2d" / "cases" / "dnaedit_sd35_real_minimal.yaml"
REAL_2D_CONFIG = REPO_ROOT / "benchmarks" / "2d" / "configs" / "flowedit_sd35_real_minimal.yaml"
LEGACY_2D_CONFIG = REPO_ROOT / "benchmarks" / "2d" / "configs" / "flowedit_sd35_base.yaml"
README_FILE = REPO_ROOT / "README.md"
BOOTSTRAP_CN_FILE = REPO_ROOT / "docs" / "BOOTSTRAP_CN.md"
BENCHMARK_3D_DIR = REPO_ROOT / "benchmarks" / "3d_smoke"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise AssertionError(f"YAML root must be mapping: {path}")
    return data


class TestMinimalReal2DContract(unittest.TestCase):
    def test_real_2d_case_manifest_contract(self) -> None:
        self.assertTrue(REAL_2D_CASE.exists(), f"missing case file: {REAL_2D_CASE}")
        case = _load_yaml(REAL_2D_CASE)

        for key in ("case_id", "task", "description", "inputs", "outputs"):
            self.assertIn(key, case, f"case missing key: {key}")

        self.assertEqual(case["task"], "sd35_flowedit_dnaedit_2d_real_minimal")

        inputs = case["inputs"]
        outputs = case["outputs"]
        self.assertIsInstance(inputs, dict)
        self.assertIsInstance(outputs, dict)
        for key in ("source_image", "seed"):
            self.assertIn(key, inputs, f"inputs missing key: {key}")
        self.assertIn("edited_image", outputs)
        prompts = case.get("prompts", {})
        self.assertIn("source", prompts)
        self.assertIn("target", prompts)

        source_image = Path(str(inputs["source_image"]))
        edited_image = Path(str(outputs["edited_image"]))
        self.assertFalse(source_image.is_absolute(), "source_image should stay repo-relative")
        self.assertFalse(edited_image.is_absolute(), "edited_image should stay repo-relative")
        self.assertEqual(edited_image.suffix.lower(), ".png")

    def test_real_2d_config_contract(self) -> None:
        self.assertTrue(REAL_2D_CONFIG.exists(), f"missing config file: {REAL_2D_CONFIG}")
        config = _load_yaml(REAL_2D_CONFIG)

        for key in ("experiment", "model", "pipeline", "dnaedit", "runtime"):
            self.assertIn(key, config, f"config missing section: {key}")

        experiment = config["experiment"]
        self.assertEqual(experiment.get("benchmark"), "2d")
        self.assertEqual(experiment.get("mode"), "real_2d_minimal")

        self.assertEqual(config["model"].get("family"), "sd3.5")
        self.assertEqual(config["model"].get("key"), "sd35-medium-turbo-open")
        self.assertEqual(config["pipeline"].get("edit_method"), "flowedit")
        self.assertIn("steps", config["dnaedit"])

        validation = config.get("validation", {})
        smoke = validation.get("smoke", {})
        self.assertTrue(smoke.get("record_vram_usage"), "smoke must record vram usage")
        metrics = smoke.get("required_metrics", [])
        self.assertIn("peak_vram_mb", metrics)
        self.assertIn("allocated_vram_mb", metrics)
        self.assertIn("reserved_vram_mb", metrics)

    def test_legacy_2d_config_has_smoke_vram_requirement(self) -> None:
        self.assertTrue(LEGACY_2D_CONFIG.exists(), f"missing config file: {LEGACY_2D_CONFIG}")
        config = _load_yaml(LEGACY_2D_CONFIG)
        smoke = config.get("validation", {}).get("smoke", {})
        self.assertTrue(smoke.get("record_vram_usage"), "base 2d config must record vram usage")

    def test_runner_entrypoint_contract(self) -> None:
        runner_file = REPO_ROOT / "src" / "flowedit_sd35" / "runner.py"
        self.assertTrue(runner_file.exists(), "runner.py must exist")

    def test_repo_docs_are_2d_only_and_include_vram_requirement(self) -> None:
        readme = README_FILE.read_text(encoding="utf-8")
        bootstrap_cn = BOOTSTRAP_CN_FILE.read_text(encoding="utf-8")
        self.assertNotIn("3d", readme.lower())
        self.assertNotIn("3d", bootstrap_cn.lower())
        self.assertIn("vram", readme.lower())
        self.assertIn("vram", bootstrap_cn.lower())

    def test_benchmark_tree_has_no_3d_directory(self) -> None:
        self.assertFalse(BENCHMARK_3D_DIR.exists(), "benchmarks must stay 2d-only")


if __name__ == "__main__":
    unittest.main()
