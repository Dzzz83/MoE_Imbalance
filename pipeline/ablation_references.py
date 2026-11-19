"""Pinned Stage A references and compatibility checks for Stage B."""

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch

from pipeline.ablation_weights import baseline_scores, reassign_weights, transform_weights
from pipeline.run_ablation import _completed_run
from pipeline import run_weight_study as assignment
from pipeline.runtime import file_sha256

PARENT_COMMIT = "0d4cf3e90be73513ddbe5c731698610b0285773c"
PARENT_SOURCE_COMMIT = "a792b81ef737e8fba54033ffb4cc72a92c6b2e2a"
PARENT_MANIFEST_SHA256 = "f607071e8f8977be013376aa87b411680cca16fd314a26b7c854f6162121ab20"
PARENT_ARCHIVE_SHA256 = "23f217c28b72e82b91428a9024ec4cc890cd510470de4fe5d59e01ed67d51194"
ASSIGNMENT_CONFIG = assignment.resolve_config({})

# Only dispatch and priority reassignment are extended in these modules. Keep the
# actual shared continuation and scoring source identical to the parent. Hash
# source text rather than AST dumps, whose representation can vary by Python version.
PARENT_FUNCTION_HASHES = {
    "run_weight_study.py": {
        "_tensor_hash": "ec3bca9b22192de33eeee19a6a208ef5806c674707628c0001f9b8f8b96df847",
        "environment": "db89a081ea53054b64b0a6047ddff9d38ad5e46c4f4486a4eddb57055122f548",
        "_continuation": "6dec1f2bd4d34ef9b8fa3b6ab12d300216a858a69823f7a711902ab463faf511",
    },
    "ablation_weights.py": {
        "class_balanced_weights": "da3bb2d38d91ffcd061f380a2d436f71c3d6e52944166419b3a2d46f2e28c23c",
        "transform_weights": "d4ea16b3c7ce58a06ea45809ad0bb329de4cba5bba45c5266ed00887138771fd",
        "weight_diagnostics": "b4a4e1b4430c801d9054e63e15276125da9e3d43a8ce0cb2f0de1f081597e1d6",
        "baseline_scores": "533e097da55f375b48589e18ee20c0c1cba088e260c4140cd3a3a076ecf73b49",
        "_midranks": "eb7574601d5da36cc07f7ea07d315ed393539057531e2540980fb5a8efe1dff1",
        "_quarter_membership": "381ed297585fffd55d059ef1d1792e3c3fa06ce00a9a7cc87f38b0f975facc11",
        "assignment_audit": "12646733dd15ae4fb8b487ac645fe17ffea4be232a966d57d83cf8746b009556",
    },
}
PARENT_EXTENDED_MODULES = {
    "run_weight_study.py": "c35cf71c5ea293ae150b069cf5c16760cac13943cd12cf039c1784cdeb1fe8ff",
    "ablation_weights.py": "576c64d4e79e57e9411107423d5abd64919a17bcb9db8df88895db3a2f770f9b",
}
RUN_FILES = {
    "config.json", "weights.pt", "weight_diagnostics.json", "starting_metrics.json",
    "history.json", "metrics.json", "final.pt",
    "prototype_weight_diagnostics.json", "assignment_finetuning_diagnostics.json",
}


def validate_parent_manifest(root):
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file() or file_sha256(path) != PARENT_MANIFEST_SHA256:
        raise ValueError("Pinned Stage A manifest is missing or changed")
    manifest = json.loads(path.read_text())
    if manifest["source_commit"] != PARENT_SOURCE_COMMIT:
        raise ValueError("Stage A source commit changed")
    for relative, digest in manifest["report_hashes"].items():
        parts = PurePosixPath(relative).parts
        if not parts or ".." in parts or PurePosixPath(relative).is_absolute():
            raise ValueError("Invalid parent report path")
        path = root / "reports" / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Stage A report missing or changed: {relative}")
    assets = json.loads((root / "reports/study_assets.json").read_text())
    if assets.get("complete", {}).get("sha256") != PARENT_ARCHIVE_SHA256:
        raise ValueError("Stage A complete archive identity changed")
    return manifest


def validate_shared_implementation(parent):
    root = Path(__file__).parent
    for name, digest in parent["implementation"].items():
        if name in PARENT_EXTENDED_MODULES:
            if digest != PARENT_EXTENDED_MODULES[name]:
                raise ValueError(f"Unexpected parent implementation: {name}")
        elif file_sha256(root / name) != digest:
            raise ValueError(f"Shared training implementation changed: {name}; record a reference-rerun budget")
    for name, expected in PARENT_FUNCTION_HASHES.items():
        source = (root / name).read_text()
        lines = source.splitlines(keepends=True)
        functions = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
                function_source = "".join(lines[start - 1:node.end_lineno])
                functions[node.name] = hashlib.sha256(function_source.encode()).hexdigest()
        if any(functions.get(function) != digest for function, digest in expected.items()):
            raise ValueError(f"Shared training functions changed: {name}; record a reference-rerun budget")


def validate_environment(parent_environment, device):
    current = assignment.environment(torch.device(device))
    # Compiler/build text can differ, but Python's minor version is part of the recipe.
    keys = parent_environment.keys() - {"python"}
    mismatches = {k: {"parent": parent_environment[k], "current": current.get(k)}
                  for k in keys if parent_environment[k] != current.get(k)}
    if parent_environment["python"].split()[0].split(".")[:2] != current["python"].split()[0].split(".")[:2]:
        mismatches["python"] = {"parent": parent_environment["python"], "current": current["python"]}
    if mismatches:
        raise ValueError("Stage A training environment mismatch; restore the parent environment or record "
                         f"a reference-rerun budget before training: {json.dumps(mismatches, sort_keys=True)}")
    return current


def load_parent(root, inputs):
    """Validate every reference receipt, score, weight vector and training setting."""
    root = Path(root)
    manifest = validate_parent_manifest(root)
    reports = root / "reports"
    parent = json.loads((reports / "study_config.json").read_text())
    if parent["config"] != ASSIGNMENT_CONFIG or parent["inputs"] != inputs["provenance"]:
        raise ValueError("Stage A training settings or baseline/cache identities changed")
    validate_shared_implementation(parent)
    records = json.loads((reports / "study_results.json").read_text())
    status = json.loads((reports / "study_status.json").read_text())
    if (status["status"] != "complete" or status["error"] is not None
            or assignment.summarize_assignment(records, parent["config"])["status"] != "complete"):
        raise ValueError("Stage A references must contain all 27 completed runs and three baselines")
    indexed = {(r["seed"], r["condition"], r["shuffle_seed"]): r for r in records}
    scores_by_seed, strong_by_seed = {}, {}
    for seed in parent["config"]["seeds"]:
        bundle, checkpoint = inputs["bundles"][seed], inputs["checkpoints"][seed]
        scores = torch.load(reports / f"diagnostics/seed{seed}/scores.pt", map_location="cpu", weights_only=True)
        margins, correct = baseline_scores(bundle["features"], bundle["labels"], checkpoint["model_state_dict"])
        expected_scores = {"margins": margins, "correct": correct, "tie_order": torch.randperm(
            len(margins), generator=torch.Generator().manual_seed(parent["config"]["tie_seed"]))}
        if scores.keys() != expected_scores.keys() or any(not torch.equal(scores[k], v) for k, v in expected_scores.items()):
            raise ValueError("Stage A saved fitting scores or tie permutation changed")
        strong = transform_weights(bundle["ot_weights"], bundle["labels"], "strong_ot")
        for condition, shuffle in assignment.conditions(parent["config"]):
            name = condition if shuffle is None else f"{condition}_shuffle{shuffle}"
            run = reports / f"runs/seed{seed}/{name}"
            if condition in ("easy_strong", "boundary_strong", "hard_strong"):
                weights = reassign_weights(strong, bundle["labels"], margins, condition, scores["tie_order"])
            elif condition == "shuffled_strong_ot":
                weights = transform_weights(strong, bundle["labels"], "shuffled_ot", shuffle)
            else:
                weights = transform_weights(bundle["ot_weights"], bundle["labels"], condition)
            identity = {
                "seed": seed, "condition": condition, "shuffle_seed": shuffle, "config": parent["config"],
                "files": inputs["provenance"]["seed_files"][str(seed)],
                "weights_sha256": assignment._tensor_hash(weights),
                "margins_sha256": assignment._tensor_hash(margins),
                "original_class_counts": inputs["split"]["original_class_counts"],
            }
            receipt_path = run / "complete.json"
            if not receipt_path.is_file():
                raise ValueError(f"Missing Stage A completion receipt: {name}, seed {seed}")
            receipt = json.loads(receipt_path.read_text())
            if set(receipt["files"]) != RUN_FILES or not _completed_run(run, identity):
                raise ValueError("Incomplete Stage A reference receipt")
            saved = torch.load(run / "weights.pt", map_location="cpu", weights_only=True)
            if not torch.equal(saved["weights"], weights) or not torch.equal(saved["labels"], bundle["labels"]):
                raise ValueError("Stage A reference weights changed")
            record = indexed[seed, condition, shuffle]
            history = json.loads((run / "history.json").read_text())
            if (record["metrics"] != json.loads((run / "metrics.json").read_text())
                    or len(history) != parent["config"]["classifier"]["epochs"]
                    or history[-1]["development"] != record["metrics"]
                    or json.loads((run / "starting_metrics.json").read_text()) != indexed[seed, "baseline", None]["metrics"]):
                raise ValueError("Stage A reference metrics or schedule changed")
        scores_by_seed[seed] = scores
        strong_by_seed[seed] = torch.load(reports / f"runs/seed{seed}/strong_ot/weights.pt",
                                          map_location="cpu", weights_only=True)["weights"]
    return {"records": records, "scores": scores_by_seed, "strong": strong_by_seed,
            "environment": parent["environment"], "provenance": {
                "results_commit": PARENT_COMMIT, "source_commit": PARENT_SOURCE_COMMIT,
                "manifest_sha256": PARENT_MANIFEST_SHA256,
                "archive": json.loads((reports / "study_assets.json").read_text())["complete"],
                "report_hashes": manifest["report_hashes"], "validation": "passed",
            }}
