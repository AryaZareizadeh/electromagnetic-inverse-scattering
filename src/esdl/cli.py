"""Explicit, repeatable entry points for the recovered research workflow."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

from .config import ProjectPaths
from .data import SPLIT_FILES, encode_domains, encode_fields, load_split


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _output_directory(args, paths, label: str) -> Path:
    path = Path(args.output) if args.output else paths.root / "runs" / (
        label + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    path = path.absolute()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, content) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _data(args, paths, split=None):
    return load_split(args.data_dir or paths.data, split or args.split, args.samples)


def _model(args, paths):
    from .checkpoints import load_forward_model

    return load_forward_model(args.checkpoint or paths.checkpoint, args.device)


def _inspect(args, paths):
    from .checkpoints import load_tensor
    import torch

    result = {"root": str(paths.root), "datasets": {}, "artifacts": {}}
    for split, filenames in SPLIT_FILES.items():
        arrays = {}
        for name in filenames:
            if name is None:
                continue
            path = paths.data / name
            if not path.is_file():
                arrays[name] = {"missing": True}
                continue
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            arrays[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                            "size_bytes": path.stat().st_size}
        result["datasets"][split] = arrays
    if paths.checkpoint.is_file():
        checkpoint = torch.load(paths.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("state_dict", checkpoint)
        optimizer = checkpoint.get("optimizer", {})
        result["artifacts"][paths.checkpoint.name] = {
            "parameters": sum(value.numel() for value in state.values()),
            "shapes": {key: list(value.shape) for key, value in state.items()},
            "saved_learning_rates": [group["lr"] for group in optimizer.get("param_groups", [])],
            "size_bytes": paths.checkpoint.stat().st_size,
        }
        del checkpoint, state
    for path in (paths.matrices, paths.initial):
        if path.is_file():
            tensor = load_tensor(path)
            result["artifacts"][path.name] = {
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "logical_bytes": tensor.numel() * tensor.element_size(),
                "storage_bytes": tensor.untyped_storage().nbytes(),
                "size_bytes": path.stat().st_size,
            }
    print(json.dumps(result, indent=2))


def _generate(args, paths):
    from .forward import ForwardConfig, simulate_fields
    from .generation import generate_cylinders, generate_mnist

    output = _output_directory(args, paths, "generated-" + args.kind)
    if args.kind == "cylinders":
        raw = generate_cylinders(args.count, seed=args.seed)
        labels, indices, split = None, None, "cylinders"
    else:
        raw, labels, indices = generate_mnist(args.count, split=args.split, seed=args.seed,
                                             download_dir=paths.root / "data" / "raw")
        split = args.split
    config = ForwardConfig(formulation=args.formulation)
    fields = simulate_fields(raw, config)
    domain_name, field_name, label_name = SPLIT_FILES[split]
    np.save(output / domain_name, encode_domains(raw))
    np.save(output / field_name, encode_fields(fields))
    if labels is not None:
        np.save(output / label_name, labels)
        np.save(output / "source_indices.npy", indices)
    _write_json(output / "metadata.json", {"kind": args.kind, "split": split,
                "count": args.count, "seed": args.seed, "forward_config": asdict(config),
                "domain_channels": ["zero", "relative_permittivity"],
                "field_channels": ["real", "imaginary"]})
    print(f"Generated {args.count} samples in {output}")


def _train_forward(args, paths):
    from .training import train_forward_model

    train = _data(args, paths, "train")
    test = load_split(args.data_dir or paths.data, "test", args.test_samples)
    output = _output_directory(args, paths, "forward")
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 5e-5 if args.full_batch else 1e-6
    result = train_forward_model(train.domains, train.fields, test.domains, test.fields, output,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=learning_rate,
        device=args.device, seed=args.seed, resume=args.resume, full_batch=args.full_batch,
        resume_learning_rate=args.resume_learning_rate)
    print(json.dumps(result, indent=2))


def _evaluate(args, paths):
    from .metrics import regression_metrics
    from .training import predict_fields

    data = _data(args, paths)
    model = _model(args, paths)
    predictions = predict_fields(model, data.domains, batch_size=args.batch_size, device=args.device)
    metrics = regression_metrics(predictions, data.fields)
    output = _output_directory(args, paths, "evaluation")
    np.save(output / "predicted_fields.npy", predictions)
    _write_json(output / "metrics.json", metrics)
    if args.plot:
        from .plotting import save_comparison, save_error_histogram

        save_comparison(output / "comparison.png", data.domains[0], data.fields[0], predictions[0])
        save_error_histogram(output / "field_errors.png", data.fields, predictions)
    print(json.dumps({"output": str(output), "samples": len(predictions), "metrics": metrics}, indent=2))


def _train_descent(args, paths):
    from .descent import load_initial_guess, train_descent

    data = _data(args, paths, "train")
    model = _model(args, paths)
    initial = load_initial_guess(args.initial or paths.initial, len(data.domains), args.device)
    output = _output_directory(args, paths, "descent")
    matrices, history = train_descent(model, data.domains, data.fields, initial,
        iterations=args.iterations, alpha=args.alpha, tolerance=args.tolerance,
        batch_size=args.batch_size, device=args.device, output_path=output / "Ks.pt")
    _write_json(output / "history.json", history)
    _write_json(output / "config.json", {"alpha": args.alpha, "tolerance": args.tolerance,
        "maximum_iterations": args.iterations, "learned_iterations": len(matrices),
        "samples": len(data.domains), "checkpoint": str(args.checkpoint or paths.checkpoint),
        "initial": str(args.initial or paths.initial)})
    print(f"Learned {len(matrices)} matrices in {output}")


def _reconstruct(args, paths):
    from .descent import load_descent_matrices, load_initial_guess, reconstruct
    from .metrics import regression_metrics
    from .training import predict_fields

    data = None
    if args.fields:
        fields = np.load(args.fields, allow_pickle=False)
        if fields.ndim == 3:
            fields = encode_fields(fields)
        if fields.ndim != 4 or fields.shape[1:] != (2, 20, 20) or np.iscomplexobj(fields):
            raise ValueError("Observed fields must be (N, 2, 20, 20) real or (N, 20, 20) complex")
        fields = fields[:args.samples] if args.samples else fields
        if len(fields) == 0 or not np.isfinite(fields).all():
            raise ValueError("Observed fields must contain nonempty finite samples")
    else:
        data = _data(args, paths)
        fields = data.fields
    model = _model(args, paths)
    matrices = load_descent_matrices(args.matrices or paths.matrices)
    initial = load_initial_guess(args.initial or paths.initial, len(fields), args.device)
    result = reconstruct(model, fields, initial, matrices, iterations=args.iterations,
                          batch_size=args.batch_size, device=args.device).detach().cpu().numpy()
    predictions = predict_fields(model, result, batch_size=args.batch_size, device=args.device)
    metrics = {"field": regression_metrics(predictions, fields)}
    if data is not None:
        metrics["domain"] = regression_metrics(result, data.domains)
    output = _output_directory(args, paths, "reconstruction")
    np.save(output / "reconstructed_domains.npy", result)
    np.save(output / "predicted_fields.npy", predictions)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "config.json", {"samples": len(fields),
        "iterations": args.iterations if args.iterations is not None else len(matrices),
        "checkpoint": str(args.checkpoint or paths.checkpoint),
        "matrices": str(args.matrices or paths.matrices), "initial": str(args.initial or paths.initial)})
    if args.plot and data is not None:
        from .plotting import save_comparison

        save_comparison(output / "comparison.png", data.domains[0], fields[0], predictions[0], result[0])
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Project directory containing data and artifacts")
    parser.add_argument("--threads", type=_positive, help="PyTorch CPU thread count")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Inspect dataset and trained-artifact metadata")
    commands.add_parser("verify-assets", help="Verify the original files' SHA-256 hashes")
    generate = commands.add_parser("generate", help="Generate data in a new directory")
    generate.add_argument("--kind", choices=("mnist", "cylinders"), default="cylinders")
    generate.add_argument("--split", choices=("train", "test"), default="train")
    generate.add_argument("--count", type=_positive, default=32)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--formulation", choices=("legacy", "free-space"), default="legacy")
    generate.add_argument("--output", type=Path)
    for name in ("train-forward", "evaluate", "train-descent", "reconstruct"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--data-dir", type=Path, help="Directory with the canonical array filenames")
        cmd.add_argument("--samples", type=_positive, help="Limit samples for a small run")
        cmd.add_argument("--batch-size", type=_positive, default=100)
        cmd.add_argument("--device", default="cpu", help="cpu, auto, cuda, or cuda:N")
        cmd.add_argument("--output", type=Path, help="New output directory; must not already exist")
        if name == "train-forward":
            cmd.add_argument("--epochs", type=_positive, default=400)
            cmd.add_argument("--test-samples", type=_positive)
            cmd.add_argument("--learning-rate", type=float)
            cmd.add_argument("--full-batch", action="store_true")
            cmd.add_argument("--seed", type=int, default=0)
            cmd.add_argument("--resume", type=Path)
            cmd.add_argument("--resume-learning-rate", type=float,
                             help="Override the resumed Adam learning rate")
        else:
            cmd.add_argument("--checkpoint", type=Path)
        if name in ("evaluate", "reconstruct"):
            cmd.add_argument("--split", choices=tuple(SPLIT_FILES), default="test")
            cmd.add_argument("--plot", action="store_true")
        if name in ("train-descent", "reconstruct"):
            cmd.add_argument("--initial", type=Path)
            cmd.add_argument("--iterations", type=_positive, default=800 if name == "train-descent" else None)
        if name == "train-descent":
            cmd.add_argument("--alpha", type=float, default=0.5)
            cmd.add_argument("--tolerance", type=float, default=0.02)
        if name == "reconstruct":
            cmd.add_argument("--matrices", type=Path)
            cmd.add_argument("--fields", type=Path, help="External observed fields; no ground truth required")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.from_root(args.root)
    try:
        if args.threads:
            import torch

            torch.set_num_threads(args.threads)
        if args.command == "verify-assets":
            from .assets import verify_assets

            results = verify_assets(paths.root)
            for result in results:
                print(f"{'OK' if result['ok'] else 'FAILED'} {result['path']}")
            return 0 if all(item["ok"] for item in results) else 1
        handlers = {"inspect": _inspect, "generate": _generate, "train-forward": _train_forward,
                    "evaluate": _evaluate, "train-descent": _train_descent, "reconstruct": _reconstruct}
        handlers[args.command](args, paths)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
