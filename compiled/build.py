#!/usr/bin/env python3
"""Compile GTA-PS Python targets with Nuitka.

The pipeline intentionally keeps generated C/C++ and executable outputs under
code/compiled/artifacts so source directories stay clean.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


COMPILED_ROOT = Path(__file__).resolve().parent
CODE_ROOT = COMPILED_ROOT.parent
DEFAULT_ROOTS = ("misc", "experiments", "src")
DEFAULT_PYTHONPATH = (CODE_ROOT / "src", CODE_ROOT / "threads", CODE_ROOT / "misc")
EXCLUDED_DIR_NAMES = {
    ".agent",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "compiled",
    "gta_ps.egg-info",
}


@dataclass(frozen=True)
class Target:
    name: str
    source: str
    path: str


def _safe_name(relative_path: Path) -> str:
    without_suffix = relative_path.with_suffix("").as_posix()
    name = re.sub(r"[^A-Za-z0-9]+", "__", without_suffix).strip("_")
    return name or "target"


def _under_excluded_dir(path: Path) -> bool:
    try:
        rel = path.relative_to(CODE_ROOT)
    except ValueError:
        return True
    return any(part in EXCLUDED_DIR_NAMES for part in rel.parts)


def _iter_python_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix == ".py" and not _under_excluded_dir(root):
            yield root
        return

    if not root.exists():
        raise SystemExit(f"Target root does not exist: {root}")

    for path in root.rglob("*.py"):
        if not _under_excluded_dir(path):
            yield path


def discover_targets(roots: list[str]) -> list[Target]:
    seen: set[Path] = set()
    targets: list[Target] = []

    for root_name in roots:
        root = (CODE_ROOT / root_name).resolve()
        for path in _iter_python_files(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rel = resolved.relative_to(CODE_ROOT)
            targets.append(
                Target(
                    name=_safe_name(rel),
                    source=rel.as_posix(),
                    path=str(resolved),
                )
            )

    return sorted(targets, key=lambda item: item.source)


def select_targets(targets: list[Target], selectors: list[str]) -> list[Target]:
    if not selectors:
        return targets

    by_source = {target.source: target for target in targets}
    by_name = {target.name: target for target in targets}
    selected: list[Target] = []

    for selector in selectors:
        normalized = selector.replace("\\", "/")
        target = by_source.get(normalized) or by_name.get(normalized)
        if target is None:
            matches = [
                candidate
                for candidate in targets
                if candidate.source.endswith(normalized) or candidate.name == normalized
            ]
            if len(matches) == 1:
                target = matches[0]
        if target is None:
            raise SystemExit(f"Unknown target: {selector}")
        if target not in selected:
            selected.append(target)

    return selected


def write_manifest(path: Path, targets: list[Target]) -> None:
    payload = {
        "code_root": str(CODE_ROOT),
        "platform": platform.platform(),
        "target_count": len(targets),
        "targets": [asdict(target) for target in targets],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    path_entries = [str(path) for path in DEFAULT_PYTHONPATH]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        path_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def split_nuitka_options(values: list[str]) -> list[str]:
    options: list[str] = []
    for value in values:
        options.extend(shlex.split(value))
    return options


def check_nuitka(python_cmd: str) -> None:
    cmd = shlex.split(python_cmd) + ["-m", "nuitka", "--version"]
    result = subprocess.run(
        cmd,
        cwd=CODE_ROOT,
        env=build_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Nuitka is not available for this interpreter. Install it with:\n"
            "  python -m pip install -r compiled/requirements-build.txt\n\n"
            f"Interpreter command: {python_cmd}\n"
            f"Output:\n{result.stdout}"
        )
    print(result.stdout.strip())


def nuitka_command(args: argparse.Namespace, target: Target) -> list[str]:
    output_root = Path(args.output_root).resolve()
    command = shlex.split(args.python) + ["-m", "nuitka"]

    if args.standalone:
        command.append("--standalone")
    if args.onefile:
        command.append("--onefile")
    if args.clean:
        command.append("--remove-output")
    if args.assume_yes_for_downloads:
        command.append("--assume-yes-for-downloads")
    if args.windows_mingw64 and os.name == "nt":
        command.append("--mingw64")

    command.extend(
        [
            "--follow-imports",
            f"--jobs={args.jobs}",
            f"--output-dir={output_root}",
            f"--output-filename={target.name}",
            "--include-package=gta_ps",
        ]
    )

    for plugin in args.enable_plugin:
        command.append(f"--enable-plugin={plugin}")

    command.extend(split_nuitka_options(args.nuitka_option))
    command.append(target.path)
    return command


def run_build(args: argparse.Namespace, targets: list[Target]) -> int:
    env = build_environment()
    output_root = Path(args.output_root).resolve()
    if not args.dry_run:
        check_nuitka(args.python)
        output_root.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[Target, int]] = []
    for index, target in enumerate(targets, start=1):
        command = nuitka_command(args, target)
        print(f"[{index}/{len(targets)}] {target.source} -> {target.name}")
        if args.dry_run:
            print(shlex.join(command))
            continue

        result = subprocess.run(command, cwd=CODE_ROOT, env=env)
        if result.returncode != 0:
            failures.append((target, result.returncode))
            if not args.keep_going:
                break

    if failures:
        print("\nFailed targets:")
        for target, returncode in failures:
            print(f"  {target.source} ({target.name}) exited {returncode}")
        return 1

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and compile GTA-PS Python scripts with Nuitka.",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Relative code/ roots or files to compile. Defaults: misc experiments src.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Compile one discovered target by relative path or generated name. May repeat.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered targets and exit without compiling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Nuitka commands without compiling.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Nuitka availability for the selected interpreter.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help='Python command used to invoke Nuitka. Example: "py -3".',
    )
    parser.add_argument(
        "--output-root",
        default=str(COMPILED_ROOT / "artifacts"),
        help="Directory for Nuitka build and dist outputs.",
    )
    parser.add_argument(
        "--manifest",
        default=str(COMPILED_ROOT / "manifest.json"),
        help="Path for the discovered target manifest.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write the discovered target manifest.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max((os.cpu_count() or 2) - 1, 1),
        help="Compiler jobs passed to Nuitka.",
    )
    parser.add_argument(
        "--standalone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build standalone executables. Enabled by default.",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Build one-file executables instead of standalone directories.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Ask Nuitka to remove previous output for each target before building.",
    )
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue compiling remaining targets after a failure.",
    )
    parser.add_argument(
        "--windows-mingw64",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="On Windows, ask Nuitka to use MinGW64.",
    )
    parser.add_argument(
        "--assume-yes-for-downloads",
        action="store_true",
        help="Allow Nuitka to answer yes to dependency download prompts.",
    )
    parser.add_argument(
        "--enable-plugin",
        action="append",
        default=[],
        help="Enable a Nuitka plugin by name. May repeat.",
    )
    parser.add_argument(
        "--nuitka-option",
        action="append",
        default=[],
        help="Raw Nuitka option(s) appended after defaults. May repeat.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    targets = select_targets(discover_targets(args.roots), args.target)

    if not args.no_manifest:
        write_manifest(Path(args.manifest), targets)

    if args.list:
        for target in targets:
            print(f"{target.name}\t{target.source}")
        print(f"\n{len(targets)} target(s)")
        return 0

    if args.check:
        check_nuitka(args.python)
        return 0

    return run_build(args, targets)


if __name__ == "__main__":
    raise SystemExit(main())
