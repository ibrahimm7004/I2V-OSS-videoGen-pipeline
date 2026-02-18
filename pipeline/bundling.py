from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def _bundle_members(run_dir: Path) -> list[Path]:
    members: list[Path] = []
    for candidate in [run_dir / "final_stitched.mp4", run_dir / "manifest.json"]:
        if candidate.exists():
            members.append(candidate)
    for folder_name in ["clips", "frames", "logs"]:
        folder = run_dir / folder_name
        if folder.exists():
            members.extend(sorted(path for path in folder.rglob("*") if path.is_file()))
    return members


def create_run_bundle(run_dir: Path, archive_path: Path | None = None, fmt: str = "zip") -> Path:
    run_dir = run_dir.resolve()
    if archive_path is None:
        suffix = ".zip" if fmt == "zip" else ".tar.gz"
        archive_path = run_dir / f"{run_dir.name}_bundle{suffix}"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    members = _bundle_members(run_dir)
    if fmt == "zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in members:
                archive.write(member, arcname=member.relative_to(run_dir))
    elif fmt == "tar.gz":
        with tarfile.open(archive_path, "w:gz") as archive:
            for member in members:
                archive.add(member, arcname=member.relative_to(run_dir))
    else:
        raise ValueError("Unsupported bundle format. Use 'zip' or 'tar.gz'.")
    return archive_path


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Create a single archive for a pipeline run.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run output directory (outputs/<run_id>).")
    parser.add_argument("--out", type=Path, default=None, help="Archive output path.")
    parser.add_argument("--format", choices=["zip", "tar.gz"], default="zip")
    args = parser.parse_args()

    archive = create_run_bundle(args.run_dir, args.out, args.format)
    print(f"Bundle created: {archive}")


if __name__ == "__main__":
    cli_main()

