"""Extract selected still frames from the default FourRooms showcase GIF."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
INPUT_GIF = ROOT / "plots" / "fourrooms_showcase_last400.gif"
OUTPUT_DIRS = [ROOT / "plots", PROJECT_ROOT / "presentation" / "assets"]
FRAMES = [20, 100, 400]
OUTPUT_PREFIX = "fourrooms_showcase_last400"


def main() -> None:
    if not INPUT_GIF.is_file():
        raise FileNotFoundError(f"Missing showcase GIF: {INPUT_GIF}")

    with Image.open(INPUT_GIF) as gif:
        frame_count = getattr(gif, "n_frames", 1)
        for frame_number in FRAMES:
            if frame_number < 1 or frame_number > frame_count:
                raise ValueError(f"Frame {frame_number} outside GIF range 1..{frame_count}")

            gif.seek(frame_number - 1)
            frame = gif.convert("RGBA")
            filename = f"{OUTPUT_PREFIX}_frame_{frame_number:03d}.png"
            for out_dir in OUTPUT_DIRS:
                out_dir.mkdir(parents=True, exist_ok=True)
                frame.save(out_dir / filename)
                print(out_dir / filename)


if __name__ == "__main__":
    main()
