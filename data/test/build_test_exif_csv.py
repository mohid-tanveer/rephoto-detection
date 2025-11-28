import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import ExifTags, Image


def get_repo_root():
    """return the repository root assuming this file is in data/."""
    # __file__ -> .../rephoto-detection/data/build_exif_csv.py
    # parents[1] -> .../rephoto-detection
    return Path(__file__).resolve().parents[1]


def iter_image_files(data_dir):
    """recursively yield all jpeg image files under the data directory."""
    image_paths: List[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            image_paths.append(path)
    return sorted(image_paths)


def infer_labels(path, repo_root):
    """infer semantic labels from the file path."""
    rel_path = path.relative_to(repo_root)
    parts = rel_path.parts

    label = "unknown"
    screen_source = "none"
    screen_type = "none"

    if "authentic" in parts and "re-photo" not in parts:
        label = "authentic"
    elif "re-photo" in parts:
        label = "rephoto"
        # expected structure: data/re-photo/<subset-dir>/...
        try:
            rephoto_index = parts.index("re-photo")
            subset_dir = parts[rephoto_index + 1]
            subset_dir_lower = subset_dir.lower()
            if subset_dir_lower.startswith("ai-"):
                screen_source = "ai"
            elif subset_dir_lower.startswith("authentic-"):
                screen_source = "authentic"

            if subset_dir_lower.endswith("lcd"):
                screen_type = "lcd"
            elif subset_dir_lower.endswith("oled"):
                screen_type = "oled"
        except (ValueError, IndexError):
            # valueerror if "re-photo" is not found,
            # indexerror if there is no subset directory after "re-photo"
            pass

    return {
        "label": label,
        "screen_source": screen_source,
        "screen_type": screen_type,
    }


def build_exif_tag_map():
    """build a mapping from exif tag ids to human-readable names."""
    return {tag_id: tag_name for tag_id, tag_name in ExifTags.TAGS.items()}


def rational_to_float(value):
    """convert common exif rational representations to float."""
    if value is None:
        return None
    try:
        # pillow often returns its own rational type or a tuple (num, den)
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            den = float(value.denominator)
            if den == 0:
                return None
            return float(value.numerator) / den
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            den = float(den)
            if den == 0:
                return None
            return float(num) / den
        return float(value)
    except Exception:
        return None


def extract_exif_fields(image_path, tag_map):
    """extract relevant exif fields from an image."""
    fields: Dict[str, Any] = {}

    with Image.open(image_path) as img:
        exif_raw = img._getexif() or {}

        exif_dict: Dict[str, Any] = {}
        for tag_id, value in exif_raw.items():
            name = tag_map.get(tag_id, str(tag_id))
            exif_dict[name] = value

        width, height = img.size
        fields["image_width"] = width
        fields["image_height"] = height

        # core exif features
        fields["focal_length_35mm_eq"] = exif_dict.get("FocalLengthIn35mmFilm")
        fields["f_number"] = rational_to_float(exif_dict.get("FNumber"))
        fields["exposure_time"] = rational_to_float(exif_dict.get("ExposureTime"))
        fields["shutter_speed_value"] = rational_to_float(
            exif_dict.get("ShutterSpeedValue")
        )

        iso_val = exif_dict.get(
            "PhotographicSensitivity", exif_dict.get("ISOSpeedRatings")
        )
        if isinstance(iso_val, (list, tuple)):
            iso_val = iso_val[0] if iso_val else None
        fields["iso"] = iso_val

        fields["metering_mode"] = exif_dict.get("MeteringMode")

        # device / lens metadata
        fields["device_model"] = exif_dict.get("Model")
        fields["device_make"] = exif_dict.get("Make")
        fields["lens_model"] = exif_dict.get("LensModel")

        # capture context
        fields["datetime_original"] = exif_dict.get("DateTimeOriginal")
        fields["orientation"] = exif_dict.get("Orientation")
        fields["flash"] = exif_dict.get("Flash")

    return fields


def to_serializable(value):
    """convert values to something csv-writable."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return value


def build_rows():
    """build metadata rows for all images under data/."""
    repo_root = get_repo_root()
    data_dir = repo_root / "test"
    tag_map = build_exif_tag_map()

    rows: List[Dict[str, Any]] = []
    image_paths = iter_image_files(data_dir)

    for img_path in image_paths:
        rel_path = img_path.relative_to(repo_root.parent).as_posix()
        filename = img_path.name

        labels = infer_labels(img_path, repo_root)

        exif_fields: Dict[str, Any] = {}
        try:
            exif_fields = extract_exif_fields(img_path, tag_map)
        except Exception:
            # if there is any issue reading exif, continue with whatever we have
            exif_fields = {}

        row: Dict[str, Any] = {
            "filepath": rel_path,
            "filename": filename,
            "label": labels["label"],
            "screen_source": labels["screen_source"],
            "screen_type": labels["screen_type"],
        }
        row.update(exif_fields)

        rows.append(row)

    return rows


def write_csv(rows, output_path):
    """write rows to a csv file with a stable column order."""
    if not rows:
        print("no images found under data/, nothing to write")
        return

    # explicit column order for id / label fields
    base_columns = [
        "filepath",
        "filename",
        "label",
        "screen_source",
        "screen_type",
    ]

    # remaining columns collected from the first row
    remaining_columns = [c for c in rows[0].keys() if c not in base_columns]
    all_columns = base_columns + remaining_columns

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for row in rows:
            serializable_row = {k: to_serializable(row.get(k)) for k in all_columns}
            writer.writerow(serializable_row)


if __name__ == "__main__":
    """entry point for building the exif metadata csv."""
    repo_root = get_repo_root()
    output_path = repo_root / "test" / "test_exif_metadata.csv"

    print("scanning images under data/...")
    rows = build_rows()
    print(f"found {len(rows)} images, writing metadata to {output_path}")
    write_csv(rows, output_path)
    print("done.")
