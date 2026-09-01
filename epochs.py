from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


# =========================
# Configuration
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = (SCRIPT_DIR / ".." / "rubin_mpc" / "parquet").resolve()

# Accept either variable name.
# If set, it can be a parquet filename, a glob pattern, or a full path.
PARQUET_NAME = (os.environ.get("PARQUET_NAME") or os.environ.get("PARQUET") or "").strip()

NUM_DAYS = 30
EPOCHS_NUM = 3

THRESHOLD = 30
MIN_BANDS = 2
REQUIRED_BANDS_FOR_THRESHOLD = ["r", "i", "g", "u", "z"]

# object name -> list of object output folders, one per parquet where it was found
ObjectOutputMap = Dict[str, List[Path]]


# =========================
# Input discovery
# =========================
def parquet_name_matches(path: Path) -> bool:
    if not PARQUET_NAME:
        return True
    return path.name == PARQUET_NAME or path.match(PARQUET_NAME)


def unique_existing_paths(paths: List[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if p.exists() and key not in seen:
            seen.add(key)
            out.append(rp)
    return sorted(out, key=lambda x: str(x))


def get_requested_parquet_paths() -> List[Path]:
    """Use run_preprocessing.REQUESTED when available; otherwise fall back safely."""
    try:
        import run_preprocessing as rp  # type: ignore

        rp_requested = getattr(rp, "REQUESTED", None)
        rp_base = getattr(rp, "BASE_DIR", None)
    except Exception:
        rp_requested = None
        rp_base = None

    if rp_requested and rp_base:
        requested_paths = [Path(rp_base) / d / "parquet" / f for d, f in rp_requested]
        requested_paths = [p for p in requested_paths if parquet_name_matches(p)]
        return unique_existing_paths(requested_paths)

    # Full path or relative path supplied directly.
    if PARQUET_NAME:
        direct = Path(PARQUET_NAME)
        candidates: List[Path] = []
        if direct.exists():
            candidates.append(direct)

        candidates.append(BASE_DIR / PARQUET_NAME)
        candidates.extend(BASE_DIR.rglob(PARQUET_NAME))
        return unique_existing_paths(candidates)

    # No explicit list/name: only auto-select if there is exactly one parquet.
    direct_parquets = sorted(BASE_DIR.glob("*.parquet"))
    if len(direct_parquets) == 1:
        return unique_existing_paths(direct_parquets)

    recursive_parquets = sorted(BASE_DIR.rglob("*.parquet"))
    if len(recursive_parquets) == 1:
        return unique_existing_paths(recursive_parquets)

    return []


# =========================
# Helpers
# =========================
def utc_to_mjd(dt_utc: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_utc, errors="coerce", utc=True)
    mask_nat = dt.isna()

    dt_naive_utc = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    ns = dt_naive_utc.astype("datetime64[ns]").astype("int64")
    seconds = ns.astype("float64") / 1e9
    seconds[mask_nat.to_numpy()] = float("nan")

    jd = seconds / 86400.0 + 2440587.5
    mjd = jd - 2400000.5
    return pd.Series(mjd, index=dt_utc.index, dtype="float64")


def normalize_band_label(band: str) -> str:
    s = str(band).strip()
    if not s:
        return s
    if s.startswith("L") and len(s) >= 2:
        s = s[1:]
    return s.lower()


def normalize_object_key(value, mode: str) -> str | None:
    if pd.isna(value):
        return None

    if mode == "permid":
        try:
            return str(int(value))
        except Exception:
            return str(value).strip()

    return str(value).strip()


def discover_objects_from_all_csv(output_dir: Path, mode: str) -> ObjectOutputMap:
    """Discover objects from the authoritative *_ALL.csv outputs."""
    found: ObjectOutputMap = {}
    label = "provid" if mode == "provid" else "permid"

    if not output_dir.exists():
        print(f"Found 0 *_ALL.csv files in {label}")
        print(f"[WARNING] Output folder does not exist: {output_dir}")
        return found

    all_csvs = sorted(output_dir.rglob("*_ALL.csv"))
    print(f"Found {len(all_csvs)} *_ALL.csv files in {label}: {output_dir}")

    id_col = "provid" if mode == "provid" else "permid"

    for csv_path in all_csvs:
        try:
            df_head = pd.read_csv(csv_path, nrows=1)
        except Exception:
            continue

        if id_col not in df_head.columns:
            continue

        obj_name = normalize_object_key(df_head[id_col].iloc[0], mode=mode)
        if not obj_name:
            continue

        found.setdefault(obj_name, [])
        found[obj_name].append(csv_path.parent)

    for object_name in list(found.keys()):
        found[object_name] = sorted(dict.fromkeys(found[object_name]), key=lambda p: str(p))

    return found


def merge_object_maps(target: ObjectOutputMap, source: ObjectOutputMap) -> None:
    """Merge without losing objects that occur in more than one parquet output folder."""
    for object_name, dirs in source.items():
        target.setdefault(str(object_name), [])
        target[str(object_name)].extend(dirs)
        target[str(object_name)] = sorted(dict.fromkeys(target[str(object_name)]), key=lambda p: str(p))


def discover_objects_from_output_dirs(output_dirs: List[Path], mode: str) -> ObjectOutputMap:
    combined: ObjectOutputMap = {}
    for output_dir in output_dirs:
        discovered = discover_objects_from_all_csv(output_dir, mode=mode)
        merge_object_maps(combined, discovered)
    return combined


def get_arrow_type(dataset: ds.Dataset, column_name: str):
    return dataset.schema.field(column_name).type


def cast_filter_value(value: str, arrow_type):
    if pa.types.is_integer(arrow_type):
        return int(value)
    if pa.types.is_floating(arrow_type):
        return float(value)
    return str(value)


def get_time_column(dataset: ds.Dataset) -> str:
    schema_names = set(dataset.schema.names)

    if "obs_time" in schema_names:
        return "obs_time"
    if "obstime" in schema_names:
        return "obstime"
    if "obstime_text" in schema_names:
        return "obstime_text"

    raise KeyError("No time column found. Expected one of: obs_time, obstime, obstime_text")


def load_objects_batch(dataset: ds.Dataset, id_column: str, object_values: List[str]) -> Dict[str, pd.DataFrame]:
    """Load many objects in one Arrow scan instead of one scan per object."""
    object_values = sorted({str(v) for v in object_values}, key=str)
    if not object_values:
        return {}

    mode = "provid" if id_column == "provid" else "permid"
    time_col = get_time_column(dataset)
    arrow_type = get_arrow_type(dataset, id_column)
    typed_values = [cast_filter_value(v, arrow_type) for v in object_values]

    table = dataset.to_table(
        columns=[id_column, time_col, "mag", "rmsmag", "band"],
        filter=ds.field(id_column).isin(typed_values),
    )
    df = table.to_pandas()

    if df.empty:
        return {v: pd.DataFrame() for v in object_values}

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["rmsmag"] = pd.to_numeric(df["rmsmag"], errors="coerce")
    df["band_norm"] = df["band"].apply(normalize_band_label)

    df = df.dropna(subset=[time_col, "mag", "band_norm"]).copy()
    if df.empty:
        return {v: pd.DataFrame() for v in object_values}

    df = df.sort_values([id_column, time_col])
    df["mjd"] = utc_to_mjd(df[time_col])
    df = df.dropna(subset=["mjd"]).copy()
    if df.empty:
        return {v: pd.DataFrame() for v in object_values}

    df["mjd_day"] = df["mjd"].floordiv(1).astype("int64")
    df["epoch_bin"] = df["mjd_day"].floordiv(NUM_DAYS).astype("int64")
    df["object_key"] = df[id_column].apply(lambda x: normalize_object_key(x, mode=mode))

    result: Dict[str, pd.DataFrame] = {v: pd.DataFrame() for v in object_values}
    for object_key, group in df.groupby("object_key", sort=False):
        if object_key is None:
            continue
        result[str(object_key)] = group.drop(columns=["object_key"]).copy()

    return result


def process_group(
    dataset: ds.Dataset,
    objects_map: ObjectOutputMap,
    id_column: str,
    label: str,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    print(f"\n=== Processing {label} objects ({len(objects_map)} objects found) ===")

    missing = 0
    epoch_counts: Dict[str, int] = {}
    band_counts: Dict[str, Dict[str, int]] = {}

    cached_data = load_objects_batch(dataset, id_column=id_column, object_values=list(objects_map.keys()))
    print(f"Loaded {len(cached_data)} {label} objects in batch")

    for object_name in sorted(objects_map.keys(), key=str):
        obj = cached_data.get(str(object_name), pd.DataFrame())

        if obj.empty:
            missing += 1
            continue

        epoch_counts[str(object_name)] = int(obj["epoch_bin"].nunique(dropna=True))
        band_counts[str(object_name)] = obj["band_norm"].value_counts(dropna=True).astype(int).to_dict()

    print(f"Done: {missing} skipped.")
    return epoch_counts, band_counts


def at_least_min_required_bands_meet_threshold(
    band_count_map: Dict[str, int],
    threshold: int = THRESHOLD,
) -> bool:
    if not band_count_map:
        return False

    ok = 0
    for band in REQUIRED_BANDS_FOR_THRESHOLD:
        if int(band_count_map.get(band, 0)) >= threshold:
            ok += 1

    return ok >= MIN_BANDS


def total_obs_for_object(object_name: str, band_counts: Dict[str, Dict[str, int]]) -> int:
    return int(sum(band_counts.get(str(object_name), {}).values()))


def parquet_names_for_object(
    object_name: str,
    objects_map: ObjectOutputMap,
    parquet_by_dir: Dict[Path, Path],
) -> str:
    names: set[str] = set()

    for object_output_dir in objects_map.get(str(object_name), []):
        try:
            parquet_dir = object_output_dir.parent.parent.resolve()
        except Exception:
            continue

        parquet_path = parquet_by_dir.get(parquet_dir)
        if parquet_path is not None:
            names.add(parquet_path.name)

    if not names:
        return "unknown parquet"

    return ", ".join(sorted(names, key=str))


def epoch_line(
    object_name: str,
    epoch_counts: Dict[str, int],
    band_counts: Dict[str, Dict[str, int]],
    objects_map: ObjectOutputMap,
    parquet_by_dir: Dict[Path, Path],
) -> str:
    n_epochs = int(epoch_counts.get(str(object_name), 0))
    total_obs = total_obs_for_object(object_name, band_counts)
    sources = parquet_names_for_object(object_name, objects_map, parquet_by_dir)
    return f"  {object_name} ({n_epochs} epochs, {total_obs} total obs) ({sources})"


def save_epochs_to_txt(
    prov_hits: List[str],
    perm_hits: List[str],
    prov_epoch_counts: Dict[str, int],
    perm_epoch_counts: Dict[str, int],
    prov_band_counts: Dict[str, Dict[str, int]],
    perm_band_counts: Dict[str, Dict[str, int]],
    provid_objects: ObjectOutputMap,
    permid_objects: ObjectOutputMap,
    parquet_by_dir: Dict[Path, Path],
    output_path: Path,
) -> None:
    """Save the epoch objects to a txt file with details."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Objects with >= {EPOCHS_NUM} epochs (separated by at least {NUM_DAYS} days)\n")
        f.write("=" * 60 + "\n")

        f.write(f"Provid ({len(prov_hits)}):\n")
        for object_name in prov_hits:
            f.write(
                epoch_line(
                    object_name,
                    prov_epoch_counts,
                    prov_band_counts,
                    provid_objects,
                    parquet_by_dir,
                )
                + "\n"
            )

        f.write(f"\nPermid ({len(perm_hits)}):\n")
        for object_name in perm_hits:
            f.write(
                epoch_line(
                    object_name,
                    perm_epoch_counts,
                    perm_band_counts,
                    permid_objects,
                    parquet_by_dir,
                )
                + "\n"
            )


def save_threshold_to_txt(
    prov_band_hits: List[str],
    perm_band_hits: List[str],
    prov_band_counts: Dict[str, Dict[str, int]],
    perm_band_counts: Dict[str, Dict[str, int]],
    provid_objects: ObjectOutputMap,
    permid_objects: ObjectOutputMap,
    parquet_by_dir: Dict[Path, Path],
    output_path: Path,
) -> None:
    """Save the threshold objects to a txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(
            f"Objects where >= {MIN_BANDS} of {','.join(REQUIRED_BANDS_FOR_THRESHOLD)} "
            f"have total n_obs >= {THRESHOLD}\n"
        )
        f.write("=" * 60 + "\n")

        f.write(f"Provid ({len(prov_band_hits)}):\n")
        for object_name in prov_band_hits:
            total_obs = total_obs_for_object(object_name, prov_band_counts)
            sources = parquet_names_for_object(object_name, provid_objects, parquet_by_dir)
            f.write(f"  {object_name} ({prov_band_counts[object_name]}) Total obs = {total_obs} ({sources})\n")

        f.write(f"\nPermid ({len(perm_band_hits)}):\n")
        for object_name in perm_band_hits:
            total_obs = total_obs_for_object(object_name, perm_band_counts)
            sources = parquet_names_for_object(object_name, permid_objects, parquet_by_dir)
            f.write(f"  {object_name} ({perm_band_counts[object_name]}) Total obs = {total_obs} ({sources})\n")


# =========================
# Main
# =========================
def main() -> None:
    parquet_paths = get_requested_parquet_paths()

    if not parquet_paths:
        raise FileNotFoundError(
            "No parquet files found. Check run_preprocessing.REQUESTED or set PARQUET_NAME/PARQUET."
        )

    print(f"Processing {len(parquet_paths)} parquet file(s):")
    for path in parquet_paths:
        print(f"  {path}")

    parquet_by_dir: Dict[Path, Path] = {p.parent.resolve(): p for p in parquet_paths}
    dataset = ds.dataset([str(p) for p in parquet_paths], format="parquet")

    provid_dirs = [p.parent / "provisional_designation_outputs" for p in parquet_paths]
    permid_dirs = [p.parent / "numbered_asteroids_outputs" for p in parquet_paths]

    provid_objects = discover_objects_from_output_dirs(provid_dirs, mode="provid")
    permid_objects = discover_objects_from_output_dirs(permid_dirs, mode="permid")

    prov_epoch_counts, prov_band_counts = process_group(
        dataset=dataset,
        objects_map=provid_objects,
        id_column="provid",
        label="provid",
    )

    perm_epoch_counts, perm_band_counts = process_group(
        dataset=dataset,
        objects_map=permid_objects,
        id_column="permid",
        label="permid",
    )

    prov_hits = sorted([k for k, v in prov_epoch_counts.items() if v >= EPOCHS_NUM], key=str)
    perm_hits = sorted([k for k, v in perm_epoch_counts.items() if v >= EPOCHS_NUM], key=str)

    print(f"\nObjects with >= {EPOCHS_NUM} epochs (separated by at least {NUM_DAYS} days):")
    print(f"Provid ({len(prov_hits)}):")
    for object_name in prov_hits:
        print(epoch_line(object_name, prov_epoch_counts, prov_band_counts, provid_objects, parquet_by_dir))

    print(f"Permid ({len(perm_hits)}):")
    for object_name in perm_hits:
        print(epoch_line(object_name, perm_epoch_counts, perm_band_counts, permid_objects, parquet_by_dir))

    prov_band_hits = sorted(
        [k for k, m in prov_band_counts.items() if at_least_min_required_bands_meet_threshold(m, THRESHOLD)],
        key=str,
    )
    perm_band_hits = sorted(
        [k for k, m in perm_band_counts.items() if at_least_min_required_bands_meet_threshold(m, THRESHOLD)],
        key=str,
    )

    print(
        f"\nObjects where >= {MIN_BANDS} of {','.join(REQUIRED_BANDS_FOR_THRESHOLD)} "
        f"have total n_obs >= {THRESHOLD}:"
    )

    print(f"Provid ({len(prov_band_hits)}):")
    for object_name in prov_band_hits:
        total_obs = total_obs_for_object(object_name, prov_band_counts)
        sources = parquet_names_for_object(object_name, provid_objects, parquet_by_dir)
        print(f"  {object_name} ({prov_band_counts[object_name]}) Total obs = {total_obs} ({sources})")

    print(f"Permid ({len(perm_band_hits)}):")
    for object_name in perm_band_hits:
        total_obs = total_obs_for_object(object_name, perm_band_counts)
        sources = parquet_names_for_object(object_name, permid_objects, parquet_by_dir)
        print(f"  {object_name} ({perm_band_counts[object_name]}) Total obs = {total_obs} ({sources})")

    epochs_txt = SCRIPT_DIR / "epochs.txt"
    save_epochs_to_txt(
        prov_hits=prov_hits,
        perm_hits=perm_hits,
        prov_epoch_counts=prov_epoch_counts,
        perm_epoch_counts=perm_epoch_counts,
        prov_band_counts=prov_band_counts,
        perm_band_counts=perm_band_counts,
        provid_objects=provid_objects,
        permid_objects=permid_objects,
        parquet_by_dir=parquet_by_dir,
        output_path=epochs_txt,
    )
    print(f"\nSaved epochs to: {epochs_txt}")

    threshold_txt = SCRIPT_DIR / "found_objects.txt"
    save_threshold_to_txt(
        prov_band_hits=prov_band_hits,
        perm_band_hits=perm_band_hits,
        prov_band_counts=prov_band_counts,
        perm_band_counts=perm_band_counts,
        provid_objects=provid_objects,
        permid_objects=permid_objects,
        parquet_by_dir=parquet_by_dir,
        output_path=threshold_txt,
    )
    print(f"Saved threshold objects to: {threshold_txt}")


if __name__ == "__main__":
    main()
