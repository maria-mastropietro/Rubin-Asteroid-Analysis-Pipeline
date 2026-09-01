import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import warnings

import pandas as pd
from astroquery.jplhorizons import Horizons
from astropy.utils.data import conf

warnings.filterwarnings("ignore")

PROVISIONAL_RE = re.compile(r"^\d{4}\s+[A-Z]{1,2}\d*$")


class SearchNEA_MBO:
    """Search choose*.txt files for permid and provid IDs and comment non-NEA/non-MBO objects."""

    def __init__(self, max_workers=4, max_retries=5, retry_delay=10, a_jup=5.2034):
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.a_jup = a_jup
        self.cache: dict[str, str] = {}
        self.classifications: dict[str, dict] = {}
        conf.remote_timeout = 60

        self.nea_classes = {"Amor", "Apollo", "Aten", "Atira"}
        self.saved_classes = self.nea_classes | {"MBO", "Not a NEA or MBO", "Unclassified"}

    @staticmethod
    def is_provisional(designation: str) -> bool:
        return bool(PROVISIONAL_RE.match(designation))

    @staticmethod
    def safe_id(text: str) -> str:
        return text.strip()

    @staticmethod
    def parse_choose_line(line: str) -> str | None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return None
        content = stripped.split("#", 1)[0].strip()
        if not content:
            return None
        return content

    def classify_from_elements(self, a: float, e: float, q: float) -> str:
        Q = a * (1 + e)
        if q >= 1.3 and 2.0 < a < 3.3:
            return "MBO"
        elif q >= 1.3:
            return "Not a NEA or MBO"
        elif a > 1.0 and 1.017 < q < 1.3:
            return "Amor"
        elif a > 1.0 and q < 1.017:
            return "Apollo"
        elif a < 1.0 and Q > 0.983:
            return "Aten"
        elif a < 1.0 and Q < 0.983:
            return "Atira"
        else:
            return "Unclassified"

    def tisserand(self, a: float, e: float, i_deg: float) -> float:
        i = math.radians(i_deg)
        return self.a_jup / a + 2.0 * math.cos(i) * math.sqrt(a / self.a_jup * (1.0 - e ** 2))

    def classify_via_jpl(self, object_id: str) -> tuple[str, float | None, float | None, float | None, float | None, float | None]:
        object_id = str(object_id).strip()
        if object_id in self.cache and not self.cache[object_id].startswith("Error"):
            label = self.cache[object_id]
            return label, None, None, None, None, None

        for attempt in range(1, self.max_retries + 1):
            try:
                obj = Horizons(id=object_id, id_type="smallbody", location="500@10", epochs=None)
                el = obj.elements()
                a = float(el["a"][0])
                e = float(el["e"][0])
                q = float(el["q"][0])
                i = float(el["incl"][0])
                T = float(el["P"][0]) / 365.25
                label = self.classify_from_elements(a, e, q)
                self.cache[object_id] = label
                self.save_cache()
                return label, a, e, q, i, T
            except Exception as ex:
                err = str(ex)
                is_timeout = "ConnectTimeoutError" in err or "timed out" in err.lower()
                is_ratelimit = "<!DOCTYPE" in err or "<html" in err.lower()
                if (is_timeout or is_ratelimit) and attempt < self.max_retries:
                    wait = self.retry_delay * attempt
                    print(f"  {object_id}: retrying in {wait}s (attempt {attempt}/{self.max_retries})")
                    time.sleep(wait)
                    continue
                if is_timeout:
                    result = "Error: timeout"
                elif is_ratelimit:
                    result = "Error: rate-limited"
                else:
                    result = f"Error: {err[:80]}"

                self.cache[object_id] = result
                self.save_cache()
                return result, None, None, None, None, None

        result = "Error: max retries exceeded"
        self.cache[object_id] = result
        self.save_cache()
        return result, None, None, None, None, None

    def cache_file_path(self) -> str:
        return os.path.join("search_NEA_MBO_cache.json")

    def disk_cache(self) -> None:
        cache_path = self.cache_file_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"Loaded cached results from {cache_path}")
            except Exception:
                self.cache = {}

    def save_cache(self) -> None:
        cache_path = self.cache_file_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, sort_keys=True)

    def discover_choose_files(self, root_dir: Path) -> list[Path]:
        return sorted(root_dir.rglob("choose*.txt"))

    def collect_object_ids(self, file_path: Path) -> list[str]:
        ids = []
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                object_id = self.parse_choose_line(line)
                if object_id is None:
                    continue
                # Split by tab and take only the first column (object name)
                parts = object_id.split("\t")
                object_name = parts[0].strip()
                if object_name:
                    ids.append(object_name)
        return ids

    def classify_ids(self, object_ids: list[str]) -> None:
        self.disk_cache()
        unique_ids = sorted(set(object_ids))
        print(f"Classifying {len(unique_ids)} unique IDs...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.classify_via_jpl, oid): oid for oid in unique_ids}
            for future in as_completed(futures):
                oid = futures[future]
                try:
                    label, a, e, q, i, T = future.result()
                except Exception as ex:
                    label = f"Error: {ex}"
                self.classifications[oid] = {
                    "id": oid,
                    "label": label,
                    "group": self.group_from_label(label),
                }
                print(f"{oid}: {label}")

    def group_from_label(self, label: str) -> str | None:
        if label in self.nea_classes:
            return "NEA"
        if label == "MBO":
            return "MBO"
        if label in {"Not a NEA or MBO", "Unclassified"}:
            return "Not_NEA_or_MBO"
        if label.startswith("Error"):
            return "Error"
        return None

    def annotate_file(self, file_path: Path) -> Path:
        lines: list[str] = []
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                object_id = self.parse_choose_line(line)
                if object_id is None:
                    lines.append(line)
                    continue
                classification = self.classifications.get(object_id)
                if classification is None:
                    lines.append(line)
                    continue
                group = classification["group"]
                if group == "Not_NEA_or_MBO":
                    normalized = line.rstrip("\n")
                    if not normalized.lstrip().startswith("#"):
                        comment = f"# {normalized}\n"
                        lines.append(comment)
                    else:
                        lines.append(line)
                else:
                    lines.append(line)

        with file_path.open("w", encoding="utf-8") as fh:
            fh.writelines(lines)
        return file_path

    def write_summary(self, summary_path: Path, summary: dict[str, int]) -> None:
        with summary_path.open("w", encoding="utf-8") as fh:
            fh.write("Search NEA/MBO summary\n")
            for key, value in summary.items():
                fh.write(f"{key}: {value}\n")

    def run(self, root_dir: Path, inplace: bool = False) -> None:
        choose_files = self.discover_choose_files(root_dir)
        if not choose_files:
            raise FileNotFoundError(f"No choose*.txt files found under {root_dir}")

        all_ids: list[str] = []
        for file_path in choose_files:
            ids = self.collect_object_ids(file_path)
            all_ids.extend(ids)
            print(f"Found {len(ids)} IDs in {file_path}")

        self.classify_ids(all_ids)

        summary = {"NEA": 0, "MBO": 0, "Not a NEA or MBO": 0, "Unclassified": 0, "Errors": 0}
        for stats in self.classifications.values():
            label = stats["label"]
            if label in self.nea_classes:
                summary["NEA"] += 1
            elif label == "MBO":
                summary["MBO"] += 1
            elif label == "Not a NEA or MBO":
                summary["Not a NEA or MBO"] += 1
            elif label == "Unclassified":
                summary["Unclassified"] += 1
            else:
                summary["Errors"] += 1

        output_files = []
        for file_path in choose_files:
            annotated_path = self.annotate_file(file_path)
            output_files.append(str(annotated_path))
            print(f"Annotated file written to: {annotated_path}")

        summary_path = root_dir / "search_NEA_MBO_summary.txt"
        self.write_summary(summary_path, summary)
        print(f"Summary written to: {summary_path}")

        result_path = root_dir / "search_NEA_MBO_results.json"
        with result_path.open("w", encoding="utf-8") as fh:
            json.dump(self.classifications, fh, indent=2, sort_keys=True)
        print(f"Classification results written to: {result_path}")

        print("\nTotals:")
        for key, value in summary.items():
            print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search choose*.txt files for permid/provid IDs and comment non-NEA/non-MBO entries.")
    parser.add_argument("root", nargs="?", default=".", help="Root folder to search for choose*.txt files")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original choose files with commented non-NEA/non-MBO lines")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel Horizons queries")
    parser.add_argument("--retries", type=int, default=5, help="Max retries for Horizons queries")
    parser.add_argument("--retry-delay", type=int, default=10, help="Seconds between retry attempts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    searcher = SearchNEA_MBO(max_workers=args.workers, max_retries=args.retries, retry_delay=args.retry_delay)
    root_dir = Path(args.root).resolve()
    searcher.run(root_dir, inplace=args.inplace)


if __name__ == "__main__":
    main()
