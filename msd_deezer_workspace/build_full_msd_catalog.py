"""Build the full 1M-row MSD catalog CSV from msd_summary_file.h5 + artist_term.db.

Produces a CSV with the exact same 58-column layout as running
    python msd_deezer_pipeline.py --extract-only
over an extracted full MSD tree, but without needing to extract the 210 GB
of A-Z tarballs.

Rows are sorted by msd_track_id so msd_row_index matches the order the
pipeline would produce from sorted(subset_dir.rglob("*.h5")).

msd_artist_mbtags_with_count is left blank because per-track mbtag counts
only exist inside each track's HDF5 file (the downstream Deezer matcher
does not read this column).

Inputs:
    D:/msd_targz/AdditionalFiles/msd_summary_file.h5
    D:/msd_targz/AdditionalFiles/artist_term.db

Output:
    D:/msd_targz/msd_full_catalog.csv
"""
from __future__ import annotations

import csv
import sqlite3
import time
from pathlib import Path

import h5py

from msd_deezer_pipeline import (
    CSV_FIELDNAMES,
    clean_optional_text,
    make_blank_row,
    normalize_scalar,
)

SUMMARY_H5 = Path("D:/msd_targz/AdditionalFiles/msd_summary_file.h5")
ARTIST_TERM_DB = Path("D:/msd_targz/AdditionalFiles/artist_term.db")
OUT_CSV = Path("D:/msd_targz/msd_full_catalog.csv")


def load_artist_tags(db_path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    terms: dict[str, list[str]] = {}
    for artist_id, term in cur.execute("SELECT artist_id, term FROM artist_term"):
        terms.setdefault(artist_id, []).append(term)

    mbtags: dict[str, list[str]] = {}
    for artist_id, mbtag in cur.execute("SELECT artist_id, mbtag FROM artist_mbtag"):
        mbtags.setdefault(artist_id, []).append(mbtag)

    con.close()
    return terms, mbtags


def reconstruct_file_path(track_id: str) -> str:
    if len(track_id) < 5:
        return ""
    return f"{track_id[2]}/{track_id[3]}/{track_id[4]}/{track_id}.h5"


def join_tag_list(tags: list[str]) -> str:
    cleaned = []
    for t in tags:
        text = normalize_scalar(t)
        if text:
            cleaned.append(text)
    return "|".join(cleaned)


def main() -> int:
    if not SUMMARY_H5.exists():
        raise FileNotFoundError(SUMMARY_H5)
    if not ARTIST_TERM_DB.exists():
        raise FileNotFoundError(ARTIST_TERM_DB)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"loading artist tag tables from {ARTIST_TERM_DB} ...", flush=True)
    artist_terms, artist_mbtags = load_artist_tags(ARTIST_TERM_DB)
    print(
        f"  artists with terms : {len(artist_terms):,}\n"
        f"  artists with mbtags: {len(artist_mbtags):,}\n"
        f"  loaded in {time.time()-t0:.1f}s",
        flush=True,
    )

    with h5py.File(SUMMARY_H5, "r") as f:
        meta = f["metadata/songs"]
        ana = f["analysis/songs"]
        mb = f["musicbrainz/songs"]
        n = meta.shape[0]
        print(f"summary h5: {n:,} rows", flush=True)

        # Pass 1: stream through the summary h5 in file order, writing rows
        # to a temp CSV with msd_row_index left blank. We'll sort and fill
        # row indices in pass 2.
        temp_unsorted = OUT_CSV.with_suffix(OUT_CSV.suffix + ".unsorted.tmp")
        t1 = time.time()
        written = 0
        with temp_unsorted.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()

            CHUNK = 20000
            for start in range(0, n, CHUNK):
                stop = min(start + CHUNK, n)
                meta_chunk = meta[start:stop]
                ana_chunk = ana[start:stop]
                mb_chunk = mb[start:stop]

                for i in range(stop - start):
                    m = meta_chunk[i]
                    a = ana_chunk[i]
                    y = mb_chunk[i]

                    track_id = normalize_scalar(a["track_id"])
                    artist_id = normalize_scalar(m["artist_id"])

                    row = make_blank_row()
                    row["msd_row_index"] = ""  # filled after sorting
                    row["msd_file_path"] = reconstruct_file_path(track_id)
                    row["msd_track_id"] = track_id
                    row["msd_song_id"] = normalize_scalar(m["song_id"])
                    row["msd_title"] = normalize_scalar(m["title"])
                    row["msd_artist_name"] = normalize_scalar(m["artist_name"])
                    row["msd_release"] = normalize_scalar(m["release"])
                    row["msd_year"] = normalize_scalar(y["year"])
                    row["msd_genre"] = normalize_scalar(m["genre"])
                    row["msd_artist_id"] = artist_id
                    row["msd_artist_mbid"] = normalize_scalar(m["artist_mbid"])
                    row["msd_artist_7digitalid"] = normalize_scalar(m["artist_7digitalid"])
                    row["msd_artist_playmeid"] = normalize_scalar(m["artist_playmeid"])
                    row["msd_track_7digitalid"] = normalize_scalar(m["track_7digitalid"])
                    row["msd_release_7digitalid"] = normalize_scalar(m["release_7digitalid"])
                    row["msd_artist_location"] = normalize_scalar(m["artist_location"])
                    row["msd_artist_latitude"] = normalize_scalar(m["artist_latitude"])
                    row["msd_artist_longitude"] = normalize_scalar(m["artist_longitude"])
                    row["msd_artist_familiarity"] = normalize_scalar(m["artist_familiarity"])
                    row["msd_artist_hotttnesss"] = normalize_scalar(m["artist_hotttnesss"])
                    row["msd_song_hotttnesss"] = normalize_scalar(m["song_hotttnesss"])
                    row["msd_duration"] = normalize_scalar(a["duration"])
                    row["msd_tempo"] = normalize_scalar(a["tempo"])
                    row["msd_loudness"] = normalize_scalar(a["loudness"])
                    row["msd_key"] = normalize_scalar(a["key"])
                    row["msd_key_confidence"] = normalize_scalar(a["key_confidence"])
                    row["msd_mode"] = normalize_scalar(a["mode"])
                    row["msd_mode_confidence"] = normalize_scalar(a["mode_confidence"])
                    row["msd_time_signature"] = normalize_scalar(a["time_signature"])
                    row["msd_time_signature_confidence"] = normalize_scalar(a["time_signature_confidence"])
                    row["msd_danceability"] = normalize_scalar(a["danceability"])
                    row["msd_energy"] = normalize_scalar(a["energy"])
                    row["msd_end_of_fade_in"] = normalize_scalar(a["end_of_fade_in"])
                    row["msd_start_of_fade_out"] = normalize_scalar(a["start_of_fade_out"])
                    row["msd_analysis_sample_rate"] = normalize_scalar(a["analysis_sample_rate"])
                    row["msd_audio_md5"] = normalize_scalar(a["audio_md5"])
                    row["msd_artist_terms"] = join_tag_list(artist_terms.get(artist_id, []))
                    row["msd_artist_mbtags"] = join_tag_list(artist_mbtags.get(artist_id, []))
                    row["msd_artist_mbtags_with_count"] = ""

                    writer.writerow({k: clean_optional_text(v) for k, v in row.items()})
                    written += 1

                if written % 100000 == 0 or stop == n:
                    print(
                        f"  pass1 {written:,}/{n:,}  ({time.time()-t1:.1f}s elapsed)",
                        flush=True,
                    )

    # Pass 2: load the unsorted CSV, sort by msd_track_id, reassign
    # msd_row_index to sequential integers, write final CSV atomically.
    print("sorting rows by msd_track_id ...", flush=True)
    t2 = time.time()
    with temp_unsorted.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != CSV_FIELDNAMES:
            raise RuntimeError(
                f"unexpected header in temp CSV (got {len(header)} cols, "
                f"expected {len(CSV_FIELDNAMES)})"
            )
        rows = list(reader)
    print(f"  loaded {len(rows):,} rows in {time.time()-t2:.1f}s", flush=True)

    track_id_col = CSV_FIELDNAMES.index("msd_track_id")
    row_index_col = CSV_FIELDNAMES.index("msd_row_index")

    t3 = time.time()
    rows.sort(key=lambda r: r[track_id_col])
    print(f"  sorted in {time.time()-t3:.1f}s", flush=True)

    t4 = time.time()
    for i, r in enumerate(rows):
        r[row_index_col] = str(i)

    temp_final = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    with temp_final.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDNAMES)
        writer.writerows(rows)
    temp_final.replace(OUT_CSV)
    print(f"  wrote final CSV in {time.time()-t4:.1f}s", flush=True)

    try:
        temp_unsorted.unlink()
    except OSError:
        pass

    total_elapsed = time.time() - t0
    size_mb = OUT_CSV.stat().st_size / (1024 * 1024)
    print(
        f"\nDONE: wrote {len(rows):,} rows to {OUT_CSV}  "
        f"({size_mb:.1f} MB, {total_elapsed:.1f}s total)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
