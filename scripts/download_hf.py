#!/usr/bin/env python3
"""Reliable Hugging Face / ModelScope downloader for slow HTTP proxies.

Large model files are fetched in parallel byte ranges and resumed from a
``.parts`` directory.  SHA256 metadata from ModelScope or Hugging Face is
verified after all parts are assembled.

Examples:
    python scripts/download_hf.py modelscope model black-forest-labs/FLUX.2-klein-base-4B \
        --local-dir weights/FLUX.2-klein-base-4B
    python scripts/download_hf.py hf dataset i1-datasets/i1-pexels-512-resolution-1m-tfrecord \
        --local-dir data/raw/i1_pexels_512 --endpoint https://hf-mirror.com
"""

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def request_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ecflux-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def list_hf(repo: str, repo_type: str, endpoint: str, revision: str) -> list[dict]:
    kind = "datasets" if repo_type == "dataset" else "models"
    quoted = urllib.parse.quote(repo, safe="/")
    url = (
        f"{endpoint.rstrip('/')}/api/{kind}/{quoted}/tree/"
        f"{urllib.parse.quote(revision)}?recursive=true&expand=true"
    )
    entries = request_json(url)
    files = []
    for item in entries:
        if item.get("type") != "file":
            continue
        lfs = item.get("lfs") or {}
        files.append(
            {
                "path": item["path"],
                "size": int(lfs.get("size") or item.get("size") or 0),
                "sha256": lfs.get("oid") or "",
            }
        )
    return files


def list_modelscope(repo: str, repo_type: str, revision: str) -> list[dict]:
    kind = "datasets" if repo_type == "dataset" else "models"
    quoted = urllib.parse.quote(repo, safe="/")
    if repo_type == "dataset":
        url = (
            f"https://www.modelscope.cn/api/v1/{kind}/{quoted}/repo/tree"
            f"?Revision={urllib.parse.quote(revision)}&Root=&Recursive=true&PageNumber=1&PageSize=1000"
        )
    else:
        url = (
            f"https://www.modelscope.cn/api/v1/{kind}/{quoted}/repo/files"
            f"?Revision={urllib.parse.quote(revision)}&Recursive=true"
        )
    payload = request_json(url)
    if payload.get("Code") != 200:
        raise RuntimeError(f"ModelScope list failed: {payload.get('Message')}")
    files = []
    for item in payload["Data"]["Files"]:
        if item.get("Type") != "blob":
            continue
        files.append(
            {
                "path": item["Path"],
                "size": int(item.get("Size") or 0),
                "sha256": item.get("Sha256") or "",
            }
        )
    return files


def download_url(source: str, repo: str, repo_type: str, path: str, endpoint: str, revision: str) -> str:
    if source == "modelscope":
        kind = "datasets" if repo_type == "dataset" else "models"
        quoted_repo = urllib.parse.quote(repo, safe="/")
        quoted_path = urllib.parse.quote(path, safe="/")
        return (
            f"https://www.modelscope.cn/api/v1/{kind}/{quoted_repo}/repo"
            f"?Revision={urllib.parse.quote(revision)}&FilePath={quoted_path}"
        )
    quoted_repo = urllib.parse.quote(repo, safe="/")
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"{endpoint.rstrip('/')}/{prefix}{quoted_repo}/resolve/{urllib.parse.quote(revision)}/{path}"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_range(url: str, start: int, end: int, part: str) -> None:
    expected = end - start + 1
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > expected:
        os.remove(part)
        have = 0
    if have == expected:
        return

    error = None
    for attempt in range(20):
        try:
            headers = {"Range": f"bytes={start + have}-{end}", "User-Agent": "ecflux-downloader/1.0"}
            with requests.Session() as session:
                session.trust_env = True
                with session.get(
                    url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True
                ) as response:
                    # Some mirrors ignore Range for small objects. Accept that only
                    # at offset zero and never copy beyond this part's assigned range.
                    if response.status_code == 206:
                        offset = 0
                    elif response.status_code == 200 and start + have == 0:
                        offset = 0
                    else:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    remaining = expected - have
                    mode = "ab" if have else "wb"
                    with open(part, mode) as file:
                        for block in response.iter_content(1024 * 1024):
                            if not block:
                                continue
                            block = block[:remaining]
                            file.write(block)
                            remaining -= len(block)
                            if remaining <= 0:
                                break
            have = os.path.getsize(part)
            if have == expected:
                return
            error = RuntimeError(f"short range: {have}/{expected}")
        except Exception as exc:
            error = exc
            time.sleep(min(2 * (attempt + 1), 30))
    raise RuntimeError(f"range {start}-{end} failed: {error}")


def get_size(session: requests.Session, url: str) -> int:
    with session.head(url, timeout=(30, 120), allow_redirects=True) as response:
        response.raise_for_status()
        return int(response.headers["Content-Length"])


def download_file(
    url: str,
    output: str,
    expected_size: int,
    expected_sha256: str,
    chunk_size: int,
    workers: int,
) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if os.path.exists(output):
        size_ok = os.path.getsize(output) == expected_size
        hash_ok = not expected_sha256 or sha256_file(output) == expected_sha256
        if size_ok and hash_ok:
            print(f"[skip] {output}", flush=True)
            return

    if not expected_size:
        with requests.Session() as session:
            session.trust_env = True
            expected_size = get_size(session, url)
    part_dir = output + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    n_parts = (expected_size + chunk_size - 1) // chunk_size
    parts = [os.path.join(part_dir, f"{index:05d}.part") for index in range(n_parts)]

    done_before = sum(
        min(os.path.getsize(part), min((index + 1) * chunk_size, expected_size) - index * chunk_size)
        for index, part in enumerate(parts)
        if os.path.exists(part)
    )
    print(
        f"[get] {os.path.basename(output)} {expected_size / 2**30:.2f} GiB "
        f"({n_parts} parts, resumed {done_before / 2**30:.2f} GiB)",
        flush=True,
    )
    started = time.time()

    def task(index: int) -> None:
        start = index * chunk_size
        end = min(start + chunk_size, expected_size) - 1
        fetch_range(url, start, end, parts[index])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task, index): index for index in range(n_parts)}
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == n_parts or completed % max(1, n_parts // 20) == 0:
                current = sum(os.path.getsize(part) for part in parts if os.path.exists(part))
                elapsed = max(time.time() - started, 1)
                speed = (current - done_before) / elapsed / 2**20
                print(
                    f"  {os.path.basename(output)}: {current / 2**30:.2f}/{expected_size / 2**30:.2f} GiB "
                    f"({completed}/{n_parts}, {speed:.1f} MiB/s)",
                    flush=True,
                )

    temporary = output + ".tmp"
    with open(temporary, "wb") as destination:
        for part in parts:
            with open(part, "rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    actual_size = os.path.getsize(temporary)
    if actual_size != expected_size:
        os.remove(temporary)
        shutil.rmtree(part_dir)
        raise RuntimeError(f"{output}: expected {expected_size} bytes, got {actual_size}")
    if expected_sha256:
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            os.remove(temporary)
            shutil.rmtree(part_dir)
            raise RuntimeError(f"{output}: sha256 mismatch {actual_sha256} != {expected_sha256}")
    os.replace(temporary, output)
    shutil.rmtree(part_dir)
    print(f"[ok] {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["hf", "modelscope"])
    parser.add_argument("repo_type", choices=["model", "dataset"])
    parser.add_argument("repo")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default="master" if "modelscope" else "main")
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--exclude-pattern", action="append", default=[])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()

    if args.source == "modelscope":
        files = list_modelscope(args.repo, args.repo_type, args.revision)
    else:
        files = list_hf(args.repo, args.repo_type, args.endpoint, args.revision)

    if args.allow_pattern:
        files = [
            item for item in files
            if any(fnmatch.fnmatch(item["path"], pattern) for pattern in args.allow_pattern)
        ]
    if args.exclude_pattern:
        files = [
            item for item in files
            if not any(fnmatch.fnmatch(item["path"], pattern) for pattern in args.exclude_pattern)
        ]

    print(f"{len(files)} files, {sum(item['size'] for item in files) / 2**30:.2f} GiB", flush=True)
    for item in sorted(files, key=lambda entry: entry["size"]):
        output = os.path.join(args.local_dir, item["path"])
        url = download_url(
            args.source, args.repo, args.repo_type, item["path"], args.endpoint, args.revision
        )
        download_file(
            url,
            output,
            item["size"],
            item["sha256"],
            args.chunk_size,
            args.workers,
        )


if __name__ == "__main__":
    main()
