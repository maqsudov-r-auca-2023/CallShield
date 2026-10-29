"""Download the FraudZen dataset (INRIA, DOI 10.57745/TAV6PQ) file by file via the
Dataverse API, with retries and resume - the single bundled zip is fragile on slow links.

    python scripts/download_fraudzen.py
"""
import sys
import time
import zipfile
from pathlib import Path

import requests

BASE = "https://entrepot.recherche.data.gouv.fr"
DOI = "doi:10.57745/TAV6PQ"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "fraudzen"
OUT.mkdir(parents=True, exist_ok=True)


def fetch(url: str, dest: Path, size: int | None, tries: int = 8) -> None:
    for attempt in range(1, tries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if size and have == size:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 200 and have:  # server ignored Range -> restart
                    have = 0
                r.raise_for_status()
                with open(dest, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if not size or dest.stat().st_size == size:
                return
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}/{tries} for {dest.name}: {e}", flush=True)
            time.sleep(min(30, 3 * attempt))
    sys.exit(f"failed to download {dest.name}")


meta = requests.get(f"{BASE}/api/datasets/:persistentId/", params={"persistentId": DOI}, timeout=60).json()
files = meta["data"]["latestVersion"]["files"]
print(f"FraudZen: {len(files)} files")
for f in files:
    df = f["dataFile"]
    sub = OUT / f.get("directoryLabel", "")
    sub.mkdir(parents=True, exist_ok=True)
    dest = sub / df["filename"]
    size = df.get("filesize")
    print(f"- {dest.relative_to(OUT)} ({(size or 0)/1e6:.1f} MB)", flush=True)
    # format=original returns the uploaded file, not Dataverse's tab-delimited conversion
    fetch(f"{BASE}/api/access/datafile/{df['id']}?format=original", dest, None if df.get("originalFileSize") else size)
    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.with_suffix(""))
print("[ok] FraudZen downloaded to", OUT)
