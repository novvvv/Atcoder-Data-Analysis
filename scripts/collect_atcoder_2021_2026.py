#!/usr/bin/env python3
"""AtCoder sponsored enterprise ABC collection pipeline (2021~2026).

Scope:
- Japanese major-enterprise sponsored ABC contests only (see config/sponsor_whitelist.yaml)
- Excludes generic ABC, ARC, AHC, standalone finals

Sources:
- Problems/tags/solver_count: AtCoder AI Tagged Problems API
- Contest title/year: mirror cache + atcoder.jp contest pages
- Difficulty: kenkoooo AtCoder Problems problem-models.json (primary); mirror fallback
- Contest points (配点): extracted from JA statement HTML (separate slot-score proxy)
- Statements: atcoder.jp task pages (incremental crawl)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

SPIDERS_DATA_URL = "https://atcoder-for-chinese-developers.github.io/spiders/data.json"
KENKOOOO_MODELS_URL = "https://kenkoooo.com/atcoder/resources/problem-models.json"
KENKOOOO_MODELS_CACHE = "kenkoooo_problem_models.json"
AI_TAGGED_ALL_URL = "https://atcoder-tagged-backend.onrender.com/problems/all"
PARSER_VERSION = "2.3.0-sponsor-abc-kenkoooo-diff"
CONTEST_POINTS_RE = re.compile(r"配点\s*:\s*<var>(\d+)</var>")
STATEMENT_LANG = "ja"
HTML_RAW_SUBDIR = "atcoder_html_ja"


@dataclass
class RunConfig:
    start_year: int
    end_year: int
    max_html_pages: int
    sleep_sec: float
    out_dir: Path
    whitelist_path: Path


@dataclass
class SponsorRules:
    markers: list[str]
    enterprises: list[dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> SponsorRules:
        if yaml is None:
            raise RuntimeError("PyYAML required: pip install pyyaml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            markers=list(raw.get("sponsor_markers", [])),
            enterprises=list(raw.get("enterprises", [])),
        )

    def match(self, title: str) -> str | None:
        if not title:
            return None
        if not any(m.lower() in title.lower() for m in self.markers):
            return None
        for ent in self.enterprises:
            for kw in ent.get("keywords", []):
                if kw in title:
                    return str(ent.get("id", "unknown"))
        return None


def fetch_json(url: str, retries: int = 3, timeout: int = 60) -> Any:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            if i < retries - 1:
                time.sleep(1.0 + i)
    raise RuntimeError(f"Failed to fetch JSON: {url}") from last_err


def fetch_text(url: str, retries: int = 2, timeout: int = 30) -> str | None:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", "ignore")
        except Exception as err:  # noqa: BLE001
            last_err = err
            if i < retries - 1:
                time.sleep(0.8 + i)
    print(f"[warn] fetch failed: {url} ({last_err})")
    return None


def normalize_ai_tags(raw_tags: Any) -> list[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, list):
        return [str(x).strip() for x in raw_tags if str(x).strip()]
    if isinstance(raw_tags, str):
        return [x.strip() for x in raw_tags.split(",") if x.strip()]
    return []


def to_year(epoch_second: Any) -> int | None:
    if not isinstance(epoch_second, (int, float)) or epoch_second <= 0:
        return None
    return datetime.fromtimestamp(epoch_second, tz=timezone.utc).year


def resolve_start_epoch(meta: dict[str, Any]) -> float | None:
    """Prefer mirror epoch; else parse start_date from atcoder.jp cache."""
    epoch = meta.get("start_epoch_second")
    if isinstance(epoch, (int, float)) and epoch > 0:
        return float(epoch)
    start_date = meta.get("start_date")
    if isinstance(start_date, str) and len(start_date) >= 10:
        try:
            dt = datetime.strptime(start_date[:10], "%Y-%m-%d").replace(
                hour=12, minute=0, second=0, tzinfo=timezone.utc
            )
            return float(dt.timestamp())
        except ValueError:
            pass
    return None


def period_from_start_date(start_date: str | None) -> float | None:
    """Calendar half-year label: H1 -> YYYY.0, H2 -> YYYY.5."""
    if not isinstance(start_date, str) or len(start_date) < 7:
        return None
    try:
        year = int(start_date[:4])
        month = int(start_date[5:7])
    except ValueError:
        return None
    half = 0.0 if month <= 6 else 0.5
    return float(year) + half


def parse_contest_page(html: str) -> dict[str, Any]:
    title_m = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.S)
    title = re.sub(r"<[^>]+>", " ", title_m.group(1)).strip() if title_m else None
    date_m = re.search(r"Contest Duration:.*?(\d{4}-\d{2}-\d{2})", html, flags=re.S)
    start_date = date_m.group(1) if date_m else None
    year = int(start_date[:4]) if start_date else None
    return {"title": title, "start_date": start_date, "year": year}


def load_mirror_index() -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    print("[info] Loading mirror index...")
    spiders_data = fetch_json(SPIDERS_DATA_URL)
    contests: dict[str, dict[str, Any]] = {}
    difficulties: dict[str, float] = {}

    for category, contest_map in spiders_data.items():
        if not isinstance(contest_map, dict):
            continue
        for contest_id, contest in contest_map.items():
            contests[contest_id] = {
                "title": contest.get("title"),
                "year": to_year(contest.get("start_epoch_second")),
                "start_epoch_second": contest.get("start_epoch_second"),
                "category": category,
            }
            problems = contest.get("problems", {})
            if isinstance(problems, dict):
                for problem_id, problem in problems.items():
                    diff = problem.get("difficulty")
                    if diff is not None:
                        difficulties[problem_id] = float(diff)
    return contests, difficulties


def parse_kenkoooo_difficulty(model: Any) -> float | None:
    if not isinstance(model, dict):
        return None
    for key in ("difficulty", "rawDifficulty"):
        value = model.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def download_kenkoooo_problem_models(cache_path: Path) -> None:
    """Fetch problem-models.json via Node (urllib often gets 403)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    node_script = """
const fs = require('fs');
(async () => {
  const url = process.argv[1];
  const out = process.argv[2];
  const res = await fetch(url, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
      'Accept': 'application/json,*/*',
      'Referer': 'https://kenkoooo.com/atcoder/',
    },
  });
  if (!res.ok) {
    console.error('HTTP', res.status);
    process.exit(2);
  }
  fs.writeFileSync(out, Buffer.from(await res.arrayBuffer()));
})();
"""
    print(f"[info] Downloading kenkoooo problem-models.json -> {cache_path}")
    result = subprocess.run(
        ["node", "-e", node_script, KENKOOOO_MODELS_URL, str(cache_path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"kenkoooo download failed: {stderr or result.returncode}")


def load_kenkoooo_difficulty_map(
    cache_path: Path,
    refresh: bool = False,
) -> dict[str, float]:
    if refresh or not cache_path.exists() or cache_path.stat().st_size < 1000:
        download_kenkoooo_problem_models(cache_path)
    else:
        print(f"[info] Using cached kenkoooo models: {cache_path}")

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    difficulties: dict[str, float] = {}
    for problem_id, model in payload.items():
        diff = parse_kenkoooo_difficulty(model)
        if diff is not None:
            difficulties[problem_id] = diff
    print(f"[info] kenkoooo difficulty map: {len(difficulties)} problems")
    return difficulties


def resolve_difficulty(
    problem_id: str,
    kenkoooo_map: dict[str, float],
    mirror_map: dict[str, float],
) -> float | None:
    if problem_id in kenkoooo_map:
        return kenkoooo_map[problem_id]
    return mirror_map.get(problem_id)


def recompute_difficulty_100(df: pd.DataFrame) -> pd.DataFrame:
    difficulty = pd.to_numeric(df["difficulty"], errors="coerce")
    valid = difficulty.notna()
    if valid.any():
        dmin = difficulty[valid].min()
        dmax = difficulty[valid].max()
        if dmax > dmin:
            df["difficulty_100"] = ((difficulty - dmin) / (dmax - dmin) * 100).round(3)
        else:
            df["difficulty_100"] = np.where(valid, 50.0, np.nan)
    else:
        df["difficulty_100"] = np.nan
    return df


def apply_kenkoooo_difficulty(
    df: pd.DataFrame,
    kenkoooo_map: dict[str, float],
    mirror_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    mirror_map = mirror_map or {}
    df["difficulty"] = df["problem_id"].map(
        lambda pid: resolve_difficulty(str(pid), kenkoooo_map, mirror_map)
    )
    filled = int(df["difficulty"].notna().sum())
    print(f"[info] difficulty assigned: {filled}/{len(df)} rows")
    return recompute_difficulty_100(df)


def load_contest_cache(path: Path) -> dict[str, dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_contest_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_contest_meta(
    contest_id: str,
    mirror_contests: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    sleep_sec: float,
) -> dict[str, Any]:
    if contest_id in cache and cache[contest_id].get("title"):
        cached = dict(cache[contest_id])
        epoch = resolve_start_epoch(cached)
        if epoch is not None:
            cached["start_epoch_second"] = epoch
        return cached

    if contest_id in mirror_contests and mirror_contests[contest_id].get("title"):
        meta = {
            "title": mirror_contests[contest_id]["title"],
            "year": mirror_contests[contest_id].get("year"),
            "start_epoch_second": mirror_contests[contest_id].get("start_epoch_second"),
            "source": "mirror",
        }
        epoch = resolve_start_epoch(meta)
        if epoch is not None:
            meta["start_epoch_second"] = epoch
        cache[contest_id] = meta
        return meta

    url = f"https://atcoder.jp/contests/{contest_id}"
    html = fetch_text(url)
    if html:
        parsed = parse_contest_page(html)
        meta = {
            "title": parsed.get("title"),
            "year": parsed.get("year"),
            "start_date": parsed.get("start_date"),
            "source": "atcoder.jp",
        }
        epoch = resolve_start_epoch(meta)
        if epoch is not None:
            meta["start_epoch_second"] = epoch
        cache[contest_id] = meta
        time.sleep(sleep_sec)
        return meta

    cache[contest_id] = {"title": None, "year": None, "source": "failed"}
    return cache[contest_id]


def build_sponsored_abc_dataframe(config: RunConfig, rules: SponsorRules) -> pd.DataFrame:
    print("[info] Fetching AI tagged problems...")
    ai_payload = fetch_json(AI_TAGGED_ALL_URL)
    ai_items = ai_payload.get("message", {}).get("items", [])

    mirror_contests, mirror_difficulty_map = load_mirror_index()
    kenkoooo_cache = config.out_dir / "raw" / KENKOOOO_MODELS_CACHE
    kenkoooo_map = load_kenkoooo_difficulty_map(kenkoooo_cache)
    contest_cache_path = config.out_dir / "raw" / "atcoder_contests.json"
    contest_cache = load_contest_cache(contest_cache_path)

    # Group ABC problems by contest
    by_contest: dict[str, list[dict[str, Any]]] = {}
    for item in ai_items:
        contest_id = str(item.get("contest_id", ""))
        if not contest_id.startswith("abc"):
            continue
        by_contest.setdefault(contest_id, []).append(item)

    print(f"[info] Resolving metadata for {len(by_contest)} ABC contests...")
    sponsored_contests: dict[str, dict[str, Any]] = {}
    for i, contest_id in enumerate(sorted(by_contest.keys())):
        meta = resolve_contest_meta(
            contest_id, mirror_contests, contest_cache, config.sleep_sec
        )
        title = meta.get("title") or ""
        year = meta.get("year")
        sponsor = rules.match(title)
        if sponsor is None:
            continue
        if year is None or year < config.start_year or year > config.end_year:
            continue
        sponsored_contests[contest_id] = {
            **meta,
            "sponsor": sponsor,
            "contest_title": title,
        }
        if (i + 1) % 50 == 0:
            print(f"[info] scanned {i + 1}/{len(by_contest)} contests...")
            save_contest_cache(contest_cache_path, contest_cache)

    save_contest_cache(contest_cache_path, contest_cache)
    print(f"[info] sponsored contests in range: {len(sponsored_contests)}")

    crawl_time = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for contest_id, cmeta in sorted(sponsored_contests.items()):
        for item in by_contest[contest_id]:
            problem_id = item.get("id")
            if not problem_id:
                continue
            ai_tags = normalize_ai_tags(item.get("tags"))
            records.append(
                {
                    "platform": "atcoder",
                    "category": "abc",
                    "assessment_type": "public_sponsored_abc",
                    "sponsor": cmeta["sponsor"],
                    "contest_id": contest_id,
                    "contest_title": cmeta.get("contest_title"),
                    "year": cmeta.get("year"),
                    "start_date": cmeta.get("start_date"),
                    "start_epoch_second": resolve_start_epoch(cmeta),
                    "period": period_from_start_date(cmeta.get("start_date"))
                    or (
                        float(cmeta["year"])
                        if cmeta.get("year") is not None
                        else None
                    ),
                    "problem_id": problem_id,
                    "problem_index": item.get("problem_index"),
                    "problem_name": item.get("name"),
                    "problem_title": item.get("title"),
                    "difficulty": resolve_difficulty(
                        str(problem_id), kenkoooo_map, mirror_difficulty_map
                    ),
                    "solver_count": item.get("solver_count"),
                    "ai_tags": ai_tags,
                    "ai_tag_count": len(ai_tags),
                    "problem_url": f"https://atcoder.jp/contests/{contest_id}/tasks/{problem_id}",
                    "source_url": AI_TAGGED_ALL_URL,
                        "parser_version": PARSER_VERSION,
                        "statement_lang": STATEMENT_LANG,
                        "crawl_time": crawl_time,
                    }
                )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    df = recompute_difficulty_100(df)
    df["clean_text_len"] = np.nan
    df["clean_text"] = None
    df["contest_points"] = np.nan
    df["acceptance_rate"] = np.nan
    return df


def extract_task_statement_html(page_html: str) -> str | None:
    """Extract Japanese (lang-ja) statement only; English section is excluded."""
    ja_before_en = re.search(
        r'<span class="lang-ja">(.*?)</span>\s*<span class="lang-en">',
        page_html,
        flags=re.S,
    )
    if ja_before_en:
        return ja_before_en.group(1)

    ja_only = re.search(r'<span class="lang-ja">(.*?)</span>', page_html, flags=re.S)
    if ja_only:
        return ja_only.group(1)
    return None


def html_to_clean_text(html_fragment: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_fragment)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_contest_points(html_fragment: str) -> float | None:
    """AtCoder contest score allocation (配点) from JA statement HTML."""
    match = CONTEST_POINTS_RE.search(html_fragment)
    if not match:
        return None
    return float(match.group(1))


def backfill_contest_points_from_cache(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    if "contest_points" not in df.columns:
        df["contest_points"] = np.nan

    filled = 0
    for idx, row in df.iterrows():
        if pd.notna(df.at[idx, "contest_points"]):
            continue
        cache_path = raw_dir / f"{row['problem_id']}.html"
        if not cache_path.exists():
            continue
        points = extract_contest_points(
            cache_path.read_text(encoding="utf-8", errors="ignore")
        )
        if points is not None:
            df.at[idx, "contest_points"] = points
            filled += 1

    total = int(df["contest_points"].notna().sum())
    print(
        f"[info] contest_points: {total}/{len(df)} filled "
        f"({filled} newly parsed from cache)"
    )
    return df


def apply_statement_fields(df: pd.DataFrame, idx: int, html_fragment: str) -> None:
    clean_text = html_to_clean_text(html_fragment)
    if clean_text:
        df.at[idx, "clean_text"] = clean_text
        df.at[idx, "clean_text_len"] = len(clean_text)
    points = extract_contest_points(html_fragment)
    if points is not None:
        df.at[idx, "contest_points"] = points


def crawl_and_clean_statements(df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    if config.max_html_pages == 0:
        limit = len(df)
    elif config.max_html_pages < 0 or df.empty:
        return df
    else:
        limit = config.max_html_pages

    print(
        f"[info] Crawling JA statements only (limit={limit}, dir={HTML_RAW_SUBDIR})..."
    )
    raw_dir = config.out_dir / "raw" / HTML_RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    crawl_count = 0
    skipped = 0
    for idx, row in df.iterrows():
        if crawl_count >= limit:
            break

        problem_id = row["problem_id"]
        cache_path = raw_dir / f"{problem_id}.html"
        if cache_path.exists() and cache_path.stat().st_size > 50:
            html_fragment = cache_path.read_text(encoding="utf-8", errors="ignore")
            apply_statement_fields(df, idx, html_fragment)
            skipped += 1
            continue

        url = row["problem_url"]
        html = fetch_text(url)
        if not html:
            continue

        statement_html = extract_task_statement_html(html)
        if not statement_html:
            continue

        apply_statement_fields(df, idx, statement_html)
        if pd.isna(df.at[idx, "clean_text_len"]):
            continue

        cache_path.write_text(statement_html, encoding="utf-8")

        crawl_count += 1
        if crawl_count % 20 == 0:
            print(f"[info] crawled {crawl_count}, cache-hit {skipped}...")
        time.sleep(config.sleep_sec)

    print(f"[info] crawling done: new={crawl_count}, cache-hit={skipped}")
    return df


def backfill_period_metadata_on_csv(
    csv_path: Path,
    cache_path: Path,
) -> pd.DataFrame:
    """Fill start_date / start_epoch_second / period from contest cache."""
    df = pd.read_csv(csv_path)
    cache = load_contest_cache(cache_path)
    if "start_date" not in df.columns:
        df["start_date"] = np.nan
    if "period" not in df.columns:
        df["period"] = np.nan

    for idx, row in df.iterrows():
        cid = str(row["contest_id"])
        meta = cache.get(cid, {})
        if meta.get("start_date") and pd.isna(df.at[idx, "start_date"]):
            df.at[idx, "start_date"] = meta["start_date"]
        epoch = resolve_start_epoch(meta)
        if epoch is not None:
            df.at[idx, "start_epoch_second"] = epoch
        period = period_from_start_date(
            df.at[idx, "start_date"] if pd.notna(df.at[idx, "start_date"]) else meta.get("start_date")
        )
        if period is not None:
            df.at[idx, "period"] = period
        elif pd.notna(row.get("year")):
            df.at[idx, "period"] = float(int(row["year"]))

    df.to_csv(csv_path, index=False)
    print(f"[info] backfilled period metadata: {csv_path}")
    return df


def print_gap_report(df: pd.DataFrame, config: RunConfig) -> None:
    print("\n[gap report]")
    for year in range(config.start_year, config.end_year + 1):
        sub = df[df["year"] == year]
        contests = sub["contest_id"].nunique()
        problems = len(sub)
        crawled = sub["clean_text_len"].notna().sum()
        print(f"  {year}: contests={contests}, problems={problems}, crawled={crawled}")
    print("[sponsor breakdown]")
    print(df.groupby("sponsor")["problem_id"].count().sort_values(ascending=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--max-html-pages",
        type=int,
        default=0,
        help="0 = crawl all filtered problems (incremental)",
    )
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--whitelist",
        default="config/sponsor_whitelist.yaml",
    )
    parser.add_argument(
        "--points-only",
        action="store_true",
        help="Backfill contest_points from cached JA HTML and update cleaned CSV",
    )
    parser.add_argument(
        "--difficulty-only",
        action="store_true",
        help="Refresh kenkoooo difficulty on cleaned CSV (uses cached JSON if present)",
    )
    parser.add_argument(
        "--backfill-metadata-only",
        action="store_true",
        help="Fill start_date/start_epoch/period on existing cleaned CSV from contest cache",
    )
    parser.add_argument(
        "--refresh-kenkoooo",
        action="store_true",
        help="Force re-download kenkoooo problem-models.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    whitelist_path = root / args.whitelist
    config = RunConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        max_html_pages=args.max_html_pages,
        sleep_sec=args.sleep_sec,
        out_dir=root / args.out_dir,
        whitelist_path=whitelist_path,
    )

    rules = SponsorRules.load(config.whitelist_path)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (config.out_dir / "processed").mkdir(parents=True, exist_ok=True)

    atcoder_dir = config.out_dir / "processed" / "atcoder"
    atcoder_dir.mkdir(parents=True, exist_ok=True)
    (atcoder_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (atcoder_dir / "summaries" / "yearly").mkdir(parents=True, exist_ok=True)
    final_path = atcoder_dir / "datasets" / "atcoder_sponsored_abc_2021_2026_cleaned.csv"
    raw_dir = config.out_dir / "raw" / HTML_RAW_SUBDIR
    kenkoooo_cache = config.out_dir / "raw" / KENKOOOO_MODELS_CACHE

    contest_cache_path = config.out_dir / "raw" / "atcoder_contests.json"

    if args.backfill_metadata_only:
        if not final_path.exists():
            print(f"[error] cleaned CSV not found: {final_path}")
            return
        backfill_period_metadata_on_csv(final_path, contest_cache_path)
        return

    if args.difficulty_only:
        if not final_path.exists():
            print(f"[error] cleaned CSV not found: {final_path}")
            return
        _, mirror_difficulty_map = load_mirror_index()
        kenkoooo_map = load_kenkoooo_difficulty_map(
            kenkoooo_cache, refresh=args.refresh_kenkoooo
        )
        df = pd.read_csv(final_path)
        df = apply_kenkoooo_difficulty(df, kenkoooo_map, mirror_difficulty_map)
        df.to_csv(final_path, index=False)
        print(f"[info] cleaned updated: {final_path}")
        _write_yearly_summary(df, config)
        return

    if args.points_only:
        if not final_path.exists():
            print(f"[error] cleaned CSV not found: {final_path}")
            return
        df = pd.read_csv(final_path)
        df = backfill_contest_points_from_cache(df, raw_dir)
        df.to_csv(final_path, index=False)
        print(f"[info] cleaned updated: {final_path}")
        _write_yearly_summary(df, config)
        return

    df = build_sponsored_abc_dataframe(config, rules)
    if df.empty:
        print("[error] No sponsored ABC records found.")
        return

    meta_path = atcoder_dir / "datasets" / "atcoder_sponsored_abc_2021_2026_metadata.csv"
    df.to_csv(meta_path, index=False)
    print(f"[info] metadata saved: {meta_path} ({len(df)} rows)")

    df = crawl_and_clean_statements(df, config)
    df = backfill_contest_points_from_cache(df, raw_dir)

    df.to_csv(final_path, index=False)
    print(f"[info] cleaned saved: {final_path}")
    _write_yearly_summary(df, config)
    print_gap_report(df, config)


def _write_yearly_summary(df: pd.DataFrame, config: RunConfig) -> None:
    summary = (
        df.groupby("year", dropna=True)
        .agg(
            contests=("contest_id", "nunique"),
            problems=("problem_id", "count"),
            avg_contest_points=("contest_points", "mean"),
            avg_difficulty=("difficulty", "mean"),
            avg_ai_tag_count=("ai_tag_count", "mean"),
            crawled_text_count=("clean_text_len", lambda s: int(s.notna().sum())),
            avg_clean_text_len=("clean_text_len", "mean"),
        )
        .reset_index()
    )
    summary_path = (
        config.out_dir
        / "processed"
        / "atcoder"
        / "summaries"
        / "yearly"
        / "atcoder_sponsored_abc_2021_2026_yearly_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"[info] summary saved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
