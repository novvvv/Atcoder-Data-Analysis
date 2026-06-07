"""Programmers 기업 코딩테스트 통계 집계 (파이프라인 3단계).

파이프라인 구성:
  1단계 수집  : crawl.py + detail.py
  2단계 정제  : clean.py
  3단계 집계  : analyze.py  ← 현재 파일

입력: programmers_clean.csv
출력:
  - programmers_yearly_summary.csv       연도별 집계
  - programmers_company_summary.csv      연도×기업 교차 집계
  - programmers_level_distribution.csv  연도별 레벨 분포
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


# ─── 파싱 헬퍼 ─────────────────────────────────────────────────────────────

def parse_level(text: object) -> int | None:
    """'Lv. 3' → 3"""
    if pd.isna(text):
        return None
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else None


def parse_rate(text: object) -> float | None:
    """'42%' → 42.0"""
    if pd.isna(text):
        return None
    m = re.search(r"(\d+)", str(text))
    return float(m.group(1)) if m else None


def parse_count(text: object) -> int:
    """'12,583명' → 12583"""
    if pd.isna(text):
        return 0
    return int(re.sub(r"[^0-9]", "", str(text)) or 0)


def iqr_trimmed_mean(series: pd.Series) -> float:
    """IQR 기준으로 극단값을 제외한 평균.

    상·하위 25% 밖 값을 제거하여 소수 문제의 이상치가
    연도별 추세를 왜곡하지 않도록 한다.
    """
    s = series.dropna()
    if s.empty:
        return np.nan
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    trimmed = s[(s >= q1) & (s <= q3)]
    return float(trimmed.mean()) if not trimmed.empty else np.nan


# ─── 집계 함수 ──────────────────────────────────────────────────────────────

def aggregate_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 평균 지문 길이·레벨·정답률·완료 인원 집계."""
    yearly = (
        df.groupby("연도", as_index=False)
        .agg(
            문제수=("문제명", "count"),
            평균지문길이=("지문글자수", iqr_trimmed_mean),
            평균레벨=("레벨_수", "mean"),
            평균정답률=("정답률_수", "mean"),
            평균완료인원=("완료한사람_수", "mean"),
        )
        .sort_values("연도")
    )
    yearly["평균지문길이"] = yearly["평균지문길이"].round(1)
    yearly["평균레벨"] = yearly["평균레벨"].round(2)
    yearly["평균정답률"] = yearly["평균정답률"].round(1)
    yearly["평균완료인원"] = yearly["평균완료인원"].round(0).astype("Int64")
    return yearly


def aggregate_company_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """연도 × 기업 교차 집계."""
    result = (
        df.groupby(["연도", "기업명"], as_index=False)
        .agg(
            문제수=("문제명", "count"),
            평균지문길이=("지문글자수", iqr_trimmed_mean),
            평균레벨=("레벨_수", "mean"),
            평균정답률=("정답률_수", "mean"),
        )
        .sort_values(["연도", "기업명"])
    )
    result["평균지문길이"] = result["평균지문길이"].round(1)
    result["평균레벨"] = result["평균레벨"].round(2)
    result["평균정답률"] = result["평균정답률"].round(1)
    return result


def aggregate_level_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 레벨 분포 (문제 수)."""
    return (
        df.groupby(["연도", "레벨_수"])["문제명"]
        .count()
        .rename("문제수")
        .reset_index()
        .sort_values(["연도", "레벨_수"])
    )


# ─── 메인 ──────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv("programmers_clean.csv")
    print(f"[info] 로드 완료: {len(df)}개 문제")

    # 수치 변환
    df["레벨_수"] = df["레벨"].apply(parse_level)
    df["정답률_수"] = df["정답률"].apply(parse_rate)
    df["완료한사람_수"] = df["완료한사람"].apply(parse_count)
    df["지문글자수"] = pd.to_numeric(df["지문글자수"], errors="coerce")

    # ── 1. 연도별 집계 ──────────────────────────────────────────────────────
    yearly = aggregate_yearly(df)
    print("\n=== 연도별 집계 ===")
    print(yearly.to_string(index=False))
    yearly.to_csv("programmers_yearly_summary.csv", index=False, encoding="utf-8-sig")

    # ── 2. 연도 × 기업 교차 집계 ────────────────────────────────────────────
    company_yearly = aggregate_company_yearly(df)
    print("\n=== 연도 × 기업 집계 ===")
    print(company_yearly.to_string(index=False))
    company_yearly.to_csv("programmers_company_summary.csv", index=False, encoding="utf-8-sig")

    # ── 3. 연도별 레벨 분포 ─────────────────────────────────────────────────
    level_dist = aggregate_level_distribution(df)
    print("\n=== 연도별 레벨 분포 ===")
    print(level_dist.to_string(index=False))
    level_dist.to_csv("programmers_level_distribution.csv", index=False, encoding="utf-8-sig")

    print("\n[완료]")
    print("  → programmers_yearly_summary.csv")
    print("  → programmers_company_summary.csv")
    print("  → programmers_level_distribution.csv")


if __name__ == "__main__":
    main()
