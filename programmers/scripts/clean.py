"""Programmers 기업 코딩테스트 데이터 정제 (파이프라인 2단계).

입력: data/programmers_data.csv
출력: data/programmers_clean.csv

필터링 기준:
  - 연도: 2021-2026
  - 완료 인원: 100명 이상
  - 기업: 카카오, 현대모비스, 프로그래머스
"""

from __future__ import annotations

import re
import pandas as pd


# 출처 키워드 -> 정규화된 기업명
TARGET_COMPANIES = {
    "카카오": "카카오",
    "KAKAO": "카카오",
    "현대모비스": "현대모비스",
    "프로그래머스": "프로그래머스",
}


def extract_year(text: object) -> int | None:
    """출처 문자열에서 2021-2026 연도를 추출한다."""
    if pd.isna(text):
        return None
    match = re.search(r"(202[1-6])", str(text))
    return int(match.group(1)) if match else None


def parse_count(text: object) -> int:
    """'12,583명' 형식의 문자열을 정수로 변환한다."""
    if pd.isna(text):
        return 0
    return int(re.sub(r"[^0-9]", "", str(text)) or 0)


def filter_company(text: object) -> str | None:
    """출처 문자열에서 대상 기업명을 반환하고, 해당 없으면 None을 반환한다."""
    if pd.isna(text):
        return None
    for keyword, company_name in TARGET_COMPANIES.items():
        if keyword in str(text):
            return company_name
    return None


# ─── 로드 ────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/programmers_data.csv")
print(f"정제 전 총 {len(df)}개")
print(df.head(3))
print(df["출처"].unique())

# ─── 1단계: 연도 필터링 ──────────────────────────────────────────────────────
df["연도"] = df["출처"].apply(extract_year)

before = len(df)
df = df.dropna(subset=["연도"])
df["연도"] = df["연도"].astype(int)
print(f"\n[1단계] 연도 필터링: {before}개 -> {len(df)}개")

# ─── 2단계: 완료 인원 100명 미만 제거 ────────────────────────────────────────
df["완료한사람_수"] = df["완료한사람"].apply(parse_count)

before = len(df)
df = df[df["완료한사람_수"] >= 100]
print(f"[2단계] 100명 미만 드롭: {before}개 -> {len(df)}개")

# ─── 3단계: 기업 필터링 ──────────────────────────────────────────────────────
df["기업명"] = df["출처"].apply(filter_company)

before = len(df)
df = df.dropna(subset=["기업명"])
print(f"[3단계] 기업 필터링: {before}개 -> {len(df)}개")

# ─── 저장 ────────────────────────────────────────────────────────────────────
df_final = df[[
    "연도", "기업명", "문제명", "출처",
    "레벨", "완료한사람", "정답률", "url",
]].sort_values(["연도", "기업명"])

print(f"\n최종 데이터: {len(df_final)}개")
print(df_final.head(10))

df_final.to_csv("../data/programmers_clean.csv", index=False, encoding="utf-8-sig")
print("\nprogrammers_clean.csv 저장 완료!")
