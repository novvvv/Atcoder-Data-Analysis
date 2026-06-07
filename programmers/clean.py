import pandas as pd
import re

# CSV 불러오기
df = pd.read_csv("programmers_data.csv")
print(f"정제 전 총 {len(df)}개")
print(df.head(3))

print(df["출처"].unique())

# ─────────────────────────────────────
# 1단계. 연도 추출 및 2021~2026 필터링
# ─────────────────────────────────────
# 연도 추출
def extract_year(text):
    if pd.isna(text):
        return None
    match = re.search(r'(202[1-6])', text)  # 2021~2026 숫자 찾기
    return int(match.group(1)) if match else None

df["연도"] = df["출처"].apply(extract_year)

# 연도 없는 행 드롭 (2021~2026 범위 밖)
before = len(df)
df = df.dropna(subset=["연도"])
df["연도"] = df["연도"].astype(int)
print(f"\n[1단계] 연도 필터링: {before}개 → {len(df)}개")

# ─────────────────────────────────────
# 2단계. 완료한 사람 100명 미만 드롭
# ─────────────────────────────────────
# 숫자로 변환
def parse_count(text):
    if pd.isna(text):
        return 0
    return int(re.sub(r'[^0-9]', '', str(text)) or 0)

df["완료한사람_수"] = df["완료한사람"].apply(parse_count)

before = len(df)
df = df[df["완료한사람_수"] >= 100]
print(f"[2단계] 100명 미만 드롭: {before}개 → {len(df)}개")

# ─────────────────────────────────────
# 3단계. 기업 필터링
# ─────────────────────────────────────
target_companies = {
    "카카오": "카카오",
    "KAKAO": "카카오",   
    "현대모비스": "현대모비스",
    "프로그래머스": "프로그래머스"
}

def filter_company(text):
    if pd.isna(text):
        return None
    for keyword, company_name in target_companies.items():
        if keyword in text:
            return company_name  
    return None

df["기업명"] = df["출처"].apply(filter_company)

before = len(df)
df = df.dropna(subset=["기업명"])
print(f"[3단계] 기업 필터링: {before}개 → {len(df)}개")

# ─────────────────────────────────────
# 최종 정리 및 저장
# ─────────────────────────────────────
# 필요한 컬럼만 정리
df_final = df[[
    "연도",
    "기업명",
    "문제명",
    "출처",
    "레벨",
    "완료한사람",
    "정답률",
    "url",
]].sort_values(["연도", "기업명"])

print(f"\n최종 데이터: {len(df_final)}개")
print(df_final.head(10))

df_final.to_csv("programmers_clean.csv", index=False, encoding="utf-8-sig")
print("\nprogrammers_clean.csv 저장 완료!")
