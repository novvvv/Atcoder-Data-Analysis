"""Programmers 기업 코딩테스트 지문 길이 수집 (파이프라인 3단계).

입력 : data/programmers_clean.csv
출력 : data/programmers_clean.csv (지문글자수 컬럼 추가 후 덮어쓰기)
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

# ─── 로드 ────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/programmers_clean.csv")
print(f"총 {len(df)}개 문제 지문 수집 시작...")

text_lengths = []

# ─── 지문 길이 수집 ───────────────────────────────────────────────────────────
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    for i, row in df.iterrows():
        url = row["url"]

        if pd.isna(url) or url == "":
            text_lengths.append(None)
            continue

        try:
            page.goto(url)
            page.wait_for_timeout(2000)

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Programmers 문제 지문 마크다운 렌더링 컨테이너
            content = soup.select_one("div.markdown.solarized-dark")

            if content:
                # BeautifulSoup Tag를 문자열로 변환 후 HTML 태그 일괄 제거
                pure_text = re.sub(r"<[^>]+>", "", str(content)).strip()
                text_lengths.append(len(pure_text))
            else:
                text_lengths.append(None)

            print(f"[{i+1}/{len(df)}] {row['문제명']} -> {text_lengths[-1]}자")

        except Exception as e:
            print(f"[{i+1}/{len(df)}] {row['문제명']} -> 에러: {e}")
            text_lengths.append(None)

        time.sleep(1)  # 서버 부하 방지

    browser.close()

# ─── 저장 ────────────────────────────────────────────────────────────────────
df["지문글자수"] = text_lengths
df.to_csv("../data/programmers_clean.csv", index=False, encoding="utf-8-sig")
print(f"\n완료! programmers_clean.csv 업데이트됨")
