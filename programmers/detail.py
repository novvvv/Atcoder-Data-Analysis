from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

# 기존 CSV 불러오기
df = pd.read_csv("programmers_clean.csv")
print(f"총 {len(df)}개 문제 지문 수집 시작...")

text_lengths = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    
    for i, row in df.iterrows():
        url = row["url"]
        
        # URL 없으면 스킵
        if pd.isna(url) or url == "":
            text_lengths.append(None)
            continue
        
        try:
            page.goto(url)
            page.wait_for_timeout(2000)
            
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
            
            # 지문 영역 찾기
            content = soup.select_one("div.markdown.solarized-dark")
            
            if content:
                # HTML 태그 전부 제거 후 순수 글자 수 카운트
                pure_text = re.sub(r'<[^>]+>', '', str(content))
                # 공백/줄바꿈 정리
                pure_text = pure_text.strip()
                text_lengths.append(len(pure_text))
            else:
                text_lengths.append(None)
            
            # 진행상황 출력
            print(f"[{i+1}/{len(df)}] {row['문제명']} → {text_lengths[-1]}자")
        
        except Exception as e:
            print(f"[{i+1}/{len(df)}] {row['문제명']} → 에러: {e}")
            text_lengths.append(None)
        
        time.sleep(1)  # 서버 부하 방지
    
    browser.close()

# 지문 글자수 컬럼 추가 후 저장
df["지문글자수"] = text_lengths
df.to_csv("programmers_clean.csv", index=False, encoding="utf-8-sig")
print(f"\n완료! programmers_clean.csv 업데이트됨")