from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import time

results = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    
    page_num = 1
    prev_titles = None  # 이전 페이지 문제명 저장용

    while True:
        url = f"https://school.programmers.co.kr/learn/challenges?order=recent&page={page_num}"
        print(f"{page_num}페이지 수집 중...")
        
        page.goto(url)
        page.wait_for_timeout(2000)
        
        html = page.content()
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("tbody tr")
        
        if not rows:
            print("마지막 페이지 도달!")
            break
        
        # 현재 페이지 문제명 추출
        current_titles = [row.select_one("td.title a").text.strip() 
                            for row in rows if row.select_one("td.title a")]
        
        # 이전 페이지랑 같으면 종료
        if current_titles == prev_titles:
            print("마지막 페이지 도달!")
            break
        
        prev_titles = current_titles  # 현재 페이지 저장
        
        for row in rows:
            title_td = row.select_one("td.title a")
            title = title_td.text.strip() if title_td else ""
            href = title_td.get("href", "")
            url = f"https://school.programmers.co.kr{href}" if href else ""
            
            part_title = row.select_one("small.part-title")
            source = part_title.text.strip() if part_title else ""
            
            level_td = row.select_one("td.level")
            level = level_td.text.strip() if level_td else ""
            
            finished_td = row.select_one("td.finished-count")
            finished = finished_td.text.strip() if finished_td else ""
            
            rate_td = row.select_one("td.acceptance-rate")
            rate = rate_td.text.strip() if rate_td else ""
            
            if not title:
                continue
            
            results.append({
                "문제명": title,
                "출처": source,
                "레벨": level,
                "완료한사람": finished,
                "정답률": rate,
                "url": url,
            })
        
        print(f"   → 누적 {len(results)}개")
        page_num += 1
        time.sleep(1)
    
    browser.close()

df = pd.DataFrame(results)
print(df.head(10))
df.to_csv("programmers_data.csv", index=False, encoding="utf-8-sig")
print(f"\n총 {len(df)}개 수집 완료!")