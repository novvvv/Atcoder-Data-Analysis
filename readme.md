# 알고리즘 코딩테스트의 난이도 트렌드 분석

Atcoder 기반 기업 알고리즘 대회 문항 특성 변화 분석 (2021-2025)

고급 프로그래밍 설계 텀 프로젝트 — 최도일(202001836) · 박서연(202202152)

---

## 연구 개요

**문제 정의** — 취업 준비생에게 기업 코딩테스트 난이도 트렌드와 알고리즘 태그 경향을 데이터 기반으로 제시한다.

| RQ | 질문 | 핵심 결과 |
|----|------|-----------|
| RQ1 | 난이도 인플레이션 구간이 존재하는가? | 2021→2023년 평균 난이도 약 310p 하락, 이후 수렴 안정화 |
| RQ2 | 기업 스폰서 대회 태그 경향은? | simulation·grid·BFS 급상승 / DP·sorting·combinatorics 하락 |
| RQ3 | 취준생 학습 우선순위는? | 최우선: implementation·simulation·greedy / 추가: 빠른 지문 독해 훈련 |

---

## 프로젝트 구조

```
Atcoder-Data-Analysis/
├── config/
│   └── sponsor_whitelist.yaml          # 일본 대기업 협찬 대회 필터 (15개 스폰서)
│
├── data/
│   ├── raw/
│   │   └── atcoder_html_ja/            # HTML 크롤링 원본 (atcoder.jp 일본어 지문)
│   └── processed/
│       ├── atcoder/
│       │   ├── datasets/               # 정제 메인 데이터 CSV
│       │   ├── summaries/              # 집계 CSV (yearly / halfyearly / tags / paper_g)
│       │   └── figures/                # 분석 차트 PNG (overview / paper / paper_g / tags)
│       ├── kakao/                      # 카카오 기출 비교 데이터 & 차트
│       └── archive/                    # 구버전 (전체 ABC 기준) 데이터
│
├── notebooks/
│   ├── atcoder_visualize.ipynb         # AtCoder 분석 시각화
│   └── programmers_analysis.ipynb      # Programmers 분석 시각화
│
├── programmers/                        # Programmers 기업 기출 서브 파이프라인
│   ├── scripts/
│   │   ├── crawl.py                    # 1단계: 문제 목록 수집 → data/programmers_data.csv
│   │   ├── clean.py                    # 2단계: 연도·기업 필터링 → data/programmers_clean.csv
│   │   ├── detail.py                   # 3단계: 지문 길이 수집 → programmers_clean.csv 업데이트
│   │   └── analyze.py                  # 4단계: 집계 CSV 3종 생성
│   └── data/
│       ├── programmers_data.csv        # 원본 수집 데이터
│       ├── programmers_clean.csv       # 정제 데이터 (지문 길이 포함)
│       ├── programmers_yearly_summary.csv
│       ├── programmers_company_summary.csv
│       └── programmers_level_distribution.csv
│
├── scripts/
│   ├── collect_atcoder_2021_2026.py    # AtCoder 메인 수집 파이프라인
│   └── visualize_atcoder.py            # 논문용 차트 생성 (C–F 슬롯 기준 7종)
│
└── readme.md
```

---

## 데이터 수집 파이프라인

### AtCoder (`scripts/collect_atcoder_2021_2026.py`)

| 소스 | 수집 내용 | 방법 |
|------|-----------|------|
| AtCoder AI Tagged API | 문제 메타데이터, AI 알고리즘 태그, solver_count | REST API |
| AtCoder.jp | 일본어 지문 (배점, 지문 길이) | HTML 크롤링 (BeautifulSoup) |
| Kenkoooo problem-models.json | 문제별 객관 난이도 추정치 (difficulty) | JSON 다운로드 |

**필터링 기준**
- 대상 스폰서: `config/sponsor_whitelist.yaml` 에 정의된 일본 대기업 15개사 (Toyota, 도쿄해상, Panasonic 등)
- 슬롯 범위: C–F (중~고난도, 기업 코딩테스트 수준)
- 기간: 2021–2025

**최종 분석 대상** — 전체 644문제 (87대회) → C–F 슬롯 **348문항**  
연도별: 2021(40) / 2022(76) / 2023(116) / 2024(72) / 2025(44)

**주요 출력**
```
data/processed/atcoder/datasets/       # 문항 단위 정제 데이터
data/processed/atcoder/summaries/      # 연도별·반기별·태그 집계
data/raw/atcoder_html_ja/              # HTML 원본 (증분 크롤링)
```

### Programmers (`programmers/scripts/`)

카카오·현대모비스·프로그래머스 기업 기출 데이터 수집 (2021–2026, 100명 이상 완료 기준)

```
crawl.py → clean.py → detail.py → analyze.py
```

---

## 시각화 (`scripts/visualize_atcoder.py`)

논문용 차트 7종 (`data/processed/atcoder/figures/paper/`):

| 파일 | 내용 |
|------|------|
| `yearly_difficulty_cf.png` | 연도별 난이도 추세 (IQR 절사평균) |
| `yearly_statement_length_cf.png` | 연도별 지문 길이 추세 |
| `halfyearly_difficulty_cf.png` | 반기별 난이도 추세 |
| `halfyearly_statement_length_cf.png` | 반기별 지문 길이 추세 |
| `top_tags_cf_frequency.png` | 상위 15개 알고리즘 태그 빈도 |
| `tag_yearly_prevalence_cf.png` | 연도×태그 히트맵 |
| `tag_share_delta_cf.png` | 태그 비중 변화 (2021 vs 2025) |

---

## 주요 분석 결과

### 난이도 추세 (RQ1)
- 2021년 평균 ~1,380 → 2023년 ~1,070으로 **약 310p 하락** (난이도 인플레이션 구간)
- 2024–2025년: 1,050–1,090 수렴 → 안정화 단계

### 지문 길이 추세
- 2021년 ~785자 → 2025년 ~1,072자 (**5년간 약 36% 증가**)
- 수치적 난이도↓ + 지문 길이↑ → 독해력의 중요성 증가

### 알고리즘 태그 경향 (RQ2)
| 방향 | 태그 | 변화 |
|------|------|------|
| 상승 | simulation | +15.4%p |
| 상승 | grid | +8.9%p |
| 상승 | bfs | +6.6%p |
| 하락 | dp | -11.4%p |
| 하락 | sorting | -8.9%p |
| 하락 | combinatorics | -8.2%p |

### 취준생 학습 가이드 (RQ3)
| 우선순위 | 태그 | 근거 |
|----------|------|------|
| 최우선 | implementation, simulation, greedy | 빈도 TOP + 상승 트렌드 |
| 우선 | grid, bfs, prefix sum, string | 최근 상승 태그 |
| 유지 | dp, counting, sorting | 절대 빈도 높으나 하락 추세 |
| 축소 가능 | modular arithmetic, combinatorics, graph theory | 빈도 낮고 하락 추세 |

---

## 데이터 출처

- [AtCoder AI Tagged API](https://atcoder.jp) — AI 알고리즘 태그, 문제 메타데이터
- [Kenkoooo AtCoder Problems](https://kenkoooo.com/atcoder) — 난이도 추정치 (problem-models.json)
- [AtCoder.jp](https://atcoder.jp) — 일본어 지문, 배점 데이터
- [Programmers](https://school.programmers.co.kr) — 국내 기업 기출 데이터
