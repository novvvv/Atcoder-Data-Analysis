# AtCoder (협찬 ABC 2021–2025)

## `datasets/` — 원본·정제 테이블

| 파일 | 설명 |
|------|------|
| `atcoder_sponsored_abc_2021_2026_cleaned.csv` | 문항 단위 메인 데이터 |
| `atcoder_sponsored_abc_2021_2026_metadata.csv` | 메타만 (지문 전) |
| `atcoder_sponsored_abc_2021_2026_with_period.csv` | `period` 포함 (2025.0 / 2025.5 등) |

## `summaries/` — 집계 CSV

| 폴더 | 내용 |
|------|------|
| `yearly/` | 연도별 요약, 지문·난이도·배점 |
| `halfyearly/` | 반기 요약 (`halfyearly_trends_summary.csv`) |
| `tags/` | AI 태그 빈도·연도별 점유·추세 |

## `figures/` — PNG

| 폴더 | 내용 |
|------|------|
| `overview/` | `yearly_trends.png`, `halfyearly_trends.png` |
| `metrics/` | 지문 길이·난이도·배점 (연도/반기, C–F) |
| `tags/` | 태그 히트맵·추세·facets 등 |
