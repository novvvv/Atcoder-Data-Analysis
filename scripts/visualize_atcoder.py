#!/usr/bin/env python3
"""Generate only paper-facing AtCoder C-F figures.

Paper outputs (7 pngs):
  - yearly_statement_length_cf.png
  - yearly_difficulty_cf.png
  - halfyearly_statement_length_cf.png
  - halfyearly_difficulty_cf.png
  - top_tags_cf_frequency.png
  - tag_yearly_prevalence_cf.png
  - tag_share_delta_cf.png
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CF_SLOTS = {"C", "D", "E", "F"}
CFG_SLOTS = {"C", "D", "E", "F", "G"}
COLOR_TEXT = "#111111"
COLOR_LINE = "#111111"
COLOR_GRID = "#d9d9d9"
COLOR_BAR = "#1f1f1f"
COLOR_BAR_LIGHT = "#8c8c8c"
COLOR_BG = "#ffffff"
COLOR_HEATMAP_TEXT_LIGHT = "#f5f5f5"
COLOR_HEATMAP_TEXT_DARK = "#222222"


def setup_minimal_monotone_style() -> None:
    """Apply a clean black/gray monochrome style globally."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_BG,
            "axes.edgecolor": COLOR_TEXT,
            "axes.labelcolor": COLOR_TEXT,
            "axes.titlecolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "text.color": COLOR_TEXT,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def style_axis_minimal(ax: plt.Axes, *, y_grid: bool = True) -> None:
    """Give each axis a minimal paper-like style."""
    ax.spines["left"].set_color(COLOR_TEXT)
    ax.spines["bottom"].set_color(COLOR_TEXT)
    if y_grid:
        ax.grid(axis="y")


def parse_ai_tags(value: object) -> list[str]:
    """ai_tags 컬럼 값을 '항상 리스트' 형태로 통일한다.

    CSV에서는 ai_tags가 여러 형태로 들어올 수 있다.
    - 이미 리스트인 경우: ["dp", "graph"]
    - 문자열로 저장된 경우: "['dp', 'graph']"
    - 비어 있거나 NaN인 경우

    이후 태그 집계를 쉽게 하려고 어떤 입력이 와도
    최종적으로 ["tag1", "tag2", ...] 형태를 반환한다.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass

    text = text.strip("[]")
    return [x.strip().strip("'").strip('"') for x in text.split(",") if x.strip()]


def iqr_trimmed_mean(values: pd.Series) -> float:
    """IQR(사분위 범위) 기준으로 극단값을 제외한 평균.

    쉽게 말해:
    - 너무 작은 값(하위 25% 밖)
    - 너무 큰 값(상위 25% 밖)
    을 빼고 가운데 구간만 평균낸다.

    이유:
    특정 문제 1~2개가 지나치게 길거나 짧아도
    전체 연도 추세가 과도하게 흔들리지 않게 하기 위해서다.
    """
    series = values.dropna()
    if series.empty:
        return np.nan
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    trimmed = series[(series >= q1) & (series <= q3)]
    if trimmed.empty:
        return np.nan
    return float(trimmed.mean())


def period_from_start_date(start_date: object) -> float | None:
    """대회 시작일을 반기 표기(YYYY.0 / YYYY.5)로 변환.

    예:
    - 2024-03-10 -> 2024.0 (상반기)
    - 2024-09-01 -> 2024.5 (하반기)

    반기 그래프 x축을 만들기 위한 보조 함수다.
    """
    if start_date is None or (isinstance(start_date, float) and pd.isna(start_date)):
        return None
    text = str(start_date).strip()
    if len(text) < 7:
        return None
    try:
        year = int(text[:4])
        month = int(text[5:7])
    except ValueError:
        return None
    return float(year) + (0.0 if month <= 6 else 0.5)


def attach_period_columns(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """df에 period(반기) 컬럼이 없거나 비어 있을 때 채워 넣는다.

    동작 순서:
    1) contest 캐시(json)에서 start_date를 찾는다.
    2) start_date -> period(YYYY.0/YYYY.5) 변환.
    3) 그래도 비어 있으면 year를 float로 넣어 최소한 값은 보장.

    목적:
    halfyearly_* 그래프를 그리려면 period가 필요하기 때문.
    """
    work = df.copy()
    if "period" in work.columns:
        work["period"] = pd.to_numeric(work["period"], errors="coerce")
    else:
        work["period"] = np.nan

    if "start_date" not in work.columns:
        work["start_date"] = np.nan

    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for idx, row in work.iterrows():
            cid = str(row["contest_id"])
            meta = cache.get(cid, {})
            if pd.isna(work.at[idx, "start_date"]) and meta.get("start_date"):
                work.at[idx, "start_date"] = meta["start_date"]
            if pd.isna(work.at[idx, "period"]):
                period = period_from_start_date(meta.get("start_date"))
                if period is not None:
                    work.at[idx, "period"] = period

    missing = work["period"].isna()
    if missing.any():
        work.loc[missing, "period"] = (
            pd.to_numeric(work.loc[missing, "year"], errors="coerce").astype(float)
        )
    return work


def filter_by_slots(df: pd.DataFrame, slots: set[str]) -> pd.DataFrame:
    """문제 슬롯(A~F 중 문자) 기준으로 필요한 행만 필터링.

    여기서는 CF_SLOTS={"C","D","E","F"}를 넣어서
    논문 범위(C~F) 문제만 남긴다.
    """
    if "problem_index" not in df.columns:
        raise ValueError("problem_index column is required for slot filtering")
    slot = df["problem_index"].astype(str).str.upper()
    return df[slot.isin(slots)].copy()


def _save_yearly_metric_chart(
    yearly: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
    *,
    show_value_labels: bool = False,
    show_trend_bounds: bool = False,
    show_trend_line: bool = False,
    compact_labels: bool = False,
    show_n: bool = True,
    label_y_offsets: list[float] | None = None,
) -> None:
    """연도별 지표 선그래프를 그리는 공통 템플릿.

    이 함수 하나로
    - yearly_statement_length_cf
    - yearly_difficulty_cf
    두 그림을 같은 스타일로 만든다.

    점 위의 n=... 표시는 해당 연도 문제 수(표본 크기)다.
    show_trend_bounds=True이면 연도별 평균값 중 최저·최고 수준을 점선으로 표시한다.
    """
    if yearly.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    xmin = yearly["year"].min()
    xmax = yearly["year"].max()
    if show_trend_bounds:
        avg_low = float(yearly[value_col].min())
        avg_high = float(yearly[value_col].max())
        ax.axhline(avg_low, color=COLOR_BAR_LIGHT, linestyle="--", linewidth=1.2, zorder=1)
        ax.axhline(avg_high, color=COLOR_BAR_LIGHT, linestyle="--", linewidth=1.2, zorder=1)
    if show_trend_line and len(yearly) >= 2:
        coeffs = np.polyfit(yearly["year"], yearly[value_col], 1)
        trend_years = np.linspace(xmin, xmax, 50)
        trend_vals = np.polyval(coeffs, trend_years)
        ax.plot(
            trend_years,
            trend_vals,
            linestyle="--",
            color=COLOR_BAR_LIGHT,
            linewidth=1.6,
            zorder=2,
        )
    ax.plot(
        yearly["year"],
        yearly[value_col],
        marker="o",
        linewidth=2.2,
        color=COLOR_LINE,
        markerfacecolor=COLOR_BG,
        markeredgecolor=COLOR_LINE,
        markeredgewidth=1.6,
        zorder=3,
    )
    compact_offsets = [(0, 7), (0, 7), (0, 7), (0, 7), (0, 7)]
    for i, (_, row) in enumerate(yearly.iterrows()):
        if show_value_labels and label_y_offsets is not None:
            dy = label_y_offsets[min(i, len(label_y_offsets) - 1)]
            label_y = row[value_col] + dy
            ax.text(
                row["year"],
                label_y,
                f"{row[value_col]:.0f}",
                ha="center",
                va="bottom" if dy >= 0 else "top",
                fontsize=8,
                color=COLOR_TEXT,
                zorder=4,
            )
            continue
        if compact_labels and show_value_labels:
            dx, dy = compact_offsets[min(i, len(compact_offsets) - 1)]
            ax.annotate(
                f"{row[value_col]:.0f}",
                (row["year"], row[value_col]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="center",
                fontsize=7,
                color="#555555",
            )
            continue
        if show_n:
            ax.annotate(
                f"n={int(row['problems'])}",
                (row["year"], row[value_col]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color=COLOR_TEXT,
            )
        if show_value_labels:
            ax.annotate(
                f"{row[value_col]:.0f}",
                (row["year"], row[value_col]),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color=COLOR_TEXT,
            )
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(yearly["year"])
    if show_value_labels and label_y_offsets is not None:
        label_ys = [
            row[value_col] + label_y_offsets[min(i, len(label_y_offsets) - 1)]
            for i, (_, row) in enumerate(yearly.iterrows())
        ]
        ymin = min(float(yearly[value_col].min()), min(label_ys)) - 25
        ymax = max(float(yearly[value_col].max()), max(label_ys)) + 20
        ax.set_ylim(ymin, ymax)
    if show_trend_bounds and compact_labels:
        ax.set_xlim(xmin - 0.2, xmax + 0.15)
    elif show_trend_bounds:
        ax.set_xlim(xmin - 0.3, xmax + 0.55)
    style_axis_minimal(ax, y_grid=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[info] saved: {output_path}")


def parse_slots_arg(slots_text: str) -> set[str]:
    """--slots 인자를 슬롯 집합으로 변환한다. 예: 'C,D,E,F,G'"""
    slots = {part.strip().upper() for part in slots_text.split(",") if part.strip()}
    if not slots:
        raise ValueError("At least one slot must be provided.")
    return set(slots)


def slot_file_suffix(slots: set[str]) -> str:
    """출력 파일 접미사를 슬롯 범위에 맞게 결정한다."""
    if slots == CF_SLOTS:
        return "cf"
    if slots == CFG_SLOTS:
        return "cfg"
    ordered = "".join(slot for slot in ["C", "D", "E", "F", "G"] if slot in slots).lower()
    return ordered or "all"


def export_paper_summaries(
    df_slots: pd.DataFrame,
    prevalence: pd.DataFrame,
    trend_summary: pd.DataFrame,
    summaries_dir: Path,
    file_suffix: str,
) -> None:
    """paper/paper_g용 집계 CSV를 summaries/<subdir>/에 저장한다."""
    summaries_dir.mkdir(parents=True, exist_ok=True)

    yearly_stmt = (
        df_slots.groupby("year", as_index=False)
        .agg(
            problems=("problem_id", "count"),
            avg_clean_text_len=("clean_text_len", iqr_trimmed_mean),
        )
        .sort_values("year")
    )
    yearly_stmt.to_csv(
        summaries_dir / f"yearly_statement_length_{file_suffix}.csv",
        index=False,
    )

    yearly_diff = (
        df_slots.groupby("year", as_index=False)
        .agg(
            problems=("problem_id", "count"),
            avg_difficulty=("difficulty", iqr_trimmed_mean),
        )
        .sort_values("year")
    )
    yearly_diff.to_csv(
        summaries_dir / f"yearly_difficulty_{file_suffix}.csv",
        index=False,
    )

    sponsor = (
        df_slots.groupby("sponsor", as_index=False)
        .agg(
            contests=("contest_id", "nunique"),
            problems=("problem_id", "nunique"),
        )
        .sort_values("problems", ascending=False)
    )
    sponsor["problem_pct"] = (sponsor["problems"] / sponsor["problems"].sum() * 100).round(1)
    sponsor["contest_pct"] = (sponsor["contests"] / sponsor["contests"].sum() * 100).round(1)
    sponsor.to_csv(summaries_dir / f"sponsor_breakdown_{file_suffix}.csv", index=False)

    slot = (
        df_slots.groupby("problem_index", as_index=False)
        .agg(problems=("problem_id", "nunique"))
        .sort_values("problem_index")
    )
    slot["pct"] = (slot["problems"] / slot["problems"].sum() * 100).round(1)
    slot.to_csv(summaries_dir / f"slot_breakdown_{file_suffix}.csv", index=False)

    if not prevalence.empty:
        prevalence.to_csv(
            summaries_dir / f"tag_yearly_prevalence_{file_suffix}.csv",
            index=False,
        )
    if not trend_summary.empty:
        trend_summary.to_csv(
            summaries_dir / f"tag_trend_summary_{file_suffix}.csv",
            index=False,
        )

    print(f"[info] saved summaries to: {summaries_dir}")


def plot_yearly_statement_length_cf(
    df_cf: pd.DataFrame, output_dir: Path, *, file_suffix: str = "cf"
) -> None:
    """C~F 문제의 연도별 지문 길이 추세를 저장한다.

    출력:
    - yearly_statement_length_cf.png
    """
    yearly = (
        df_cf.groupby("year", as_index=False)
        .agg(
            problems=("problem_id", "count"),
            avg_clean_text_len=("clean_text_len", iqr_trimmed_mean),
        )
        .sort_values("year")
    )
    _save_yearly_metric_chart(
        yearly,
        "avg_clean_text_len",
        output_dir / f"yearly_statement_length_{file_suffix}.png",
        "Statement Length",
        "Characters (JA)",
        show_n=False,
        show_value_labels=True,
        compact_labels=True,
    )


def plot_yearly_difficulty_cf(
    df_cf: pd.DataFrame, output_dir: Path, *, file_suffix: str = "cf"
) -> None:
    """C~F 문제의 연도별 난이도 추세를 저장한다.

    출력:
    - yearly_difficulty_cf.png
    """
    yearly = (
        df_cf.groupby("year", as_index=False)
        .agg(
            problems=("problem_id", "count"),
            avg_difficulty=("difficulty", iqr_trimmed_mean),
        )
        .sort_values("year")
    )
    _save_yearly_metric_chart(
        yearly,
        "avg_difficulty",
        output_dir / f"yearly_difficulty_{file_suffix}.png",
        "Kenkoooo Difficulty",
        "Difficulty",
        show_n=False,
        show_value_labels=True,
        compact_labels=True,
    )


def _plot_period_axis(ax: plt.Axes, periods: pd.Series) -> None:
    """반기 축 라벨(2021.0, 2021.5, ...)을 보기 좋게 배치."""
    labels = [f"{p:.1f}" for p in periods]
    xs = np.arange(len(periods))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")


def plot_halfyear_metric_cf(
    df_cf: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """C~F 반기 지표 그래프를 그리는 공통 함수.

    value_col에 따라 2개의 paper 그림을 만든다.
    - clean_text_len -> halfyearly_statement_length_cf.png
    - difficulty     -> halfyearly_difficulty_cf.png
    """
    half = (
        df_cf.groupby("period", as_index=False)
        .agg(problems=("problem_id", "count"), metric=(value_col, iqr_trimmed_mean))
        .sort_values("period")
    )
    if half.empty:
        return
    xs = np.arange(len(half))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        xs,
        half["metric"],
        marker="o",
        linewidth=2.2,
        color=COLOR_LINE,
        markerfacecolor=COLOR_BG,
        markeredgecolor=COLOR_LINE,
        markeredgewidth=1.6,
    )
    for i, (_, row) in enumerate(half.iterrows()):
        ax.annotate(
            f"n={int(row['problems'])}",
            (i, row["metric"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color=COLOR_TEXT,
        )
    ax.set_title(title)
    _plot_period_axis(ax, half["period"])
    ax.set_ylabel(ylabel)
    style_axis_minimal(ax, y_grid=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[info] saved: {output_path}")


def _normalize_tag(tag: str) -> str:
    """태그 문자열 정리: 소문자화 + 공백 정규화."""
    tag = tag.lower().strip()
    tag = re.sub(r"\s+", " ", tag)
    return tag


def prepare_tag_df(df: pd.DataFrame) -> pd.DataFrame:
    """문제 단위 ai_tags를 '태그 1행 1개' 구조로 펼친다.

    예를 들어 한 문제 태그가 ['dp','graph']면
    같은 problem_id가 2행으로 늘어난다.

    이유:
    태그 빈도/점유율 계산(groupby)이 쉬워진다.
    """
    work = df.copy()
    work["ai_tags_parsed"] = work["ai_tags"].apply(parse_ai_tags)
    tag_df = work.explode("ai_tags_parsed")
    tag_df = tag_df.dropna(subset=["ai_tags_parsed"])
    tag_df["tag"] = tag_df["ai_tags_parsed"].astype(str).map(_normalize_tag)
    tag_df = tag_df[tag_df["tag"] != ""]
    return tag_df


def plot_top_tags_cf_frequency(
    tag_df_cf: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    *,
    file_suffix: str = "cf",
) -> None:
    """C~F 상위 태그 빈도 막대그래프.

    출력:
    - top_tags_cf_frequency.png
    """
    top = (
        tag_df_cf.groupby("tag", as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(top_n)
        .sort_values("size", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["tag"], top["size"], color=COLOR_BAR, edgecolor=COLOR_TEXT, linewidth=0.8)
    ax.set_title(f"Top {top_n} AI Tags (Frequency)")
    ax.set_xlabel("Count")
    ax.set_ylabel("Tag")
    style_axis_minimal(ax, y_grid=True)
    fig.tight_layout()
    fig_path = output_dir / f"top_tags_{file_suffix}_frequency.png"
    fig.savefig(fig_path, dpi=180, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[info] saved: {fig_path}")


def build_tag_yearly_prevalence_cf(
    df_cf: pd.DataFrame, tag_df_cf: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    """연도별 태그 점유율 테이블 생성.

    share_pct 계산식:
    (해당 연도에서 해당 태그가 붙은 문제 수 / 해당 연도 C~F 전체 문제 수) * 100

    그리고 전체 기간 누적 빈도 기준으로 상위 top_n 태그만 남긴다.
    """
    per_problem = (
        tag_df_cf.groupby(["problem_id", "year"], as_index=False)["tag"]
        .agg(lambda tags: sorted(set(tags)))
        .explode("tag")
    )
    problems_per_year = (
        df_cf.groupby("year")["problem_id"].nunique().rename("problems_in_year")
    )
    yearly = (
        per_problem.groupby(["tag", "year"], as_index=False)
        .agg(problem_count=("problem_id", "nunique"))
        .sort_values(["tag", "year"])
    )
    yearly = yearly.merge(problems_per_year.reset_index(), on="year", how="left")
    yearly["share_pct"] = (
        yearly["problem_count"] / yearly["problems_in_year"] * 100
    ).round(2)

    top_tags = (
        yearly.groupby("tag")["problem_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    return yearly[yearly["tag"].isin(top_tags)].copy()


def build_tag_trend_summary(prevalence: pd.DataFrame) -> pd.DataFrame:
    """태그별 시작연도 대비 마지막연도 변화량(delta_pp) 요약.

    delta_pp = 마지막연도 점유율 - 시작연도 점유율
    단, 표본이 너무 작은 태그(총 10문제 미만)는 제외한다.
    """
    rows: list[dict[str, object]] = []
    for tag, group in prevalence.groupby("tag"):
        g = group.sort_values("year")
        total = int(g["problem_count"].sum())
        if total < 10:
            continue
        y0 = int(g["year"].min())
        y1 = int(g["year"].max())
        s0 = float(g.loc[g["year"] == y0, "share_pct"].iloc[0])
        s1 = float(g.loc[g["year"] == y1, "share_pct"].iloc[0])
        rows.append({"tag": tag, "delta_pp": round(s1 - s0, 2)})
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values("delta_pp", ascending=False).reset_index(drop=True)


def plot_tag_yearly_prevalence_cf(
    prevalence: pd.DataFrame, output_dir: Path, *, file_suffix: str = "cf"
) -> None:
    """태그-연도 히트맵 생성.

    출력:
    - tag_yearly_prevalence_cf.png

    읽는 법:
    - 행: 태그
    - 열: 연도
    - 색/숫자: 해당 연도 점유율(%)
    """
    pivot = prevalence.pivot(index="tag", columns="year", values="share_pct")
    tag_order = (
        prevalence.groupby("tag")["problem_count"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    pivot = pivot.loc[tag_order]
    n_rows = len(pivot)
    n_cols = len(pivot.columns)
    fig, ax = plt.subplots(figsize=(9.0, max(7.5, 0.52 * n_rows)))
    vmin = 0.0
    vmax = float(np.nanmax(pivot.values)) if np.isfinite(np.nanmax(pivot.values)) else 1.0
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(int(y)) for y in pivot.columns])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=8, length=0)
    ax.set_xlabel("Year", labelpad=10)
    ax.set_ylabel("")
    ax.set_title(
        f"AI tag prevalence by year ({int(prevalence['year'].min())}–{int(prevalence['year'].max())})",
        pad=14,
    )
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=COLOR_BG, linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, row_vals in enumerate(pivot.values):
        for j, val in enumerate(row_vals):
            if pd.notna(val):
                # Dark cell -> bright text, light cell -> dark text.
                ratio = (val - vmin) / (vmax - vmin + 1e-9)
                txt_color = COLOR_HEATMAP_TEXT_LIGHT if ratio > 0.5 else COLOR_HEATMAP_TEXT_DARK
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, fraction=0.045)
    cbar.set_label("Share of problems (%)", rotation=0, labelpad=8)
    cbar.ax.tick_params(labelsize=8, pad=4)
    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.16, top=0.90)
    ax.text(-0.08, 1.03, "Tag", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=COLOR_TEXT)
    fig_path = output_dir / f"tag_yearly_prevalence_{file_suffix}.png"
    fig.savefig(fig_path, dpi=180, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[info] saved: {fig_path}")


def plot_tag_share_delta_cf(
    trend_summary: pd.DataFrame, output_dir: Path, *, file_suffix: str = "cf"
) -> None:
    """태그 점유율 증가/감소 막대그래프 생성.

    출력:
    - tag_share_delta_cf.png

    의미:
    - 빨강(+) : 기간 동안 점유율 증가
    - 파랑(-) : 기간 동안 점유율 감소
    """
    if trend_summary.empty:
        print("[warn] empty tag trend summary; skipped tag_share_delta_cf")
        return
    top_k = min(12, len(trend_summary))
    rising = trend_summary.head(top_k // 2)
    falling = trend_summary.tail(top_k // 2).sort_values("delta_pp")
    plot_df = pd.concat([rising, falling], ignore_index=True).sort_values("delta_pp")
    colors = [COLOR_BAR if v >= 0 else COLOR_BAR_LIGHT for v in plot_df["delta_pp"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["tag"], plot_df["delta_pp"], color=colors, edgecolor=COLOR_TEXT, linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in problem share (percentage points)")
    ax.set_ylabel("AI tag")
    ax.set_title("Tag share change")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        x = row["delta_pp"]
        ax.text(
            x + (0.4 if x >= 0 else -0.4),
            i,
            f"{x:+.1f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
            color=COLOR_TEXT,
        )
    style_axis_minimal(ax, y_grid=True)
    fig.tight_layout()
    fig_path = output_dir / f"tag_share_delta_{file_suffix}.png"
    fig.savefig(fig_path, dpi=180, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[info] saved: {fig_path}")


def main() -> None:
    """paper용 7개 그림 생성 파이프라인의 시작점.

    큰 흐름:
    1) 입력 CSV 로드
    2) period 보정 + 연도 필터
    3) C~F 문제만 추출
    4) 지문/난이도 yearly + halfyearly 그래프 생성
    5) 태그 빈도/히트맵/변화량 그래프 생성
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="data/processed/atcoder/datasets/atcoder_sponsored_abc_2021_2026_cleaned.csv",
        help="Path to cleaned AtCoder CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/atcoder/figures",
        help="Figure root; paper figures are written to <output-dir>/paper",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Top N tags to plot")
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument(
        "--slots",
        default="C,D,E,F",
        help="Comma-separated problem slots to include (e.g. C,D,E,F,G)",
    )
    parser.add_argument(
        "--paper-subdir",
        default="paper",
        help="Output subdirectory under --output-dir (e.g. paper, paper_g)",
    )
    args = parser.parse_args()
    setup_minimal_monotone_style()

    input_csv = Path(args.input_csv)
    figures_root = Path(args.output_dir)
    paper_dir = figures_root / args.paper_subdir
    paper_dir.mkdir(parents=True, exist_ok=True)

    atcoder_dir = (
        input_csv.parent.parent if input_csv.parent.name == "datasets" else input_csv.parent
    )
    summaries_dir = atcoder_dir / "summaries" / args.paper_subdir
    data_dir = atcoder_dir.parent.parent
    contest_cache_path = data_dir / "raw" / "atcoder_contests.json"

    slots = parse_slots_arg(args.slots)
    file_suffix = slot_file_suffix(slots)
    slot_label = ",".join(slot for slot in ["C", "D", "E", "F", "G"] if slot in slots)

    df = pd.read_csv(input_csv)
    df = attach_period_columns(df, contest_cache_path)
    for col in ["year", "period", "difficulty", "clean_text_len"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if args.max_year is not None:
        df = df[df["year"] <= args.max_year].copy()

    df_slots = filter_by_slots(df, slots)
    if df_slots.empty:
        raise SystemExit(f"No rows found after filtering slots: {slot_label}")
    print(f"[info] slots={slot_label}, rows={len(df_slots)}, output={paper_dir}")

    # 1) 연도별 지문 길이
    plot_yearly_statement_length_cf(df_slots, paper_dir, file_suffix=file_suffix)
    # 2) 연도별 난이도
    plot_yearly_difficulty_cf(df_slots, paper_dir, file_suffix=file_suffix)
    # 3) 반기별 지문 길이
    plot_halfyear_metric_cf(
        df_slots,
        "clean_text_len",
        paper_dir / f"halfyearly_statement_length_{file_suffix}.png",
        "Statement Length by Half-Year",
        "Characters (JA)",
    )
    # 4) 반기별 난이도
    plot_halfyear_metric_cf(
        df_slots,
        "difficulty",
        paper_dir / f"halfyearly_difficulty_{file_suffix}.png",
        "Kenkoooo Difficulty by Half-Year",
        "Difficulty",
    )

    tag_df_slots = prepare_tag_df(df_slots)
    prevalence = pd.DataFrame()
    trend = pd.DataFrame()
    if tag_df_slots.empty:
        print("[warn] No tag data found; skipped tag figures.")
    else:
        # 5) 태그 빈도 Top-N
        plot_top_tags_cf_frequency(
            tag_df_slots, paper_dir, args.top_n, file_suffix=file_suffix
        )
        prevalence = build_tag_yearly_prevalence_cf(df_slots, tag_df_slots, args.top_n)
        if prevalence.empty:
            print("[warn] Empty yearly prevalence; skipped remaining tag figures.")
        else:
            # 6) 태그 연도별 점유율 히트맵
            plot_tag_yearly_prevalence_cf(
                prevalence, paper_dir, file_suffix=file_suffix
            )
            # 7) 태그 점유율 변화량(증가/감소)
            trend = build_tag_trend_summary(prevalence)
            plot_tag_share_delta_cf(trend, paper_dir, file_suffix=file_suffix)

    export_paper_summaries(df_slots, prevalence, trend, summaries_dir, file_suffix)


if __name__ == "__main__":
    main()
