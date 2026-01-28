# app.py
# --------------------------------------------
# Excelアップロード → 条件シートの列位置に基づく図番グルーピング → 日/週/月集計 → 多軸グラフ（Streamlit）
# 仕様要点：
# - 日付列デフォルトは「状態」
# - 条件シート：B列=出荷品番、C列以降=すべてグラフ番号（列名は何でもOK）
# - 左軸（棒）：生産済・生産時間[分]、右軸（線）：工数
# - 異常値フィルタ、粒度（日/週/月）、期間指定、移動平均、CSVダウンロード
# --------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Excelグラフ化ツール（条件シート対応）", layout="wide")
st.title("📊 Excelグラフ化ツール（条件シート対応）")

st.markdown(
    "- **総集計（全データ合算）** → **各グラフ名（条件シート C列以降）ごと**に該当「出荷品番」を**合算して**表示  \n"
    "- 左軸（棒）: **生産済**・**生産時間[分]** ／ 右軸（線）: **工数**  \n"
    "- 条件シートは **B列=出荷品番**、**C列以降=グラフ番号（列名は任意）** として自動解釈します"
)

# -----------------------------
# ユーティリティ
# -----------------------------
DATE_CANDIDATES = ["状態", "生産日", "出荷日", "更新日時"]

def parse_datetime_series(s: pd.Series) -> pd.Series:
    """日付/日時っぽい列をdatetimeへ。Excelシリアル/文字列/NaTに対応。"""
    if s is None:
        return pd.Series([], dtype="datetime64[ns]")
    if np.issubdtype(s.dtype, np.datetime64):
        try:
            return s.dt.tz_localize(None)
        except Exception:
            return s
    # 数値（Excel日数シリアル対応）
    if np.issubdtype(s.dtype, np.number):
        try:
            return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
        except Exception:
            pass
    # 文字列など
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def ensure_numeric(s: pd.Series, fill=0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def compute_minutes(soyo_time: pd.Series, mode: str) -> pd.Series:
    x = ensure_numeric(soyo_time, fill=0)
    return x * 1440.0 if mode == "excel_time" else x

def pick_default_date_col(df: pd.DataFrame) -> str:
    if "状態" in df.columns:
        return "状態"
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    return df.columns[0] if len(df.columns) else "日付"

def normalize_conditions_by_position(cond_raw: pd.DataFrame):
    """
    条件シートを列位置で正規化：
      - 出荷品番: 物理B列（index=1）
      - グラフ番号: 物理C列（index=2）以降をすべて対象
    戻り値: (正規化DataFrame, グラフ列名リスト)
    """
    if cond_raw is None or cond_raw.empty:
        return pd.DataFrame(columns=["出荷品番"]), []

    cond = cond_raw.copy()
    cols = list(cond.columns)

    # B列 → 出荷品番
    if len(cols) >= 2:
        cond.rename(columns={cols[1]: "出荷品番"}, inplace=True)
    else:
        cond["出荷品番"] = np.nan  # 足りない場合は空列

    # C以降 → すべてグラフ列として扱う（列名は何でも可）
    graph_cols = []
    if len(cols) >= 3:
        graph_cols = cols[2:]  # 2番目以降全て
        # 全部空の列は除外
        keep = []
        for c in graph_cols:
            series = cond[c]
            if series.notna().any() and series.astype(str).str.strip().replace("nan", "").ne("").any():
                keep.append(c)
        graph_cols = keep

    # 最低限の列だけ残す
    keep_show = ["出荷品番"] + graph_cols
    cond = cond[[c for c in keep_show if c in cond.columns]].copy()

    # 出荷品番 正規化
    if "出荷品番" in cond.columns:
        cond["出荷品番"] = cond["出荷品番"].astype(str).str.strip()

    return cond, graph_cols

def build_graph_map_dynamic(cond: pd.DataFrame, graph_cols: list[str]) -> dict:
    """
    グラフ名（セル値）→ {出荷品番,...} の辞書を生成。
    graph_cols の各列に書かれたセルの値を“グラフ名”として扱う。
    """
    mapping: dict[str, set] = {}
    if "出荷品番" not in cond.columns or not graph_cols:
        return mapping

    for _, row in cond.iterrows():
        item = str(row["出荷品番"]).strip()
        if not item or item.lower() == "nan":
            continue
        for c in graph_cols:
            g = row.get(c, None)
            if pd.isna(g):
                continue
            gname = str(g).strip()
            if gname == "" or gname.lower() == "nan":
                continue
            mapping.setdefault(gname, set()).add(item)
    return mapping

def aggregate_timeseries(df: pd.DataFrame, date_col: str, freq: str, ma_window: int | None) -> pd.DataFrame:
    """
    日付列で集計（freq='D'|'W'|'M'）。工数=生産時間[分]/生産済（0除算=0）。
    """
    _df = df.copy()
    _df[date_col] = parse_datetime_series(_df[date_col])
    _df = _df.dropna(subset=[date_col])

    _df["生産済"] = ensure_numeric(_df.get("生産済", pd.Series(dtype=float)), 0)
    _df["生産時間[分]"] = ensure_numeric(_df.get("生産時間[分]", pd.Series(dtype=float)), 0)

    _df = _df.set_index(date_col).sort_index()
    grouped = _df.resample(freq).agg({"生産済": "sum", "生産時間[分]": "sum"})
    grouped["工数"] = np.where(grouped["生産済"] > 0, grouped["生産時間[分]"] / grouped["生産済"], 0.0)

    if ma_window and ma_window > 1:
        grouped["工数_MA"] = grouped["工数"].rolling(ma_window, min_periods=1).mean()
    else:
        grouped["工数_MA"] = np.nan

    grouped = grouped.reset_index().rename(columns={date_col: "日付"})
    # 日付列を確実にdatetime型に保持
    if "日付" in grouped.columns:
        grouped["日付"] = pd.to_datetime(grouped["日付"])
    return grouped

def alt_dual_axis_chart(agg_df: pd.DataFrame, title: str):
    """
    左軸：棒（生産済・生産時間[分]）/ 右軸：線（工数 or MA）
    """
    if agg_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_text(text="データなし")
    
    # 日付列を確実にdatetime型に変換
    _df = agg_df.copy()
    if "日付" in _df.columns:
        _df["日付"] = pd.to_datetime(_df["日付"], errors='coerce')

    # 数値が無限大でないか確認
    _df = _df.replace([np.inf, -np.inf], np.nan)

    base = alt.Chart(_df).properties(title=title, height=360, width=800)

    # 棒グラフ1：生産済
    bar1 = (
        base
        .mark_bar(opacity=0.7, color='#4472C4')
        .encode(
            x=alt.X("日付:T", title="日付"),
            y=alt.Y("生産済:Q", title="生産済", axis=alt.Axis(orient="left")),
            tooltip=[
                alt.Tooltip("日付:T", title="日付", format="%Y/%m/%d"),
                alt.Tooltip("生産済:Q", title="生産済", format=".0f"),
            ],
        )
    )

    # 棒グラフ2：生産時間[分]（透明度を下げて重ねる）
    bar2 = (
        base
        .mark_bar(opacity=0.5, color='#70AD47')
        .encode(
            x=alt.X("日付:T"),
            y=alt.Y("生産時間[分]:Q"),
            tooltip=[
                alt.Tooltip("日付:T", title="日付", format="%Y/%m/%d"),
                alt.Tooltip("生産時間[分]:Q", title="生産時間[分]", format=".0f"),
            ],
        )
    )

    # 右軸：工数（移動平均があればそちらを優先）
    y_line_field = "工数_MA" if "工数_MA" in _df.columns and _df["工数_MA"].notna().any() else "工数"
    
    # 右軸用の工数ライン
    line = (
        base.mark_line(point=True, color="#F39C12", size=3)
        .encode(
            x=alt.X("日付:T", title=""),
            y=alt.Y(f"{y_line_field}:Q", axis=alt.Axis(title="工数", titleColor="#F39C12", labelColor="#F39C12", orient="right")),
            tooltip=[
                alt.Tooltip("日付:T", title="日付", format="%Y/%m/%d"),
                alt.Tooltip(f"{y_line_field}:Q", title="工数", format=".4f"),
            ],
        )
    )

    # 左軸と右軸を独立させて合成
    chart = (bar1 + bar2 + line).resolve_scale(y="independent")
    return chart

# -----------------------------
# サイドバー：オプション
# -----------------------------
with st.sidebar:
    st.header("⚙️ オプション")

    uploaded = st.file_uploader("Excelファイル（.xlsx）をアップロード", type=["xlsx"])

    st.markdown("**所要時間の単位**")
    time_mode = st.radio(
        "所要時間の換算方法",
        options=[("Excelの時間（×1440で分に換算）", "excel_time"), ("すでに分（換算なし）", "minutes")],
        format_func=lambda x: x[0],
        index=0,
    )[1]

    st.markdown("**異常値フィルタ（'異常値'列）**")
    abnormal_filter = st.radio(
        "フィルタ",
        options=[("全て", "all"), ("正常のみ（0）", "normal"), ("異常のみ（1）", "abnormal")],
        format_func=lambda x: x[0],
        horizontal=True,
        index=0,
    )[1]

    freq_options = [("日次", "D"), ("週次", "W"), ("月次", "M")]
    freq_choice = st.selectbox("集計粒度", options=freq_options, format_func=lambda x: x[0], index=0)
    freq = freq_choice[1]  # タプルの2番目の要素（文字列）を取得

    ma_on = st.checkbox("工数の移動平均を表示", value=False)
    ma_window = st.slider("移動平均ウィンドウ（日数換算）", min_value=2, max_value=28, value=7) if ma_on else None

    st.divider()
    st.caption("※ ヘッダー行（0始まり）を調整できます。最上段が見出しでない場合にご利用ください。")
    cond_header_idx = st.number_input("条件シートのヘッダー行", min_value=0, max_value=50, value=0, step=1)
    data_header_idx = st.number_input("データシート（39）のヘッダー行", min_value=0, max_value=50, value=0, step=1)

if not uploaded:
    st.info("左のサイドバーから Excel ファイル（.xlsx）をアップロードしてください。")
    st.stop()

# -----------------------------
# Excel読込
# -----------------------------
with st.spinner("Excelを読み込み中…"):
    try:
        xl = pd.ExcelFile(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"Excelの読み込みに失敗しました: {e}")
        st.stop()

    # 条件シート名の推定
    cond_sheet_name = "条件シート" if "条件シート" in xl.sheet_names else next((s for s in xl.sheet_names if "条件" in s), None)
    if not cond_sheet_name:
        st.error(f"条件シートが見つかりません。存在するシート: {xl.sheet_names}")
        st.stop()

    # データシート名
    data_sheet_name = "39" if "39" in xl.sheet_names else xl.sheet_names[0]

    try:
        cond_raw = xl.parse(cond_sheet_name, header=int(cond_header_idx))
        data_raw = xl.parse(data_sheet_name, header=int(data_header_idx))
    except Exception as e:
        st.error(f"シートの読み取りに失敗しました: {e}")
        st.stop()

# 条件シート正規化（列位置ベース）
cond, graph_cols = normalize_conditions_by_position(cond_raw)
gmap = build_graph_map_dynamic(cond, graph_cols)
graph_names = sorted(gmap.keys())

# データ前処理
data = data_raw.copy()

# 日付列の既定は「状態」
date_col_default = pick_default_date_col(data)
date_options = [c for c in DATE_CANDIDATES if c in data.columns]
if date_col_default not in date_options and date_col_default in data.columns:
    date_options.append(date_col_default)
date_col = st.selectbox(
    "日付列を選択（既定=状態）",
    options=date_options or list(data.columns),
    index=(date_options or list(data.columns)).index(date_col_default) if (date_options or list(data.columns)) else 0
)

# 数値化
data["生産済"] = ensure_numeric(data.get("生産済", pd.Series(dtype=float)), 0)
data["生産時間[分]"] = compute_minutes(data.get("所要時間", pd.Series(dtype=float)), time_mode)

# 異常値フィルタ
if "異常値" in data.columns:
    if abnormal_filter == "normal":
        data = data[data["異常値"].fillna(0) == 0]
    elif abnormal_filter == "abnormal":
        data = data[data["異常値"].fillna(0) == 1]
else:
    st.warning("注意：'異常値' 列が見つからないため、異常値フィルタは無効です。")

# 期間フィルタUI
dt_series = parse_datetime_series(data.get(date_col))
if dt_series.notna().any():
    min_d, max_d = dt_series.min().date(), dt_series.max().date()
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("開始日", value=min_d, min_value=min_d, max_value=max_d)
    with c2:
        end_date = st.date_input("終了日", value=max_d, min_value=min_d, max_value=max_d)
    mask = (dt_series.dt.date >= start_date) & (dt_series.dt.date <= end_date)
    data = data.loc[mask].copy()
else:
    st.warning("選択した日付列を日時に解釈できませんでした。日付列の選択を見直してください。")

# プレビュー
with st.expander("データプレビュー（先頭50行）", expanded=False):
    st.caption(f"シート: {data_sheet_name} / 行数: {len(data)}")
    st.dataframe(data.head(50), use_container_width=True)

with st.expander("条件シートプレビュー（先頭50行）", expanded=False):
    st.caption(f"シート: {cond_sheet_name} / 行数: {len(cond)} / グラフ列数: {len(graph_cols)}")
    st.dataframe(cond.head(50), use_container_width=True)

# ---- 総集計（全データ合算） ----
st.subheader("① 総集計（全データ合算）")
st.caption(f"集計対象データ件数: {len(data)} 件")
overall_agg = aggregate_timeseries(data, date_col=date_col, freq=freq, ma_window=(ma_window if ma_on else None))
st.caption(f"集計結果: {len(overall_agg)} 行")
if overall_agg.empty:
    st.warning("⚠️ 集計結果が空です。日付データや数値データを確認してください。")
    st.dataframe(data[[date_col, "生産済", "生産時間[分]"]].head(10))
else:
    st.dataframe(overall_agg.head(10))
st.altair_chart(alt_dual_axis_chart(overall_agg, "総集計"), use_container_width=True)
st.download_button(
    "総集計CSVをダウンロード",
    data=overall_agg.to_csv(index=False).encode("utf-8-sig"),
    file_name="overall_aggregate.csv",
    mime="text/csv"
)

# ---- 各グラフ名ごと ----
st.subheader("② 各グラフ名（条件シート C列以降）ごとの集計")
if not graph_names:
    st.info("条件シートにグラフ名（C列以降のセル値）が見つかりませんでした。")
else:
    if "出荷品番" not in data.columns:
        st.error("データシートに '出荷品番' 列が見つかりません。列名をご確認ください。")
    else:
        for gname in graph_names:
            items = sorted(gmap[gname])
            st.markdown(f"### グラフ名：**{gname}**")
            st.caption(f"対象 出荷品番（{len(items)}件）：{', '.join(items[:30])}{' ...' if len(items) > 30 else ''}")

            # 出荷品番一致で抽出（型ブレ対策で文字列比較）
            sub = data[data["出荷品番"].astype(str).str.strip().isin(items)].copy()
            if sub.empty:
                st.warning("該当データなし")
                continue

            agg = aggregate_timeseries(sub, date_col=date_col, freq=freq, ma_window=(ma_window if ma_on else None))
            st.altair_chart(alt_dual_axis_chart(agg, f"{gname}"), use_container_width=True)
            st.download_button(
                f"{gname} の集計CSVをダウンロード",
                data=agg.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"aggregate_{gname}.csv",
                mime="text/csv"
            )