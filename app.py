# app.py
# --------------------------------------------
# Excelアップロード → 条件シートの列位置に基づく図番グルーピング → 日/週/月集計 → 多軸グラフ（Streamlit）
# 仕様要点：
# - 日付列デフォルトは「状態」
# - 条件シート：B列=出荷品番、C列以降=すべてグラフ番号（列名は何でもOK）
# - 左軸（棒）：生産済・生産時間[分]、右軸（線）：工数
# - 異常値フィルタ、粒度（日/週/月）、期間指定、CSVダウンロード
# --------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        cond["出荷品番"] = cond["出荷品番"].astype(str).str.strip()
    else:
        cond["出荷品番"] = np.nan

    # C以降 → すべてグラフ列として扱う（列名は何でも可）
    # 空列フィルタを一度だけ実行
    graph_cols = []
    if len(cols) >= 3:
        graph_cols = [c for c in cols[2:] if cond[c].notna().any()]

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

def aggregate_timeseries(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    """
    日付列で集計（freq='D'|'W'|'M'）。工数=生産時間[分]/生産済（0除算=0）。
    """
    _df = df.copy()
    _df[date_col] = parse_datetime_series(_df[date_col])
    _df = _df.dropna(subset=[date_col])

    if _df.empty:
        return pd.DataFrame()

    _df["生産済"] = ensure_numeric(_df.get("生産済", pd.Series(dtype=float)), 0)
    _df["生産時間[分]"] = ensure_numeric(_df.get("生産時間[分]", pd.Series(dtype=float)), 0)
    _df["基準時間[分]"] = ensure_numeric(_df.get("基準時間[分]", pd.Series(dtype=float)), 0)
    _df["能率[%]"] = ensure_numeric(_df.get("能率[%]", pd.Series(dtype=float)), 0)

    _df = _df.set_index(date_col).sort_index()
    grouped = _df.resample(freq).agg({"生産済": "sum", "生産時間[分]": "sum", "基準時間[分]": "sum", "能率[%]": "mean"})
    grouped["工数"] = np.where(grouped["生産済"] > 0, grouped["生産時間[分]"] / grouped["生産済"], 0.0)

    grouped = grouped.reset_index().rename(columns={date_col: "日付"})
    # 日付列を確実にdatetime型に保持
    if "日付" in grouped.columns:
        grouped["日付"] = pd.to_datetime(grouped["日付"], errors="coerce")
    return grouped

def build_summary_stats(agg_df: pd.DataFrame, columns_list: list = None) -> dict:
    """
    集計結果から統計情報を抽出
    入力：
      - agg_df: 集計済みDataFrame
      - columns_list: ['工数', '能率[%]']など対象列のリスト
    出力：
      {
        '工数': {'合計': 100.5, '平均': 10.05, '最大': 25.3, '最小': 2.1},
        '能率[%]': {'合計': 920.0, '平均': 92.0, '最大': 98.5, '最小': 85.0}
      }
    """
    if columns_list is None:
        columns_list = ['工数', '能率[%]']
    
    summary = {}
    for col in columns_list:
        if col in agg_df.columns:
            valid_data = agg_df[col].dropna()
            if len(valid_data) > 0:
                summary[col] = {
                    '合計': valid_data.sum(),
                    '平均': valid_data.mean(),
                    '最大': valid_data.max(),
                    '最小': valid_data.min()
                }
    return summary

def display_summary_metrics(agg_df: pd.DataFrame, columns_list: list = None):
    """
    統計情報をStreamlit metricsで表示
    入力：
      - agg_df: 集計済みDataFrame
      - columns_list: ['工数', '能率[%]']など対象列のリスト
    処理：
      - DataFrame空チェック → メッセージ表示で return
      - st.columns(2) で左右2列を作成
      - 各列に工数 / 能率[%] を表示
      - 各指標（合計・平均・最大・最小）を st.metric() で積み重ね
    """
    if columns_list is None:
        columns_list = ['工数', '能率[%]']
    
    if agg_df.empty or len(agg_df) == 0:
        st.info("集計結果がありません")
        return
    
    stats = build_summary_stats(agg_df, columns_list)
    
    if not stats:
        st.info("集計結果がありません")
        return
    
    # 2列レイアウト
    cols = st.columns(len(columns_list))
    
    for idx, col_name in enumerate(columns_list):
        if col_name in stats:
            with cols[idx]:
                st.metric(f"{col_name} - 合計", f"{stats[col_name]['合計']:.1f}")
                st.metric(f"{col_name} - 平均", f"{stats[col_name]['平均']:.1f}")
                st.metric(f"{col_name} - 最大", f"{stats[col_name]['最大']:.1f}")
                st.metric(f"{col_name} - 最小", f"{stats[col_name]['最小']:.1f}")

def alt_dual_axis_chart(agg_df: pd.DataFrame, title: str, show_items: dict = None, y_autorange: bool = False):
    """
    Plotlyを使った多軸グラフ
    左軸：棒（生産済・生産時間[分]・基準時間[分]）/ 右軸1：工数 / 右軸2：能率[%]
    show_items: 表示要素の辞書
    y_autorange: Trueで Y軸ズーム許可、Falseで固定
    """
    if show_items is None:
        show_items = {"生産済": True, "生産時間[分]": True, "基準時間[分]": True, "工数": True, "能率[%]": True}
    
    if agg_df.empty:
        return go.Figure().add_annotation(text="データなし", showarrow=False)
    
    _df = agg_df.copy()
    if "日付" in _df.columns:
        _df["日付"] = pd.to_datetime(_df["日付"], errors='coerce')
    _df = _df.replace([np.inf, -np.inf], np.nan)

    # Plotly図を作成（3つのY軸：左、右1、右2）
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 左軸：棒グラフ（生産済、生産時間[分]、基準時間[分]）
    bar_configs = [
        ("生産済", '#4472C4', 0.7),
        ("生産時間[分]", '#70AD47', 0.6),
        ("基準時間[分]", '#FFC000', 0.6)
    ]
    
    for item_name, color, opacity in bar_configs:
        if show_items.get(item_name, True) and item_name in _df.columns:
            fig.add_trace(
                go.Bar(
                    x=_df["日付"],
                    y=_df[item_name],
                    name=item_name,
                    marker_color=color,
                    opacity=opacity,
                    yaxis='y'
                ),
                secondary_y=False
            )
    
    # 右軸：工数ライン
    if show_items.get("工数", True) and "工数" in _df.columns:
        fig.add_trace(
            go.Scatter(
                x=_df["日付"],
                y=_df["工数"],
                name="工数",
                mode='lines+markers',
                line=dict(color='#F39C12', width=3),
                yaxis='y2'
            ),
            secondary_y=True
        )
    
    # 右軸2：能率[%]ライン（別スケール）
    if show_items.get("能率[%]", True) and "能率[%]" in _df.columns:
        fig.add_trace(
            go.Scatter(
                x=_df["日付"],
                y=_df["能率[%]"],
                name="能率[%]",
                mode='lines+markers',
                line=dict(color='#E74C3C', width=3, dash='dash'),
                yaxis='y3'
            )
        )

    # レイアウト設定
    fig.update_layout(
        title=title,
        xaxis=dict(title="日付", domain=[0, 0.88], tickformat="%m月%d日<br>%Y年"),
        yaxis=dict(title="生産済・時間[分]", side='left', fixedrange=not y_autorange),
        yaxis2=dict(title="工数", side='right', overlaying='y', title_font=dict(color='#F39C12'), tickfont=dict(color='#F39C12'), fixedrange=not y_autorange),
        yaxis3=dict(title="能率[%]", side='right', overlaying='y', anchor='free', position=1.0, title_font=dict(color='#E74C3C'), tickfont=dict(color='#E74C3C'), fixedrange=not y_autorange),
        margin=dict(r=150),
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# -----------------------------
# サイドバー：オプション
# -----------------------------
with st.sidebar:
    st.header("⚙️ オプション")

    uploaded = st.file_uploader("Excelファイル（.xlsx）をアップロード", type=["xlsx"])

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

    st.divider()
    st.markdown("**グラフ表示要素**")
    show_seisansu = st.checkbox("生産済", value=True)
    show_seisan_time = st.checkbox("生産時間[分]", value=True)
    show_kijun_time = st.checkbox("基準時間[分]", value=True)
    show_kosuu = st.checkbox("工数", value=True)
    show_nouritsu = st.checkbox("能率[%]", value=True)

    st.divider()
    st.markdown("**グラフ操作**")
    y_autorange_mode = st.checkbox("Y軸自動スケール", value=False)

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

# 日付列候補を構築：DATE_CANDIDATES優先、その後中身が日付の列を補完
date_options = [c for c in DATE_CANDIDATES if c in data.columns]
for c in data.columns:
    if c not in date_options and parse_datetime_series(data[c]).notna().any():
        date_options.append(c)
if not date_options and len(data.columns) > 0:
    date_options = [data.columns[0]]

if date_col_default not in date_options and date_col_default in data.columns:
    date_options.append(date_col_default)
    
date_col = st.selectbox(
    "日付列を選択（既定=状態）",
    options=date_options or list(data.columns),
    index=(date_options or list(data.columns)).index(date_col_default) if (date_options or list(data.columns)) else 0
)

# 数値化
data["生産済"] = ensure_numeric(data.get("生産済", pd.Series(dtype=float)), 0)
data["生産時間[分]"] = compute_minutes(data.get("所要時間", pd.Series(dtype=float)), "excel_time")

# 基準時間[分]の計算（Excel形式 × 86400 / 60）
if "基準時間" in data.columns:
    data["基準時間[分]"] = ensure_numeric(data.get("基準時間", pd.Series(dtype=float)), 0) * 86400 / 60
else:
    data["基準時間[分]"] = 0.0

# 能率[%]の計算（基準時間[分] / 生産時間[分] × 100）
data["能率[%]"] = np.where(
    data["生産時間[分]"] > 0,
    (data["基準時間[分]"] / data["生産時間[分]"]) * 100,
    0.0
)

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
st.divider()
# ---- 総集計（全データ合算） ----
st.subheader("① 総集計（全データ合算）")

st.caption(f"集計対象データ件数: {len(data)} 件")

# 品番選択UI（総集計用）
all_hinban = sorted(data["出荷品番"].astype(str).str.strip().unique()) if "出荷品番" in data.columns else []
with st.expander("🔧 表示条件（品番）", expanded=False):
    if all_hinban:
        selected_hinban_overall = st.multiselect(
            "表示する品番を選択",
            options=all_hinban,
            default=all_hinban,
            key="overall_hinban_select"
        )
    else:
        selected_hinban_overall = []

# フィルタリング
if selected_hinban_overall:
    data_filtered_overall = data[data["出荷品番"].astype(str).str.strip().isin(selected_hinban_overall)].copy()
else:
    data_filtered_overall = data.copy()

# CSV用は全データで集計
overall_agg = aggregate_timeseries(data, date_col=date_col, freq=freq)
# グラフ用は選択された品番のみで集計
overall_agg_filtered = aggregate_timeseries(data_filtered_overall, date_col=date_col, freq=freq)
st.caption(f"集計結果: {len(overall_agg)} 行")

# デバッグ：集計後のカラムを表示
with st.expander("🔍 デバッグ：集計後のカラム一覧", expanded=False):
    st.write("**存在するカラム:**")
    st.write(overall_agg.columns.tolist())
    st.write("**データ型:**")
    st.write(overall_agg.dtypes)
    st.write("**能率[%]の値（先頭10行）:**")
    st.write(overall_agg[["日付", "能率[%]"]].head(10))

if not selected_hinban_overall:
    st.warning("⚠️ 品番を選択してください")
elif overall_agg_filtered.empty:
    st.warning("⚠️ 集計結果が空です。日付データや数値データを確認してください。")
    st.dataframe(data_filtered_overall[[date_col, "生産済", "生産時間[分]"]].head(10))
else:
    st.dataframe(overall_agg_filtered.head(10))
    display_summary_metrics(overall_agg_filtered, ['工数', '能率[%]'])
    st.plotly_chart(alt_dual_axis_chart(overall_agg_filtered, "総集計", show_items={
    "生産済": show_seisansu,
    "生産時間[分]": show_seisan_time,
    "基準時間[分]": show_kijun_time,
    "工数": show_kosuu,
    "能率[%]": show_nouritsu
}, y_autorange=y_autorange_mode), use_container_width=True, config={"scrollZoom": True})
st.download_button(
    "総集計CSVをダウンロード",
    data=overall_agg.to_csv(index=False).encode("utf-8-sig"),
    file_name="overall_aggregate.csv",
    mime="text/csv"
)

st.divider()

# ---- 各グラフ名ごと ----
st.subheader("② 各グラフ名（条件シート C列以降）ごとの集計")
if not graph_names:
    st.info("条件シートにグラフ名（C列以降のセル値）が見つかりませんでした。")
else:
    if "出荷品番" not in data.columns:
        st.error("データシートに '出荷品番' 列が見つかりません。列名をご確認ください。")
    else:
        # グラフ名選択
        selected_gname = st.selectbox("表示するグラフを選択", options=graph_names, key="graph_select")
        
        st.subheader(f"📊 {selected_gname}")
        
        items = sorted(gmap[selected_gname])

        
        # 表示条件を expander で折りたたみ
        with st.expander("🔧 表示条件（品番・日付・集計）", expanded=False):
            selected_items = st.multiselect(
                "表示する品番を選択",
                options=items,
                default=items,
                key=f"hinban_select_{selected_gname}"
            )
            
            st.caption(f"対象 出荷品番（{len(items)}件）：{', '.join(items[:30])}{' ...' if len(items) > 30 else ''}")

        # CSV用：条件シート設定通りの全品番データ
        sub_all = data[data["出荷品番"].astype(str).str.strip().isin(items)].copy()
        if sub_all.empty:
            st.warning(f"⚠️  '{selected_gname}': 該当出荷品番データなし")
        elif not selected_items:
            st.warning(f"⚠️  '{selected_gname}': 品番を選択してください")
        else:
            # グラフ用：選択された品番のみ
            sub = data[data["出荷品番"].astype(str).str.strip().isin(selected_items)].copy()
            if sub.empty:
                st.warning(f"⚠️  '{selected_gname}': 選択した品番のデータなし")
            else:
                # CSV用集計（全品番）
                agg_all = aggregate_timeseries(sub_all, date_col=date_col, freq=freq)
                # グラフ用集計（選択品番）
                agg = aggregate_timeseries(sub, date_col=date_col, freq=freq)
                
                # 集計結果が空の場合のチェック
                if agg.empty:
                    st.error(f"❌ '{selected_gname}': 集計結果が空です（日付・数値データを確認してください）")
                else:
                    display_summary_metrics(agg, ['工数', '能率[%]'])
                    st.plotly_chart(alt_dual_axis_chart(agg, f"{selected_gname}", show_items={
                        "生産済": show_seisansu,
                        "生産時間[分]": show_seisan_time,
                        "基準時間[分]": show_kijun_time,
                        "工数": show_kosuu,
                        "能率[%]": show_nouritsu
                    }, y_autorange=y_autorange_mode), use_container_width=True, config={"scrollZoom": True})
                    st.download_button(
                        f"{selected_gname} の集計CSVをダウンロード（全品番）",
                        data=agg_all.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"aggregate_{selected_gname}.csv",
                        mime="text/csv"
                    )