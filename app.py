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

from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

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

    if is_datetime64_any_dtype(s):

        try:

            return s.dt.tz_localize(None)

        except Exception:

            return s

    # 数値（Excel日数シリアル対応）

    if is_numeric_dtype(s):

        try:

            return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")

        except Exception:

            pass

    # 文字列など

    parsed = pd.to_datetime(s, errors="coerce", utc=True)

    try:

        return parsed.dt.tz_localize(None)

    except Exception:

        return parsed



def ensure_numeric(s: pd.Series, fill=0) -> pd.Series:

    return pd.to_numeric(s, errors="coerce").fillna(fill)



def compute_minutes(soyo_time: pd.Series, mode: str) -> pd.Series:

    x = ensure_numeric(soyo_time, fill=0)

    return x * 1440.0 if mode == "excel_time" else x


def normalize_resample_freq(freq: str) -> str:

    """pandas 3 で変更された旧オフセット別名を互換変換する。"""

    return {
        "M": "ME",
        "Q": "QE",
        "Y": "YE",
        "BM": "BME",
        "BQ": "BQE",
        "BY": "BYE",
    }.get(freq, freq)


def normalize_graph_key(val) -> str:

    """グラフ番号の表記ゆれを統一（例: 24.0 → 24）"""

    if pd.isna(val):

        return ""

    if isinstance(val, (int, np.integer)):

        return str(val)

    if isinstance(val, (float, np.floating)):

        return str(int(val)) if float(val).is_integer() else str(val)

    s = str(val).strip()

    try:

        f = float(s)

        return str(int(f)) if f.is_integer() else s

    except Exception:

        return s



def apply_quick_date_filter(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:

    """

    クイック期間ボタン + 日付ピッカーで期間を決定し、dfをフィルタして返す。

    - df.index は DatetimeIndex を想定（必要なら変換）

    - セッション状態でクイック選択と手動日付入力の競合を回避

    """

    if df is None or df.empty:

        return df



    _df = df.copy()

    if not isinstance(_df.index, pd.DatetimeIndex):

        _df.index = pd.to_datetime(_df.index, errors="coerce")

    _df = _df[_df.index.notna()].sort_index()

    if _df.empty:

        return _df



    if getattr(_df.index, "tz", None) is not None:

        _df.index = _df.index.tz_localize(None)



    today = pd.Timestamp.today().normalize()

    date_min = _df.index.min().normalize()

    date_max = _df.index.max().normalize()



    quick_options = ["指定なし", "直近7日", "直近30日", "今月", "先月"]

    quick_key = f"{key_prefix}_quick"

    start_key = f"{key_prefix}_start"

    end_key = f"{key_prefix}_end"

    last_quick_key = f"{key_prefix}_last_quick"



    if quick_key not in st.session_state:

        st.session_state[quick_key] = "指定なし"

    if start_key not in st.session_state:

        st.session_state[start_key] = date_min.date()

    if end_key not in st.session_state:

        st.session_state[end_key] = date_max.date()

    if last_quick_key not in st.session_state:

        st.session_state[last_quick_key] = st.session_state[quick_key]



    st.markdown("**期間選択**")

    st.radio("クイック期間選択", options=quick_options, key=quick_key, horizontal=True)



    if st.session_state[quick_key] != st.session_state[last_quick_key]:

        q = st.session_state[quick_key]

        if q == "直近7日":

            start = today - pd.Timedelta(days=6)

            end = today

        elif q == "直近30日":

            start = today - pd.Timedelta(days=29)

            end = today

        elif q == "今月":

            start = today.replace(day=1)

            end = today

        elif q == "先月":

            first_this_month = today.replace(day=1)

            end = first_this_month - pd.Timedelta(days=1)

            start = end.replace(day=1)

        else:

            start = date_min

            end = date_max



        st.session_state[start_key] = start.date()

        st.session_state[end_key] = end.date()

        st.session_state[last_quick_key] = q



    col1, col2 = st.columns(2)

    with col1:

        st.date_input("開始日", value=st.session_state[start_key], key=start_key)

    with col2:

        st.date_input("終了日", value=st.session_state[end_key], key=end_key)



    start_date = st.session_state[start_key]

    end_date = st.session_state[end_key]



    if start_date and end_date and start_date > end_date:

        start_date, end_date = end_date, start_date



    idx_dates = _df.index.date

    mask = (idx_dates >= start_date) & (idx_dates <= end_date)

    return _df.loc[mask]



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

    graph_cols = []

    if len(cols) >= 3:

        graph_cols = list(cols[2:])



    # 最低限の列だけ残す

    keep_show = ["出荷品番"] + graph_cols

    cond = cond[[c for c in keep_show if c in cond.columns]].copy()



    # 出荷品番 正規化

    if "出荷品番" in cond.columns:

        cond["出荷品番"] = cond["出荷品番"].astype(str).str.strip()



    return cond, graph_cols



def build_graph_map_dynamic(cond: pd.DataFrame, graph_cols: list[str], name_map: dict = None) -> dict:
    """
    グラフ名（セル値）→ {出荷品番,...} の辞書を生成。
    graph_cols の各列に書かれたセルの値を“グラフ名”として扱う。
    name_map: {グラフ番号: グラフ名} の辞書。指定があれば番号を名前に変換する。
    """
    mapping: dict[str, set] = {}
    if "出荷品番" not in cond.columns or not graph_cols:
        return mapping

    if name_map is None:
        name_map = {}

    for _, row in cond.iterrows():
        item = str(row["出荷品番"]).strip()
        if not item or item.lower() == "nan":
            continue
        for c in graph_cols:
            g = row.get(c, None)
            if pd.isna(g):
                continue
            g_raw = normalize_graph_key(g)
            if g_raw == "" or g_raw.lower() == "nan":
                continue
            
            # グラフ名変換（マッピングにあれば置換、なければそのまま）
            gname = name_map.get(g_raw, g_raw)
            
            mapping.setdefault(gname, set()).add(item)
    return mapping



def aggregate_timeseries(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:

    """

    日付列で集計（freq='D'|'W'|'ME'）。工数=生産時間[分]/生産済（0除算=0）。

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

    grouped = _df.resample(normalize_resample_freq(freq)).agg({"生産済": "sum", "生産時間[分]": "sum", "基準時間[分]": "sum", "能率[%]": "mean"})

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

        '集計期間': '2024/01/01 ~ 2024/12/31',

        '生産済': {'合計': 1000},

        '生産時間[分]': {'合計': 15000},

        '工数': {'平均': 10.05, '最大': 25.3, '最小': 2.1},

        '能率[%]': {'平均': 92.0}

      }

    """

    if columns_list is None:

        columns_list = ['工数', '能率[%]']

    

    summary = {}

    

    # 集計期間（日付列の最小～最大）

    if "日付" in agg_df.columns:

        date_series = pd.to_datetime(agg_df["日付"], errors='coerce').dropna()

        if len(date_series) > 0:

            start_date = date_series.min().strftime('%m%d')

            end_date = date_series.max().strftime('%m%d')

            summary["集計期間"] = f"{start_date}-{end_date}"

        else:

            summary["集計期間"] = "N/A"

    else:

        summary["集計期間"] = "N/A"

    

    # 生産済の合計

    seisan_sum = ensure_numeric(agg_df.get("生産済", pd.Series(dtype=float)), 0).sum()

    summary["生産済"] = {'合計': seisan_sum}

    

    # 生産時間[分]の合計

    seisan_time_sum = ensure_numeric(agg_df.get("生産時間[分]", pd.Series(dtype=float)), 0).sum()

    summary["生産時間[分]"] = {'合計': seisan_time_sum}

    

    # 工数の平均：生産時間合計 ÷ 生産済合計

    if "工数" in columns_list:

        kosuu_avg = seisan_time_sum / seisan_sum if seisan_sum > 0 else 0.0



        kosuu_series = agg_df.get("工数", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()

        kosuu_max = kosuu_series.max() if len(kosuu_series) > 0 else 0.0

        kosuu_min = kosuu_series.min() if len(kosuu_series) > 0 else 0.0



        summary["工数"] = {

            '平均': kosuu_avg,

            '最大': kosuu_max,

            '最小': kosuu_min

        }



    # 能率の平均：基準時間合計 ÷ 生産時間合計 × 100

    if "能率[%]" in columns_list:

        kijun_sum = ensure_numeric(agg_df.get("基準時間[分]", pd.Series(dtype=float)), 0).sum()

        nouritsu_avg = (kijun_sum / seisan_time_sum * 100) if seisan_time_sum > 0 else 0.0

        summary["能率[%]"] = {

            '平均': nouritsu_avg

        }

    return summary



def display_summary_metrics(agg_df: pd.DataFrame, columns_list: list = None, freq: str = "D"):

    """

    統計情報をStreamlit metricsで表示

    入力：

      - agg_df: 集計済みDataFrame

      - columns_list: ['工数', '能率[%]']など対象列のリスト

    処理：

      - DataFrame空チェック → メッセージ表示で return

      - st.columns(7) で7列を作成

      - 集計期間、生産済-合計、生産時間[分]-合計、工数-平均、工数-最大、工数-最小、能率[%]-平均 を横に並べて表示

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

    

    # 7列レイアウト：集計期間、生産済-合計、生産時間[分]-合計、工数-平均、工数-最大、工数-最小、能率[%]-平均

    cols = st.columns([2, 1, 1, 1, 1, 1, 1])

    

    # 1列目：集計期間

    with cols[0]:

        st.metric("期間", stats.get("集計期間", "N/A"))

    

    # 2列目：生産済-合計

    with cols[1]:

        if "生産済" in stats:

            st.metric("生産済 - 合計", f"{stats['生産済']['合計']:.0f}")

    

    # 3列目：生産時間[分]-合計

    with cols[2]:

        if "生産時間[分]" in stats:

            st.metric("生産時間[分] - 合計", f"{stats['生産時間[分]']['合計']:.1f}")

    

    # 4～7列目：工数と能率

    metrics_to_display = [

        ("工数", "平均", 3),

        ("工数", "最大", 4),

        ("工数", "最小", 5),

        ("能率[%]", "平均", 6)

    ]

    

    for col_name, stat_name, col_idx in metrics_to_display:

        if col_name in stats and stat_name in stats[col_name]:

            with cols[col_idx]:

                st.metric(f"{col_name} - {stat_name}", f"{stats[col_name][stat_name]:.1f}")



    # 週次・月次集計の場合、直近3期間分を内訳表示

    if freq in ["W", "M"]:

        period_label = "3週" if freq == "W" else "3ヶ月"

        st.markdown("---")

        st.markdown(f"**▼ 直近{period_label}分の内訳**")

        

        # 日付降順（新しい順）で3件取得

        if "日付" in agg_df.columns:

            recent_df = agg_df.sort_values("日付", ascending=False).head(3)

            

            for _, row in recent_df.iterrows():

                cols = st.columns([2, 1, 1, 1, 1, 1, 1])

                

                # 日付整形

                d_val = row["日付"]

                if pd.notna(d_val):

                    if normalize_resample_freq(freq) == "ME":

                        d_str = d_val.strftime('%Y年%m月')

                    else:

                        d_str = d_val.strftime('%Y-%m-%d')

                else:

                    d_str = "N/A"

                

                # 値取得

                seisan = row.get("生産済", 0)

                time_min = row.get("生産時間[分]", 0)

                kosu = row.get("工数", 0)

                eff = row.get("能率[%]", 0)



                # 行表示

                with cols[0]:

                    st.write(f"**{d_str}**")

                with cols[1]:

                    st.write(f"{seisan:,.0f}")

                with cols[2]:

                    st.write(f"{time_min:,.1f}")

                with cols[3]:

                    st.write(f"{kosu:.1f}")

                with cols[4]:

                    st.write("-") # 集計済の単位データのため最大最小は算出不可

                with cols[5]:

                    st.write("-")

                with cols[6]:

                    st.write(f"{eff:.1f}")



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

                connectgaps=True,

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

                connectgaps=True,

                yaxis='y3'

            )

        )



    # 目標線 105% (能率軸 y3)
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="paper",
        y0=105, y1=105, yref="y3",
        line=dict(color="red", width=2),
    )
    fig.add_annotation(
        x=0.02, xref="paper",
        y=105, yref="y3",
        text="目標 105%",
        showarrow=False,
        font=dict(color="red"),
        yshift=10
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



    uploaded = st.file_uploader("ミノベサン（.xlsx）をアップロード", type=["xlsx"])
    uploaded_sino = st.file_uploader("追加データ（オプション: sinoPなど）", type=["xlsx"], key="sino_uploader")



    st.markdown("**異常値フィルタ（'異常値'列）**")

    abnormal_filter = st.radio(

        "フィルタ",

        options=[("全て", "all"), ("正常のみ（0）", "normal"), ("異常のみ（1）", "abnormal")],

        format_func=lambda x: x[0],

        horizontal=True,

        index=0,

    )[1]



    freq_options = [("日次", "D"), ("週次", "W"), ("月次", "ME")]

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
    # 既にセッションにデータがある場合は続行（ページ戻り対応）
    if "source_data_raw" in st.session_state:
        data_raw = st.session_state["source_data_raw"]
        gmap = st.session_state["source_gmap"]
        graph_name_map = st.session_state.get("source_graph_name_map", {})
        data_sheet_name = st.session_state.get("source_data_sheet_name", "39")
        cond_sheet_name = st.session_state.get("source_cond_sheet_name", "条件シート")
        cond = st.session_state.get("source_cond", pd.DataFrame())
        graph_cols = st.session_state.get("source_graph_cols", [])
        graph_names = sorted(gmap.keys())
        skip_loading = True
    else:
        st.info("左のサイドバーから Excel ファイル（.xlsx）をアップロードしてください。")
        st.stop()
else:
    skip_loading = False

if not skip_loading:
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
            
            # 列名の空白除去（不整合防止）
            if not data_raw.empty:
                data_raw.columns = [str(c).strip() for c in data_raw.columns]
        except Exception as e:
            st.error(f"シートの読み取りに失敗しました: {e}")
            st.stop()

        # グラフ名シートの読み込み（オプション）
        graph_name_map = {}
        if "グラフ名" in xl.sheet_names:
            try:
                # ユーザー指定：A列=グラフ番号, B列=グラフ名
                gname_df = xl.parse("グラフ名")
                if len(gname_df.columns) >= 2:
                    for _, row in gname_df.iterrows():
                        k = normalize_graph_key(row[0])
                        v = str(row[1]).strip()
                        if k and k.lower() != "nan" and v and v.lower() != "nan":
                            graph_name_map[k] = v
            except Exception:
                pass
        
        # -----------------------------
        # 追加: 手動アップロードされたファイルの読み込みと統合
        # -----------------------------
        sino_mapping_list = []  # (出荷品番, グラフ番号) のリスト

        if uploaded_sino:
            try:
                # Sheet "33" を読み込み
                sino_df = pd.read_excel(uploaded_sino, sheet_name="33")
                if not sino_df.empty:
                    # 列名正規化
                    sino_df.columns = [str(c).strip() for c in sino_df.columns]
                    # "条件" 列が無い場合、シート名 "33" を条件番号として付与
                    if "条件" not in sino_df.columns:
                        sino_df["条件"] = 33
                    data_raw = pd.concat([data_raw, sino_df], ignore_index=True)
                    # マッピング情報の抽出
                    if "出荷品番" in sino_df.columns:
                        for _, r in sino_df.iterrows():
                            sino_mapping_list.append((r["出荷品番"], r["条件"]))
            except Exception as e:
                st.warning(f"追加データの読み込みに失敗しました: {e}")

if not skip_loading:
    # 条件シート正規化（列位置ベース）
    cond, graph_cols = normalize_conditions_by_position(cond_raw)
    gmap = build_graph_map_dynamic(cond, graph_cols, name_map=graph_name_map)

    # sinoPのマッピング情報を追記
    for item_val, g_val in sino_mapping_list:
        item = str(item_val).strip()
        if not item or item.lower() == "nan":
            continue
        g_raw = normalize_graph_key(g_val)
        if g_raw == "" or g_raw.lower() == "nan":
            continue
        gname = graph_name_map.get(g_raw, g_raw)
        gmap.setdefault(gname, set()).add(item)

    graph_names = sorted(gmap.keys())

    # 保存（後の復元用）
    st.session_state.update({
        "source_data_raw": data_raw,
        "source_gmap": gmap,
        "source_graph_name_map": graph_name_map,
        "source_data_sheet_name": data_sheet_name,
        "source_cond_sheet_name": cond_sheet_name,
        "source_cond": cond,
        "source_graph_cols": graph_cols,
    })

# データ前処理（常に生データから開始）
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

# 分析ページ用に、フィルタ前の計算済みデータを保存
st.session_state["data_full_calculated"] = data.copy()



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


st.session_state.update({"data": data, "gmap": gmap, "date_col": date_col})

# プレビュー

if skip_loading:
    data_sheet_name = "（セッションから復元）"
    cond_sheet_name = "（セッションから復元）"
    cond = pd.DataFrame() # プレビュー用の空枠
    graph_cols = []

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

        # 品番リストが変更されたときに選択状態をリセットするために、keyにハッシュを含める
        import hashlib
        options_hash = hashlib.md5(str(all_hinban).encode()).hexdigest()
        
        selected_hinban_overall = st.multiselect(
            "表示する品番を選択",
            options=all_hinban,
            default=all_hinban,
            key=f"overall_hinban_select_{options_hash}"
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

    if not overall_agg.empty and "日付" in overall_agg.columns and "能率[%]" in overall_agg.columns:
        st.write(overall_agg[["日付", "能率[%]"]].head(10))
    else:
        st.write("表示できるデータがありません")



if not selected_hinban_overall:

    st.warning("⚠️ 品番を選択してください")

elif overall_agg_filtered.empty:

    st.warning("⚠️ 集計結果が空です。日付データや数値データを確認してください。")

    with st.expander("🔍 総集計の行データ（先頭10行）", expanded=False):

        st.dataframe(data_filtered_overall[[date_col, "生産済", "生産時間[分]"]].head(10))

else:

    with st.expander("🔍 総集計の行データ（先頭10行）", expanded=False):

        st.dataframe(overall_agg_filtered.head(10))

    display_summary_metrics(overall_agg_filtered, ['工数', '能率[%]'], freq=freq)

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

        # 全グラフ表示モード切り替え

        show_all = st.checkbox("全グラフをまとめて表示", value=False)



        if show_all:

            target_graphs = graph_names

        else:

            # 単一選択

            target_graphs = [st.selectbox("表示するグラフを選択", options=graph_names, key="graph_select")]



        # ループで描画

        for i, selected_gname in enumerate(target_graphs):

            if i > 0:

                st.markdown("---")

            

            st.subheader(f"📊 {selected_gname}")

            

            items = sorted(gmap[selected_gname])



            # 表示条件を expander で折りたたみ

            with st.expander(f"🔧 {selected_gname}: 表示条件（品番・日付・集計）", expanded=False):

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

                    with st.expander(f"🔍 {selected_gname}: 選択品番の生データ（先頭50行）", expanded=False):

                        st.caption(f"件数: {len(sub)} 行")

                        st.dataframe(sub.head(50), use_container_width=True)



                    # CSV用集計（全品番）

                    agg_all = aggregate_timeseries(sub_all, date_col=date_col, freq=freq)

                    # グラフ用集計（選択品番）

                    agg = aggregate_timeseries(sub, date_col=date_col, freq=freq)

                    

                    # 集計結果が空の場合のチェック

                    if agg.empty:

                        st.error(f"❌ '{selected_gname}': 集計結果が空です（日付・数値データを確認してください）")

                    else:

                        # 期間フィルタ（クイック期間 + 日付ピッカー）

                        if "日付" in agg.columns:

                            agg_indexed = agg.set_index("日付")

                        else:

                            agg_indexed = agg.copy()



                        filtered_indexed = apply_quick_date_filter(agg_indexed, key_prefix=f"period_{selected_gname}")

                        filtered_agg = filtered_indexed.reset_index()

                        if "日付" in filtered_agg.columns:

                            filtered_agg["日付"] = pd.to_datetime(filtered_agg["日付"], errors="coerce")



                        if filtered_agg.empty:

                            st.warning(f"⚠️  '{selected_gname}': 指定期間内のデータがありません")

                        else:

                            display_summary_metrics(filtered_agg, ['工数', '能率[%]'], freq=freq)

                            st.plotly_chart(alt_dual_axis_chart(filtered_agg, f"{selected_gname}", show_items={

                                "生産済": show_seisansu,

                                "生産時間[分]": show_seisan_time,

                                "基準時間[分]": show_kijun_time,

                                "工数": show_kosuu,

                                "能率[%]": show_nouritsu

                            }, y_autorange=y_autorange_mode), use_container_width=True, config={"scrollZoom": True}, key=f"chart_{selected_gname}")

                        st.download_button(

                            f"{selected_gname} の集計CSVをダウンロード（全品番）",

                            data=agg_all.to_csv(index=False).encode("utf-8-sig"),

                            file_name=f"aggregate_{selected_gname}.csv",

                            mime="text/csv",

                            key=f"btn_{selected_gname}"

                        )

# -----------------------------
# ③ 取引先別集計
# -----------------------------
st.divider()
st.subheader("③ 取引先別集計")

if "取引先" not in data.columns:
    st.info("データに '取引先' 列が見つかりません。")
else:
    # 取引先リスト取得（空白除外）
    clients = sorted([str(x).strip() for x in data["取引先"].unique() if str(x).strip() != "" and str(x).lower() != "nan"])
    
    if not clients:
        st.warning("有効な取引先データが見つかりません。")
    else:
        # 全取引先表示モード切り替え
        show_all_clients = st.checkbox("全取引先をまとめて表示", value=False, key="client_show_all")

        if show_all_clients:
            target_clients = clients
        else:
            # 単一選択
            target_clients = [st.selectbox("表示する取引先を選択", options=clients, key="client_select")]

        # ループで描画
        for i, selected_client in enumerate(target_clients):
            if i > 0:
                st.markdown("---")
            
            st.subheader(f"🏢 {selected_client}")
            
            # フィルタリング（取引先一致）
            sub_client = data[data["取引先"].astype(str).str.strip() == selected_client].copy()
            
            if sub_client.empty:
                st.warning(f"⚠️  '{selected_client}': データなし")
            else:
                with st.expander(f"🔍 {selected_client}: 生データ（先頭50行）", expanded=False):
                    st.caption(f"件数: {len(sub_client)} 行")
                    st.dataframe(sub_client.head(50), use_container_width=True)

                # 集計実行
                agg_client = aggregate_timeseries(sub_client, date_col=date_col, freq=freq)
                
                # 集計結果空チェック
                if agg_client.empty:
                    st.error(f"❌ '{selected_client}': 集計結果が空です（日付・数値データを確認してください）")
                else:
                    display_summary_metrics(agg_client, ['工数', '能率[%]'], freq=freq)
                    st.plotly_chart(alt_dual_axis_chart(agg_client, f"{selected_client}（取引先別）", show_items={
                        "生産済": show_seisansu,
                        "生産時間[分]": show_seisan_time,
                        "基準時間[分]": show_kijun_time,
                        "工数": show_kosuu,
                        "能率[%]": show_nouritsu
                    }, y_autorange=y_autorange_mode), use_container_width=True, config={"scrollZoom": True}, key=f"chart_client_{selected_client}")
                    
                    st.download_button(
                        f"{selected_client} の集計CSVをダウンロード",
                        data=agg_client.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"aggregate_client_{selected_client}.csv",
                        mime="text/csv",
                        key=f"btn_client_{selected_client}"
                    )


# -----------------------------
# ④ 作業者別集計
# -----------------------------
st.divider()
st.subheader("④ 作業者別集計")

if "作業者" not in data.columns:
    st.info("データに '作業者' 列が見つかりません。")
else:
    # 作業者リスト取得（空白除外）
    workers = sorted([
        str(x).strip()
        for x in data["作業者"].unique()
        if str(x).strip() != "" and str(x).lower() != "nan"
    ])
    
    if not workers:
        st.warning("有効な作業者データが見つかりません。")
    else:
        # 全作業者表示モード切り替え
        show_all_workers = st.checkbox("全作業者をまとめて表示", value=False, key="worker_show_all")

        if show_all_workers:
            target_workers = workers
        else:
            # 単一選択
            target_workers = [
                st.selectbox("表示する作業者を選択", options=workers, key="worker_select")
            ]

        # ループで描画
        for i, selected_worker in enumerate(target_workers):
            if i > 0:
                st.markdown("---")
            
            st.subheader(f"👷 {selected_worker}")
            
            # フィルタリング（作業者一致）
            sub_worker = data[
                data["作業者"].astype(str).str.strip() == selected_worker
            ].copy()
            
            if sub_worker.empty:
                st.warning(f"⚠️  '{selected_worker}': データなし")
            else:
                with st.expander(f"🔍 {selected_worker}: 生データ（先頭50行）", expanded=False):
                    st.caption(f"件数: {len(sub_worker)} 行")
                    st.dataframe(sub_worker.head(50), use_container_width=True)

                # 集計実行
                agg_worker = aggregate_timeseries(sub_worker, date_col=date_col, freq=freq)
                
                if agg_worker.empty:
                    st.error(f"❌ '{selected_worker}': 集計結果が空です（日付・数値データを確認してください）")
                else:
                    display_summary_metrics(agg_worker, ['工数', '能率[%]'], freq=freq)

                    st.plotly_chart(
                        alt_dual_axis_chart(
                            agg_worker,
                            f"{selected_worker}（作業者別）",
                            show_items={
                                "生産済": show_seisansu,
                                "生産時間[分]": show_seisan_time,
                                "基準時間[分]": show_kijun_time,
                                "工数": show_kosuu,
                                "能率[%]": show_nouritsu
                            },
                            y_autorange=y_autorange_mode
                        ),
                        use_container_width=True,
                        config={"scrollZoom": True},
                        key=f"chart_worker_{selected_worker}"
                    )
                    
                    st.download_button(
                        f"{selected_worker} の集計CSVをダウンロード",
                        data=agg_worker.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"aggregate_worker_{selected_worker}.csv",
                        mime="text/csv",
                        key=f"btn_worker_{selected_worker}"
                    )
