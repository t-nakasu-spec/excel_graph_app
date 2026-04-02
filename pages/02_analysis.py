import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ページ設定
st.set_page_config(page_title="📊 分析ページ", layout="wide")
st.title("📊 分析ページ")

# 共有関数の再定義（app.py 互換）
def parse_datetime_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype="datetime64[ns]")
    if np.issubdtype(s.dtype, np.datetime64):
        try:
            return s.dt.tz_localize(None)
        except Exception:
            return s
    if np.issubdtype(s.dtype, np.number):
        try:
            return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce")

def ensure_numeric(s: pd.Series, fill=0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def aggregate_timeseries(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
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
    grouped = _df.resample(freq).agg({"生産済": "sum", "生産時間[分]": "sum", "基準時間[分]": "sum"})
    grouped["能率[%]"] = np.where(grouped["生産時間[分]"] > 0, (grouped["基準時間[分]"] / grouped["生産時間[分]"]) * 100, np.nan)
    grouped["工数"] = np.where(grouped["生産済"] > 0, grouped["生産時間[分]"] / grouped["生産済"], np.nan)
    grouped = grouped.reset_index().rename(columns={date_col: "日付"})
    if "日付" in grouped.columns:
        grouped["日付"] = pd.to_datetime(grouped["日付"], errors="coerce")
    return grouped

def alt_dual_axis_chart(agg_df: pd.DataFrame, title: str, show_items: dict = None, height=500, show_legend=True):
    if show_items is None:
        show_items = {"生産済": True, "生産時間[分]": True, "基準時間[分]": True, "工数": True, "能率[%]": True}
    if agg_df.empty:
        return go.Figure().add_annotation(text="データなし", showarrow=False)
    
    _df = agg_df.copy()
    if "日付" in _df.columns:
        _df["日付"] = pd.to_datetime(_df["日付"], errors='coerce')
    _df = _df.replace([np.inf, -np.inf], np.nan)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_configs = [("生産済", '#4472C4', 0.7), ("生産時間[分]", '#70AD47', 0.6), ("基準時間[分]", '#FFC000', 0.6)]
    for item_name, color, opacity in bar_configs:
        if show_items.get(item_name, True) and item_name in _df.columns:
            fig.add_trace(go.Bar(x=_df["日付"], y=_df[item_name], name=item_name, marker_color=color, opacity=opacity, yaxis='y'), secondary_y=False)
    
    if show_items.get("工数", True) and "工数" in _df.columns:
        fig.add_trace(go.Scatter(x=_df["日付"], y=_df["工数"], name="工数", mode='lines+markers', line=dict(color='#F39C12', width=3), connectgaps=True, yaxis='y2'), secondary_y=True)
    
    if show_items.get("能率[%]", True) and "能率[%]" in _df.columns:
        fig.add_trace(go.Scatter(
            x=_df["日付"], 
            y=_df["能率[%]"], 
            name="能率[%]", 
            mode='lines+markers', 
            line=dict(color='#E74C3C', width=3), # 実線に変更
            connectgaps=True,
            yaxis='y3'
        ))

    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=105, y1=105, yref="y3", line=dict(color="red", width=2))
    
    fig.update_layout(
        title=title, height=height, showlegend=show_legend,
        xaxis=dict(title="日付", domain=[0, 0.88], tickformat="%m/%d"),
        yaxis=dict(title="生産済/時間", side='left'),
        yaxis2=dict(
            title="工数" if show_items.get("工数", True) else None, 
            side='right', 
            overlaying='y', 
            title_font=dict(color='#F39C12'), 
            tickfont=dict(color='#F39C12'),
            showticklabels=show_items.get("工数", True)
        ),
        yaxis3=dict(
            title="能率[%]" if height > 300 else None, 
            side='right', 
            overlaying='y', 
            anchor='free', 
            position=1.0, 
            title_font=dict(color='#E74C3C', size=10 if height <= 300 else 14), 
            tickfont=dict(color='#E74C3C', size=9 if height <= 300 else 12), 
            range=[0, 130]
        ),
        margin=dict(l=50, r=80 if height <= 300 else 100, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    return fig

# データ取得
if "data" not in st.session_state:
    st.warning("先にメインページでファイルをアップロードしてください。")
    st.stop()

# データのコピーと情報の取得
df_raw = st.session_state.get("data_full_calculated", st.session_state.get("data", pd.DataFrame())).copy()
gmap = st.session_state.get("gmap", {})
date_col = st.session_state.get("date_col", "日付")

if df_raw.empty:
    st.warning("データが空です。")
    st.stop()

# サイドバーUI
analysis_mode = st.sidebar.radio(
    "分析モード選択",
    ["個別グラフ分析（月間）", "項目別サマリー（月間）", "取引先別生産時間分析"]
)

# 分析対象月の選択（動的）
df_raw[date_col] = parse_datetime_series(df_raw[date_col])
all_dates = df_raw[date_col].dropna()
if all_dates.empty:
    st.warning("日付データがありません。")
    st.stop()

# 利用可能な年月を抽出 (YYYY-MM形式)
available_months = sorted(list(set(all_dates.dt.strftime("%Y-%m"))), reverse=True)
selected_month_str = st.sidebar.selectbox("分析対象年月を選択", options=available_months, index=0)

# 対象月の範囲計算
sel_dt = datetime.strptime(selected_month_str, "%Y-%m")
display_month = f"{sel_dt.year}年{sel_dt.month}月"
month_start = pd.Timestamp(sel_dt).replace(day=1)
month_end = (month_start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)

# データフォルダ
df_selected = df_raw[(df_raw[date_col] >= month_start) & (df_raw[date_col] <= month_end)].copy()

if df_selected.empty:
    st.warning(f"選択された月（{selected_month_str}）のデータが見つかりません。")
    st.stop()

if analysis_mode == "個別グラフ分析（月間）":
    st.subheader(f"📊 {display_month} グラフ分析レポート")
    
    # グラフ名ごとの集計と能率平均の算出
    graph_data_list = []
    for gname, items in gmap.items():
        sub = df_selected[df_selected["出荷品番"].astype(str).str.strip().isin(items)].copy()
        if not sub.empty:
            agg = aggregate_timeseries(sub, date_col=date_col, freq="D")
            if not agg.empty:
                # 期間全体の加重平均能率
                s_time = sub["生産時間[分]"].sum()
                s_kijun = sub["基準時間[分]"].sum()
                avg_nouritsu = (s_kijun / s_time * 100) if s_time > 0 else np.nan
                graph_data_list.append({"gname": gname, "agg": agg, "avg_nouritsu": avg_nouritsu})
    
    # 能率平均昇順でソート
    graph_data_list = sorted(graph_data_list, key=lambda x: x["avg_nouritsu"])
    
    if not graph_data_list:
        st.warning("対象となるグラフデータがありません。")
    else:
        # 1行3列レイアウトで表示
        cols = st.columns(3)
        for i, item in enumerate(graph_data_list):
            with cols[i % 3]:
                fig = alt_dual_axis_chart(
                    item["agg"], 
                    title=f"{item['gname']} (能率: {item['avg_nouritsu']:.1f}%)", 
                    show_items={"工数": False, "能率[%]": True, "生産済": False, "生産時間[分]": False, "基準時間[分]": False},
                    height=250, 
                    show_legend=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")

elif analysis_mode == "項目別サマリー（月間）":
    st.subheader(f"📋 {display_month} 項目別サマリー集計")
    
    summary_list = []
    for gname, items in gmap.items():
        sub = df_selected[df_selected["出荷品番"].astype(str).str.strip().isin(items)].copy()
        if not sub.empty:
            s_seisan = sub["生産済"].sum()
            s_time = sub["生産時間[分]"].sum()
            s_kijun = sub["基準時間[分]"].sum()
            avg_kosuu = (s_time / s_seisan) if s_seisan > 0 else 0.0
            nouritsu = (s_kijun / s_time * 100) if s_time > 0 else 0.0
            
            summary_list.append({
                "グラフ名": gname,
                "生産済": s_seisan,
                "生産時間[分]": s_time,
                "工数平均": avg_kosuu,
                "能率[%]": nouritsu
            })
    
    if not summary_list:
        st.warning("対象となるサマリーデータがありません。")
    else:
        summary_df = pd.DataFrame(summary_list)
        # 能率昇順
        summary_df = summary_df.sort_values("能率[%]", ascending=True)
        
        # フォーマット適用
        st.dataframe(
            summary_df.style.format({
                "生産済": "{:,.0f}",
                "生産時間[分]": "{:,.1f}",
                "工数平均": "{:,.2f}",
                "能率[%]": "{:,.2f}"
            }),
            use_container_width=True,
            height=600 # 十分な高さを確保
        )

elif analysis_mode == "取引先別生産時間分析":
    st.subheader(f"📋 {display_month} 取引先別生産時間[分]集計")
    
    # 取引先列の存在チェック
    if "取引先" not in df_selected.columns:
        st.warning("データに '取引先' 列が見つかりません。")
    else:
        # 取引先ごとの生産時間[分]を集計
        client_summary = df_selected.groupby("取引先").agg({
            "生産時間[分]": "sum"
        }).reset_index()
        
        # 空白・NaNを除外
        client_summary = client_summary[
            client_summary["取引先"].astype(str).str.strip().ne("") & 
            client_summary["取引先"].astype(str).str.strip().ne("nan")
        ]
        
        # 生産時間[分]の降順でソート
        client_summary = client_summary.sort_values("生産時間[分]", ascending=False)
        
        if client_summary.empty:
            st.warning("対象となる取引先データがありません。")
        else:
            # 集計表の表示
            st.dataframe(
                client_summary.style.format({
                    "生産時間[分]": "{:,.1f}"
                }),
                use_container_width=True,
                height=400
            )
            
            # 横棒グラフの表示
            fig = go.Figure(go.Bar(
                x=client_summary["生産時間[分]"],
                y=client_summary["取引先"],
                orientation='h',
                marker_color='#70AD47'
            ))
            
            fig.update_layout(
                title=f"{display_month} 取引先別 生産時間[分]",
                xaxis_title="生産時間[分]",
                yaxis_title="取引先",
                height=max(400, len(client_summary) * 30),
                yaxis=dict(autorange="reversed")  # 上から順に並ぶ
            )
            
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
