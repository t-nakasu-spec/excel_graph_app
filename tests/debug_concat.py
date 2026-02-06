import pandas as pd
import numpy as np

def debug_concat_logic():
    print("--- Debugging Concat Logic ---")
    
    # 1. Mock Data (Main)
    data_raw_mock = pd.DataFrame({
        "生産日": [pd.Timestamp("2025-04-01")],
        "出荷品番": ["ITEM-A"],
        "生産済": [100],
        "所要時間": [0.5], # Excel time
        "状態": [pd.Timestamp("2025-04-01")],
    })
    
    # 2. Mock SinoP Data (Based on sinoP読み込み.md)
    # The markdown shows columns: "生産日", "生産No", "生産担当", "取引先", "項目", "出荷品番", "品番担当", ...
    # "所要時間", "状態", "基準時間", "異常値" are present.
    # Note: "所要時間" in markdown seems to be float.
    
    sino_df_mock = pd.DataFrame({
        "生産日": [pd.Timestamp("2025-04-02")],
        "出荷品番": ["ITEM-B"],
        "生産済": [50],
        "所要時間": [0.2], 
        "状態": [pd.Timestamp("2025-04-02")],
        # "条件" is missing initially
    })
    
    print("Main columns:", data_raw_mock.columns.tolist())
    print("Sino columns:", sino_df_mock.columns.tolist())
    
    # Simulate Logic in app.py
    
    # Normalize columns (strip)
    sino_df_mock.columns = [str(c).strip() for c in sino_df_mock.columns]
    
    # Add '条件'
    if "条件" not in sino_df_mock.columns:
        sino_df_mock["条件"] = 33
        
    # Concat
    combined = pd.concat([data_raw_mock, sino_df_mock], ignore_index=True)
    
    print("\n--- Combined Data ---")
    print(combined)
    print("Len:", len(combined))
    
    # Check aggregation readiness
    # In app.py: data["生産済"] = ensure_numeric(...)
    
    def ensure_numeric(s, fill=0):
        return pd.to_numeric(s, errors="coerce").fillna(fill)
        
    combined["生産済"] = ensure_numeric(combined.get("生産済", pd.Series(dtype=float)), 0)
    
    print("\nTotal Seisan:", combined["生産済"].sum())
    
    # Check if column alignment failed (e.g. if one has "生産済" and other has "生産済 ")
    # But we stripped sino_df columns. Did we strip data_raw columns? 
    # In app.py, we don't explicitly strip data_raw columns before concat!
    # If the main excel has "生産済 " (with space), and sino has "生産済", they won't merge.
    
    # Let's inspect column names in actual implementation logic in app.py
    # data_raw comes from xl.parse.
    
    print("\nWarning: If 'data_raw' columns have spaces, concat might misalign.")

if __name__ == "__main__":
    debug_concat_logic()
