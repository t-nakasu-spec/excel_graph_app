import pandas as pd
import os

def normalize_graph_key(val) -> str:
    """グラフ番号の表記ゆれを統一（例: 24.0 → 24）"""
    if pd.isna(val):
        return ""
    if isinstance(val, (int, float)):
         # Check if it's integer-like
         try:
             if float(val).is_integer():
                 return str(int(val))
         except:
             pass
         return str(val)
    
    s = str(val).strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else s
    except Exception:
        return s

def verify_logic():
    sino_path = "sinoP読み込み.xlsx"
    if not os.path.exists(sino_path):
        print("Skipping: file not found")
        return

    # Simulate basic state
    cond_raw = pd.DataFrame() # Empty condition sheet
    data_raw = pd.DataFrame() # Empty main data
    graph_name_map = {} # Empty map for now, so "33" stays "33"
    gmap = {}

    sino_mapping_list = []

    print("--- Starting Logic Simulation ---")
    try:
        sino_df = pd.read_excel(sino_path, sheet_name="33")
        if not sino_df.empty:
            sino_df.columns = [str(c).strip() for c in sino_df.columns]
            
            if "条件" not in sino_df.columns:
                print("'条件' column missing, adding 33.")
                sino_df["条件"] = 33
            
            data_raw = pd.concat([data_raw, sino_df], ignore_index=True)

            if "出荷品番" in sino_df.columns:
                print(f"Extraction items... {len(sino_df)} rows")
                for _, r in sino_df.iterrows():
                    sino_mapping_list.append((r["出荷品番"], r["条件"]))
    except Exception as e:
        print(f"Error: {e}")
        return

    # Simulate gmap update
    for item_val, g_val in sino_mapping_list:
        item = str(item_val).strip()
        if not item or item.lower() == "nan": continue
        
        g_raw = normalize_graph_key(g_val)
        gname = graph_name_map.get(g_raw, g_raw)
        gmap.setdefault(gname, set()).add(item)
    
    print(f"Total graph groups: {len(gmap)}")
    if "33" in gmap:
        print(f"Graph '33' has {len(gmap['33'])} items.")
        print("Sample items:", list(gmap['33'])[:3])
    else:
        print("Graph '33' NOT found in gmap!")

if __name__ == "__main__":
    verify_logic()
