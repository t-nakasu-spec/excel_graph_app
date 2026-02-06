import pandas as pd
import os

def test_sino_loading():
    sino_path = "sinoP読み込み.xlsx"
    if not os.path.exists(sino_path):
        print(f"Skipping test: {sino_path} not found.")
        return

    print(f"Loading {sino_path}...")
    try:
        # Load sheet "33"
        df = pd.read_excel(sino_path, sheet_name="33")
        print("Columns found:", df.columns.tolist())
        
        # Check for '条件' column
        if "条件" not in df.columns:
            print("WARNING: '条件' column not found in sheet '33'. Available columns:", df.columns.tolist())
        else:
            print("'条件' column found.")
            print("Sample values in '条件':", df["条件"].dropna().unique()[:5])

        if "出荷品番" in df.columns:
             print("'出荷品番' column found.")
        else:
             print("WARNING: '出荷品番' column not found.")

    except Exception as e:
        print(f"Error loading excel: {e}")

if __name__ == "__main__":
    test_sino_loading()
