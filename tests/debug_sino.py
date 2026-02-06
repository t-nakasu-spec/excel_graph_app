import pandas as pd
import os

def test_sino_loading_debug():
    sino_path = "sinoP読み込み.xlsx"
    if not os.path.exists(sino_path):
        print("File not found")
        return

    try:
        # Load header=0
        df = pd.read_excel(sino_path, sheet_name="33", header=0)
        print("--- Header 0 ---")
        print(df.columns)
        print(df.head(3))
        
        # Load header=1 just in case
        df2 = pd.read_excel(sino_path, sheet_name="33", header=1)
        print("\n--- Header 1 ---")
        print(df2.columns)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sino_loading_debug()
