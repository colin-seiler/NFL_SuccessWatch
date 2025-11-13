import pandas as pd
import numpy as np
import os

def clean_data(df):
    df_clean = df.copy()
    
    no_play = df_clean['play_type'] != 'no_play'
    kneel = df_clean.get('qb_kneel', 0) == 0
    spike = df_clean.get('qb_spike', 0) == 0
    df_clean = df_clean[no_play & kneel & spike]
    
    df_clean = df_clean[df_clean['down'].notna()]
    
    df_clean = df_clean.dropna(subset=['down', 'ydstogo', 'yardline_100'])
    
    return df_clean

def main():
    for year in range(2016, 2025):
        input_path = f'data/raw/pbp_data/pbp_{year}.csv'
        output_path = f'data/cleaned/pbp_{year}_cleaned.csv'
        
        if os.path.exists(input_path):
            print(f"Cleaning {year} data...")
            df = pd.read_csv(input_path, low_memory=False)
            df_clean = clean_data(df)
            df_clean.to_csv(output_path, index=False)
            print(f"Saved cleaned data: {output_path}")

if __name__ == "__main__":
    main()