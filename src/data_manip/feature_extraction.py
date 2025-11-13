import pandas as pd
import numpy as np
import os

def extract_features(df):
    """
    Feature engineering based on friend's EDA
    """
    df_feat = df.copy()
    
    df_feat['ydstosuccess'] = np.where(
        df_feat['down'] == 1, df_feat['ydstogo'] * 0.5,
        np.where(df_feat['down'] == 2, df_feat['ydstogo'] * 0.7, 
                df_feat['ydstogo'])
    )
    
    df_feat['previous_success'] = df_feat.groupby(['nflverse_game_id', 'drive'])['success'].shift(1)
    df_feat['previous_success'].fillna(1, inplace=True)
    
    df_feat['score_differential'] = df_feat['posteam_score'] - df_feat['defteam_score']
    df_feat['in_redzone'] = (df_feat['yardline_100'] <= 20).astype(int)
    
    if 'offense_personnel' in df_feat.columns:
        df_feat['has_te'] = df_feat['offense_personnel'].str.contains('TE', na=False).astype(int)
        df_feat['rb_count'] = df_feat['offense_personnel'].str.count('RB').fillna(0)
    
    return df_feat

def main():
    for year in range(2016, 2025):
        input_path = f'data/cleaned/pbp_{year}_cleaned.csv'
        output_path = f'data/interim/pbp_{year}_featured.csv'
        
        if os.path.exists(input_path):
            print(f"Adding features to {year} data...")
            df = pd.read_csv(input_path, low_memory=False)
            df_feat = extract_features(df)
            df_feat.to_csv(output_path, index=False)
            print(f"Saved featured data: {output_path}")

if __name__ == "__main__":
    main()