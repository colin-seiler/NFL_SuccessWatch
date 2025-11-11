import numpy as np
import pandas as pd
import glob
import sys
from datetime import datetime


import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)


IN_PATH = "data/raw/pbp_data/"
OUT_PATH = "data/cleaned/pbp/"
PBP = "/*.csv"

def clean_data(off_personnel=True, add_features=True, off_form=True, off_form_personnel=True, wp_trim=0, input_path=IN_PATH+PBP):
    try:
        dfs = []
        for file in sorted(glob.glob(input_path)):
            print(f"📂 Loading: {file}")
            temp = pd.read_csv(file, low_memory=False)
            dfs.append(temp)

        df = pd.concat(dfs, ignore_index=True)
        print("✅ All files loaded!")
        shape = df.shape
        print(f'\nDataFrame Size: {shape} \n')
    except:
        print('❌ Files not loaded, please try again')
        sys.exit()

    #DROP NA DATA
    formation = df['offense_formation'].notna()
    personnel = df['offense_personnel'].notna()
    no_play = df['play_type'] != 'no_play'
    kneel = df['qb_kneel'] == 0
    spike = df['qb_spike'] == 0
    pt2 = df['down'].notna()

    df = df[pt2 & formation & personnel & no_play & kneel & spike].drop(columns=['qb_kneel','qb_spike'])
    df = df.copy()

    # Drop columns with >95% missing data
    missing_ratio = df.isna().mean()
    df = df.drop(columns=missing_ratio[missing_ratio > 0.8].index)
    df = df.copy()

    useless_keywords = ['players','ngs','air','time','pressure','route',
                        'old','total_home','total_away','yac','safety',
                        'timeout','opp','extra_point','two_point','vegas',
                        'kickoff','punt','incomplete','touchback','fumble',
                        'interception','tackle','receiver','rusher','passer',
                        'first','special','jersey','drive_','coach','aborted',
                        'fantasy','xyac','xpass','oe','out','qb','name','desc',
                        'deleted','clock','series','cp','replay','lateral',
                        'touchdown','team_score','after','coverage','man_zone',
                        'pass_','rush_','_post','sack','return','_nfl']
    
    drop_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in useless_keywords)]
    timestamp = datetime.now().strftime("%m_%d")
    print(f"🧹 Dropping {len(drop_cols)} columns!\n💾 Saved to logs/{timestamp}_dropped_cols.txt")

    try:
        with open(f"logs/{timestamp}_dropped_cols.txt", 'w') as f:
            for col in drop_cols:
                f.write(col+"\n")
    except:
        print('Unable to save dropped columns in log')

    df = df.drop(columns=drop_cols)
    df = df.copy()

    shape = df.shape
    print(f'\nNew DataFrame Size: {shape} \n')

    
    try:
        timestamp = datetime.now().strftime("%m_%d")
        out_file = f"clean_{timestamp}.csv"
        output_path = OUT_PATH+out_file
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {out_file} to: {OUT_PATH}")
    except:
        print("Unable to save CSV to output path specified")

if __name__ == "__main__":
    clean_data()