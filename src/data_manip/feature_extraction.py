import numpy as np
import pandas as pd
import glob
import sys
from datetime import datetime
import yaml

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)


def fix_personnel(dataframe):
    df = dataframe.copy()

    personnel_dict = {
        'OL':'OL',
        'C':'OL',
        'G':'OL',
        'T':'OL',
        'FB':'RB',
        'HB':'RB',
        'RB':'RB',
        'QB':'QB',
        'WR':'WR',
        'TE':'TE',
    }

    def parse_personnel(x):
        parsed = {'OL': 0, 'RB': 0, 'QB': 0, 'WR': 0, 'TE': 0, 'OTHER': 0}
        for part in x.split(','):
            try:
                count, pos = part.strip().split(' ')
                pos = personnel_dict.get(pos.strip(), 'OTHER')
                parsed[pos] += int(count)
            except ValueError:
                parsed['OTHER'] += 1
        if parsed['OL'] == 0:
            parsed['OL'] = 5
        if parsed['QB'] == 0:
            parsed['QB'] = 1
        return parsed

    df['personnel_dict'] = df['offense_personnel'].apply(parse_personnel)

    expanded = pd.DataFrame(df['personnel_dict'].tolist(), index=df.index).fillna(0)
    df = pd.concat([df, expanded], axis=1)
    df = df.copy()

    df['extra_lineman_flag'] = np.where(df['OL'] > 5, 1, 0)
    df['bigs'] = df['OL']+df['TE']-5

    df['personnel_num'] = df.apply(lambda x: f'{x['bigs']}{x['RB']}', axis=1)
    df = df.copy()

    return df

#Trim DataFrame by certain win_perc
def win_trim(dataframe, win_perc=0):
    df = dataframe.copy()
    df = df[(win_perc < df['wp']) & (df['wp'] < (1-win_perc))]
    df = df.copy()

#Add ydstoscucces
def yardstosuccess(dataframe):
    df = dataframe.copy()

    #Yards to success, a combination of down and yards
    df['ydstosuccess'] = np.where(df['down']==1, df['ydstogo']*.4, np.where(df['down']==2, df['ydstogo']*.6, df['ydstogo']))
    df = df.copy()
    
    return df

def prev_success(dataframe):
    df = dataframe.copy()
    #Define previous success based on previous plays success
    df['previous_success'] = df.groupby(['nflverse_game_id','fixed_drive'])['success'].shift(1)
    df['previous_success'] = df['previous_success'].fillna(0)
    df = df.copy()

    return df

def formation_personnel_success(dataframe):
    df = dataframe.copy()

    formation_map = {
        'I_FORM':'UNDER_CENTER',
        'UNDER CENTER':'UNDER_CENTER',
        'SINGLEBACK':'UNDER_CENTER',
        'JUMBO':'UNDER_CENTER',
        'PISTOL':'PISTOL',
        'EMPTY':'SHOTGUN',
        'SHOTGUN':'SHOTGUN',
        'WILDCAT':'SHOTGUN'
    }

    df['offense_formation'] = df['offense_formation'].map(formation_map)

    succ = (
        df.groupby(['offense_formation', 'personnel_num'])
        .agg(
            plays=('success', 'count'),
            fp_success=('success', 'mean'),
            fp_epa=('epa', 'mean')
        )
    .reset_index()
    )

    global_mean = df['success'].mean()
    succ['smoothed_fp_success'] = (
        (succ['plays'] * succ['fp_success'] + 500 * global_mean)
        / (succ['plays'] + 500)
    )

    df = df.merge(succ[['offense_formation','personnel_num','fp_success','smoothed_fp_success','fp_epa']], how='left', on=['offense_formation','personnel_num'])
    df = df.copy()

    return df

def formation_success(dataframe):
    df = dataframe.copy()

    f_success = (
        df.groupby('offense_formation')
        .agg(
            plays=('success', 'count'),
            f_success=('success', 'mean'),
            f_epa=('epa', 'mean')
        )
    .reset_index()
    )   

    global_success = df['success'].mean()

    f_success['smoothed_f_success'] = (
        (f_success['plays'] * f_success['f_success'] + 500 * global_success)
        / (f_success['plays'] + 500)
    )

    df = df.merge(f_success[['offense_formation','f_success','smoothed_f_success','f_epa']], how='left', on=['offense_formation'])
    df = df.copy()

    return df

def yardage_bins(dataframe, bin_size, bin_count):
    df = dataframe.copy()

    bin_count-=1
    yard_bins = list(np.arange(0, bin_count * bin_size + bin_size, bin_size))+[np.inf]
    yard_labels = [f'{str(yard_bins[i])}to{str(yard_bins[i+1])}' for i in range(len(yard_bins)-1)]

    df['yard_group'] = pd.cut(df['ydstosuccess'], bins=yard_bins, labels=yard_labels, right=False)
    return df

def engineer_features(cfg):
    toggles = cfg["features"]["toggles"]
    vals = cfg["features"]["values"]

    filename = cfg['file']
    input_dir = cfg['input_dir']
    output_dir = cfg['output_dir']
    log_dir = cfg['log_dir']

    bin_size = cfg['bin_size']
    bin_count = cfg['bin_count']

    read_file = input_dir+filename
    try:
        print(f"📂 Loading: {read_file}")
        df = pd.read_csv(read_file, dtype={'personnel_num': 'string'}, low_memory=False)
        print("✅ All files loaded!")
    except:
        print('❌ Files not loaded, please try again')
        sys.exit()

    if toggles.get("build_off", False):
        print(f'🏈 Building Offense Personnel Counts')
        df = fix_personnel(df)
        df = df.copy()
    if toggles.get("build_ydstosuccess", False):
        print(f'🏈 Building "ydstosuccess" Feature')
        df = yardstosuccess(df)
        df = df.copy()
    if toggles.get("build_fp_succ", False):
        print(f'🏈 Building Formation Personnel Success and EPA')
        df = formation_personnel_success(df)
        df = df.copy()
    if toggles.get("build_f_succ", False):
        print(f'🏈 Building Formation Success and EPA')
        df = formation_success(df)
        df = df.copy()
    if toggles.get("build_yardbins", False):
        print(f'🏈 Building Yard Bins')
        df = yardage_bins(df, bin_size, bin_count)
        df = df.copy()
    if toggles.get("build_prev", False):
        print(f'🏈 Building Previous Success')
        df = prev_success(df)
        df = df.copy()
    

    if vals.get("win_trim", 0):
        print(f'🏈 Trimming plays based on WP: {vals.get("win_trim")}')
        df = win_trim(df)
        df = df.copy()

    try:
        timestamp = datetime.now().strftime("%m_%d")
        out_file = f"featured_{timestamp}.csv"
        output_path = output_dir+out_file
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {out_file} to: {output_dir}")
    except:
        print("Unable to save CSV to output path specified")
    

if __name__ == "__main__":
    #CFG file is located at cfg/features.yml
    with open("cfg/features.yml") as f:
        cfg = yaml.safe_load(f)

    engineer_features(cfg)