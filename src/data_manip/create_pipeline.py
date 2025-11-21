import numpy as np
import pandas as pd
import yaml
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

def get_config(config):
    in_file = config['input_dir']+config['file']
    out = config['output_dir']
    model_type = config['model_type']

    keep = config['features']['keep']
    drop = config['features']['drop']
    target = config['features']['drop']
    categorical = config['features']['categorical']
    normalize = config['features']['normalize']

    return [in_file, out, model_type], [keep, drop, target, categorical, normalize]

def prep(keep, drop, target, categorical, normalize, dataframe, mdl_type):
    df = dataframe.copy()

    df = df[keep]

    if mdl_type.lower() == 'log' or mdl_type.lower() == 'gd':
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe_matrix = ohe.fit_transform(df[categorical])
        ohe_cols = ohe.get_feature_names_out(categorical)
        ohe_df = pd.DataFrame(ohe_matrix, columns=ohe_cols)

        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(df[normalize])
        scaled_df = pd.DataFrame(scaled_matrix, columns=normalize)

        other_cols = df.drop(columns = categorical+normalize)

        df = pd.concat([other_cols, scaled_df, ohe_df], axis=1)
    elif mdl_type.lower() == 'rdf':
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe_matrix = ohe.fit_transform(df[categorical])
        ohe_cols = ohe.get_feature_names_out(categorical)
        ohe_df = pd.DataFrame(ohe_matrix, columns=ohe_cols)

        other_cols = df.drop(columns = categorical)

        df = pd.concat([other_cols, ohe_df], axis=1)
    else:
        print(f'❌ INCORRECT MODEL INPUT: {mdl_type} - must be log, gd, or rdf')
        sys.exit()

    train = df[df['season'] < 2022]
    test = df[df['season'] > 2021]

    train = train.drop(columns = drop)
    test = test.drop(columns = drop)

    return train, test


if __name__ == "__main__":
    try:
        cfg_path = 'cfg/pipeline.yml'
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except:
        print('❌ Unable to load config, please check config path')
        print(f'Current path: {cfg_path}')
        sys.exit()

    files, params = get_config(cfg)

    try:
        in_file = files[0]
        print(f"📂 Loading: {in_file}")
        df = pd.read_csv(in_file)
        print("✅ DataFrame loaded!")
    except:
        print('❌ Unable to load file, please try again')
        sys.exit()

    model_type = files[2]
    train, test = prep(*params, df, model_type)
    
    try:
        out = files[1]
        print(f"📂 Saving Training Set to {out}")
        train.to_csv(out+model_type+"_train.csv", index=False)
        print(f"💾 Saved {model_type}_train.csv to: {out}")
        print(f"📂 Saving Testing Set to {out}")
        test.to_csv(out+model_type+"_test.csv", index=False)
        print(f"💾 Saved {model_type}_test.csv to: {out}")
    except:
        print("❌ Unable to save CSV to output path specified")

    