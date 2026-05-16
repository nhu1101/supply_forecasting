import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import sys 
import json 


## Add lag, avm features: 
def feature_engineering(data):
    df = data.copy()

    # convert to datetime
    df['block_start_time'] = pd.to_datetime(df['block_start_time'])

    df['date'] = df['block_start_time'].dt.date

    # sort (VERY IMPORTANT for lag)
    df = df.sort_values('block_start_time')

    # ========================
    # 1. Time-based features
    # ========================

    df['month'] = df['block_start_time'].dt.month
    df['weekday'] = df['block_start_time'].dt.weekday   # Monday=0
    df['hour'] = df['block_start_time'].dt.hour

    # week of month
    df['week_of_month'] = df['block_start_time'].apply(
        lambda x: (x.day - 1) // 7 + 1
    )

    # ========================
    # 2. Lag features
    # ========================

    # IMPORTANT: group by date if each day is independent
    df['lag_1'] = df.groupby(['hex_id','vehicle_type'])['suppliers_online'].shift(1)

    df['lag_2'] = df.groupby(['hex_id','vehicle_type'])['suppliers_online'].shift(2)
    df['lag_3'] = df.groupby(['hex_id','vehicle_type'])['suppliers_online'].shift(3)
    df['lag_6'] = df.groupby(['hex_id','vehicle_type'])['suppliers_online'].shift(6)
    df['lag_144'] = df.groupby(['hex_id','vehicle_type'])['suppliers_online'].shift(144)  
    df['lag_143'] = df.groupby(['hex_id','vehicle_type'])['suppliers_online'].shift(143) 


    df['diff_2'] = df['lag_1'] - df['lag_2']

    df['ma_3'] = (
        df.groupby(['hex_id','vehicle_type'])['suppliers_online']
        .transform(lambda x: x.shift(1).rolling(3).mean())
    )

    df['ma_6'] = (
        df.groupby(['hex_id','vehicle_type'])['suppliers_online']
        .transform(lambda x: x.shift(1).rolling(6).mean())
    )

    #detect abnormal spike:
    df['ratio_to_mean'] = df['lag_1'] / (df['ma_6'] + 1e-5)

    df['std_3'] = df.groupby(['hex_id','vehicle_type'])['suppliers_online']\
                    .transform(lambda x: x.shift(1).rolling(3).std())

    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7)
    df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)


    # define peak condition
    df['is_peak'] = (
        ((df['hour'] >= 8) & (df['hour'] < 11)) | #add 8AM for capture orders in pre-preak hour 
        ((df['hour'] >= 13) & (df['hour'] < 15))
    ).astype(int)

    # df_encode  = pd.get_dummies(df, columns=['vehicle_type'])
    # df_encode['vehicle_type'] = df['vehicle_type']

    return df


# make sure sorted
def split_train_test_time(data, ratio, peak):

    df = data.copy()

# df_train = df[['order_date','total_demands','month','weekday','hour','week_of_month','lag_1','lag_2','lag_3','ma_3','std_3','is_peak']]
    df_train = df[df['is_peak']==peak].sort_values(['block_start_time'])

    # get unique dates
    dates = df_train['date'].sort_values().unique()

    # 80% train, 20% test
    split_idx = int(len(dates) * ratio)

    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    # split dataset
    train_df = df_train[df_train['date'].isin(train_dates)]
    test_df = df_train[df_train['date'].isin(test_dates)]

    print(train_df['date'].min(), "→", train_df['date'].max())
    print(test_df['date'].min(), "→", test_df['date'].max())

    return train_df, test_df


def fill_time_template(
    df,
    min_datetime,
    max_datetime,
    interval='10min'
):

    data = df.copy()
    # Ensure datetime type
    data['block_start_time'] = pd.to_datetime(data['block_start_time'])

    # Create full 10-minute interval range
    time_range = pd.DataFrame({
        'block_start_time': pd.date_range(
            start=min_datetime,
            end=max_datetime,
            freq=interval
        )
    })

    # Create unique combinations
    group_template = data[['hex_id', 'vehicle_type']].drop_duplicates()

    # Cross join
    group_template['key'] = 1
    time_range['key'] = 1

    template = pd.merge(group_template,time_range,on='key').drop('key', axis=1)

    # Merge with original data
    data_merge = pd.merge(template, data,
        on=[
            'hex_id',
            'vehicle_type',
            'block_start_time'
        ],
        how='left'
    )

    # Fill missing values
    fill_cols = ['suppliers_online']

    for col in fill_cols:
        if col in data_merge.columns:
            data_merge[col] = data_merge[col].fillna(0)

    return data_merge


def impute_missing_values(data):
    df = data.copy()
    FEATURES = [
    'lag_1',
    'lag_2',
    'lag_3',
    'lag_6',
    'lag_144',
    'lag_143',
    'diff_2',
    'ma_3',
    'ma_6',
    'hour_sin',
    'hour_cos',
    'weekday_sin',
    'weekday_cos', 
    ]

    df = df.sort_values(['hex_id', 'block_start_time'])

    df[FEATURES] = df.groupby('hex_id')[FEATURES].transform(lambda x: x.ffill())

    df_final = df.dropna().reset_index(drop=True)

    return df_final 