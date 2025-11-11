import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['FIRST_ORDER_DATE'] = pd.to_datetime(df['FIRST_ORDER_DATE'])
    df['LATEST_ORDER_DATE'] = pd.to_datetime(df['LATEST_ORDER_DATE'])
    df['TENURE_IN_COMPLETE_DAYS'] = (df['LATEST_ORDER_DATE'] - df['FIRST_ORDER_DATE']).dt.days
    return df

def create_target(df):
    df['is_bad_status'] = (df['STATUS'].isin(['suspended', 'canceled'])).astype(int)
    return df

def feature_engineering(df):
    df['GMV_LEVEL_NUM'] = df['GMV_10K_LEVEL_VOICE_PLATFORM'].str.extract(r'level_(\d+)').astype(int)
    df['mistakes_per_order'] = df['TOTAL_MISTAKES_COUNT'] / (df['ORDER_COUNT_VOICE_PLATFORM'] + 1)
    df['hours_per_tenure'] = df['WEEKLY_SERVICE_HOURS'] / (df['TENURE_IN_COMPLETE_DAYS'] + 1)
    df['zip_region'] = df['ZIP_CODE'].astype(str).str[:1].astype(int)

    # One-hot encoding
    cat_cols = ['CUISINE_TYPE', 'SERVICE_TYPE']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df, encoded_df], axis=1)

    # Tenure bins
    df['tenure_bin'] = pd.cut(df['TENURE_IN_COMPLETE_DAYS'], bins=[0, 365, 730, np.inf], labels=['short', 'medium', 'long'])
    df = pd.get_dummies(df, columns=['tenure_bin'], prefix='tenure_bin')
    df['is_short_tenure'] = (df['TENURE_IN_COMPLETE_DAYS'] < 365).astype(int)

    return df, encoder

def get_features_and_target(df):
    feature_cols = [
        'TENURE_IN_COMPLETE_DAYS', 'SERVICE_RATE_PERCENT', 'ORDER_COUNT_VOICE_PLATFORM',
        'BUSINESS_PHONE_COUNT', 'WEEKLY_SERVICE_HOURS', 'TOTAL_MISTAKES_COUNT',
        'AVG_CALL_DURATION_SECONDS', 'GMV_LEVEL_NUM', 'mistakes_per_order',
        'hours_per_tenure', 'zip_region', 'is_short_tenure'
    ]
    cat_encoded = [col for col in df.columns if col.startswith(('CUISINE_TYPE_', 'SERVICE_TYPE_', 'tenure_bin_'))]
    feature_cols += cat_encoded

    X = df[feature_cols].fillna(0)
    y = df['is_bad_status']
    return X, y
