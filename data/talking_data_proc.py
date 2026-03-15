import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def process_talking_data(input_path='e:/NOP Assignment/fraud_optimizer_project/data/train.csv'):
    print(f"Processing data from {input_path}...")
    
    # Load 2M rows (assuming generate_talking_data already created it)
    df = pd.read_csv(input_path)
    
    print("Feature engineering...")
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['click_hour'] = df['click_time'].dt.hour
    df['click_day'] = df['click_time'].dt.day
    df['click_dayofweek'] = df['click_time'].dt.dayofweek
    
    # Group counts (Frequency encoding)
    df['ip_count'] = df.groupby('ip')['ip'].transform('count')
    
    # Combined group counts
    df['app_channel_count'] = df.groupby(['app', 'channel'])['app'].transform('count')
    df['ip_app_count'] = df.groupby(['ip', 'app'])['ip'].transform('count')
    df['ip_app_os_count'] = df.groupby(['ip', 'app', 'os'])['ip'].transform('count')
    
    # Drop unused columns
    # click_time, attributed_time are not needed for training
    # attributed_time might be missing or empty for is_attributed=0
    df = df.drop(['click_time', 'attributed_time'], axis=1, errors='ignore')
    
    # Final features (10 total as requested)
    # ip, app, device, os, channel, click_hour, click_day, click_dayofweek, ip_count, app_channel_count
    # Note: ip_app_count etc can be included if desired, but user specifically asked for these 10:
    target_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_hour', 'click_day', 
                   'click_dayofweek', 'ip_count', 'app_channel_count']
    
    X = df[target_cols]
    y = df['is_attributed']
    
    print(f"X shape: {X.shape}, y distribution: {np.bincount(y)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save artifacts
    output_dir = 'e:/NOP Assignment/fraud_optimizer_project/data'
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(target_cols, f)
        
    print("Preprocessed data and scaler saved.")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    process_talking_data()
