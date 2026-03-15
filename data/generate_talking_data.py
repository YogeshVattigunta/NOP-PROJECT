import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_talking_data(num_rows=2_000_000, output_path='e:/NOP Assignment/fraud_optimizer_project/data/train.csv'):
    print(f"Generating {num_rows} rows of synthetic TalkingData...")
    np.random.seed(42)
    
    # TalkingData Schema: ip, app, device, os, channel, click_time, attributed_time, is_attributed
    
    data = {
        'ip': np.random.randint(1, 100000, num_rows),
        'app': np.random.randint(1, 100, num_rows),
        'device': np.random.randint(1, 10, num_rows),
        'os': np.random.randint(1, 50, num_rows),
        'channel': np.random.randint(1, 500, num_rows),
    }
    
    # Generate click_time over a few days
    start_date = datetime(2017, 11, 6)
    data['click_time'] = [start_date + timedelta(seconds=np.random.randint(0, 86400*4)) for _ in range(num_rows)]
    
    # ~0.25% fraud rate
    data['is_attributed'] = np.random.choice([0, 1], size=num_rows, p=[0.9975, 0.0025])
    
    # attributed_time is only for is_attributed == 1
    data['attributed_time'] = [
        (ct + timedelta(seconds=np.random.randint(1, 3600))).strftime('%Y-%m-%d %H:%M:%S') 
        if ia == 1 else '' 
        for ct, ia in zip(data['click_time'], data['is_attributed'])
    ]
    
    data['click_time'] = [ct.strftime('%Y-%m-%d %H:%M:%S') for ct in data['click_time']]
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_talking_data()
