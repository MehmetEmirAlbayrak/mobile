import pandas as pd
import numpy as np
import os
from src.config import DATA_DIR, LABEL_COLUMN

def generate_dummy_data(num_samples=1000):
    print(f"Generating {num_samples} dummy samples...")
    
    # Common ISCX features (subset)
    features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd Header Length', 'Bwd Header Length',
        'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
        'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 
        'Avg Bwd Segment Size'
    ]
    
    data = np.random.rand(num_samples, len(features)) * 1000
    
    # Create classes
    classes = ['VPN_Netflix', 'VPN_Skype', 'NonVPN_Youtube', 'NonVPN_Web', 'Chat']
    labels = np.random.choice(classes, size=num_samples)
    
    df = pd.DataFrame(data, columns=features)
    df[LABEL_COLUMN] = labels
    
    output_path = os.path.join(DATA_DIR, 'dummy_traffic_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dummy data saved to {output_path}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_dummy_data()

