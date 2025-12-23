import pandas as pd
import numpy as np
import os
import glob
from scipy.io import arff

DATA_DIR = "data"

def inspect_data():
    print("Inspecting ARFF files...")
    arff_files = glob.glob(os.path.join(DATA_DIR, "**/*.arff"), recursive=True)
    
    if not arff_files:
        print("No ARFF files found.")
        return

    print(f"Found {len(arff_files)} files.")
    
    labels = []
    for file in arff_files:
        try:
            data, meta = arff.loadarff(file)
            df = pd.DataFrame(data)
            
            # Decode byte strings
            for col in df.select_dtypes([object]):
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            
            # Check for label column
            possible_labels = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
            if possible_labels:
                label_col = possible_labels[0]
                unique_labels = df[label_col].unique()
                print(f"File: {os.path.basename(file)} | Shape: {df.shape} | Label Col: {label_col} | Unique Labels: {unique_labels}")
                labels.extend(df[label_col].tolist())
            else:
                print(f"File: {os.path.basename(file)} | Label column NOT found.")
                
        except Exception as e:
            print(f"Error reading {os.path.basename(file)}: {e}")

    if labels:
        print("\nOverall Label Distribution:")
        print(pd.Series(labels).value_counts())

if __name__ == "__main__":
    inspect_data()

