import pickle
import numpy as np
from pprint import pprint

def analyze_pickle_structure(data, level=0, max_items=3):
    prefix = "  " * level
    
    if isinstance(data, dict):
        print(f"{prefix}Dictionary with {len(data)} keys:")
        for key in list(data.keys())[:max_items]:
            print(f"{prefix}- {key}:")
            if isinstance(data[key], (dict, list, np.ndarray)):
                analyze_pickle_structure(data[key], level + 1)
            else:
                print(f"{prefix}  Value type: {type(data[key])}")
    
    elif isinstance(data, list):
        print(f"{prefix}List with {len(data)} items:")
        for item in data[:max_items]:
            analyze_pickle_structure(item, level + 1)
    
    elif isinstance(data, np.ndarray):
        print(f"{prefix}NumPy Array - Shape: {data.shape}, dtype: {data.dtype}")
        if len(data) > 0 and level < 2:
            print(f"{prefix}First element: {data[0]}")
    
    else:
        print(f"{prefix}Type: {type(data)}")

if __name__ == "__main__":
    filename = "traj_100_v2.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    print("\nPickle File Structure Analysis:")
    print("===============================")
    analyze_pickle_structure(data)