# prediction/sample_data/generate_samples.py
import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_transactions(n_samples: int = 50) -> pd.DataFrame:
    """Generate realistic sample transaction data"""
    np.random.seed(42)
    
    samples = []
    for i in range(n_samples):
        transaction = {
            'TransactionID': f"TXN_{10000 + i}",
            'Time': np.random.uniform(0, 172000),  # Similar to your training data
            'Amount': np.random.uniform(1, 5000),
        }
        
        # Add V1-V28 features with realistic distributions
        for j in range(1, 29):
            # Create some realistic patterns
            if j in [1, 2, 3, 4]:
                # These might be more important features - wider range
                transaction[f'V{j}'] = np.random.normal(0, 2)
            elif j in [14, 15, 16, 17]:
                # These might be moderate importance
                transaction[f'V{j}'] = np.random.normal(0, 1.5)
            else:
                # Less important features
                transaction[f'V{j}'] = np.random.normal(0, 1)
        
        samples.append(transaction)
    
    df = pd.DataFrame(samples)
    
    # Reorder columns to have TransactionID first, then Time, Amount, then V1-V28
    columns_order = ['TransactionID', 'Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    df = df[columns_order]
    
    return df

def save_sample_data():
    """Generate and save sample data for testing"""
    samples_dir = Path("prediction/sample_data")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    sample_df = generate_sample_transactions(50)
    
    # Save to CSV
    sample_path = samples_dir / "sample_transactions.csv"
    sample_df.to_csv(sample_path, index=False)
    
    # Create a smaller test batch
    test_batch = generate_sample_transactions(10)
    test_batch_path = samples_dir / "test_batch.csv"
    test_batch.to_csv(test_batch_path, index=False)
    
    print(f"✅ Generated sample data:")
    print(f"   - {sample_path} ({len(sample_df)} transactions)")
    print(f"   - {test_batch_path} ({len(test_batch)} transactions)")
    
    return sample_df

if __name__ == "__main__":
    save_sample_data()