import pandas as pd
import os

def load_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✅ Loaded data from {input_path} — shape: {df.shape}")
    return df

def preprocess_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Example preprocessing steps
    df = df.copy()
    
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert categorical values to numeric
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    
    # Encode diagnosis (target) as categorical codes
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].astype('category').cat.codes
    
    print("✅ Data preprocessing complete.")
    return df

def save_preprocessed_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved at: {output_path}")

def preprocess_data(df, label_column='diagnosis'):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Check class distribution before applying stratify
    value_counts = y.value_counts()
    if value_counts.min() < 2:
        print("⚠️ Warning: Some classes have fewer than 2 samples — disabling stratify.")
        stratify_option = None
    else:
        stratify_option = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )
    print("✅ Data split into train/test sets.")
    return X_train, X_test, y_train, y_test

def main():
    input_path = 'data/raw/iem_dataset.csv'
    output_path = 'data/processed/iem_processed.csv'

    try:
        df = load_data(input_path)
        df = preprocess_and_clean(df)
        save_preprocessed_data(df, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
