import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_data(data_url: str) -> pd.DataFrame:
    try:
        print("\n📌 Loading data from URL...")
        df = pd.read_csv(data_url)
        print("✅ Data loaded successfully!")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        print("\n📌 Preprocessing data...")
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        print("✅ Preprocessing completed!")
        return final_df
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        print("\n📌 Saving processed data...")

        # ✅ Go to project root (one level up from src/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ✅ Create data/raw folder inside project root
        data_dir = os.path.join(base_dir, data_path, "raw")
        os.makedirs(data_dir, exist_ok=True)

        # ✅ Save files
        train_data.to_csv(os.path.join(data_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_dir, "test.csv"), index=False)

        print(f"✅ Data saved at: {data_dir}")
    except Exception as e:
        print(f"❌ Error during saving data: {e}")
        raise


def main():
    try:
        print("\n🚀 Data Ingestion Pipeline Started")

        df = load_data('https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

        # ✅ Save to ML Pipeline/data/raw
        save_data(train_data, test_data, data_path='data')

        print("\n🎉 Pipeline Completed Successfully!")
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")


if __name__ == '__main__':
    main()
