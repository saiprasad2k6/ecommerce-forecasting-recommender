from pathlib import Path

from ecommerce_forecasting_recommender.data.download import bootstrap_sample_raw_data

if __name__ == "__main__":
    bootstrap_sample_raw_data(Path("data/raw"))
