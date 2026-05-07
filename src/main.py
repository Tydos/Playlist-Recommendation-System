from src.utils.logging import get_logger
from src.utils.config import load_config
from src.track_etl import run_full_etl

from src.utils.benchmark import benchmark

logger = get_logger("train")
config = load_config()
DATASET_PATH = config.get("dataset_path", "dataset/data")
PROCESSED_DATA_PATH = config.get("processed_data_path", "processed_tracks.parquet")
NUM_RECORDS = config.get("num_records")

if __name__ == "__main__":
    success, duration = benchmark(run_full_etl, DATASET_PATH, PROCESSED_DATA_PATH, NUM_RECORDS)
    if success:
        logger.info(f"Full ETL pipeline completed successfully in {duration:.2f} seconds")
    else:
        logger.error(f"Full ETL pipeline failed after {duration:.2f} seconds")