from backend.utils.logging import get_logger
from backend.utils.config import load_config
from backend.etl.track_etl import run_full_etl
from backend.utils.benchmark import benchmark

logger = get_logger("run_etl")
config = load_config()
DATASET_PATH = config.get("dataset_path", "dataset/data")
OUTPUT_PATH = config.get("output_path", "output")
NUM_RECORDS = config.get("num_json")

if __name__ == "__main__":
    success, duration = benchmark(run_full_etl, DATASET_PATH, OUTPUT_PATH, NUM_RECORDS)
    if success:
        logger.info(f"Full ETL pipeline completed successfully in {duration:.2f} seconds")
    else:
        logger.error(f"Full ETL pipeline failed after {duration:.2f} seconds")
