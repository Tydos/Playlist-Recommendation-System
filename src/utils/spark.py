import os
from pathlib import Path
from pyspark.sql import SparkSession
from src.utils.config import load_config

# temp workaround for windows
def _windows_hadoop_jvm_opts() -> str:
    """Return extra JVM opts that point the driver at the bundled winutils."""
    hadoop_home = str(Path(__file__).parents[2] / "hadoop")
    hadoop_bin = str(Path(hadoop_home) / "bin")
    os.environ["HADOOP_HOME"] = hadoop_home
    # Use forward slashes — backslashes in -D values confuse the JVM arg parser.
    home_fwd = hadoop_home.replace("\\", "/")
    bin_fwd = hadoop_bin.replace("\\", "/")
    return f"-Dhadoop.home.dir={home_fwd} -Djava.library.path={bin_fwd}"

def spark_session() -> SparkSession:
    config = load_config()

    driver_memory = config.get("spark_driver_memory", "8g")
    shuffle_partitions = config.get("spark_shuffle_partitions", 8)
    parquet_block_size = config.get("spark_parquet_block_size", 32 * 1024 * 1024)

    builder = (
        SparkSession.builder
        .appName("TrackETL")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.parquet.block.size", parquet_block_size)
    )

    if os.name == "nt":
        builder = builder.config("spark.driver.extraJavaOptions", _windows_hadoop_jvm_opts())

    return builder.getOrCreate()