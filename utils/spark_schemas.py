from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def get_track_schema():
    return StructType([
        StructField("playlists", ArrayType(
            StructType([
                StructField("pid", IntegerType()),
                StructField("tracks", ArrayType(
                    StructType([
                        StructField("track_uri", StringType()),
                        StructField("artist_name", StringType()),
                        StructField("track_name", StringType()),
                        StructField("album_name", StringType()),
                        StructField("album_uri", StringType()),
                        StructField("artist_uri", StringType()),
                    ])
                ))
            ])
        ))
    ])
