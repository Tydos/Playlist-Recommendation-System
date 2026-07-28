import etl.run_etl as run_etl


def test_dataset_path_and_output_path_are_configured():
    assert run_etl.DATASET_PATH
    assert run_etl.OUTPUT_PATH
