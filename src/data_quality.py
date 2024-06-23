from pathlib import Path

import great_expectations as gx
from great_expectations.data_context import FileDataContext
from great_expectations.datasource.fluent import PandasDatasource, BatchRequest
from great_expectations.validator.validator import Validator


def load_context_and_sample_data(gx_root_dir: str, sample_data_path: str):
    """
    Load Great Expectations context, and add sample asset as data source.
    """
    context = gx.get_context(project_root_dir=gx_root_dir, mode="file")
    context.add_or_update_expectation_suite("sample_validation")
    ds: PandasDatasource = context.sources.add_or_update_pandas(name="sample_data")
    da = ds.add_csv_asset(
        name="sample_file",
        filepath_or_buffer=Path(sample_data_path),
    )
    return context, da


def define_expectations(context: FileDataContext, batch_request: BatchRequest) -> Validator:
    """
    Define expectations for the data.
    """
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="sample_validation",
    )

    validator.expect_column_values_to_not_be_null(column="FlightDate", meta={"dimension": "Completeness"})
    validator.expect_column_values_to_be_between(column="FlightDate", min_value="2022-01-01", max_value="2022-12-31", meta={"dimension": "Timelessness"})
    validator.expect_column_values_to_match_regex(column="FlightDate", regex="^\d{4}-\d{2}-\d{2}$", meta={"dimension": "Validity"})

    validator.expect_column_values_to_not_be_null(column="Operating_Airline", meta={"dimension": "Completeness"})
    validator.expect_column_values_to_match_regex(column="Operating_Airline", regex="^[A-Z0-9]{2}$", meta={"dimension": "Validity"})
    validator.expect_column_unique_value_count_to_be_between(column="Operating_Airline", min_value=1, max_value=100, meta={"dimension": "Uniqueness"})

    validator.expect_column_values_to_not_be_null(column="OriginAirportID", meta={"dimension": "Completeness"})
    validator.expect_column_unique_value_count_to_be_between(column="OriginAirportID", min_value=1, max_value=1000000, meta={"dimension": "Uniqueness"})
    validator.expect_column_unique_value_count_to_be_between(column="OriginAirportID", min_value=1, max_value=10000, meta={"dimension": "Uniqueness"})

    validator.expect_column_values_to_not_be_null(column="DestAirportID", meta={"dimension": "Completeness"})
    validator.expect_column_unique_value_count_to_be_between(column="DestAirportID", min_value=1, max_value=1000000, meta={"dimension": "Uniqueness"})
    validator.expect_column_unique_value_count_to_be_between(column="DestAirportID", min_value=1, max_value=10000, meta={"dimension": "Uniqueness"})

    validator.expect_column_values_to_not_be_null(column="Cancelled", meta={"dimension": "Completeness"})
    validator.expect_column_values_to_be_in_set(column="Cancelled", value_set=[False, True], meta={"dimension": "Validity"})

    validator.expect_column_values_to_not_be_null(column="CRSDepTime", meta={"dimension": "Completeness"})
    validator.expect_column_max_to_be_between(column="CRSDepTime", min_value=0, max_value=2400, meta={"dimension": "Accuracy"})

    validator.expect_column_values_to_not_be_null(column="CRSArrTime", meta={"dimension": "Completeness"})
    validator.expect_column_max_to_be_between(column="CRSArrTime", min_value=0, max_value=2400, meta={"dimension": "Accuracy"})

    validator.expect_column_values_to_match_regex(column="Tail_Number", regex="^[A-Z0-9]{5,6}$", meta={"dimension": "Validity"})
    validator.expect_column_unique_value_count_to_be_between(column="Tail_Number", min_value=1, max_value=10000, meta={"dimension": "Uniqueness"})

    validator.expect_column_values_to_not_be_null(column="CRSElapsedTime", meta={"dimension": "Completeness"})
    validator.expect_column_max_to_be_between(column="CRSElapsedTime", min_value=0, max_value=8000, meta={"dimension": "Consistency"})

    validator.expect_column_max_to_be_between(column="DepDelay", min_value=-1000, max_value=10000, meta={"dimension": "Consistency"})

    validator.expect_column_max_to_be_between(column="ActualElapsedTime", min_value=0, max_value=8000, meta={"dimension": "Consistency"})

    validator.expect_column_values_to_not_be_null(column="Distance", meta={"dimension": "Completeness"})
    validator.expect_column_values_to_be_between(column="Distance", min_value=0, max_value=8000, meta={"dimension": "Consistency"})

    return validator
