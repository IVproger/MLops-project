from pathlib import Path

import great_expectations as gx
from great_expectations.datasource.fluent import PandasDatasource


def load_context_and_sample_data(gx_root_dir: str, sample_data_path: str):
    """
    Load Great Expectations context, and add sample asset as data source.
    """
    context = gx.get_context(project_root_dir=gx_root_dir, mode="file")
    ds: PandasDatasource = context.sources.add_or_update_pandas(name="sample_data")
    da = ds.add_csv_asset(
        name="sample_file",
        filepath_or_buffer=Path(sample_data_path),
    )
    return context, da
