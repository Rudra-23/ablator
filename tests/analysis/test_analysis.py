from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    PlotAnalysis,
    RunConfig,
    TrainConfig,
)
from ablator.analysis.results import Results
from ablator.config.mp import ParallelConfig, SearchSpace
from ablator.main.mp import ParallelTrainer

from ablator.analysis.plot.utils import parse_name_remap
from ablator.analysis.main import Analysis


def get_best(x: pd.DataFrame, task_type: str):
    if task_type == "regression":
        return x.sort_values("val_rmse", na_position="last").iloc[0]
    else:
        return x.sort_values("val_acc", na_position="first").iloc[-1]


@pytest.fixture()
def results(tmp_path: Path, wrapper, make_config, working_dir):
    return _results(tmp_path, wrapper, make_config, working_dir)


def _results(tmp_path: Path, wrapper, make_config, working_dir) -> Results:
    search_space = {
        "train_config.optimizer_config.arguments.lr": SearchSpace(
            value_range=[0, 19],
            value_type="float",
            n_bins=10,
        ),
        "model_config.mock_param": SearchSpace(
            categorical_values=list(range(10)),
        ),
    }
    config = make_config(tmp_path.joinpath("test_exp"), search_space=search_space)
    ablator = ParallelTrainer(wrapper=wrapper, run_config=config)
    ablator.launch(working_dir)
    res = Results(config, ablator.experiment_dir)
    return res


def test_analysis(tmp_path: Path, results: Results):
    PlotAnalysis(results, optim_metrics={"val_loss": "min"})
    categorical_name_remap = {
        "model_config.mock_param": "Some Parameter",
    }
    numerical_name_remap = {
        "train_config.optimizer_config.arguments.lr": "Learning Rate",
    }
    analysis = PlotAnalysis(
        results,
        save_dir=tmp_path.as_posix(),
        cache=True,
        optim_metrics={"val_loss": "min"},
    )
    attribute_name_remap = {**categorical_name_remap, **numerical_name_remap}
    analysis.make_figures(
        metric_name_remap={
            "val_loss": "Val. Loss",
        },
        attribute_name_remap=attribute_name_remap,
    )
    assert all(
        tmp_path.joinpath("violinplot", "val_loss", f"{file_name}.png").exists()
        for file_name in categorical_name_remap
    )

    assert all(
        tmp_path.joinpath("linearplot", "val_loss", f"{file_name}.png").exists()
        for file_name in numerical_name_remap
    )
    pass

    try:
        no_cache = PlotAnalysis(
            df,
            save_dir=tmp_path.as_posix(),
            cache=False,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )


        # checking whether if cache = False, clears memory.
        assert no_cache.cache == None

        # Ensuring that the incorrect path provided must not exists.
        assert not Path("/random_folder/nonexistent_folder/").parent.exists()

        incorrect_path = PlotAnalysis(
            df,
            save_dir="/nonexistent_folder/file.txt",
            cache=False,
            optim_metrics={"val_acc": Optim.min},
            numerical_attributes=list(numerical_name_remap.keys()),
            categorical_attributes=list(categorical_name_remap.keys()),
        )
    except FileNotFoundError as f:
        # Expecting the constructor to fail because of incorrect_path. 
        assert str(f).startswith("Save directory does not exist")

def test_name_remap(tmp_path: Path):
    name_map = {'a': 'A', 'b': 'B'}
    defaults = None
    assert parse_name_remap(defaults, name_map) == name_map

    name_map = {'a': 'A', 'b': 'B'}
    defaults = ['a', 'b', 'c', 'd']
    expected_output = {'a': 'A', 'b': 'B', 'c': 'c', 'd': 'd'}
    assert parse_name_remap(defaults, name_map) == expected_output

    name_map = None
    defaults = ['a', 'b', 'c']
    assert parse_name_remap(defaults, name_map) == {'a': 'a', 'b': 'b', 'c': 'c'}

    try:
        parse_name_remap(None, None)
    except NotImplementedError as err:
        if str(err) != "`defaults` or `name_map` argument required.":
            raise err

def test_get_best_results_by_metrics(tmp_path: Path):
    results_csv = Path(__file__).parent.parent.joinpath("assets", "results.csv")
    df = pd.read_csv(results_csv)

    metric_map = {"val_acc": Optim.max, "val_loss" :Optim.min}
    report_results = Analysis._get_best_results_by_metric(df, metric_map)

    # check whether each "metric" from metric_map has all the results.
    assert (df['path'].unique().count() == v for _ , v in dict(report_results["best"].value_counts()))

    # check for Max values
    data1 = df.groupby("path")["val_acc"].agg(lambda x: x.max(skipna=True)).reset_index(name="val_acc")    
    data2 = report_results[report_results["best"] == "val_acc"][["path", "val_acc"]].reset_index(drop=True)
    merged_df = pd.merge(data1, data2, on="path", suffixes=("_df1", "_df2"))

    assert not (merged_df["val_acc_df1"] != merged_df["val_acc_df2"]).any()

    # check for Min values
    data1 = df.groupby("path")["val_loss"].agg(lambda x: x.min(skipna=True)).reset_index(name="val_loss")    
    data2 = report_results[report_results["best"] == "val_loss"][["path", "val_loss"]].reset_index(drop=True)
    merged_df = pd.merge(data1, data2, on="path", suffixes=("_df1", "_df2"))

    assert not (merged_df["val_loss_df1"] != merged_df["val_loss_df2"]).any()

if __name__ == "__main__":
    import shutil
    from tests.ray_models.model import (
        WORKING_DIR,
        TestWrapper,
        MyCustomModel,
        _make_config,
    )

    tmp_path = Path("/tmp/save_dir")
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(exist_ok=True)
    res = _results(tmp_path, TestWrapper(MyCustomModel), _make_config, WORKING_DIR)
    test_analysis(tmp_path, res)
