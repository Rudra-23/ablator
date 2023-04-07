import logging
from logging import warning
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set

import numpy as np
import pandas as pd
from trainer.analysis.plot.utils import parse_name_remap
import copy
from trainer.analysis.results import Results


from joblib import Memory
import trainer.analysis.utils as autils
from trainer.config.run import ParallelConfig, Optim
from trainer.utils.config import flatten_nested_dict

logger = logging.getLogger(__name__)


class Analysis:
    def __init__(
        self,
        results: Results,
        save_dir: Optional[str] = None,
        cache=False,
        aux_metrics: Optional[Dict[str, Optim]] = None,
    ) -> None:
        self._results = results

        self.config = results.config
        self.metric_map = copy.deepcopy(self._results.metric_map)
        if aux_metrics is not None:
            # TODO assert types of `aux_metrics`
            self.metric_map = aux_metrics
        self.save_dir: Optional[Path] = None
        self.cache: Optional[Memory] = None
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            if not self.save_dir.parent.exists():
                raise ValueError(
                    f"Save directory does not exist. `{self.save_dir.parent}`"
                )
            self.save_dir.mkdir(exist_ok=True)
            self.cache = Memory(Path(save_dir).joinpath(".cache"), verbose=0)
            if not cache:
                self.cache.clear()
                self.cache = None
        self.categorical_attributes = self.results.categorical_attributes
        self.numerical_attributes = self.results.numerical_attributes
        self.experiment_attributes: List[str] = self.categorical_attributes + self.numerical_attributes

        self.results: pd.DataFrame = self._results.data[
            self.experiment_attributes
            + list(self.metric_map.keys())
            + ["path", "index"]
        ]

    @property
    def metric_names(self):
        return list(self.metric_map.keys())


    @classmethod
    def _get_best_results_by_metric(
        cls,
        raw_results: pd.DataFrame,
        metric_map: Dict[str, Optim],
    ):
        def _best_perf(row: pd.DataFrame, name, obj_fn):
            if Optim(obj_fn) == Optim.min:
                return row.sort_values(name, na_position="last").iloc[0]
            else:
                return row.sort_values(name, na_position="first").iloc[-1]

        _ress = []
        for name, obj_fn in metric_map.items():
            res = (
                raw_results.groupby("path")
                .apply(lambda x: _best_perf(x, name, obj_fn))
                .reset_index(drop=True)
            )
            res["best"] = name
            _ress.append(res)
        report_results = pd.concat(_ress).reset_index(drop=True)

        return report_results

    @classmethod
    def _remap_results(
        cls,
        attributes: pd.DataFrame,
        metrics: pd.DataFrame,
        metric_map: Dict[str, Optim],
        metric_name_remap: Optional[Dict[str, str]] = None,
        attribute_name_remap: Optional[Dict[str, str]] = None,
    ):
        metric_name_remap = parse_name_remap(metrics.columns, metric_name_remap)
        attribute_name_remap = parse_name_remap(
            attributes.columns, attribute_name_remap
        )
        metric_map = {
            metric_name_remap[metric_name]: direction
            for metric_name, direction in metric_map.items()
            if metric_name in metric_name_remap
        }

        attributes = attributes[list(attribute_name_remap.keys())]
        metrics = metrics[list(metric_name_remap.keys())]
        attributes.columns = [attribute_name_remap[c] for c in attributes.columns]
        metrics.columns = [metric_name_remap[c] for c in metrics.columns]
        return attributes, metrics, metric_map

    def _init_directories(self, save_dir):
        report_dir = Path(save_dir)
        if not report_dir.parent.exists():
            raise FileNotFoundError(f"{report_dir.parent}")
        report_dir.mkdir(exist_ok=True)
        return report_dir
