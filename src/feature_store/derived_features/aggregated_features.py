import warnings
from typing import Any, List, Literal, Mapping

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class GroupFeatureGenerator(BaseEstimator, TransformerMixin):
    """Class used for generating group features as median or rank per group.

    Attributes:
        group_cols (List): List of columns used for calculating the aggregated value.
        metric (str): The metric to be used for replacement, can be one of
            ['mean', 'median', 'rank']
        values_by_group (Mapping[str, Any]): a mapping of column
            name to dictionary mapping of the metric values for each group in
            the column. Set in fit method.
        target_name (str): Target column name used for grouping. Default to the y passed in fit.
    """

    def __init__(
        self,
        group_cols: List,
        group_metric: Literal["mean", "median", "rank", "min", "max", "percentile"],
        percentile_q: float | None = None,
        target_name: str = "y",
    ):
        if group_metric not in ("mean", "median", "rank", "min", "max", "percentile"):
            raise ValueError(f"Invalid metric {group_metric}.")

        self.group_cols = group_cols
        self.group_metric = group_metric
        self.target_name = target_name
        self.percentile_q = percentile_q

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.values_by_group: Mapping[str, Any] = {}
        X = X.copy()
        if self.target_name == "y":
            self.target_name = str(y.name)
            X[self.target_name] = y

        X_group = X.groupby(self.group_cols)[self.target_name]
        X_agg: pd.Series

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            match self.group_metric:
                case "mean" | "median" | "min" | "max":
                    X_agg = X_group.agg(self.group_metric)  # type: ignore[assignment]
                case "rank":
                    X_agg = X_group.median().rank()
                case "percentile":
                    X_agg = X_group.agg(self.percentile, q=self.percentile_q)

        self.values_by_group = X_agg.to_dict()
        return self

    def transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self, "values_by_group")
        X = X.copy()
        new_col_name = self._get_new_col_name()
        X[new_col_name] = X.set_index(self.group_cols).index.map(self.values_by_group)
        return X[[new_col_name]]

    def get_feature_names_out(self, feature_names=None):
        return [self._get_new_col_name()]

    def _get_new_col_name(self) -> str:
        """Generates the proper name for the new grouped column.

        Examples:
            >>> GroupFeatureGenerator(group_cols=['country'], target_name='sales', group_metric='mean')._get_new_col_name()
            'mean_sales_country'
            >>> GroupFeatureGenerator(group_cols=['country'], target_name='sales', group_metric='rank')._get_new_col_name()
            'rank_sales_country'
            >>> GroupFeatureGenerator(group_cols=['country'], target_name='sales', group_metric='min')._get_new_col_name()
            'min_sales_country'
            >>> GroupFeatureGenerator(group_cols=["industry", "country"], target_name='sales', group_metric='rank')._get_new_col_name()
            'rank_sales_industry_country'
            >>> GroupFeatureGenerator(group_cols=["industry", "country"], target_name='sales', group_metric='percentile', percentile_q=0.5)._get_new_col_name()
            'percentile_50_sales_industry_country'
        """
        metric = self.group_metric
        if metric == "percentile":
            metric = f"{metric}_{self.percentile_q*100:.0f}"
        new_name = "_".join([metric, self.target_name] + self.group_cols)
        return new_name

    @staticmethod
    def percentile(x: pd.DataFrame, q: float) -> pd.DataFrame:
        """Wrapper for groupby quantile aggregation function"""
        return x.quantile(q)
