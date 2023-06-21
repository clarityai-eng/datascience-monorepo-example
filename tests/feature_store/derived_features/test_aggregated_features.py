import pandas as pd
import pytest

from feature_store.derived_features import aggregated_features as agg_feat


class TestGroupFeatureGenerator:
    @pytest.fixture(scope="class")
    def X(self):
        return pd.DataFrame(
            {
                "industry": ["AA", "BB", "BB", "AA", "BB", "BB"],
                "country": ["US", "US", "US", "FR", "FR", "FR"],
                "sales": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

    @pytest.fixture(scope="class")
    def y(self, X: pd.DataFrame):
        return X.sales

    def test_group_feature_generator_median_groupping(self, X: pd.DataFrame, y: pd.Series):
        group_feature_generator = agg_feat.GroupFeatureGenerator(group_cols=["country"], group_metric="median")
        X_transformed = group_feature_generator.fit_transform(X, y)
        expected_median_sales_country = pd.Series([2.0, 2.0, 2.0, 5.0, 5.0, 5.0])
        assert X_transformed["median_sales_country"].equals(expected_median_sales_country)
        assert group_feature_generator.values_by_group == {"US": 2.0, "FR": 5.0}

    def test_group_feature_generator_mean_groupping(self, X: pd.DataFrame, y: pd.Series):
        group_feature_generator = agg_feat.GroupFeatureGenerator(group_cols=["country"], group_metric="mean")
        X_transformed = group_feature_generator.fit_transform(X, y)
        expected_mean_sales_country = pd.Series([2.0, 2.0, 2.0, 5.0, 5.0, 5.0])
        assert X_transformed["mean_sales_country"].equals(expected_mean_sales_country)
        assert group_feature_generator.values_by_group == {"US": 2.0, "FR": 5.0}

    def test_group_feature_generator_median_rank_groupping(self, X: pd.DataFrame, y: pd.Series):
        group_feature_generator = agg_feat.GroupFeatureGenerator(group_cols=["country"], group_metric="rank")
        X_transformed = group_feature_generator.fit_transform(X, y)
        expected_median_rank_sales_country = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        assert X_transformed["rank_sales_country"].equals(expected_median_rank_sales_country)
        assert group_feature_generator.values_by_group == {"US": 1.0, "FR": 2.0}

    def test_invalid_aggregation_metric(self, X: pd.DataFrame, y: pd.Series):
        with pytest.raises(ValueError):
            agg_feat.GroupFeatureGenerator(group_cols=["country"], group_metric="invalid")  # type: ignore # type: ignore[assignment]

    def test_group_feature_generator_median_groupping_multiple_group_cols(self, X: pd.DataFrame, y: pd.Series):
        group_feature_generator = agg_feat.GroupFeatureGenerator(
            group_cols=["industry", "country"], group_metric="median"
        )
        X_transformed = group_feature_generator.fit_transform(X, y)
        expected_median_sales_industry_country = pd.Series([1.0, 2.5, 2.5, 4.0, 5.5, 5.5])
        assert X_transformed["median_sales_industry_country"].equals(expected_median_sales_industry_country)
        assert group_feature_generator.values_by_group == {
            ("AA", "FR"): 4.0,
            ("AA", "US"): 1.0,
            ("BB", "FR"): 5.5,
            ("BB", "US"): 2.5,
        }  # type: ignore[comparison-overlap]
