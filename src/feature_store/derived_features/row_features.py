from typing import Sequence

import pandas as pd
from sklearn import set_config
from sklearn.preprocessing import FunctionTransformer

set_config(transform_output="pandas")


class FormulaTransformer(FunctionTransformer):
    """Sklearn transformers that applies a formula to a dataframe.

    Attributes:
        formula (str): The formula to be applied to the dataframe.

    Example:
        >>> X = pd.DataFrame({"revenue": [100, 200, 300], "employees": [1, 2, 3]})
        >>> t = FormulaTransformer(formula="revenue_per_capita = revenue / employees")
        >>> print(t.fit_transform(X))
           revenue_per_capita
        0               100.0
        1               100.0
        2               100.0
        >>> print(t.get_feature_names_out())
        ['revenue_per_capita']
    """

    def __init__(self, *, formula: str):
        self.formula = formula
        super().__init__(
            func=self.apply_formula,
            feature_names_out=self._get_feature_name_out_from_formula,
        )

    def apply_formula(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.eval(self.formula)[self._get_feature_name_out_from_formula()]

    def _get_feature_name_out_from_formula(self, *args, **kwargs) -> Sequence[str]:
        return [self.formula.split("=")[0].strip()]

    def _get_feature_names_in_from_formula(self, *args, **kwargs) -> Sequence[str]:
        return [col.strip() for col in self.formula.split("=")[1].split("/") if col.strip() != ""]
