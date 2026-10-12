
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TenureGroup'] = pd.cut(X['Tenure'], bins=[0,3,6,9,11],
            labels=['0-3 years','3-6 years','6-9 years','9+ years'], include_lowest=True).astype(object)
        X['AgeGroup'] = pd.cut(X['Age'], bins=[0,30,45,60,100],
            labels=['Young','Middle','Senior','Elderly'], include_lowest=True).astype(object)
        X['BalanceToSalaryRatio'] = X['Balance'] / (X['EstimatedSalary'] + 1)
        X['HighProductRisk'] = np.where(X['NumOfProducts'].notna(),
            (X['NumOfProducts'] >= 3).astype(float), np.nan)
        X['EngagementScore'] = X['IsActiveMember'] + X['NumOfProducts'] / 4
        return X
