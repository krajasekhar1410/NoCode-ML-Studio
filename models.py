import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelRunner:
    df: pd.DataFrame
    target: str
    task: str

    def run_all(self, test_size=0.2):
        X = self.df.drop(columns=[self.target]).select_dtypes(include=['number'])
        y = self.df[self.target]
        if X.shape[1] == 0:
            return {'error':'No numeric features found. Please select numeric features.'}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        results = {}
        if self.task == 'Regression':
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'RandomForest': RandomForestRegressor(n_estimators=50),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=50)
            }
            for name, m in models.items():
                try:
                    m.fit(X_train, y_train)
                    preds = m.predict(X_test)
                    results[name] = {'mse': mean_squared_error(y_test, preds), 'r2': r2_score(y_test, preds)}
                except Exception as e:
                    results[name] = {'error': str(e)}

            # simple hyperparam tuning example for RandomForest
            try:
                rf = RandomForestRegressor()
                param_dist = {'n_estimators': [50,100,200], 'max_depth': [None,5,10,20]}
                rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=4, cv=3, random_state=42)
                rs.fit(X_train, y_train)
                preds = rs.predict(X_test)
                results['RandomForest_tuned'] = {'best_params': rs.best_params_, 'mse': mean_squared_error(y_test, preds), 'r2': r2_score(y_test, preds)}
            except Exception as e:
                results['RandomForest_tuned'] = {'error': str(e)}

        elif self.task == 'Classification':
            models = {
                'LogisticRegression': LogisticRegression(max_iter=200),
                'RandomForest': RandomForestClassifier(n_estimators=50),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=50),
                'SVC': SVC(),
                'KNN': KNeighborsClassifier()
            }
            for name, m in models.items():
                try:
                    m.fit(X_train, y_train)
                    preds = m.predict(X_test)
                    results[name] = {'accuracy': accuracy_score(y_test, preds)}
                except Exception as e:
                    results[name] = {'error': str(e)}

            try:
                rf = RandomForestClassifier()
                param_dist = {'n_estimators': [50,100,200], 'max_depth': [None,5,10,20]}
                rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=4, cv=3, random_state=42)
                rs.fit(X_train, y_train)
                preds = rs.predict(X_test)
                results['RandomForest_tuned'] = {'best_params': rs.best_params_, 'accuracy': accuracy_score(y_test, preds)}
            except Exception as e:
                results['RandomForest_tuned'] = {'error': str(e)}

        else:
            # Simple time series baseline using lag features
            results['Note'] = 'Time series modelling not yet fully implemented. Use regression/time features.'

        return results
