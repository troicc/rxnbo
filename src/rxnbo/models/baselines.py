"""The models a neural network has to beat before it is worth anything.

- mean:  always predicts the training mean. R^2 = 0 on average by construction; any model
         at or below it has learnt nothing.
- ridge: linear regression with an L2 penalty, features standardised on training rows.
- rf:    random forest - the model Ahneman et al. used (R^2 ~ 0.92 on random splits).
"""
from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make(name: str, seed: int = 0):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        # the scaler sits inside the pipeline, so its mean and spread come from the
        # training rows only - fitting it on all rows would leak the test set into training
        return make_pipeline(StandardScaler(), RidgeCV(alphas=(0.1, 1, 10, 100, 1000)))
    if name == "rf":
        return RandomForestRegressor(n_estimators=500, max_features=0.3, min_samples_leaf=1,
                                     n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown baseline {name!r}")
