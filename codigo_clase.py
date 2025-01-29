def make_pipeline(estimator):

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(dtype="int"), ["Origin"]),
        ],
        remainder=StandardScaler(),
    )

    selectkbest = SelectKBest(score_func=f_regression)

    pipeline = Pipeline(
        steps=[
            ("tranformer", transformer),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline

