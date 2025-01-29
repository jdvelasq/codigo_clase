def make_pipeline(estimator):

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(dtype="int"), ["thal"]),
        ],
        remainder="passthrough",
    )

    selectkbest = SelectKBest(score_func=f_classif)

    pipeline = Pipeline(
        steps=[
            ("tranformer", transformer),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline
