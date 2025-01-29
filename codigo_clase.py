def train_logistic_regression():

    from sklearn.linear_model import LogisticRegression

    pipeline = make_pipeline(
        estimator=LogisticRegression(max_iter=10000, solver="saga"),
    )

    param_grid = {
        "selectkbest__k": range(1, 11),
        "estimator__penalty": ["l1", "l2"],
        "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100],
    }

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
    )

    train_estimator(estimator)


train_logistic_regression()
