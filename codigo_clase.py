def train_mlp_regressor():

    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error

    data, target = load_data()

    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )

    pipeline = make_pipeline(
        estimator=MLPRegressor(max_iter=30000),
    )

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid={
            "selectkbest__k": range(1, len(x_train.columns) + 1),
            "estimator__hidden_layer_sizes": [(n,) for n in range(1, 11)],
            "estimator__solver": ["adam"],
            "estimator__learning_rate_init": [0.01, 0.001, 0.0001],
        },
        cv=5,
    )

    estimator.fit(x_train, y_train)

    best_estimator = load_estimator()

    if best_estimator is not None:

        saved_mae = mean_absolute_error(
            y_true=y_test, y_pred=best_estimator.predict(x_test)
        )

        current_mae = mean_absolute_error(
            y_true=y_test, y_pred=estimator.predict(x_test)
        )

        if saved_mae < current_mae:
            estimator = best_estimator

    save_estimator(estimator)


train_mlp_regressor()
check_estimator()
