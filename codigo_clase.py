def check_estimator():

    import pickle

    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    data, target = load_data()

    x_train, x_test, y_train_true, y_test_true = make_train_test_split(
        x=data,
        y=target,
    )

    estimator = load_estimator()

    mse, mae, r2 = eval_metrics(
        y_test_true,
        estimator.predict(x_test),
    )

    report(estimator.best_estimator_, mse, mae, r2)


check_estimator()
