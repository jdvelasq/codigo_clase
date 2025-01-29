def train_logistic_regression():

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    data, target = load_data()

    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )

    pipeline = make_pipeline(
        estimator=LogisticRegression(max_iter=1000),
    )

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid={
            "transformer__norm": ["l1", "l2", None],
            "transformer__use_idf": [True, False],
            "transformer__smooth_idf": [True, False],
        },
        cv=5,
    )

    estimator.fit(x_train, y_train)

    best_estimator = load_estimator()

    if best_estimator is not None:

        saved_balanced_accuracy = balanced_accuracy_score(
            y_true=y_test, y_pred=best_estimator.predict(x_test)
        )

        current_balanced_accuracy = balanced_accuracy_score(
            y_true=y_test, y_pred=estimator.predict(x_test)
        )

        if current_balanced_accuracy < saved_balanced_accuracy:
            estimator = best_estimator

    save_estimator(estimator)


train_logistic_regression()

