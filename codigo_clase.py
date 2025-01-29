def report(
    estimator,
    accuracy_train,
    accuracy_test,
    balanced_accuracy_train,
    balanced_accuracy_test,
):

    print(estimator, ":", sep="")
    print("-" * 80)
    print(f"Balanced Accuracy: {balanced_accuracy_test} ({balanced_accuracy_train})")
    print(f"         Accuracy: {accuracy_test} ({accuracy_train})")




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

    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    (
        accuracy_train,
        accuracy_test,
        balanced_accuracy_train,
        balanced_accuracy_test,
    ) = eval_metrics(
        y_train_true,
        y_test_true,
        y_train_pred,
        y_test_pred,
    )

    report(
        estimator.best_estimator_,
        accuracy_train,
        accuracy_test,
        balanced_accuracy_train,
        balanced_accuracy_test,
    )


check_estimator()








def train_mlp_classifier():

    from sklearn.neural_network import MLPClassifier

    pipeline = make_pipeline(
        estimator=MLPClassifier(max_iter=10000),
    )

    param_grid = {
        "selectkbest__k": range(1, 11),
        "estimator__hidden_layer_sizes": [(h,) for h in range(1, 11)],
        "estimator__learning_rate_init": [0.0001, 0.001, 0.01, 0.1, 1.0],
    }

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
    )

    train_estimator(estimator)


train_mlp_classifier()
check_estimator()









