def check_estimator():

    import pickle

    import pandas as pd
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    data, target = load_data()

    x_train, x_test, y_train_true, y_test_true = make_train_test_split(
        x=data,
        y=target,
    )

    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    accuracy_train = round(accuracy_score(y_train_true, y_train_pred), 4)
    accuracy_test = round(accuracy_score(y_test_true, y_test_pred), 4)
    balanced_accuracy_train = round(
        balanced_accuracy_score(y_train_true, y_train_pred), 4
    )
    balanced_accuracy_test = round(balanced_accuracy_score(y_test_true, y_test_pred), 4)

    print(estimator.best_estimator_, ":", sep="")
    print(f"  Balanced Accuracy: {balanced_accuracy_test} ({balanced_accuracy_train})")
    print(f"           Accuracy: {accuracy_test} ({accuracy_train})")


check_estimator()
