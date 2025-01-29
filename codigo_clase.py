def eval_metrics(
    y_train_true,
    y_test_true,
    y_train_pred,
    y_test_pred,
):

    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    accuracy_train = round(accuracy_score(y_train_true, y_train_pred), 4)
    accuracy_test = round(accuracy_score(y_test_true, y_test_pred), 4)
    balanced_accuracy_train = round(
        balanced_accuracy_score(y_train_true, y_train_pred), 4
    )
    balanced_accuracy_test = round(balanced_accuracy_score(y_test_true, y_test_pred), 4)

    return (
        accuracy_train,
        accuracy_test,
        balanced_accuracy_train,
        balanced_accuracy_test,
    )
