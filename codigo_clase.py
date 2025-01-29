def train_logistic_regression():

    from sklearn.linear_model import LogisticRegression

    data, target = load_data()
    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )
    estimator = make_pipeline(estimator=LogisticRegression(max_iter=1000))
    estimator.fit(x_train, y_train)
    save_estimator(estimator)


train_logistic_regression()
