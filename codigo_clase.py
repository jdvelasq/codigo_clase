def save_estimator(estimator):

    import pickle

    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)




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




def use_estimator():

    import pickle

    import pandas as pd

    dataframe = pd.read_csv(
        "../files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )

    data = dataframe.phrase

    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    prediction = estimator.predict(data)

    return prediction


use_estimator()








