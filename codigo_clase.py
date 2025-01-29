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
