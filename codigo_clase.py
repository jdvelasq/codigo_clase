def save_estimator(estimator):

    import pickle

    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)
