def load_data():

    import pandas as pd

    dataset = pd.read_csv("heart_disease.csv")
    y = dataset.pop("target")
    x = dataset.copy()
    x["thal"] = x["thal"].map(
        lambda x: "normal" if x not in ["fixed", "fixed", "reversible"] else x
    )

    return x, y


x, y = load_data()
x

