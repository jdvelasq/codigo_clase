def make_train_test_split(x, y):

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.10,
        random_state=0,
    )
    return x_train, x_test, y_train, y_test


