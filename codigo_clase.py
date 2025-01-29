def make_grid_search(estimator, param_grid, cv=5):

    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
    )

    return grid_search
