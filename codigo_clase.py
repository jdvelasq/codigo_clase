def make_pipeline(estimator):

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.pipeline import Pipeline

    vectorizer = CountVectorizer(
        lowercase=True,
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]\w+\b",
        stop_words="english",
    )

    transformer = TfidfTransformer()

    pipeline = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("transformer", transformer),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline
    
