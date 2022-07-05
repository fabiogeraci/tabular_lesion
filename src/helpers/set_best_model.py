import sklearn


def best_model_(model=None, search: sklearn = None):
    """

    :param model:
    :param search:
    :return:
    """
    best_params_ = {key.replace("classifier__", ""): value for key, value in search.best_params_.items()}

    return model.classifier.set_params(**best_params_)
