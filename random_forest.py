from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self, **kwargs):
        # Create the model with the provided parameters
        self.model = RandomForestRegressor(**kwargs)

        # Define default test_size and random_state
        self.test_size = kwargs.get('test_size', 0.2)
        self.random_state = kwargs.get('random_state', 24)

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        return self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the fitted model.
        """
        return self.model.predict(X)

    def get_model(self):
        """
        Returns the model instance.
        """
        return self.model

    def get_params(self):
        """
        Returns model parameters.
        """
        return self.model.get_params()

    def get_test_size(self):
        """
        Returns the test size.
        """
        return self.test_size

    def get_random_state(self):
        """
        Returns the random state.
        """
        return self.random_state

    def get_model_name(self):
        """
        Returns the model class name.
        """
        return self.model.__class__.__name__
