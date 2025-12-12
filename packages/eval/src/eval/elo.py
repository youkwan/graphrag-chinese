from dataclasses import dataclass, field


@dataclass
class EloRatingSystem:
    """Manages ELO ratings for models.

    Attributes:
        k_factor (int): The K-factor used for ELO calculation. Defaults to 32.
        base_rating (int): The initial rating for new models. Defaults to 1000.
        ratings (dict[str, float]): A dictionary mapping model names to their ratings.
        history (list[dict]): A history of ELO updates.
    """

    k_factor: int = 32
    base_rating: int = 1000
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def get_rating(self, model_name: str) -> float:
        """Retrieves the current rating for a model, initializing it if necessary.

        Args:
            model_name: The name of the model.

        Returns:
            The current ELO rating of the model.
        """
        if model_name not in self.ratings:
            self.ratings[model_name] = float(self.base_rating)
        return self.ratings[model_name]

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculates the expected score for model A against model B.

        Args:
            rating_a: The rating of model A.
            rating_b: The rating of model B.

        Returns:
            The probability (0.0 to 1.0) of model A winning.
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, model_a: str, model_b: str, result: float) -> None:
        """Updates the ratings for two models based on a match result.

        Args:
            model_a: The name of the first model.
            model_b: The name of the second model.
            result: The actual score for model A (1.0 for win, 0.0 for loss, 0.5 for tie).
        """
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)

        expected_a = self.calculate_expected_score(rating_a, rating_b)
        expected_b = self.calculate_expected_score(rating_b, rating_a)

        # Result for B is 1 - result (since result is A's score)
        # If result is 0.5 (tie), both get 0.5
        actual_b = 1.0 - result

        new_rating_a = rating_a + self.k_factor * (result - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_b - expected_b)

        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b

        self.history.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "rating_a_before": rating_a,
                "rating_b_before": rating_b,
                "rating_a_after": new_rating_a,
                "rating_b_after": new_rating_b,
                "score_a": result,
                "expected_a": expected_a,
            }
        )
