import numpy as np


class Normalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        current_level_db = self._get_level_db(x)
        target_level_db = self._get_target_db()
        multiplier = self._get_multiplier(current_level_db, target_level_db)
        return multiplier * x

    @staticmethod
    def _get_level_db(x: np.ndarray) -> float:
        return 10 * np.log10(np.mean(x ** 2))

    def _get_target_db(self) -> float:
        return np.random.normal(loc=self.mean, scale=self.std)

    @staticmethod
    def _get_multiplier(current_level_db, target_level_db) -> float:
        return 10 ** ((target_level_db - current_level_db) / 20)


if __name__ == '__main__':
    x = 10 * np.random.rand(1, 48_000)  # random audio (white noise)
    normalizer = Normalizer(mean=-25, std=0)
    current_level_db = normalizer._get_level_db(x)  # noqa
    print(f'current level DB = {current_level_db}')
    normalized_x = normalizer(x)
    normalized_level_db = normalizer._get_level_db(normalized_x)  # noqa
    print(f'normalized level DB = {normalized_level_db}')
