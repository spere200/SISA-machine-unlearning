from sklearn.linear_model import SGDClassifier
import numpy as np
from collections import defaultdict


class SISA:
    def __init__(self, shards=5, slices=5) -> None:
        """shards: The number of shards and models
        \nslices: The number of slices and incremental training steps/snapshots created for each model
        """
        self.shards = shards
        self.slices = slices

        # create the weak learning models
        self.models = [SGDClassifier(loss="log_loss") for _ in range(self.shards)]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """x: The independent set of features; expects a 2d numpy array
        \ny: The dependent features; expects a 1d numpy array"""

        input_shards = np.array_split(x, self.shards)
        output_shards = np.array_split(y, self.shards)

        for i, model in enumerate(self.models):
            # split the shards into slices for incremental learning using partial_fit
            input_slices = np.array_split(input_shards[i], self.slices)
            output_slices = np.array_split(output_shards[i], self.slices)

            # perform the first partial fit, by passing the unique labels that are expected on this shard
            model.partial_fit(
                input_slices[0], output_slices[0], np.unique(output_shards[i])
            )

            # perform the remaining partial fits on the rest of the slices of the current shard
            for j in range(1, self.slices):
                model.partial_fit(input_slices[j], output_slices[j])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: The dependent feature set on which to make a prediction; expects a 2d numpy array
        \nReturns a 1d numpy array where the value at i is the prediction for row i in the input
        """
        predictions = []

        # go through each row in the input
        for row in x:
            currPred = 0
            finalPred = 0
            freq = defaultdict(int)

            # have each of the weak learning models vote on the result
            for model in self.models:
                currPred = model.predict([row])[0]
                freq[currPred] += 1

                if freq[currPred] > freq[finalPred]:
                    finalPred = currPred

            predictions.append(finalPred)

        return np.array(predictions)

    def score(self) -> float:
        pass


if __name__ == "__main__":
    test = SISA()
    print(test.models)
