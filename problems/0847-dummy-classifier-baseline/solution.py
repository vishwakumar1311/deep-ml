from collections import Counter
import math
import numpy as np


def dummy_classifier(y_train, n_test, strategy, constant=None):
    """Produce baseline predictions of length n_test using the given strategy.

    Returns a plain Python list of predicted labels.
    """
    if n_test == 0:
        return []

    # Get class counts and deterministic sorted classes
    counts = Counter(y_train)
    sorted_classes = sorted(counts.keys())

    match strategy:
        case "constant":
            return [constant] * n_test

        case "most_frequent":
            # Highest frequency first; break ties with smallest class label
            best_class = min(sorted_classes, key=lambda c: (-counts[c], c))
            return [best_class] * n_test

        case "uniform":
            k = len(sorted_classes)
            return [sorted_classes[i % k] for i in range(n_test)]

        case "stratified":
            n_train = len(y_train)
            allocations = {}
            fractional_parts = []

            for c in sorted_classes:
                expected = n_test * (counts[c] / n_train)
                base = math.floor(expected)
                allocations[c] = base
                fractional_parts.append((expected - base, c))

            # Distribute remaining slots by largest fractional part (ties broken by smaller class label)
            remainder = n_test - sum(allocations.values())
            fractional_parts.sort(key=lambda item: (-item[0], item[1]))

            for i in range(remainder):
                allocations[fractional_parts[i][1]] += 1

            # Group outputs in sorted class order
            predictions = []
            for c in sorted_classes:
                predictions.extend([c] * allocations[c])
            return predictions

        case _:
            raise ValueError(f"Unknown strategy: '{strategy}'")
