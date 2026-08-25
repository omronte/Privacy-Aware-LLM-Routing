import random


def split_dataset(dataset, seed=42):

    if not dataset:
        raise ValueError("Dataset must contain at least one sample.")

    shuffled = list(dataset)
    random.Random(seed).shuffle(shuffled)

    temp_size = max(1, round(len(shuffled) * 0.30))
    temp_size = min(temp_size, len(shuffled) - 1) if len(shuffled) > 1 else 0

    train_data = shuffled[:-temp_size] if temp_size else shuffled
    temp_data = shuffled[-temp_size:] if temp_size else []

    validation_size = round(len(temp_data) * 0.50)
    val_data = temp_data[:validation_size]
    test_data = temp_data[validation_size:]

    print(
        f"Train: {len(train_data)}"
    )

    print(
        f"Validation: {len(val_data)}"
    )

    print(
        f"Test: {len(test_data)}"
    )

    return (
        train_data,
        val_data,
        test_data
    )
