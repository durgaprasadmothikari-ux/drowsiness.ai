import csv
import math
import pickle
import random


# ==========================================
# SETTINGS
# ==========================================

DATASET_FILE = "drowsiness_dataset.csv"
MODEL_FILE = "drowsiness_model.pkl"

K = 5


# ==========================================
# LOAD DATASET
# ==========================================

features = []
labels = []


with open(
    DATASET_FILE,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        left_ear = float(
            row["left_ear"]
        )

        right_ear = float(
            row["right_ear"]
        )

        average_ear = float(
            row["average_ear"]
        )

        label = int(
            row["label"]
        )


        features.append([
            left_ear,
            right_ear,
            average_ear
        ])

        labels.append(label)


# ==========================================
# DISPLAY DATASET INFORMATION
# ==========================================

print()
print("=======================================")
print("      DRIVER DROWSINESS ML MODEL")
print("=======================================")
print()

print(
    "Total samples:",
    len(features)
)

print(
    "Awake samples:",
    labels.count(0)
)

print(
    "Drowsy samples:",
    labels.count(1)
)

print()


# ==========================================
# CHECK DATASET
# ==========================================

if len(features) < 20:

    print("ERROR: Not enough data.")

    exit()


if labels.count(0) == 0:

    print("ERROR: No Awake samples.")

    exit()


if labels.count(1) == 0:

    print("ERROR: No Drowsy samples.")

    exit()


# ==========================================
# COMBINE FEATURES AND LABELS
# ==========================================

dataset = list(
    zip(
        features,
        labels
    )
)


# ==========================================
# SHUFFLE DATA
# ==========================================

random.seed(42)

random.shuffle(
    dataset
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

split_index = int(
    len(dataset) * 0.8
)


train_data = dataset[
    :split_index
]


test_data = dataset[
    split_index:
]


print(
    "Training samples:",
    len(train_data)
)

print(
    "Testing samples:",
    len(test_data)
)

print()


# ==========================================
# EUCLIDEAN DISTANCE
# ==========================================

def euclidean_distance(
    point1,
    point2
):

    total = 0


    for i in range(
        len(point1)
    ):

        difference = (
            point1[i] -
            point2[i]
        )


        total += (
            difference *
            difference
        )


    return math.sqrt(
        total
    )


# ==========================================
# KNN PREDICTION
# ==========================================

def predict(
    sample,
    training_data
):

    distances = []


    for (
        training_features,
        training_label
    ) in training_data:

        distance = euclidean_distance(
            sample,
            training_features
        )


        distances.append(
            (
                distance,
                training_label
            )
        )


    # Sort nearest → farthest

    distances.sort(
        key=lambda x: x[0]
    )


    # Select K nearest samples

    nearest = distances[
        :K
    ]


    awake_votes = 0

    drowsy_votes = 0


    for (
        distance,
        label
    ) in nearest:

        if label == 0:

            awake_votes += 1

        else:

            drowsy_votes += 1


    if drowsy_votes > awake_votes:

        return 1

    else:

        return 0


# ==========================================
# TEST MODEL
# ==========================================

print(
    "Testing KNN model..."
)

print()


correct = 0

awake_correct = 0

drowsy_correct = 0

awake_total = 0

drowsy_total = 0


for (
    test_features,
    actual_label
) in test_data:


    predicted_label = predict(
        test_features,
        train_data
    )


    # Count total correct

    if predicted_label == actual_label:

        correct += 1


    # Awake statistics

    if actual_label == 0:

        awake_total += 1

        if predicted_label == 0:

            awake_correct += 1


    # Drowsy statistics

    else:

        drowsy_total += 1

        if predicted_label == 1:

            drowsy_correct += 1


# ==========================================
# CALCULATE ACCURACY
# ==========================================

accuracy = (
    correct /
    len(test_data)
) * 100


awake_accuracy = (
    awake_correct /
    awake_total
) * 100


drowsy_accuracy = (
    drowsy_correct /
    drowsy_total
) * 100


# ==========================================
# DISPLAY RESULTS
# ==========================================

print()
print("=======================================")
print("             MODEL RESULTS")
print("=======================================")
print()

print(
    f"Correct predictions: "
    f"{correct}/{len(test_data)}"
)

print(
    f"Overall accuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"Awake accuracy: "
    f"{awake_accuracy:.2f}%"
)

print(
    f"Drowsy accuracy: "
    f"{drowsy_accuracy:.2f}%"
)

print()


# ==========================================
# SAVE MODEL
# ==========================================

model = {

    "training_data": train_data,

    "k": K

}


with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("=======================================")
print("       MODEL SAVED SUCCESSFULLY")
print("=======================================")
print()

print(
    "File:",
    MODEL_FILE
)

print()