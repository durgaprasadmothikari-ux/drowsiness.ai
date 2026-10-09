import csv
import os
import pickle
import random
import math


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "advanced_drowsiness_dataset.csv"
MODEL_PATH = "advanced_drowsiness_model.pkl"

TEST_SIZE = 0.20
RANDOM_SEED = 42

# Number of trees
NUM_TREES = 25

# Maximum tree depth
MAX_DEPTH = 8

# Minimum samples required to split
MIN_SAMPLES_SPLIT = 4


# ============================================================
# FEATURES
# ============================================================

FEATURE_NAMES = [
    "left_ear",
    "right_ear",
    "average_ear",
    "mar",
    "blink_left",
    "blink_right",
    "jaw_open",
    "brow_left",
    "brow_right",
    "mouth_smile_left",
    "mouth_smile_right",
    "pitch",
    "yaw",
    "roll",
    "brightness",
    "contrast",
    "sharpness",
    "motion",
    "eye_closure_duration",
    "yawn_duration",
    "head_down_duration"
]


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("ERROR: Dataset not found:")
    print(DATASET_PATH)
    raise SystemExit


with open(DATASET_PATH, "r", newline="") as file:

    reader = csv.DictReader(file)
    rows = list(reader)


if len(rows) < 10:

    print("ERROR: Dataset contains too few samples.")
    raise SystemExit


print()
print("=" * 65)
print("          ADVANCED DROWSINESS MODEL TRAINING")
print("=" * 65)


print()
print("Dataset:", DATASET_PATH)
print("Total samples:", len(rows))
print("Features:", len(FEATURE_NAMES))


# ============================================================
# CONVERT DATA
# ============================================================

X = []
y = []


for row in rows:

    try:

        features = [
            float(row[name])
            for name in FEATURE_NAMES
        ]

        label = int(row["label"])

        if label not in [0, 1]:
            continue

        if not all(
            math.isfinite(value)
            for value in features
        ):
            continue

        X.append(features)
        y.append(label)

    except:

        continue


print("Valid samples:", len(X))


if len(X) < 10:

    print("ERROR: Not enough valid samples.")
    raise SystemExit


# ============================================================
# CLASS COUNTS
# ============================================================

awake_count = y.count(0)
drowsy_count = y.count(1)


print()
print("Awake samples :", awake_count)
print("Drowsy samples:", drowsy_count)


# ============================================================
# NORMALIZATION
# ============================================================

print()
print("Calculating feature normalization...")


means = []
stds = []


for column in range(len(FEATURE_NAMES)):

    values = [
        row[column]
        for row in X
    ]

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    std = math.sqrt(variance)

    if std == 0:
        std = 1.0

    means.append(mean)
    stds.append(std)


def normalize_row(row):

    return [
        (row[i] - means[i]) / stds[i]
        for i in range(len(row))
    ]


X_normalized = [
    normalize_row(row)
    for row in X
]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

random.seed(RANDOM_SEED)


indices = list(range(len(X_normalized)))

random.shuffle(indices)


test_count = int(
    len(indices) * TEST_SIZE
)


test_indices = indices[:test_count]
train_indices = indices[test_count:]


X_train = [
    X_normalized[i]
    for i in train_indices
]

y_train = [
    y[i]
    for i in train_indices
]


X_test = [
    X_normalized[i]
    for i in test_indices
]

y_test = [
    y[i]
    for i in test_indices
]


print()
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# KNN CLASSIFIER
# ============================================================

# We use a feature-rich KNN classifier here.
# It works entirely in Python and avoids dependency problems.


K = 7


def euclidean_distance(a, b):

    total = 0.0

    for i in range(len(a)):

        difference = a[i] - b[i]

        total += difference * difference

    return math.sqrt(total)


def predict_knn(sample):

    distances = []

    for index in range(len(X_train)):

        distance = euclidean_distance(
            sample,
            X_train[index]
        )

        distances.append(
            (distance, y_train[index])
        )

    distances.sort(
        key=lambda item: item[0]
    )

    nearest = distances[:K]

    awake_votes = 0
    drowsy_votes = 0

    for distance, label in nearest:

        if label == 0:
            awake_votes += 1
        else:
            drowsy_votes += 1

    if drowsy_votes > awake_votes:

        prediction = 1
        confidence = (
            drowsy_votes / K
        )

    else:

        prediction = 0
        confidence = (
            awake_votes / K
        )

    return prediction, confidence


# ============================================================
# TRAINING
# ============================================================

print()
print("Training feature-based KNN model...")
print("K =", K)


# KNN doesn't need a traditional fitting phase.
# The normalized training dataset is the learned model.


# ============================================================
# TESTING
# ============================================================

print()
print("Testing model...")


predictions = []
confidences = []


for sample in X_test:

    prediction, confidence = predict_knn(
        sample
    )

    predictions.append(prediction)
    confidences.append(confidence)


# ============================================================
# METRICS
# ============================================================

correct = 0

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


for actual, predicted in zip(
    y_test,
    predictions
):

    if actual == predicted:

        correct += 1

    if actual == 1 and predicted == 1:

        true_positive += 1

    elif actual == 0 and predicted == 0:

        true_negative += 1

    elif actual == 0 and predicted == 1:

        false_positive += 1

    elif actual == 1 and predicted == 0:

        false_negative += 1


accuracy = (
    correct / len(y_test)
) * 100


# Precision

if (
    true_positive +
    false_positive
) > 0:

    precision = (
        true_positive /
        (true_positive + false_positive)
    ) * 100

else:

    precision = 0


# Recall

if (
    true_positive +
    false_negative
) > 0:

    recall = (
        true_positive /
        (true_positive + false_negative)
    ) * 100

else:

    recall = 0


# F1 score

if precision + recall > 0:

    f1 = (
        2 *
        precision *
        recall /
        (precision + recall)
    )

else:

    f1 = 0


# ============================================================
# CLASS-SPECIFIC ACCURACY
# ============================================================

awake_total = y_test.count(0)
drowsy_total = y_test.count(1)


awake_correct = true_negative

drowsy_correct = true_positive


if awake_total > 0:

    awake_accuracy = (
        awake_correct /
        awake_total
    ) * 100

else:

    awake_accuracy = 0


if drowsy_total > 0:

    drowsy_accuracy = (
        drowsy_correct /
        drowsy_total
    ) * 100

else:

    drowsy_accuracy = 0


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 65)
print("                    MODEL RESULTS")
print("=" * 65)

print()

print(
    f"Accuracy : {accuracy:.2f}%"
)

print(
    f"Precision: {precision:.2f}%"
)

print(
    f"Recall   : {recall:.2f}%"
)

print(
    f"F1 Score : {f1:.2f}%"
)

print()

print(
    f"Awake accuracy : {awake_accuracy:.2f}%"
)

print(
    f"Drowsy accuracy: {drowsy_accuracy:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 65)
print("                  CONFUSION MATRIX")
print("=" * 65)

print()

print("                  Predicted")
print("                Awake  Drowsy")

print(
    f"Actual Awake    {true_negative:5d}  {false_positive:6d}"
)

print(
    f"Actual Drowsy   {false_negative:5d}  {true_positive:6d}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model = {

    "feature_names": FEATURE_NAMES,

    "means": means,

    "stds": stds,

    "X_train": X_train,

    "y_train": y_train,

    "k": K,

    "model_type": "normalized_knn",

    "version": "2.0"

}


with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print()
print("=" * 65)
print("MODEL SAVED")
print("=" * 65)

print()
print("File:", MODEL_PATH)

print()
print("Training complete.")
print()