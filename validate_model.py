import csv
import pickle
import os
import math


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "advanced_drowsiness_model.pkl"
VALIDATION_PATH = "validation_dataset.csv"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found:")
    print(MODEL_PATH)
    raise SystemExit


if not os.path.exists(VALIDATION_PATH):

    print("ERROR: Validation dataset not found:")
    print(VALIDATION_PATH)
    raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_PATH, "rb") as file:

    model = pickle.load(file)


FEATURE_NAMES = model["feature_names"]
MEANS = model["means"]
STDS = model["stds"]

X_TRAIN = model["X_train"]
Y_TRAIN = model["y_train"]

K = model["k"]


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

with open(
    VALIDATION_PATH,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


if len(rows) == 0:

    print("ERROR: Validation dataset is empty.")
    print()
    print("Run validation_collector.py first.")
    raise SystemExit


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("             INDEPENDENT MODEL VALIDATION")
print("=" * 70)

print()
print("Model:", MODEL_PATH)
print("Validation dataset:", VALIDATION_PATH)

print()
print("Validation samples:", len(rows))
print("Features:", len(FEATURE_NAMES))


# ============================================================
# PREPARE VALIDATION DATA
# ============================================================

X_validation = []
y_validation = []

invalid_rows = 0


for row in rows:

    try:

        features = [
            float(row[name])
            for name in FEATURE_NAMES
        ]

        label = int(row["label"])


        if label not in [0, 1]:

            invalid_rows += 1
            continue


        if not all(
            math.isfinite(value)
            for value in features
        ):

            invalid_rows += 1
            continue


        X_validation.append(features)
        y_validation.append(label)


    except:

        invalid_rows += 1


print()
print("Valid validation samples:", len(X_validation))
print("Invalid rows:", invalid_rows)


if len(X_validation) == 0:

    print()
    print("ERROR: No valid validation samples.")
    raise SystemExit


# ============================================================
# CLASS COUNTS
# ============================================================

awake_total = y_validation.count(0)
drowsy_total = y_validation.count(1)


print()
print("Awake samples :", awake_total)
print("Drowsy samples:", drowsy_total)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(features):

    normalized = []

    for i in range(
        len(features)
    ):

        value = (
            features[i] -
            MEANS[i]
        ) / STDS[i]

        normalized.append(value)

    return normalized


X_validation_normalized = [

    normalize_features(features)

    for features in X_validation

]


# ============================================================
# DISTANCE
# ============================================================

def euclidean_distance(a, b):

    total = 0.0

    for i in range(
        len(a)
    ):

        difference = (
            a[i] -
            b[i]
        )

        total += (
            difference *
            difference
        )

    return math.sqrt(total)


# ============================================================
# KNN PREDICTION
# ============================================================

def predict_knn(sample):

    distances = []


    for i in range(
        len(X_TRAIN)
    ):

        distance = euclidean_distance(

            sample,

            X_TRAIN[i]

        )

        distances.append(

            (
                distance,
                Y_TRAIN[i]
            )

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
            drowsy_votes /
            K
        )

    else:

        prediction = 0

        confidence = (
            awake_votes /
            K
        )


    return prediction, confidence


# ============================================================
# RUN VALIDATION
# ============================================================

print()
print("=" * 70)
print("Running independent validation...")
print("=" * 70)


predictions = []
confidences = []


for sample in X_validation_normalized:

    prediction, confidence = predict_knn(
        sample
    )

    predictions.append(prediction)
    confidences.append(confidence)


# ============================================================
# CONFUSION MATRIX
# ============================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


for actual, predicted in zip(
    y_validation,
    predictions
):

    if actual == 1 and predicted == 1:

        true_positive += 1

    elif actual == 0 and predicted == 0:

        true_negative += 1

    elif actual == 0 and predicted == 1:

        false_positive += 1

    elif actual == 1 and predicted == 0:

        false_negative += 1


# ============================================================
# METRICS
# ============================================================

total = len(
    y_validation
)


correct = (
    true_positive +
    true_negative
)


accuracy = (

    correct /
    total

) * 100


# Precision

if (
    true_positive +
    false_positive
) > 0:

    precision = (

        true_positive /
        (
            true_positive +
            false_positive
        )

    ) * 100

else:

    precision = 0.0


# Recall

if (
    true_positive +
    false_negative
) > 0:

    recall = (

        true_positive /
        (
            true_positive +
            false_negative
        )

    ) * 100

else:

    recall = 0.0


# F1

if (
    precision +
    recall
) > 0:

    f1 = (

        2 *
        precision *
        recall /
        (
            precision +
            recall
        )

    )

else:

    f1 = 0.0


# ============================================================
# CLASS ACCURACY
# ============================================================

if awake_total > 0:

    awake_accuracy = (

        true_negative /
        awake_total

    ) * 100

else:

    awake_accuracy = 0.0


if drowsy_total > 0:

    drowsy_accuracy = (

        true_positive /
        drowsy_total

    ) * 100

else:

    drowsy_accuracy = 0.0


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("                 VALIDATION RESULTS")
print("=" * 70)

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
print("=" * 70)
print("                    CONFUSION MATRIX")
print("=" * 70)

print()

print(
    "                 Predicted"
)

print(
    "               Awake   Drowsy"
)

print(
    f"Actual Awake    {true_negative:5d}   {false_positive:5d}"
)

print(
    f"Actual Drowsy   {false_negative:5d}   {true_positive:5d}"
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

print()
print("=" * 70)
print("                    ERROR ANALYSIS")
print("=" * 70)

print()

print(
    "False alarms:",
    false_positive
)

print(
    "Missed drowsiness:",
    false_negative
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("                  INTERPRETATION")
print("=" * 70)

print()

if false_negative > 0:

    print(
        "The model missed",
        false_negative,
        "drowsy samples."
    )

else:

    print(
        "The model did not miss any drowsy samples."
    )


if false_positive > 0:

    print(
        "The model produced",
        false_positive,
        "false alarms."
    )

else:

    print(
        "The model produced no false alarms."
    )


print()
print(
    "IMPORTANT: This validation result is based on"
)

print(
    "your newly collected validation session."
)

print(
    "It should not be interpreted as real-world"
)

print(
    "driving accuracy."
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("              VALIDATION COMPLETE")
print("=" * 70)
print()