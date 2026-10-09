import csv
import os
import pickle
import random
import math


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "combined_drowsiness_dataset.csv"
MODEL_PATH = "drowsiness_model_v3.pkl"

TEST_SIZE = 0.20
RANDOM_SEED = 42

K = 9


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
# FEATURE WEIGHTS
# ============================================================

# Higher value = greater influence on distance.

FEATURE_WEIGHTS = {

    "left_ear": 1.5,

    "right_ear": 1.5,

    "average_ear": 2.5,

    "mar": 1.4,

    "blink_left": 1.2,

    "blink_right": 1.2,

    "jaw_open": 1.4,

    "brow_left": 0.6,

    "brow_right": 0.6,

    "mouth_smile_left": 0.3,

    "mouth_smile_right": 0.3,

    "pitch": 2.0,

    "yaw": 0.7,

    "roll": 0.7,

    "brightness": 0.2,

    "contrast": 0.2,

    "sharpness": 0.2,

    "motion": 0.3,

    "eye_closure_duration": 3.0,

    "yawn_duration": 2.0,

    "head_down_duration": 3.0
}


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print()
    print("ERROR: Dataset not found:")
    print(DATASET_PATH)
    print()
    print("Run combine_datasets.py first.")
    raise SystemExit


# ============================================================
# LOAD DATA
# ============================================================

with open(
    DATASET_PATH,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


print()
print("=" * 70)
print("             DROWSINESS AI - MODEL V3")
print("=" * 70)

print()
print("Dataset:", DATASET_PATH)
print("Total rows:", len(rows))
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


print()
print("Valid samples:", len(X))


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
print("Calculating normalization parameters...")


means = []
stds = []


for column in range(
    len(FEATURE_NAMES)
):

    values = [
        row[column]
        for row in X
    ]

    mean = (
        sum(values) /
        len(values)
    )

    variance = (
        sum(
            (value - mean) ** 2
            for value in values
        ) /
        len(values)
    )

    std = math.sqrt(
        variance
    )

    if std == 0:

        std = 1.0

    means.append(mean)
    stds.append(std)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_row(row):

    normalized = []

    for i in range(
        len(row)
    ):

        value = (
            row[i] -
            means[i]
        ) / stds[i]

        normalized.append(value)

    return normalized


X_normalized = [

    normalize_row(row)

    for row in X

]


# ============================================================
# FEATURE WEIGHTS ARRAY
# ============================================================

weights = [

    FEATURE_WEIGHTS[name]

    for name in FEATURE_NAMES

]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

random.seed(
    RANDOM_SEED
)


indices = list(
    range(
        len(X_normalized)
    )
)


random.shuffle(
    indices
)


test_count = int(

    len(indices) *
    TEST_SIZE

)


test_indices = indices[
    :test_count
]

train_indices = indices[
    test_count:
]


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
# WEIGHTED DISTANCE
# ============================================================

def weighted_distance(
    a,
    b
):

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
            difference *
            weights[i]

        )


    return math.sqrt(
        total
    )


# ============================================================
# WEIGHTED KNN
# ============================================================

def predict_knn(
    sample
):

    distances = []


    for i in range(
        len(X_train)
    ):

        distance = weighted_distance(

            sample,

            X_train[i]

        )


        distances.append(

            (
                distance,
                y_train[i]
            )

        )


    distances.sort(
        key=lambda item: item[0]
    )


    nearest = distances[
        :K
    ]


    # --------------------------------------------------------
    # DISTANCE-WEIGHTED VOTING
    # --------------------------------------------------------

    awake_score = 0.0
    drowsy_score = 0.0


    for distance, label in nearest:

        # Prevent division by zero.

        vote_weight = (
            1.0 /
            (distance + 0.000001)
        )


        if label == 0:

            awake_score += (
                vote_weight
            )

        else:

            drowsy_score += (
                vote_weight
            )


    total_score = (

        awake_score +
        drowsy_score

    )


    if total_score == 0:

        return 0, 0.5


    drowsy_probability = (

        drowsy_score /
        total_score

    )


    if drowsy_probability >= 0.55:

        prediction = 1

        confidence = (
            drowsy_probability
        )

    else:

        prediction = 0

        confidence = (
            1.0 -
            drowsy_probability
        )


    return (
        prediction,
        confidence
    )


# ============================================================
# TEST MODEL
# ============================================================

print()
print("=" * 70)
print("Testing Model V3...")
print("=" * 70)


predictions = []
confidences = []


for index, sample in enumerate(
    X_test
):

    prediction, confidence = (
        predict_knn(sample)
    )

    predictions.append(
        prediction
    )

    confidences.append(
        confidence
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

true_positive = 0
true_negative = 0

false_positive = 0
false_negative = 0


for actual, predicted in zip(

    y_test,

    predictions

):

    if (
        actual == 1
        and
        predicted == 1
    ):

        true_positive += 1


    elif (
        actual == 0
        and
        predicted == 0
    ):

        true_negative += 1


    elif (
        actual == 0
        and
        predicted == 1
    ):

        false_positive += 1


    elif (
        actual == 1
        and
        predicted == 0
    ):

        false_negative += 1


# ============================================================
# METRICS
# ============================================================

total = len(
    y_test
)


correct = (

    true_positive +
    true_negative

)


accuracy = (

    correct /
    total

) * 100


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

awake_total = y_test.count(
    0
)

drowsy_total = y_test.count(
    1
)


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
print("                    MODEL V3 RESULTS")
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
    "False alarms       :",
    false_positive
)

print(
    "Missed drowsiness  :",
    false_negative
)


# ============================================================
# FEATURE WEIGHTS
# ============================================================

print()
print("=" * 70)
print("                    FEATURE WEIGHTS")
print("=" * 70)

print()


for name, weight in zip(
    FEATURE_NAMES,
    weights
):

    print(
        f"{name:25s} {weight:.2f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model = {

    "model_type":
        "weighted_distance_knn",

    "version":
        "3.0",

    "feature_names":
        FEATURE_NAMES,

    "feature_weights":
        FEATURE_WEIGHTS,

    "means":
        means,

    "stds":
        stds,

    "X_train":
        X_train,

    "y_train":
        y_train,

    "k":
        K,

    "decision_threshold":
        0.55

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
print("=" * 70)
print("                    MODEL SAVED")
print("=" * 70)

print()
print("File:")
print(MODEL_PATH)

print()
print("Model V3 training complete.")
print()