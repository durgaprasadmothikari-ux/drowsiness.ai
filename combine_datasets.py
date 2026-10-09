import csv
import os


TRAINING_DATASET = "advanced_drowsiness_dataset.csv"
VALIDATION_DATASET = "validation_dataset.csv"
OUTPUT_DATASET = "combined_drowsiness_dataset.csv"


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TRAINING_DATASET):
    print("ERROR:", TRAINING_DATASET, "not found.")
    raise SystemExit

if not os.path.exists(VALIDATION_DATASET):
    print("ERROR:", VALIDATION_DATASET, "not found.")
    raise SystemExit


# ============================================================
# READ FIRST DATASET
# ============================================================

with open(
    TRAINING_DATASET,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    training_rows = list(reader)

    fieldnames = reader.fieldnames


# ============================================================
# READ SECOND DATASET
# ============================================================

with open(
    VALIDATION_DATASET,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    validation_rows = list(reader)


# ============================================================
# COMBINE
# ============================================================

combined_rows = []

combined_rows.extend(training_rows)
combined_rows.extend(validation_rows)


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_DATASET,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        combined_rows
    )


# ============================================================
# STATISTICS
# ============================================================

awake = 0
drowsy = 0


for row in combined_rows:

    if row["label"] == "0":
        awake += 1

    elif row["label"] == "1":
        drowsy += 1


print()
print("=" * 60)
print("       DATASETS COMBINED")
print("=" * 60)

print()

print(
    "Original training samples :",
    len(training_rows)
)

print(
    "Previous validation samples:",
    len(validation_rows)
)

print(
    "Combined samples          :",
    len(combined_rows)
)

print()

print(
    "Awake samples :",
    awake
)

print(
    "Drowsy samples:",
    drowsy
)

print()

print(
    "Saved as:"
)

print(
    OUTPUT_DATASET
)

print()