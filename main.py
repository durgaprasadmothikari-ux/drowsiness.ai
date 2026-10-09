import os
import subprocess
import sys
import time


# ============================================================
# DRIVER DROWSINESS AI - MAIN LAUNCHER
# ============================================================

APP_NAME = "Driver Drowsiness AI"


def print_header():

    print()
    print("=" * 70)
    print("              DRIVER DROWSINESS AI")
    print("=" * 70)
    print()
    print("1. Start live drowsiness detection")
    print("2. Open session dashboard")
    print("3. Start detection + dashboard")
    print("4. Exit")
    print()


def check_file(filename):

    if not os.path.exists(filename):

        print()
        print("ERROR: Required file not found:")
        print(filename)
        print()

        return False

    return True


def start_detection():

    if not check_file(
        "ml_prediction_v3.py"
    ):
        return

    print()
    print("Starting Driver Drowsiness AI...")
    print()
    print("Press Q inside the camera window to stop.")
    print()

    subprocess.run(
        [
            sys.executable,
            "ml_prediction_v3.py"
        ]
    )


def start_dashboard():

    if not check_file(
        "dashboard.py"
    ):
        return

    print()
    print("Starting dashboard...")
    print()

    subprocess.run(
        [
            sys.executable,
            "dashboard.py"
        ]
    )


def start_both():

    if not check_file(
        "ml_prediction_v3.py"
    ):
        return

    if not check_file(
        "dashboard.py"
    ):
        return

    print()
    print("Starting complete system...")
    print()

    # Start dashboard first
    dashboard_process = subprocess.Popen(
        [
            sys.executable,
            "dashboard.py"
        ]
    )

    # Give dashboard time to open
    time.sleep(2)

    try:

        subprocess.run(
            [
                sys.executable,
                "ml_prediction_v3.py"
            ]
        )

    finally:

        # Close dashboard after detector exits
        dashboard_process.terminate()

        try:
            dashboard_process.wait(
                timeout=3
            )

        except subprocess.TimeoutExpired:

            dashboard_process.kill()


def main():

    while True:

        print_header()

        choice = input(
            "Select an option: "
        ).strip()


        if choice == "1":

            start_detection()


        elif choice == "2":

            start_dashboard()


        elif choice == "3":

            start_both()


        elif choice == "4":

            print()
            print(
                "Driver Drowsiness AI closed."
            )
            print()

            break


        else:

            print()
            print(
                "Invalid option."
            )

            time.sleep(1)


if __name__ == "__main__":

    main()