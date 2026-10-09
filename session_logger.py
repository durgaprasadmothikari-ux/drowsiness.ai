import csv
import os
import time
from datetime import datetime


# ============================================================
# SESSION LOGGER
# ============================================================

class SessionLogger:

    def __init__(
        self,
        filename="session_logs.csv"
    ):

        self.filename = filename

        # ----------------------------------------------------
        # SESSION INFORMATION
        # ----------------------------------------------------

        self.start_time = time.time()

        self.session_id = (
            datetime.now()
            .strftime("%Y%m%d_%H%M%S")
        )


        # ----------------------------------------------------
        # FRAME STATISTICS
        # ----------------------------------------------------

        self.total_frames = 0

        self.face_detected_frames = 0


        # ----------------------------------------------------
        # DROWSINESS STATISTICS
        # ----------------------------------------------------

        self.drowsiness_events = 0

        self.total_alert_time = 0.0

        self.current_drowsy_start = None

        self.longest_drowsy_event = 0.0


        # ----------------------------------------------------
        # BEHAVIOR STATISTICS
        # ----------------------------------------------------

        self.longest_eye_closure = 0.0

        self.longest_yawn = 0.0

        self.longest_head_down = 0.0


        # ----------------------------------------------------
        # SCORE STATISTICS
        # ----------------------------------------------------

        self.score_sum = 0.0

        self.score_samples = 0

        self.maximum_drowsiness_score = 0.0


        # ----------------------------------------------------
        # LAST STATE
        # ----------------------------------------------------

        self.previous_state = "NO FACE"


        # ----------------------------------------------------
        # CREATE CSV
        # ----------------------------------------------------

        if not os.path.exists(self.filename):

            with open(
                self.filename,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([

                    "session_id",
                    "start_time",
                    "end_time",
                    "duration_seconds",

                    "total_frames",
                    "face_detected_frames",

                    "drowsiness_events",
                    "total_alert_time",

                    "longest_drowsy_event",
                    "longest_eye_closure",
                    "longest_yawn",
                    "longest_head_down",

                    "maximum_drowsiness_score",
                    "average_drowsiness_score"

                ])


    # ========================================================
    # FRAME UPDATE
    # ========================================================

    def update(

        self,
        state,
        drowsiness_score,
        eye_closure_duration,
        yawn_duration,
        head_down_duration

    ):

        now = time.time()


        # ----------------------------------------------------
        # FRAME COUNT
        # ----------------------------------------------------

        self.total_frames += 1


        if state != "NO FACE":

            self.face_detected_frames += 1


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        self.score_sum += drowsiness_score

        self.score_samples += 1


        if (
            drowsiness_score >
            self.maximum_drowsiness_score
        ):

            self.maximum_drowsiness_score = (
                drowsiness_score
            )


        # ----------------------------------------------------
        # BEHAVIOR DURATIONS
        # ----------------------------------------------------

        if (
            eye_closure_duration >
            self.longest_eye_closure
        ):

            self.longest_eye_closure = (
                eye_closure_duration
            )


        if (
            yawn_duration >
            self.longest_yawn
        ):

            self.longest_yawn = (
                yawn_duration
            )


        if (
            head_down_duration >
            self.longest_head_down
        ):

            self.longest_head_down = (
                head_down_duration
            )


        # ----------------------------------------------------
        # DROWSINESS EVENT START
        # ----------------------------------------------------

        if (

            state == "DROWSY"

            and

            self.previous_state != "DROWSY"

        ):

            self.drowsiness_events += 1

            self.current_drowsy_start = now


        # ----------------------------------------------------
        # DROWSINESS EVENT END
        # ----------------------------------------------------

        if (

            state != "DROWSY"

            and

            self.previous_state == "DROWSY"

            and

            self.current_drowsy_start is not None

        ):

            duration = (

                now -
                self.current_drowsy_start

            )


            self.total_alert_time += duration


            if (
                duration >
                self.longest_drowsy_event
            ):

                self.longest_drowsy_event = (
                    duration
                )


            self.current_drowsy_start = None


        # ----------------------------------------------------
        # CURRENT ALERT TIME
        # ----------------------------------------------------

        if (

            state == "DROWSY"

            and

            self.current_drowsy_start is not None

        ):

            current_duration = (

                now -
                self.current_drowsy_start

            )


            if (
                current_duration >
                self.longest_drowsy_event
            ):

                self.longest_drowsy_event = (
                    current_duration
                )


        # ----------------------------------------------------
        # SAVE STATE
        # ----------------------------------------------------

        self.previous_state = state


    # ========================================================
    # FINALIZE SESSION
    # ========================================================

    def finish(self):

        end_time = time.time()


        # ----------------------------------------------------
        # CLOSE ACTIVE DROWSY EVENT
        # ----------------------------------------------------

        if (

            self.current_drowsy_start
            is not None

        ):

            duration = (

                end_time -
                self.current_drowsy_start

            )


            self.total_alert_time += duration


            if (
                duration >
                self.longest_drowsy_event
            ):

                self.longest_drowsy_event = (
                    duration
                )


            self.current_drowsy_start = None


        # ----------------------------------------------------
        # SESSION DURATION
        # ----------------------------------------------------

        duration = (

            end_time -
            self.start_time

        )


        # ----------------------------------------------------
        # AVERAGE SCORE
        # ----------------------------------------------------

        if self.score_samples > 0:

            average_score = (

                self.score_sum /
                self.score_samples

            )

        else:

            average_score = 0.0


        # ----------------------------------------------------
        # TIMES
        # ----------------------------------------------------

        start_datetime = datetime.fromtimestamp(
            self.start_time
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        end_datetime = datetime.fromtimestamp(
            end_time
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        # ----------------------------------------------------
        # SAVE SESSION
        # ----------------------------------------------------

        with open(
            self.filename,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                self.session_id,

                start_datetime,

                end_datetime,

                round(
                    duration,
                    2
                ),

                self.total_frames,

                self.face_detected_frames,

                self.drowsiness_events,

                round(
                    self.total_alert_time,
                    2
                ),

                round(
                    self.longest_drowsy_event,
                    2
                ),

                round(
                    self.longest_eye_closure,
                    2
                ),

                round(
                    self.longest_yawn,
                    2
                ),

                round(
                    self.longest_head_down,
                    2
                ),

                round(
                    self.maximum_drowsiness_score,
                    4
                ),

                round(
                    average_score,
                    4
                )

            ])


        # ----------------------------------------------------
        # RETURN SUMMARY
        # ----------------------------------------------------

        return {

            "session_id":
                self.session_id,

            "duration":
                duration,

            "total_frames":
                self.total_frames,

            "face_frames":
                self.face_detected_frames,

            "drowsiness_events":
                self.drowsiness_events,

            "total_alert_time":
                self.total_alert_time,

            "longest_drowsy_event":
                self.longest_drowsy_event,

            "longest_eye_closure":
                self.longest_eye_closure,

            "longest_yawn":
                self.longest_yawn,

            "longest_head_down":
                self.longest_head_down,

            "maximum_score":
                self.maximum_drowsiness_score,

            "average_score":
                average_score

        }


# ============================================================
# TEST LOGGER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("           SESSION LOGGER TEST")
    print("=" * 60)
    print()


    logger = SessionLogger(
        "session_logs.csv"
    )


    print("Testing logger...")


    # Simulate awake frames

    for _ in range(20):

        logger.update(

            state="AWAKE",

            drowsiness_score=0.15,

            eye_closure_duration=0.0,

            yawn_duration=0.0,

            head_down_duration=0.0

        )


    # Simulate drowsiness

    for _ in range(20):

        logger.update(

            state="DROWSY",

            drowsiness_score=0.85,

            eye_closure_duration=1.2,

            yawn_duration=0.8,

            head_down_duration=1.0

        )


    # Back to awake

    for _ in range(20):

        logger.update(

            state="AWAKE",

            drowsiness_score=0.20,

            eye_closure_duration=0.0,

            yawn_duration=0.0,

            head_down_duration=0.0

        )


    summary = logger.finish()


    print()
    print("Logger test completed.")
    print()
    print("Session ID:",
          summary["session_id"])

    print(
        "Drowsiness events:",
        summary["drowsiness_events"]
    )

    print(
        "Maximum score:",
        summary["maximum_score"]
    )

    print()
    print(
        "Created/updated:"
    )

    print(
        "session_logs.csv"
    )

    print()