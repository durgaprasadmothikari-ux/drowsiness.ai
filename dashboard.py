import csv
import os
import tkinter as tk
from tkinter import ttk
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

LOG_FILE = "session_logs.csv"

REFRESH_TIME = 2000


# ============================================================
# COLORS
# ============================================================

BG = "#080b10"
PANEL = "#0e131a"
PANEL_2 = "#111821"

TEXT = "#e8edf2"
MUTED = "#7f8b98"

GREEN = "#35e58a"
RED = "#ff4d5a"
BLUE = "#4da3ff"
YELLOW = "#ffc857"


# ============================================================
# LOAD LOGS
# ============================================================

def load_sessions():

    if not os.path.exists(LOG_FILE):
        return []

    sessions = []

    try:

        with open(
            LOG_FILE,
            "r",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                try:

                    sessions.append({

                        "session_id":
                            row["session_id"],

                        "start_time":
                            row["start_time"],

                        "duration":
                            float(
                                row["duration_seconds"]
                            ),

                        "frames":
                            int(
                                row["total_frames"]
                            ),

                        "face_frames":
                            int(
                                row["face_detected_frames"]
                            ),

                        "events":
                            int(
                                row["drowsiness_events"]
                            ),

                        "alert_time":
                            float(
                                row["total_alert_time"]
                            ),

                        "longest_event":
                            float(
                                row["longest_drowsy_event"]
                            ),

                        "longest_eye":
                            float(
                                row["longest_eye_closure"]
                            ),

                        "longest_yawn":
                            float(
                                row["longest_yawn"]
                            ),

                        "longest_head":
                            float(
                                row["longest_head_down"]
                            ),

                        "max_score":
                            float(
                                row["maximum_drowsiness_score"]
                            ),

                        "avg_score":
                            float(
                                row["average_drowsiness_score"]
                            )

                    })

                except:
                    continue

    except:
        return []


    return sessions


# ============================================================
# FORMAT TIME
# ============================================================

def format_duration(seconds):

    seconds = int(seconds)

    minutes = seconds // 60

    seconds = seconds % 60

    return f"{minutes:02d}:{seconds:02d}"


# ============================================================
# MAIN APPLICATION
# ============================================================

class Dashboard:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Driver Drowsiness AI"
        )

        self.root.geometry(
            "1150x720"
        )

        self.root.minsize(
            950,
            650
        )

        self.root.configure(
            bg=BG
        )


        # ====================================================
        # STYLE
        # ====================================================

        style = ttk.Style()

        try:
            style.theme_use(
                "clam"
            )
        except:
            pass


        style.configure(

            "Treeview",

            background=PANEL,

            foreground=TEXT,

            fieldbackground=PANEL,

            borderwidth=0,

            rowheight=32

        )


        style.configure(

            "Treeview.Heading",

            background=PANEL_2,

            foreground=TEXT,

            borderwidth=0,

            font=(
                "Segoe UI",
                10,
                "bold"
            )

        )


        # ====================================================
        # HEADER
        # ====================================================

        header = tk.Frame(

            root,

            bg=BG

        )

        header.pack(

            fill="x",

            padx=30,

            pady=(25, 15)

        )


        title = tk.Label(

            header,

            text="DRIVER DROWSINESS AI",

            bg=BG,

            fg=TEXT,

            font=(
                "Segoe UI",
                22,
                "bold"
            )

        )

        title.pack(
            side="left"
        )


        subtitle = tk.Label(

            header,

            text="SESSION ANALYTICS",

            bg=BG,

            fg=BLUE,

            font=(
                "Segoe UI",
                10,
                "bold"
            )

        )

        subtitle.pack(

            side="left",

            padx=15,

            pady=(7, 0)

        )


        # ====================================================
        # STATUS
        # ====================================================

        self.status_label = tk.Label(

            header,

            text="● SYSTEM READY",

            bg=BG,

            fg=GREEN,

            font=(
                "Segoe UI",
                10,
                "bold"
            )

        )

        self.status_label.pack(

            side="right",

            pady=5

        )


        # ====================================================
        # SUMMARY CARDS
        # ====================================================

        cards = tk.Frame(

            root,

            bg=BG

        )

        cards.pack(

            fill="x",

            padx=30,

            pady=10

        )


        self.total_sessions = self.create_card(

            cards,

            "TOTAL SESSIONS",

            BLUE

        )


        self.total_events = self.create_card(

            cards,

            "DROWSINESS EVENTS",

            RED

        )


        self.alert_time = self.create_card(

            cards,

            "TOTAL ALERT TIME",

            YELLOW

        )


        self.avg_score = self.create_card(

            cards,

            "AVG DROWSINESS",

            GREEN

        )


        # ====================================================
        # ANALYTICS AREA
        # ====================================================

        analytics = tk.Frame(

            root,

            bg=BG

        )

        analytics.pack(

            fill="both",

            expand=True,

            padx=30,

            pady=15

        )


        # ====================================================
        # RECENT SESSIONS
        # ====================================================

        sessions_panel = tk.Frame(

            analytics,

            bg=PANEL

        )

        sessions_panel.pack(

            fill="both",

            expand=True

        )


        panel_title = tk.Label(

            sessions_panel,

            text="SESSION HISTORY",

            bg=PANEL,

            fg=TEXT,

            font=(
                "Segoe UI",
                12,
                "bold"
            )

        )

        panel_title.pack(

            anchor="w",

            padx=18,

            pady=(15, 10)

        )


        columns = (

            "session",
            "date",
            "duration",
            "events",
            "alert",
            "max",
            "average"

        )


        self.tree = ttk.Treeview(

            sessions_panel,

            columns=columns,

            show="headings"

        )


        self.tree.heading(
            "session",
            text="SESSION"
        )

        self.tree.heading(
            "date",
            text="DATE"
        )

        self.tree.heading(
            "duration",
            text="DURATION"
        )

        self.tree.heading(
            "events",
            text="EVENTS"
        )

        self.tree.heading(
            "alert",
            text="ALERT TIME"
        )

        self.tree.heading(
            "max",
            text="MAX SCORE"
        )

        self.tree.heading(
            "average",
            text="AVG SCORE"
        )


        self.tree.column(
            "session",
            width=180
        )

        self.tree.column(
            "date",
            width=150
        )

        self.tree.column(
            "duration",
            width=100,
            anchor="center"
        )

        self.tree.column(
            "events",
            width=90,
            anchor="center"
        )

        self.tree.column(
            "alert",
            width=110,
            anchor="center"
        )

        self.tree.column(
            "max",
            width=100,
            anchor="center"
        )

        self.tree.column(
            "average",
            width=100,
            anchor="center"
        )


        self.tree.pack(

            fill="both",

            expand=True,

            padx=15,

            pady=(0, 15)

        )


        # ====================================================
        # DETAILS PANEL
        # ====================================================

        self.details = tk.Label(

            root,

            text="Select a session to view details.",

            bg=BG,

            fg=MUTED,

            font=(
                "Segoe UI",
                10
            ),

            justify="left"

        )

        self.details.pack(

            anchor="w",

            padx=30,

            pady=(0, 20)

        )


        self.tree.bind(

            "<<TreeviewSelect>>",

            self.show_details

        )


        # ====================================================
        # START REFRESH
        # ====================================================

        self.refresh()


    # ========================================================
    # CREATE CARD
    # ========================================================

    def create_card(

        self,
        parent,
        title,
        accent

    ):

        frame = tk.Frame(

            parent,

            bg=PANEL,

            width=250,

            height=95

        )

        frame.pack(

            side="left",

            fill="both",

            expand=True,

            padx=5

        )

        frame.pack_propagate(
            False
        )


        label = tk.Label(

            frame,

            text=title,

            bg=PANEL,

            fg=MUTED,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        )

        label.pack(

            anchor="w",

            padx=15,

            pady=(12, 2)

        )


        value = tk.Label(

            frame,

            text="0",

            bg=PANEL,

            fg=accent,

            font=(
                "Segoe UI",
                22,
                "bold"
            )

        )

        value.pack(

            anchor="w",

            padx=15

        )


        return value


    # ========================================================
    # REFRESH
    # ========================================================

    def refresh(self):

        sessions = load_sessions()


        # ----------------------------------------------------
        # CLEAR TABLE
        # ----------------------------------------------------

        for item in self.tree.get_children():

            self.tree.delete(
                item
            )


        # ----------------------------------------------------
        # INSERT SESSIONS
        # ----------------------------------------------------

        for session in reversed(
            sessions
        ):

            self.tree.insert(

                "",

                "end",

                values=(

                    session["session_id"],

                    session["start_time"],

                    format_duration(
                        session["duration"]
                    ),

                    session["events"],

                    format_duration(
                        session["alert_time"]
                    ),

                    f"{session['max_score'] * 100:.1f}%",

                    f"{session['avg_score'] * 100:.1f}%"

                )

            )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        total_sessions = len(
            sessions
        )


        total_events = sum(

            session["events"]

            for session in sessions

        )


        total_alert = sum(

            session["alert_time"]

            for session in sessions

        )


        if sessions:

            average_score = (

                sum(
                    session["avg_score"]
                    for session in sessions
                )

                /

                len(sessions)

            )

        else:

            average_score = 0.0


        self.total_sessions.config(

            text=str(
                total_sessions
            )

        )


        self.total_events.config(

            text=str(
                total_events
            )

        )


        self.alert_time.config(

            text=format_duration(
                total_alert
            )

        )


        self.avg_score.config(

            text=f"{average_score * 100:.1f}%"

        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if sessions:

            self.status_label.config(

                text="● DATA ONLINE",

                fg=GREEN

            )

        else:

            self.status_label.config(

                text="● WAITING FOR SESSION",

                fg=YELLOW

            )


        # ----------------------------------------------------
        # REFRESH AGAIN
        # ----------------------------------------------------

        self.root.after(

            REFRESH_TIME,

            self.refresh

        )


    # ========================================================
    # DETAILS
    # ========================================================

    def show_details(
        self,
        event
    ):

        selected = self.tree.selection()

        if not selected:
            return


        item = self.tree.item(
            selected[0]
        )


        values = item["values"]


        session_id = values[0]


        sessions = load_sessions()


        for session in sessions:

            if (
                session["session_id"]
                ==
                session_id
            ):

                text = (

                    f"SESSION {session_id}\n"

                    f"Duration: "
                    f"{format_duration(session['duration'])}    "

                    f"Events: "
                    f"{session['events']}    "

                    f"Alert time: "
                    f"{format_duration(session['alert_time'])}\n"

                    f"Longest eye closure: "
                    f"{session['longest_eye']:.2f}s    "

                    f"Longest yawn: "
                    f"{session['longest_yawn']:.2f}s    "

                    f"Head-down: "
                    f"{session['longest_head']:.2f}s    "

                    f"Maximum drowsiness: "
                    f"{session['max_score'] * 100:.1f}%"

                )


                self.details.config(

                    text=text,

                    fg=TEXT

                )

                break


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = Dashboard(
        root
    )

    root.mainloop()