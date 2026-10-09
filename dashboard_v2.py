import csv
import os
import tkinter as tk
from tkinter import ttk


# ============================================================
# CONFIGURATION
# ============================================================

LOG_FILE = "session_logs.csv"
REFRESH_TIME = 2000


# ============================================================
# COLORS
# ============================================================

BG = "#070a0f"
PANEL = "#0d1219"
PANEL_2 = "#121923"

TEXT = "#e9eef3"
MUTED = "#788594"

GREEN = "#35e58a"
RED = "#ff4d5a"
BLUE = "#4da3ff"
YELLOW = "#ffc857"
CYAN = "#48d7ff"


# ============================================================
# LOAD SESSION DATA
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

def format_time(seconds):

    seconds = int(seconds)

    minutes = seconds // 60

    seconds = seconds % 60

    return f"{minutes:02d}:{seconds:02d}"


# ============================================================
# DASHBOARD
# ============================================================

class Dashboard:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Driver Drowsiness AI — Analytics"
        )

        self.root.geometry(
            "1250x800"
        )

        self.root.minsize(
            1050,
            700
        )

        self.root.configure(
            bg=BG
        )


        # ====================================================
        # STYLE
        # ====================================================

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except:
            pass


        style.configure(

            "Treeview",

            background=PANEL,

            foreground=TEXT,

            fieldbackground=PANEL,

            borderwidth=0,

            rowheight=30,

            font=(
                "Segoe UI",
                9
            )

        )


        style.configure(

            "Treeview.Heading",

            background=PANEL_2,

            foreground=TEXT,

            borderwidth=0,

            font=(
                "Segoe UI",
                9,
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
            padx=28,
            pady=(22, 12)
        )


        title = tk.Label(

            header,

            text="DRIVER DROWSINESS AI",

            bg=BG,

            fg=TEXT,

            font=(
                "Segoe UI",
                21,
                "bold"
            )

        )

        title.pack(
            side="left"
        )


        subtitle = tk.Label(

            header,

            text="  /  ANALYTICS",

            bg=BG,

            fg=CYAN,

            font=(
                "Segoe UI",
                10,
                "bold"
            )

        )

        subtitle.pack(
            side="left",
            pady=(7, 0)
        )


        self.status = tk.Label(

            header,

            text="● SYSTEM READY",

            bg=BG,

            fg=GREEN,

            font=(
                "Segoe UI",
                9,
                "bold"
            )

        )

        self.status.pack(
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
            padx=23,
            pady=5
        )


        self.sessions_value = self.card(

            cards,
            "SESSIONS",
            BLUE
        )


        self.events_value = self.card(

            cards,
            "DROWSINESS EVENTS",
            RED
        )


        self.alert_value = self.card(

            cards,
            "ALERT TIME",
            YELLOW
        )


        self.average_value = self.card(

            cards,
            "AVERAGE RISK",
            GREEN
        )


        self.maximum_value = self.card(

            cards,
            "MAXIMUM RISK",
            RED
        )


        # ====================================================
        # MIDDLE AREA
        # ====================================================

        middle = tk.Frame(
            root,
            bg=BG
        )

        middle.pack(
            fill="both",
            expand=True,
            padx=28,
            pady=12
        )


        # ====================================================
        # LEFT CHART
        # ====================================================

        chart_panel = tk.Frame(

            middle,

            bg=PANEL

        )

        chart_panel.pack(

            side="left",

            fill="both",

            expand=True,

            padx=(0, 7)

        )


        tk.Label(

            chart_panel,

            text="DROWSINESS EVENTS",

            bg=PANEL,

            fg=TEXT,

            font=(
                "Segoe UI",
                11,
                "bold"
            )

        ).pack(

            anchor="w",

            padx=18,

            pady=(15, 5)

        )


        self.chart = tk.Canvas(

            chart_panel,

            bg=PANEL,

            highlightthickness=0

        )

        self.chart.pack(

            fill="both",

            expand=True,

            padx=10,

            pady=10

        )


        # ====================================================
        # RIGHT RISK PANEL
        # ====================================================

        risk_panel = tk.Frame(

            middle,

            bg=PANEL,

            width=300

        )

        risk_panel.pack(

            side="right",

            fill="y",

            padx=(7, 0)

        )

        risk_panel.pack_propagate(
            False
        )


        tk.Label(

            risk_panel,

            text="RISK ANALYSIS",

            bg=PANEL,

            fg=TEXT,

            font=(
                "Segoe UI",
                11,
                "bold"
            )

        ).pack(

            anchor="w",

            padx=18,

            pady=(15, 15)

        )


        self.risk_canvas = tk.Canvas(

            risk_panel,

            bg=PANEL,

            highlightthickness=0,

            height=120

        )

        self.risk_canvas.pack(

            fill="x",

            padx=15

        )


        self.risk_text = tk.Label(

            risk_panel,

            text="NO DATA",

            bg=PANEL,

            fg=MUTED,

            font=(
                "Segoe UI",
                16,
                "bold"
            )

        )

        self.risk_text.pack(
            pady=5
        )


        self.detail_label = tk.Label(

            risk_panel,

            text="",

            bg=PANEL,

            fg=MUTED,

            justify="left",

            font=(
                "Segoe UI",
                9
            )

        )

        self.detail_label.pack(

            anchor="w",

            padx=18,

            pady=15

        )


        # ====================================================
        # SESSION HISTORY
        # ====================================================

        history_panel = tk.Frame(

            root,

            bg=PANEL

        )

        history_panel.pack(

            fill="both",

            expand=False,

            padx=28,

            pady=(0, 20)

        )


        tk.Label(

            history_panel,

            text="RECENT SESSION HISTORY",

            bg=PANEL,

            fg=TEXT,

            font=(
                "Segoe UI",
                11,
                "bold"
            )

        ).pack(

            anchor="w",

            padx=18,

            pady=(12, 8)

        )


        columns = (

            "session",
            "date",
            "duration",
            "events",
            "alert",
            "max",
            "avg"

        )


        self.tree = ttk.Treeview(

            history_panel,

            columns=columns,

            show="headings",

            height=5

        )


        headings = {

            "session": "SESSION",

            "date": "DATE",

            "duration": "DURATION",

            "events": "EVENTS",

            "alert": "ALERT",

            "max": "MAX RISK",

            "avg": "AVG RISK"

        }


        widths = {

            "session": 180,

            "date": 155,

            "duration": 100,

            "events": 80,

            "alert": 90,

            "max": 90,

            "avg": 90

        }


        for column in columns:

            self.tree.heading(

                column,

                text=headings[column]

            )

            self.tree.column(

                column,

                width=widths[column],

                anchor="center"

            )


        self.tree.pack(

            fill="x",

            padx=15,

            pady=(0, 15)

        )


        self.tree.bind(

            "<<TreeviewSelect>>",

            self.select_session

        )


        # ====================================================
        # START REFRESH
        # ====================================================

        self.refresh()


    # ========================================================
    # CARD
    # ========================================================

    def card(

        self,
        parent,
        title,
        accent

    ):

        frame = tk.Frame(

            parent,

            bg=PANEL,

            height=88

        )

        frame.pack(

            side="left",

            fill="x",

            expand=True,

            padx=5

        )

        frame.pack_propagate(
            False
        )


        tk.Label(

            frame,

            text=title,

            bg=PANEL,

            fg=MUTED,

            font=(
                "Segoe UI",
                8,
                "bold"
            )

        ).pack(

            anchor="w",

            padx=14,

            pady=(11, 0)

        )


        value = tk.Label(

            frame,

            text="0",

            bg=PANEL,

            fg=accent,

            font=(
                "Segoe UI",
                19,
                "bold"
            )

        )

        value.pack(

            anchor="w",

            padx=14,

            pady=(2, 0)

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

            self.tree.delete(item)


        # ----------------------------------------------------
        # INSERT RECENT SESSIONS
        # ----------------------------------------------------

        for session in reversed(
            sessions[-10:]
        ):

            self.tree.insert(

                "",

                "end",

                values=(

                    session["session_id"],

                    session["start_time"],

                    format_time(
                        session["duration"]
                    ),

                    session["events"],

                    format_time(
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

            s["events"]

            for s in sessions

        )


        total_alert = sum(

            s["alert_time"]

            for s in sessions

        )


        if sessions:

            avg_risk = (

                sum(
                    s["avg_score"]
                    for s in sessions
                )

                /

                len(sessions)

            )


            max_risk = max(

                s["max_score"]

                for s in sessions

            )

        else:

            avg_risk = 0
            max_risk = 0


        self.sessions_value.config(

            text=str(
                total_sessions
            )

        )


        self.events_value.config(

            text=str(
                total_events
            )

        )


        self.alert_value.config(

            text=format_time(
                total_alert
            )

        )


        self.average_value.config(

            text=f"{avg_risk * 100:.1f}%"

        )


        self.maximum_value.config(

            text=f"{max_risk * 100:.1f}%"

        )


        # ----------------------------------------------------
        # RISK DISPLAY
        # ----------------------------------------------------

        self.draw_risk(
            avg_risk
        )


        # ----------------------------------------------------
        # EVENT CHART
        # ----------------------------------------------------

        self.draw_chart(
            sessions
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if sessions:

            self.status.config(

                text="● DATA ONLINE",

                fg=GREEN

            )

        else:

            self.status.config(

                text="● WAITING FOR DATA",

                fg=YELLOW

            )


        self.root.after(

            REFRESH_TIME,

            self.refresh

        )


    # ========================================================
    # RISK DISPLAY
    # ========================================================

    def draw_risk(
        self,
        risk
    ):

        self.risk_canvas.delete(
            "all"
        )


        width = self.risk_canvas.winfo_width()


        if width < 50:
            width = 250


        bar_width = width - 30


        # Background

        self.risk_canvas.create_rectangle(

            15,

            35,

            15 + bar_width,

            58,

            fill=PANEL_2,

            outline=""

        )


        # Risk amount

        fill_width = (

            bar_width *
            max(
                0,
                min(
                    risk,
                    1
                )
            )

        )


        if risk < 0.35:

            color = GREEN
            label = "LOW RISK"

        elif risk < 0.65:

            color = YELLOW
            label = "MODERATE RISK"

        else:

            color = RED
            label = "HIGH RISK"


        self.risk_canvas.create_rectangle(

            15,

            35,

            15 + fill_width,

            58,

            fill=color,

            outline=""

        )


        self.risk_canvas.create_text(

            width / 2,

            20,

            text=f"{risk * 100:.1f}%",

            fill=TEXT,

            font=(
                "Segoe UI",
                15,
                "bold"
            )

        )


        self.risk_text.config(

            text=label,

            fg=color

        )


    # ========================================================
    # EVENT CHART
    # ========================================================

    def draw_chart(
        self,
        sessions
    ):

        self.chart.delete(
            "all"
        )


        width = self.chart.winfo_width()

        height = self.chart.winfo_height()


        if width < 100:
            return

        if height < 100:
            return


        recent = sessions[-8:]


        if not recent:

            self.chart.create_text(

                width / 2,

                height / 2,

                text="NO SESSION DATA",

                fill=MUTED,

                font=(
                    "Segoe UI",
                    11
                )

            )

            return


        max_events = max(

            1,

            max(
                s["events"]
                for s in recent
            )

        )


        chart_bottom = height - 35

        chart_top = 25

        chart_height = (
            chart_bottom -
            chart_top
        )


        bar_gap = 12

        bar_width = (

            width -
            40 -
            (
                len(recent) -
                1
            ) *
            bar_gap

        ) / len(recent)


        for index, session in enumerate(
            recent
        ):

            x1 = (

                20 +

                index *
                (
                    bar_width +
                    bar_gap
                )

            )


            x2 = x1 + bar_width


            bar_height = (

                session["events"] /
                max_events

            ) * chart_height


            y1 = (
                chart_bottom -
                bar_height
            )


            self.chart.create_rectangle(

                x1,

                y1,

                x2,

                chart_bottom,

                fill=RED,

                outline=""

            )


            self.chart.create_text(

                (x1 + x2) / 2,

                y1 - 10,

                text=str(
                    session["events"]
                ),

                fill=TEXT,

                font=(
                    "Segoe UI",
                    8,
                    "bold"
                )

            )


            self.chart.create_text(

                (x1 + x2) / 2,

                chart_bottom + 15,

                text=str(
                    index + 1
                ),

                fill=MUTED,

                font=(
                    "Segoe UI",
                    8
                )

            )


    # ========================================================
    # SESSION SELECTION
    # ========================================================

    def select_session(
        self,
        event
    ):

        selected = self.tree.selection()

        if not selected:
            return


        values = self.tree.item(
            selected[0]
        )["values"]


        session_id = values[0]


        sessions = load_sessions()


        for session in sessions:

            if (
                session["session_id"]
                ==
                session_id
            ):

                self.detail_label.config(

                    text=(

                        f"SESSION {session_id}\n\n"

                        f"Longest eye closure: "
                        f"{session['longest_eye']:.2f}s\n"

                        f"Longest yawn: "
                        f"{session['longest_yawn']:.2f}s\n"

                        f"Longest head-down period: "
                        f"{session['longest_head']:.2f}s\n\n"

                        f"Longest drowsiness event: "
                        f"{session['longest_event']:.2f}s\n"

                        f"Frames analyzed: "
                        f"{session['frames']}\n"

                        f"Face detected: "
                        f"{session['face_frames']} frames"

                    ),

                    fg=TEXT

                )

                break


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    Dashboard(
        root
    )

    root.mainloop()