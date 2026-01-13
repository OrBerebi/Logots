# gui_controller.py

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import time
from threading import Event
import recording_module

class RecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Controller")

        # Input
        self.label = tk.Label(root, text="Recording Duration (sec):")
        self.label.pack()
        self.duration_entry = tk.Entry(root)
        self.duration_entry.insert(0, "10")
        self.duration_entry.pack()

        # Buttons
        self.start_button = tk.Button(root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop", state=tk.DISABLED, command=self.stop_recording)
        self.stop_button.pack(pady=5)

        # Countdown
        self.remaining_label = tk.Label(root, text="")
        self.remaining_label.pack()

        # Internal state
        self.stop_event = Event()
        self.duration = 0
        self.start_time = None
        self.thread = None
        self.running = False
        self.recording = False
        self.speed = 75
        self.arm_angle = 90  # default position

                # Add Movement Controls
        movement_frame = ttk.LabelFrame(root, text="Movement Controls", padding=10)
        movement_frame.pack(padx=10, pady=10, fill="x")

        # Speed slider (left side)
        self.speed_slider = tk.Scale(movement_frame, from_=120, to=60,
                                     orient=tk.VERTICAL, label="Speed",
                                     command=self.update_speed)
        self.speed_slider.set(self.speed)
        self.speed_slider.grid(row=0, column=0, rowspan=3, padx=10, pady=5)

        # Arrow buttons (center in columns 1–3)
        self.forward_btn  = ttk.Button(movement_frame, text="↑")
        self.backward_btn = ttk.Button(movement_frame, text="↓")
        self.left_btn     = ttk.Button(movement_frame, text="←")
        self.right_btn    = ttk.Button(movement_frame, text="→")

        # Layout in grid (D-pad style, shifted to center)
        self.forward_btn.grid(row=0, column=2, padx=5, pady=5)
        self.left_btn.grid(row=1, column=1, padx=5, pady=5, sticky="e")
        self.right_btn.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.backward_btn.grid(row=2, column=2, padx=5, pady=5)

        # Arm angle slider (right side)
        self.arm_slider = tk.Scale(movement_frame, from_=180, to=0,
                                   orient=tk.VERTICAL, label="Arm Angle",
                                   command=self.update_arm_angle)
        self.arm_slider.set(self.arm_angle)
        self.arm_slider.grid(row=0, column=4, rowspan=3, padx=10, pady=5)

        # Bind press + release events
        self.forward_btn.bind("<ButtonPress>", lambda e: self.move_command(self.speed, self.speed))
        self.forward_btn.bind("<ButtonRelease>", lambda e: self.move_command(0, 0))

        self.backward_btn.bind("<ButtonPress>", lambda e: self.move_command(-self.speed, -self.speed))
        self.backward_btn.bind("<ButtonRelease>", lambda e: self.move_command(0, 0))

        self.left_btn.bind("<ButtonPress>", lambda e: self.move_command(-self.speed, self.speed))
        self.left_btn.bind("<ButtonRelease>", lambda e: self.move_command(0, 0))

        self.right_btn.bind("<ButtonPress>", lambda e: self.move_command(self.speed, -self.speed))
        self.right_btn.bind("<ButtonRelease>", lambda e: self.move_command(0, 0))

    def update_speed(self, val):
        self.speed = int(val)

    def update_arm_angle(self, val):
        """Update arm angle independently of movement arrows (only during recording)."""
        self.arm_angle = int(val)
        if self.running:  # Only send when recording is active
            # Wheels stay stopped, update arm only
            recording_module.send_motor(0, 0, self.arm_angle)

    def start_recording(self):
        try:
            self.duration = int(self.duration_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter a valid duration in seconds.")
            return

        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.start_time = time.time()

        self.thread = threading.Thread(target=self.run_collection)
        self.thread.start()
        self.update_timer()

    def run_collection(self):
        recording_module.run_data_collection(self.duration, self.stop_event)
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.remaining_label.config(text="Done.")

    def stop_recording(self):
        self.stop_event.set()
        self.running = False
        self.remaining_label.config(text="Stopped by user.")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_timer(self):
        if not self.running:
            return
        elapsed = int(time.time() - self.start_time)
        remaining = max(0, self.duration - elapsed)

        if self.stop_event.is_set() or remaining <= 0:
            self.remaining_label.config(text="Finalizing...")
        else:
            self.remaining_label.config(text=f"Remaining: {remaining} sec")
            self.root.after(1000, self.update_timer)

    # Movement button callbacks
    def move_command(self, left_pwm, right_pwm):
        if self.running:  # Only active during recording
            recording_module.send_motor(left_pwm, right_pwm, self.arm_angle)


if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderGUI(root)
    root.mainloop()
