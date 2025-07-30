# gui_controller.py

import tkinter as tk
from tkinter import messagebox
import threading
import time
from threading import Event
import recording_module

class RecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Data Recorder")

        # Input
        self.label = tk.Label(root, text="Recording Duration (sec):")
        self.label.pack()
        self.duration_entry = tk.Entry(root)
        self.duration_entry.insert(0, "20")
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

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderGUI(root)
    root.mainloop()
