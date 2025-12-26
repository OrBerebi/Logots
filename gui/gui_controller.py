# gui_controller.py

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import time
from threading import Event
import recording_module as rm
import transformation_mart_pipeline as t_mart 

class RecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Controller - Stream Mode")

        # --- Model Status Indicator ---
        self.status_frame = tk.Frame(root)
        self.status_frame.pack(pady=5, fill="x")
        
        self.model_status_label = tk.Label(self.status_frame, text="⏳ AI Model: Initializing...", fg="orange")
        self.model_status_label.pack()

        # Input (Optional Safety Timeout)
        self.label = tk.Label(root, text="Max Safety Duration (sec):")
        self.label.pack()
        self.duration_entry = tk.Entry(root)
        self.duration_entry.insert(0, "600") # Default to 10 minutes safety
        self.duration_entry.pack()

        # Buttons
        self.start_button = tk.Button(root, text="Start Streaming", command=self.start_recording, bg="#ddffdd")
        self.start_button.pack(pady=5, ipadx=10)

        self.stop_button = tk.Button(root, text="STOP", state=tk.DISABLED, command=self.stop_recording, bg="#ffdddd")
        self.stop_button.pack(pady=5, ipadx=20)

        # Timer Display (Stopwatch)
        self.timer_label = tk.Label(root, text="Ready", font=("Helvetica", 16))
        self.timer_label.pack(pady=10)

        # Internal state
        self.stop_event = Event()
        self.start_time = None
        self.thread = None
        self.running = False
        self.speed = 85
        self.arm_angle = 90 

        # Add Movement Controls
        movement_frame = ttk.LabelFrame(root, text="Movement Controls", padding=10)
        movement_frame.pack(padx=10, pady=10, fill="x")

        # Speed slider
        self.speed_slider = tk.Scale(movement_frame, from_=120, to=60,
                                     orient=tk.VERTICAL, label="Speed",
                                     command=self.update_speed)
        self.speed_slider.set(self.speed)
        self.speed_slider.grid(row=0, column=0, rowspan=3, padx=10, pady=5)

        # D-Pad
        self.forward_btn  = ttk.Button(movement_frame, text="↑")
        self.backward_btn = ttk.Button(movement_frame, text="↓")
        self.left_btn     = ttk.Button(movement_frame, text="←")
        self.right_btn    = ttk.Button(movement_frame, text="→")

        self.forward_btn.grid(row=0, column=2, padx=5, pady=5)
        self.left_btn.grid(row=1, column=1, padx=5, pady=5, sticky="e")
        self.right_btn.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.backward_btn.grid(row=2, column=2, padx=5, pady=5)

        # Arm slider
        self.arm_slider = tk.Scale(movement_frame, from_=180, to=0,
                                   orient=tk.VERTICAL, label="Arm Angle",
                                   command=self.update_arm_angle)
        self.arm_slider.set(self.arm_angle)
        self.arm_slider.grid(row=0, column=4, rowspan=3, padx=10, pady=5)

        # Bindings
        self.forward_btn.bind("<ButtonPress>", lambda e: self.move_command(self.speed, self.speed))
        self.forward_btn.bind("<ButtonRelease>", lambda e: self.move_command(0, 0))
        self.backward_btn.bind("<ButtonPress>", lambda e: self.move_command(-self.speed, -self.speed))
        self.backward_btn.bind("<ButtonRelease>", lambda e: self.move_command(0, 0))
        self.left_btn.bind("<ButtonPress>", lambda e: self.move_command(-self.speed, self.speed))
        self.left_btn.bind("<ButtonRelease>", lambda e: self.move_command(0, 0))
        self.right_btn.bind("<ButtonPress>", lambda e: self.move_command(self.speed, -self.speed))
        self.right_btn.bind("<ButtonRelease>", lambda e: self.move_command(0, 0))

        # Background Loader
        threading.Thread(target=self.preload_models, daemon=True).start()

    def preload_models(self):
        """Loads both Vision and Audio models in the background."""
        try:
            # 1. Load Visual Model (YOLO)
            t_mart.get_visual_model()
            
            # 2. Load Audio Model (AST Transformer)
            # This will now trigger the download/load into memory immediately
            t_mart.get_audio_model() 
            
            # Update UI on Main Thread
            self.root.after(0, lambda: self.model_status_label.config(text="✅ Vision & Audio AI: Ready", fg="green"))
            
        except Exception as e:
            print(f"Model Load Error: {e}")
            self.root.after(0, lambda: self.model_status_label.config(text=f"❌ Model Error: {str(e)[:20]}...", fg="red"))

    def update_speed(self, val):
        self.speed = int(val)

    def update_arm_angle(self, val):
        self.arm_angle = int(val)
        if self.running:
            rm.send_motor(0, 0, self.arm_angle)

    def start_recording(self):
        # We allow a "safety" duration just in case user forgets to stop
        try:
            safety_duration = int(self.duration_entry.get())
        except ValueError:
            safety_duration = 600 # Default 10 mins

        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.start_time = time.time()

        # Start the collection thread
        # Note: We pass safety_duration, but the new module largely ignores it 
        # in favor of the stop_event, though we can use it to auto-click stop if needed.
        self.thread = threading.Thread(target=self.run_collection, args=(safety_duration,))
        self.thread.start()
        
        self.update_stopwatch()

    def run_collection(self, safety_duration):
        rm.run_data_collection(safety_duration, self.stop_event)
        
        # When rm returns (after stop is pressed), reset UI
        self.root.after(0, self.on_recording_finished)

    def stop_recording(self):
        self.stop_event.set()
        self.timer_label.config(text="Stopping... (Finishing Pipeline)")
        self.stop_button.config(state=tk.DISABLED)

    def on_recording_finished(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.timer_label.config(text="Stopped / Saved.")

    def update_stopwatch(self):
        if not self.running:
            return
            
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        self.timer_label.config(text=f"Live: {mins:02}:{secs:02}")
        
        # Check if thread is still alive
        if self.thread and not self.thread.is_alive():
            self.on_recording_finished()
        else:
            self.root.after(100, self.update_stopwatch)

    def move_command(self, left_pwm, right_pwm):
        if self.running:
            rm.send_motor(left_pwm, right_pwm, self.arm_angle)

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderGUI(root)
    root.mainloop()