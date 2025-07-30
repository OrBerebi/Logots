import csv
import numpy as np

RECORD_DURATION = 20  # seconds
FPS = 5
TOTAL_FRAMES = RECORD_DURATION * FPS

with open("stg_motor_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["frame_id", "timestamp", "left_pwm", "right_pwm", "arm_angle"])

    for i in range(TOTAL_FRAMES):
        timestamp = i / FPS

        # Motor pattern: forward 5s, stop 5s, reverse 5s, stop 5s
        if 0 <= timestamp < 5:
            left_pwm = 75
            right_pwm = 75
        elif 5 <= timestamp < 10:
            left_pwm = -75
            right_pwm = 75
        elif 10 <= timestamp < 15:
            left_pwm = -75
            right_pwm = -75
        else:
            left_pwm = 0
            right_pwm = 0

        # Arm angle: sine oscillation between 0 and 180 degrees
        arm_angle = int(np.round(90 + 90 * np.sin(2 * np.pi * timestamp / RECORD_DURATION)))

        #writer.writerow([i, f"{timestamp:.2f}", left_pwm, right_pwm, f"{arm_angle:.2f}"])
        writer.writerow([i, f"{timestamp:.2f}", left_pwm, right_pwm, arm_angle])
