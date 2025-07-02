import mmap
import struct
import tkinter as tk
import time
import threading


SHARED_MEMORY_NAME = "Local\\GestureSharedMemory"
SHARED_MEMORY_SIZE = 256
shm = mmap.mmap(-1, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME, access=mmap.ACCESS_WRITE)


def send_gesture_classification(gesture_code):
    send_code = 0
    if gesture_code == "Close Left Hand":
        send_code = 1

    if gesture_code == "Close Right Hand":
        send_code = 2

    if gesture_code == "Break":
        send_code = 3

    shm.seek(0)
    shm.write(struct.pack('i', send_code))
    print(f"[SHM] Sent gesture classification: {send_code}")


def run_calibration_window():
    instructions = ["Close Left Hand", "Close Right Hand"]
    cycles = 4
    hold_time = 8
    break_time = 10

    root = tk.Tk()
    root.title("EEG Gesture Calibration")
    root.geometry("800x600")
    label = tk.Label(root, text="", font=("Arial", 24))
    label.pack(expand=True)

    def calibration_sequence():
        for i in range(cycles):
            for gesture in instructions:
                # Show gesture instruction
                for count in [gesture, "3", "2", "1"]:
                    label.config(text=count if count != gesture else gesture)
                    root.update()
                    time.sleep(1)
                label.config(text=f"GO! Hold: {gesture}")
                send_gesture_classification(gesture)
                root.update()
                time.sleep(hold_time)
                # Break
                label.config(text="Relax (Break)")
                send_gesture_classification("Break")
                root.update()
                time.sleep(break_time)
        label.config(text="Calibration Complete!")
        root.update()

    threading.Thread(target=calibration_sequence, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    run_calibration_window()
