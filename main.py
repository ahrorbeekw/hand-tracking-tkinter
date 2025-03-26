import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk

class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracking with Tkinter")

        # Mediapipe va OpenCV
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

        # GUI uchun Canvas
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Video olish
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.update()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

root = tk.Tk()
app = HandTrackingApp(root)
root.mainloop()
