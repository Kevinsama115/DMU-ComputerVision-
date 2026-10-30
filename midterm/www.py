
import cv2 as cv
import numpy as np
from tkinter import Tk, filedialog

# Select an image
root = Tk()
root.withdraw()

path = filedialog.askopenfilename(
    title="Select your image",
    filetypes=[("Images", "*.jpg *.jpeg *.png")]
)
root.destroy()

if not path:
    print("No image selected.")
else:
    img = cv.imread(path)

    if img is None:
        print("Cannot load image.")
    else:
        img_show = img.copy()

        # 0 = definite background, 1 = definite foreground,
        # 2 = probable background, 3 = probable foreground
        mask = np.full(
            img.shape[:2], cv.GC_PR_BGD, dtype=np.uint8
        )

        BrushSize = 12
        LColor = (255, 0, 0)   # Blue: object
        RColor = (0, 0, 255)   # Red: background

        drawing = None

        def painting(event, x, y, flags, param):
            global drawing

            if event == cv.EVENT_LBUTTONDOWN:
                drawing = cv.GC_FGD
            elif event == cv.EVENT_RBUTTONDOWN:
                drawing = cv.GC_BGD

            if drawing is not None:
                if drawing == cv.GC_FGD:
                    color = LColor
                else:
                    color = RColor

                if event in (
                    cv.EVENT_LBUTTONDOWN,
                    cv.EVENT_RBUTTONDOWN,
                    cv.EVENT_MOUSEMOVE
                ):
                    if event != cv.EVENT_MOUSEMOVE or (
                        flags & (
                            cv.EVENT_FLAG_LBUTTON
                            if drawing == cv.GC_FGD
                            else cv.EVENT_FLAG_RBUTTON
                        )
                    ):
                        cv.circle(
                            img_show, (x, y),
                            BrushSize, color, -1
                        )
                        cv.circle(
                            mask, (x, y),
                            BrushSize, drawing, -1
                        )

            if event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
                drawing = None

            cv.imshow("Painting", img_show)

        cv.namedWindow("Painting")
        cv.setMouseCallback("Painting", painting)
        cv.imshow("Painting", img_show)

        print("Left-click and drag: blue object")
        print("Right-click and drag: red background")
        print("Press ESC to exit.")

        while True:
            key = cv.waitKey(20) & 0xFF
            if key == 27:
                break

        cv.destroyAllWindows()
