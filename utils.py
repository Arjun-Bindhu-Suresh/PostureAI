import cv2
import time


def draw_text(
    img,
    text,
    pos,
    color=(255, 255, 255),
    scale=1.0,
    thickness=2,
    bg_color=None
):
    """
    Draw readable text. If bg_color is given, draw a small background box
    behind the text so it's visible on any video background.
    """
    x, y = pos

    if bg_color is not None:
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        top_left = (x - 8, y - th - 8)
        bottom_right = (x + tw + 8, y + baseline + 4)
        cv2.rectangle(img, top_left, bottom_right, bg_color, -1)

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def calculate_fps(prev_time: float):
    """Return (fps, new_prev_time)."""
    current = time.time()
    if prev_time == 0:
        return 0.0, current
    fps = 1.0 / (current - prev_time)
    return fps, current


def rescale_frame(frame, percent=100):
    """Resize frame by given percent, or fit to monitor resolution when percent=100."""
    if percent == 100:
        # Get monitor resolution
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the tkinter window
        monitor_width = root.winfo_screenwidth()
        monitor_height = root.winfo_screenheight()
        root.destroy()
        
        # Calculate scaling to fit monitor while maintaining aspect ratio
        frame_height, frame_width = frame.shape[:2]
        scale = min(monitor_width / frame_width, monitor_height / frame_height)
        width = int(frame_width * scale)
        height = int(frame_height * scale)
    else:
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
    
    return cv2.resize(frame, (width, height))