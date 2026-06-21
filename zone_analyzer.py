import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# the depth value in MiDaS range krti hai from 0.3 to 874 sth
# we first normalize it to get from 0  to 1 tange
def normalize_depth(depth_map):
    d_min, d_max = depth_map.min(), depth_map.max()
    return (depth_map - d_min) / (d_max - d_min + 1e-8)

# 1e-8 is to prevent it from 0/0 situation

# height and width of the depth map is taken here then it is normalized
def analyze_zones(depth_map):
    h, w = depth_map.shape
    d = normalize_depth(depth_map)

# then this normalized mapp is dividided in into mucltiple zon es
    main = d[int(h * 0.3):int(h * 0.85), :]
    floor = d[int(h * 0.85):, :]

# Top 30%    → ignored (sky or ceiling, nt important so ignored)
# 30%–85%    → main zone (where obstacles have higher chances)
# 85%–100%   → floor zone (steps, curbs)

    mh, mw = main.shape
    zones = {
        "far_left":  float(np.max(main[:, :mw//5])),
        "left":      float(np.max(main[:, mw//5:2*mw//5])),
        "center":    float(np.max(main[:, 2*mw//5:3*mw//5])),
        "right":     float(np.max(main[:, 3*mw//5:4*mw//5])),
        "far_right": float(np.max(main[:, 4*mw//5:])),
        "floor":     float(np.mean(floor)),
    }

    return zones
#Split the main zone into 5 vertical columns. For each column, we take the average depth value


#
def get_alerts(zones):
    avg = np.mean(list(zones.values()))
    threshold = avg * 1.2

## allert mei value dalkr return with what threshold value we go
# we can have multple zones with higher thn threshold value so dont use else
    alerts = []

    if zones["center"] > threshold * 1.3:
        alerts.append(("STOP — Center Blocked", "critical"))
    elif zones["center"] > threshold:
        alerts.append(("Obstacle in Center", "warning"))

    if zones["left"] > threshold:
        alerts.append(("Obstacle on Left", "warning"))
    if zones["right"] > threshold:
        alerts.append(("Obstacle on Right", "warning"))
    if zones["far_left"] > threshold:
        alerts.append(("Far Left Blocked", "warning"))
    if zones["far_right"] > threshold:
        alerts.append(("Far Right Blocked", "warning"))
    if zones["floor"] > threshold * 1.1:
        alerts.append(("Floor Obstacle Ahead", "critical"))

    if not alerts:
        alerts.append(("Path appears clear", "safe"))

    return alerts, threshold

def colorize_depth(depth_map):
    normalized = normalize_depth(depth_map)
    colored = plt.cm.magma(normalized)
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_uint8)


def alerts_to_speech(alerts):
    parts = []
    for alert_text, level in alerts:
        if level == "critical":
            parts.append("Stop. " + alert_text + ".")
        elif level == "warning":
            parts.append(alert_text + ".")
        else:
            parts.append(alert_text + ".")
    return " ".join(parts)


import cv2


def draw_zones_on_image(depth_colored_pil, zones, threshold):

    # convert PIL image to numpy array so cv2 can draw on it
    img = np.array(depth_colored_pil).copy()

    h, w = img.shape[:2]

    # Main zone boundaries (same as analyze_zones)
    y_top = int(h * 0.3)
    y_bottom = int(h * 0.85)

    # draw vertical lines splitting main zone into 5 columns
    for frac in [1/5, 2/5, 3/5, 4/5]:

        x = int(w * frac)

        cv2.line(
            img,
            (x, y_top),
            (x, y_bottom),
            (255, 255, 255),
            2
        )

    # Step 3: draw horizontal lines marking zone boundaries
    cv2.line(
        img,
        (0, y_top),
        (w, y_top),
        (255, 255, 255),
        2
    )

    cv2.line(
        img,
        (0, y_bottom),
        (w, y_bottom),
        (255, 255, 255),
        2
    )

    #draw labels for each zone

    zone_names = [
        "far_left",
        "left",
        "center",
        "right",
        "far_right"
    ]

    section_width = w // 5

    for i, zone in enumerate(zone_names):

        value = zones[zone]

        x = i * section_width + 20
        y = y_top + 40

        # Decide label and color
        if value > threshold * 1.3 and zone == "center":

            label = "STOP"

            color = (0, 0, 255)      # red

        elif value > threshold:

            label = "WARN"

            color = (0, 165, 255)    # orange

        else:

            label = "CLEAR"

            color = (0, 255, 0)      # green

        cv2.putText(
            img,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Floor zone
    floor_value = zones["floor"]

    if floor_value > threshold * 1.4:

        floor_label = "FLOOR!"

        floor_color = (0, 0, 255)

    elif floor_value > threshold:

        floor_label = "FLOOR"

        floor_color = (0, 165, 255)

    else:

        floor_label = "CLEAR"

        floor_color = (0, 255, 0)

    cv2.putText(
        img,
        floor_label,
        (20, y_bottom + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        floor_color,
        2
    )

    # convert back to PIL
    return Image.fromarray(img)