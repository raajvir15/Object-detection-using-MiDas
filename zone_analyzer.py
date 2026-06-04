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
        "far_left":  float(np.mean(main[:, :mw//5])),
        "left":      float(np.mean(main[:, mw//5:2*mw//5])),
        "center":    float(np.mean(main[:, 2*mw//5:3*mw//5])),
        "right":     float(np.mean(main[:, 3*mw//5:4*mw//5])),
        "far_right": float(np.mean(main[:, 4*mw//5:])),
        "floor":     float(np.mean(floor)),
    }
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

    if not alerts:
        alerts.append(("Path appears clear", "safe"))

    return alerts, threshold

def colorize_depth(depth_map):
    normalized = normalize_depth(depth_map)
    colored = plt.cm.magma(normalized)
    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_uint8)
