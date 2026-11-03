"""Original waypoint interpolation used by the released experiments."""
import csv
import numpy as np
INTERP_DISTANCE_RES = .01

def load_waypoints(filename):
    print(f"📁 加载路径文件: {filename}")
    with open(filename) as f:
        waypoints = list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    waypoints_np = np.array(waypoints)
    print(f"   原始路径点数: {len(waypoints)}")
    wp_distance = []
    for i in range(1, waypoints_np.shape[0]):
        wp_distance.append(np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                                   (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
    wp_distance.append(0)
    wp_interp = []
    wp_interp_hash = []
    interp_counter = 0
    for i in range(waypoints_np.shape[0] - 1):
        wp_interp.append(list(waypoints_np[i]))
        wp_interp_hash.append(interp_counter)
        interp_counter += 1
        num_pts = int(np.floor(wp_distance[i] / INTERP_DISTANCE_RES) - 1)
        if num_pts > 0:
            wp_vector = waypoints_np[i+1] - waypoints_np[i]
            wp_uvector = wp_vector / wp_distance[i]
            for j in range(num_pts):
                next_wp = INTERP_DISTANCE_RES * (j+1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp))
                interp_counter += 1
    wp_interp.append(list(waypoints_np[-1]))
    wp_interp_hash.append(interp_counter)
    print(f"   插值后路径点数: {len(wp_interp)}")
    print(f"✅ 路径加载完成\n")
    return waypoints, waypoints_np, wp_distance, wp_interp, wp_interp_hash
