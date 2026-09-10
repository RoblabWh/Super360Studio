#!/usr/bin/env python3
"""Loescht einmalig alles, was der Bag in RViz hinterlassen hat.

Laeuft als eigener Prozess in einer gesourcten ROS-Umgebung (die GUI selbst hat
kein ROS, s. ARCHITECTURE.md). Sendet auf jedes Anzeige-Topic eine leere Wolke
bzw. einen DELETEALL-Marker und beendet sich.

stdout ist Protokoll (wird von core/rviz_player.py an log_cb gereicht).
"""
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

# Punktwolken-Topics des EPIC-Bags (Voxel-/Label-Karten + Rohwolke)
CLOUD_TOPICS = ("/occ", "/pocc", "/frt", "/good_obs", "/bad_obs",
                "/quad0_pcl_render_node/cloud", "/map_generator/global_cloud",
                "/viewpoint_centers")
# Marker (Trajektorien, Roboter, Zustand)
MARKER_TOPICS = ("/robot", "/planning/state", "/planning/position_cmd_vis",
                 "/visualizer/trajectory", "/visualizer/edge",
                 "/visualizer/mesh", "/collision_count_marker")
MARKER_ARRAY_TOPICS = ("/global_tour", "/planning/travel_traj",
                       "/exploration/box", "/bubble_visualizer/frontend_traj",
                       "/viz_graph_topic", "/sf_cluster_marker")


def empty_cloud(stamp, frame="world") -> PointCloud2:
    m = PointCloud2()
    m.header = Header(stamp=stamp, frame_id=frame)
    m.height, m.width = 1, 0
    m.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    m.is_bigendian = False
    m.point_step = 12
    m.row_step = 0
    m.is_dense = True
    m.data = b""
    return m


def main() -> None:
    rclpy.init()
    node = Node("rviz_clear")
    # TRANSIENT_LOCAL: RViz fordert fuer manche Topics (z.B. /robot) "Transient
    # Local" an — ein VOLATILE-Publisher waere QoS-inkompatibel und die
    # Loesch-Nachricht kaeme nie an. Angeboten TRANSIENT_LOCAL vertraegt sich
    # mit BEIDEN Abonnenten-Varianten.
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST)
    pc = [node.create_publisher(PointCloud2, t, qos) for t in CLOUD_TOPICS]
    mk = [node.create_publisher(Marker, t, qos) for t in MARKER_TOPICS]
    ma = [node.create_publisher(MarkerArray, t, qos) for t in MARKER_ARRAY_TOPICS]

    # RViz muss die Publisher erst entdecken, sonst geht die einzige Nachricht
    # ins Leere (Discovery ist asynchron).
    deadline = time.time() + 3.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if all(p.get_subscription_count() > 0 for p in pc[:6]):
            break
    time.sleep(0.3)

    stamp = node.get_clock().now().to_msg()
    ec = empty_cloud(stamp)
    for p in pc:
        p.publish(ec)
    da = Marker()
    da.header = Header(stamp=stamp, frame_id="world")
    da.action = Marker.DELETEALL
    for p in mk:
        p.publish(da)
    arr = MarkerArray()
    arr.markers = [da]
    for p in ma:
        p.publish(arr)

    # kurz nachlaufen lassen, damit RELIABLE wirklich raus ist
    end = time.time() + 0.8
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    print(f"geleert: {len(pc)} Wolken-, {len(mk) + len(ma)} Marker-Topics",
          flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
