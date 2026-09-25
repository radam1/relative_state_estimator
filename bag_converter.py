#!/usr/bin/env python3
"""Convert a ROS 1 bag to flat CSV tables for the standalone C++ EKF.

Reads the bag directly with `rosbags` (pure Python -- no ROS install and no
compiled agiros_msgs needed, because a ROS 1 bag embeds its own message
definitions).  Writes one fixed-width CSV per stream, plus a manifest.

Because the bags predate the UWB sensor, UWB ranges are SYNTHESISED from the
ground-truth odometry at a configurable rate -- see the UWB CONFIG block below.

Usage
-----
    pip install rosbags
    python3 scripts/bag_to_csv.py --list              flight.bag
    python3 scripts/bag_to_csv.py -o data/flight_01/  flight.bag
    python3 scripts/bag_to_csv.py --self-test -o /tmp/st     # no bag needed

CSV contract (kept deliberately dumb so a ~15-line C++ reader can handle it)
---------------------------------------------------------------------------
  * ASCII, LF line endings, no quoting, no embedded commas, no empty fields
  * row 1 is a header of column names; every later row is PURELY NUMERIC
  * fixed column count per file -- variable-length arrays are flattened to
    one row per element (see uwb.csv)
  * column 0 is always `t_ns`, an integer nanosecond timestamp
  * floats are written with %.17g, which round-trips float64 exactly
  * drone identities are integer INDICES, never strings, so the reader never
    has to parse text; the index -> name mapping lives in manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# =============================================================================
# UWB CONFIG -- the bags contain no UWB, so these ranges are synthesised
# =============================================================================
UWB = dict(
    rate_hz=20.0,           # publish rate of the synthesised ranges
    noise_std=0.05,         # [m] 1-sigma Gaussian, matches the sim default
    bias=0.0,               # [m] constant offset added to every range
    seed=0,                 # RNG seed, for reproducible runs
    both_directions=True,   # True  -> 6 readings/cycle (each pair measured twice,
                            #          independent noise) as the sim publishes;
                            # False -> 3 readings/cycle, one per unordered pair
    min_range=0.05,         # [m] readings outside [min,max] are dropped
    max_range=100.0,
    # Per-drone UWB antenna offset in the DRONE BODY frame [m].
    # NOTE: an OFF-AXIS offset is what makes this system fully observable
    # (rank 48/48 instead of 44/48) -- see ekf_plan.md section 1.  Set these to
    # match where the antenna is actually mounted; all-zero is the degenerate case.
    antenna_offset={
        0: (0.00, 0.10, 0.04),
        1: (0.09, -0.05, 0.04),
        2: (-0.09, -0.05, 0.04),
    },
)

# =============================================================================
# CABLE CONFIG -- only used when the bag has no wrench_observed topic
# =============================================================================
# PREFERRED SOURCE: agiros_pilot runs an ImuBasedObserver that publishes
# agiros_msgs/Wrench on /{quad}/agiros_pilot/wrench_observed.  Its `force` is the
# external (cable) force in the DRONE BODY frame, so
#     s_i   = force / ||force||     (body-frame cable direction -- what the EKF wants)
#     tau_i = ||force||             (cable tension)
# If that topic is in the bag it is used automatically and this block is ignored.
CABLE = dict(
    rate_hz=100.0,          # synthesis rate when falling back to ground truth
    angle_noise_deg=2.0,    # 1-sigma direction error
    seed=1,
    payload_topic="/load/odometry_sensor/odometry",   # payload ground truth -> payload.csv, ALWAYS used
    payload_callerid="/gazebo",         # ekf_cable_load also publishes that topic (an echo); skip it
    # Payload-frame attach points, from flycrane.urdf/flycrane/payloads/base.usda
    attach_point={0: (0.27, 0.22, 0.14), 1: (0.27, -0.22, 0.14), 2: (-0.27, 0.0, 0.14)},
    # Cable hook offset in the DRONE BODY frame
    hook_offset={0: (0.0, 0.0, -0.03), 1: (0.0, 0.0, -0.03), 2: (0.0, 0.0, -0.03)},
)

# Drone namespaces, in the order that fixes their integer index.
# Note Falcon1's namespace really is `falcon1` (see extension.py _FLYCRANE_DRONES).
DEFAULT_DRONES = ["falcon1", "falcon2", "falcon3"]

# A drone publishes several topics of one type (e.g. four Odometry topics), and
# csv_io loads only the plain <stem>_<ns>.csv.  If one of these topics is in the
# bag it owns that file; otherwise the first topic recorded wins (see table_key).
PRIMARY_TOPIC = {
    "odom": "/{ns}/ground_truth/odometry",    # noise-free RotorS pose = ground truth
    "imu": "/{ns}/imu",                       # noisy IMU that agiros_pilot flies on
}

FLOAT_FMT = "%.17g"


# =============================================================================
# CSV writing
# =============================================================================
class Table:
    """Fixed-schema numeric table that streams straight to disk."""

    def __init__(self, path: str, columns: list[str]):
        self.path, self.columns = path, list(columns)
        self.rows = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="\n")
        self._f.write(",".join(self.columns) + "\n")

    def add(self, t_ns: int, values) -> None:
        vals = np.asarray(values, dtype=np.float64).ravel()
        if vals.size != len(self.columns) - 1:
            raise ValueError(f"{self.path}: expected {len(self.columns)-1} values, got {vals.size}")
        if not np.all(np.isfinite(vals)):
            return                       # drop non-finite rows rather than poison the C++ side
        self._f.write(str(int(t_ns)) + "," + ",".join(FLOAT_FMT % v for v in vals) + "\n")
        self.rows += 1

    def close(self) -> None:
        self._f.close()


# =============================================================================
# ROS 1 bag reading
# =============================================================================
def _stamp_ns(header) -> int:
    """ROS1 headers use secs/nsecs; rosbags may surface them as sec/nanosec."""
    s = header.stamp
    sec = getattr(s, "sec", None)
    if sec is None:
        sec = getattr(s, "secs")
        nsec = getattr(s, "nsecs")
    else:
        nsec = getattr(s, "nanosec")
    return int(sec) * 1_000_000_000 + int(nsec)


def open_bag(path):
    """Yield (topic, msgtype, msg) and expose the connection inventory."""
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import get_types_from_msg
    except ImportError:
        sys.exit("rosbags is not installed.  pip install rosbags")

    try:                                              # rosbags >= 0.10
        from rosbags.typesys import Stores, get_typestore
        ts = get_typestore(Stores.ROS1_NOETIC)
        register = ts.register
        deserialize = ts.deserialize_ros1
        known = lambda t: t in ts.types
    except ImportError:                               # rosbags < 0.10
        from rosbags.typesys import register_types
        from rosbags.serde import deserialize_cdr, ros1_to_cdr
        from rosbags.typesys.types import FIELDDEFS
        register = register_types
        deserialize = lambda raw, tp: deserialize_cdr(ros1_to_cdr(raw, tp), tp)
        known = lambda t: t in FIELDDEFS

    reader = Reader(path)
    reader.open()
    for c in reader.connections:                      # register EMBEDDED definitions
        if not known(c.msgtype):
            try:
                # rosbags >= 0.10 wraps the definition text in a MessageDefinition
                register(get_types_from_msg(getattr(c.msgdef, "data", c.msgdef), c.msgtype))
            except Exception as exc:                  # noqa: BLE001
                print(f"  ! could not register {c.msgtype}: {exc}", file=sys.stderr)
    return reader, deserialize


def list_bag(path) -> None:
    reader, _ = open_bag(path)
    try:
        print(f"{path}\n  duration {(reader.duration or 0)/1e9:.2f} s, "
              f"{reader.message_count} messages\n")
        print(f"  {'topic':<44} {'type':<34} {'count':>8}")
        for c in sorted(reader.connections, key=lambda c: c.topic):
            print(f"  {c.topic:<44} {c.msgtype:<34} {c.msgcount:>8}")
    finally:
        reader.close()


# =============================================================================
# Per-message-type extractors
# =============================================================================
IMU_COLS = ["t_ns", "ax", "ay", "az", "gx", "gy", "gz"]
ODOM_COLS = ["t_ns", "px", "py", "pz", "qx", "qy", "qz", "qw",
             "vx", "vy", "vz", "wx", "wy", "wz"]
MOTOR_COLS = ["t_ns", "w0", "w1", "w2", "w3"]
UWB_COLS = ["t_ns", "host", "peer", "range"]
CABLE_COLS = ["t_ns", "sx", "sy", "sz", "tension"]


def _xyz(v):
    return [v.x, v.y, v.z]


def extract_imu(msg):
    return _xyz(msg.linear_acceleration) + _xyz(msg.angular_velocity)


def extract_odom(msg):
    """The CSV carries a WORLD-frame velocity.  By the ROS convention the twist is
    in child_frame_id -- the body frame for RotorS (link RelativeLinearVel) -- so
    it is rotated into the world.  agiros_pilot leaves child_frame_id empty and
    publishes its world-frame QuadState velocity, which is kept as is."""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    lv, av = msg.twist.twist.linear, msg.twist.twist.angular
    v = _xyz(lv)
    if msg.child_frame_id:
        v = list(quat_to_R([q.x, q.y, q.z, q.w]) @ v)
    return _xyz(p) + [q.x, q.y, q.z, q.w] + v + _xyz(av)


def extract_quadstate(msg):
    p, q = msg.pose.position, msg.pose.orientation
    lv, av = msg.velocity.linear, msg.velocity.angular
    return _xyz(p) + [q.x, q.y, q.z, q.w] + _xyz(lv) + _xyz(av)


def extract_wrench(msg):
    """agiros_msgs/Wrench.force is the external force on the drone in its BODY
    frame (see agilib ImuBasedObserver::getAt: m*f_body - drag - thrust).  For a
    cable-suspended drone that IS the cable force, so its direction is exactly the
    body-frame cable direction the filter needs -- with no attitude in the chain."""
    f = np.array([msg.force.x, msg.force.y, msg.force.z], float)
    n = float(np.linalg.norm(f))
    if n < 1e-6:
        return [0.0, 0.0, 0.0, 0.0]          # dropped downstream as non-informative
    return list(f / n) + [n]


def extract_motors(msg):
    """MotorSpeeds.angular_velocities is a VARIABLE-LENGTH float64[].  The CSV
    schema is fixed at 4 columns, so pad or truncate -- and say so once, because
    a 6-rotor airframe would otherwise lose two columns silently."""
    w = list(getattr(msg, "angular_velocities", None) or getattr(msg, "motors", []))
    if len(w) != 4 and "motors" not in _WARNED:
        _WARNED.add("motors")
        print(f"  ! motor array has {len(w)} entries, schema is 4 "
              f"({'padding' if len(w) < 4 else 'TRUNCATING'})", file=sys.stderr)
    return (w + [0.0] * 4)[:4]


# msgtype -> (filename stem, columns, extractor)
HANDLERS = {
    "sensor_msgs/msg/Imu":            ("imu", IMU_COLS, extract_imu),
    "nav_msgs/msg/Odometry":          ("odom", ODOM_COLS, extract_odom),
    "agiros_msgs/msg/QuadState":      ("state", ODOM_COLS, extract_quadstate),
    "agiros_msgs/msg/MotorSpeeds":    ("motors", MOTOR_COLS, extract_motors),
    "agiros_msgs/msg/Wrench":         ("cable", CABLE_COLS, extract_wrench),
    "geometry_msgs/msg/PoseStamped":  ("payload", ODOM_COLS, None),
}


def _normalise(msgtype: str) -> str:
    """ROS1 types are 'pkg/Type'; rosbags may report 'pkg/msg/Type'."""
    parts = msgtype.split("/")
    return f"{parts[0]}/msg/{parts[-1]}" if len(parts) == 2 else msgtype


def table_key(stem: str, drone: str, topic: str, owner: dict[str, str]) -> str:
    """One table per (message type, drone) -- unless two DIFFERENT topics of the
    same type target the same drone, in which case disambiguate by topic so the
    two streams never interleave into one file."""
    base = f"{stem}_{drone}"
    if owner.setdefault(base, topic) == topic:
        return base
    tail = topic.strip("/").replace("/", "_")
    key = f"{base}__{tail}"
    if owner.setdefault(key, topic) == topic and key not in _WARNED:
        _WARNED.add(key)
        print(f"  ! two '{stem}' topics for {drone}: '{owner[base]}' and '{topic}'"
              f" -> writing the second to {key}.csv", file=sys.stderr)
    return key


_WARNED: set[str] = set()


def which_drone(topic: str, drones: list[str]) -> int | None:
    for i, ns in enumerate(drones):
        if f"/{ns}/" in topic or topic.startswith(f"{ns}/"):
            return i
    return None


# =============================================================================
# UWB synthesis from ground truth
# =============================================================================
def quat_to_R(q):
    """q = (x, y, z, w), normalised internally."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def slerp(q0, q1, u):
    """Shortest-arc interpolation between two (x,y,z,w) quaternions."""
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:                                   # take the short way round
        q1, d = -q1, -d
    if d > 0.9995:                                # nearly parallel -> lerp
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    return (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / s


class Pose6Track:
    """Ground-truth pose of one drone, interpolable to arbitrary times."""

    def __init__(self, t_ns, pos, quat):
        order = np.argsort(t_ns)
        self.t = np.asarray(t_ns, dtype=np.int64)[order]
        self.p = np.asarray(pos, dtype=np.float64)[order]
        self.q = np.asarray(quat, dtype=np.float64)[order]

    @property
    def span(self):
        return int(self.t[0]), int(self.t[-1])

    def at(self, t_ns: int):
        j = int(np.searchsorted(self.t, t_ns))
        if j <= 0:
            return self.p[0], self.q[0]
        if j >= len(self.t):
            return self.p[-1], self.q[-1]
        t0, t1 = float(self.t[j - 1]), float(self.t[j])
        u = 0.0 if t1 == t0 else (float(t_ns) - t0) / (t1 - t0)
        return (1 - u) * self.p[j - 1] + u * self.p[j], slerp(self.q[j - 1], self.q[j], u)


def synthesise_uwb(tracks: dict[int, Pose6Track], cfg: dict, out_path: str) -> Table:
    """Generate peer-to-peer ranges on a fixed grid from ground-truth poses.

    Mirrors the simulator: each host measures each peer independently, so a
    pair measured in both directions gets two INDEPENDENT noise draws.
    """
    ids = sorted(tracks)
    if len(ids) < 2:
        raise ValueError("need at least two ground-truth tracks to synthesise UWB")

    lo = max(tracks[i].span[0] for i in ids)      # common overlap only
    hi = min(tracks[i].span[1] for i in ids)
    if hi <= lo:
        raise ValueError("ground-truth tracks do not overlap in time")

    step_ns = int(round(1e9 / float(cfg["rate_hz"])))
    grid = np.arange(lo, hi + 1, step_ns, dtype=np.int64)
    rng = np.random.default_rng(cfg["seed"])

    pairs = ([(a, b) for a in ids for b in ids if a != b] if cfg["both_directions"]
             else [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))])

    table = Table(out_path, UWB_COLS)
    dropped = 0
    for t in grid:
        ant = {}
        for i in ids:
            p, q = tracks[i].at(int(t))
            off = np.asarray(cfg["antenna_offset"].get(i, (0.0, 0.0, 0.0)), float)
            ant[i] = p + quat_to_R(q) @ off       # antenna, not body origin
        for host, peer in pairs:
            r = float(np.linalg.norm(ant[peer] - ant[host]))
            r += cfg["bias"] + rng.normal(0.0, cfg["noise_std"]) if cfg["noise_std"] > 0 else cfg["bias"]
            if not (cfg["min_range"] <= r <= cfg["max_range"]):
                dropped += 1
                continue
            table.add(int(t), [host, peer, r])
    if dropped:
        print(f"  uwb: dropped {dropped} readings outside "
              f"[{cfg['min_range']}, {cfg['max_range']}] m")
    return table


def synthesise_cable(drone_tracks, payload_track, cfg, out_dir, drones):
    """Body-frame cable directions from ground truth, when no wrench_observed
    exists.  Mirrors the real observer's output:
        c_i = p_L + R_L rho_i      (payload attach, world)
        h_i = p_i + R_i d_i        (drone hook,   world)
        s_i = R_i^T (c_i - h_i)/|| . ||        <- BODY frame, as the filter wants
    """
    ids = sorted(drone_tracks)
    lo = max([payload_track.span[0]] + [drone_tracks[i].span[0] for i in ids])
    hi = min([payload_track.span[1]] + [drone_tracks[i].span[1] for i in ids])
    if hi <= lo:
        raise ValueError("payload and drone tracks do not overlap in time")

    step = int(round(1e9 / float(cfg["rate_hz"])))
    grid = np.arange(lo, hi + 1, step, dtype=np.int64)
    rng = np.random.default_rng(cfg["seed"])
    sigma = math.radians(cfg["angle_noise_deg"])
    out = {}
    for i in ids:
        rho = np.asarray(cfg["attach_point"].get(i, (0.0, 0.0, 0.0)), float)
        hook = np.asarray(cfg["hook_offset"].get(i, (0.0, 0.0, 0.0)), float)
        tab = Table(os.path.join(out_dir, f"cable_{drones[i]}.csv"), CABLE_COLS)
        for t in grid:
            pL, qL = payload_track.at(int(t))
            pi, qi = drone_tracks[i].at(int(t))
            Ri = quat_to_R(qi)
            e = (pL + quat_to_R(qL) @ rho) - (pi + Ri @ hook)
            n = float(np.linalg.norm(e))
            if n < 1e-9:
                continue
            s = Ri.T @ (e / n)
            if sigma > 0:
                s = s + rng.normal(0.0, sigma, 3)
                s /= np.linalg.norm(s)
            tab.add(int(t), list(s) + [0.0])      # tension unknown from geometry alone
        out[f"cable_{drones[i]}"] = tab
    return out


# =============================================================================
# Driver
# =============================================================================
def convert(bag: str, out_dir: str, drones: list[str], uwb_cfg: dict) -> None:
    reader, deserialize = open_bag(bag)
    tables: dict[str, Table] = {}
    gt: dict[int, dict] = {}
    unhandled: set[str] = set()
    owner: dict[str, str] = {}          # table key -> the topic that owns it
    payload: dict = {"t": [], "p": [], "q": []}

    topics = {c.topic for c in reader.connections}
    for ns in drones:                   # pin the primary topics before any message is seen
        for stem, fmt in PRIMARY_TOPIC.items():
            if fmt.format(ns=ns) in topics:
                owner[f"{stem}_{ns}"] = fmt.format(ns=ns)

    try:
        for conn, _, raw in reader.messages():
            mt = _normalise(conn.msgtype)
            handler = HANDLERS.get(mt)
            if handler is None:
                unhandled.add(f"{conn.topic} [{conn.msgtype}]")
                continue
            idx = which_drone(conn.topic, drones)
            if idx is None and conn.topic == CABLE["payload_topic"]:
                # ekf_cable_load republishes its last received load pose on this topic,
                # restamped with the current time -- keep only the simulator's own messages.
                callerid = getattr(conn.ext, "callerid", None)
                if callerid != CABLE["payload_callerid"]:
                    unhandled.add(f"{conn.topic} [{conn.msgtype}] (publisher {callerid}, "
                                  f"not the ground truth {CABLE['payload_callerid']})")
                    continue
                try:
                    m = deserialize(raw, conn.msgtype)
                    t_ns = _stamp_ns(m.header)
                    vals = extract_odom(m)
                    if "payload" not in tables:
                        tables["payload"] = Table(os.path.join(out_dir, "payload.csv"), ODOM_COLS)
                    tables["payload"].add(t_ns, vals)
                    payload["t"].append(t_ns)
                    payload["p"].append(vals[0:3])
                    payload["q"].append(vals[3:7])
                except Exception:                                  # noqa: BLE001
                    pass
                continue
            if idx is None:
                unhandled.add(f"{conn.topic} [{conn.msgtype}] (no drone match)")
                continue

            stem, cols, extract = handler
            if extract is None:
                unhandled.add(f"{conn.topic} [{conn.msgtype}] (no extractor for '{stem}' on a drone topic)")
                continue
            try:
                msg = deserialize(raw, conn.msgtype)
                t_ns = _stamp_ns(msg.header)
                vals = extract(msg)
            except Exception as exc:                          # noqa: BLE001
                unhandled.add(f"{conn.topic} [decode failed: {exc}]")
                continue

            key = table_key(stem, drones[idx], conn.topic, owner)
            if key not in tables:
                tables[key] = Table(os.path.join(out_dir, key + ".csv"), cols)
            tables[key].add(t_ns, vals)

            if key == f"odom_{drones[idx]}":                   # stash ground truth for UWB synthesis
                g = gt.setdefault(idx, {"t": [], "p": [], "q": []})
                g["t"].append(t_ns)
                g["p"].append(vals[0:3])
                g["q"].append(vals[3:7])
    finally:
        reader.close()

    if "payload" not in tables:
        print(f"  ! no payload ground truth ({CABLE['payload_topic']} from {CABLE['payload_callerid']})"
              " -- NO payload.csv written", file=sys.stderr)

    tracks = {i: Pose6Track(g["t"], g["p"], g["q"]) for i, g in gt.items() if len(g["t"]) >= 2}

    have_wrench = any(k.startswith("cable_") for k in tables)
    if have_wrench:
        print("  cable: using wrench_observed from the bag (body-frame, real observer)")
    elif payload["t"] and len(payload["t"]) >= 2 and len(tracks) >= 1:
        print(f"  cable: no wrench_observed -- synthesising at {CABLE['rate_hz']} Hz "
              f"from ground truth")
        tables.update(synthesise_cable(tracks, Pose6Track(payload["t"], payload["p"],
                                                          payload["q"]), CABLE, out_dir,
                                        drones))
    else:
        print("  ! no wrench_observed and no payload ground truth "
              f"(looked for {CABLE['payload_topic']}) -- NO cable directions written",
              file=sys.stderr)
    uwb_table = None
    if len(tracks) >= 2:
        uwb_table = synthesise_uwb(tracks, uwb_cfg, os.path.join(out_dir, "uwb.csv"))
        tables["uwb"] = uwb_table
    else:
        print("  ! fewer than two ground-truth tracks -- skipping UWB synthesis",
              file=sys.stderr)

    write_manifest(out_dir, tables, drones, uwb_cfg, bag)
    for t in tables.values():
        t.close()

    print(f"\nwrote {len(tables)} tables to {out_dir}")
    for k in sorted(tables):
        print(f"  {k + '.csv':<28} {tables[k].rows:>8} rows")
    if unhandled:
        print("\nskipped (no handler / no drone match):")
        for u in sorted(unhandled):
            print(f"  {u}")


def write_manifest(out_dir, tables, drones, uwb_cfg, bag) -> None:
    cfg = dict(uwb_cfg)
    cfg["antenna_offset"] = {str(k): list(v) for k, v in uwb_cfg["antenna_offset"].items()}
    man = {
        "source_bag": os.path.abspath(bag),
        "drone_index_to_name": {str(i): n for i, n in enumerate(drones)},
        "time_units": "integer nanoseconds, column t_ns",
        "float_format": FLOAT_FMT,
        "uwb_synthetic": True,
        "uwb_config": cfg,
        "tables": {k: {"columns": t.columns, "rows": t.rows} for k, t in tables.items()},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)


# =============================================================================
# Self-test: exercises UWB synthesis + the CSV contract without needing a bag
# =============================================================================
def self_test(out_dir: str) -> int:
    print("self-test: synthetic ground truth -> UWB -> CSV round trip\n")
    rng = np.random.default_rng(1)
    dur, src_hz = 4.0, 100.0
    n = int(dur * src_hz)
    t = (np.arange(n) / src_hz * 1e9).astype(np.int64) + 1_700_000_000_000_000_000
    tracks, truth = {}, {}
    for i, ang in enumerate((90.0, 210.0, 330.0)):
        a = math.radians(ang)
        base = np.array([1.2 * math.cos(a), 1.2 * math.sin(a), 2.0])
        p = base + 0.2 * np.stack([np.sin(2 * np.pi * 0.25 * np.arange(n) / src_hz),
                                   np.zeros(n), np.zeros(n)], axis=1)
        yaw = 0.3 * i + 0.1 * np.arange(n) / src_hz
        q = np.stack([np.zeros(n), np.zeros(n), np.sin(yaw / 2), np.cos(yaw / 2)], axis=1)
        tracks[i] = Pose6Track(t, p, q)
        truth[i] = (p, q)

    cfg = dict(UWB); cfg["noise_std"] = 0.0; cfg["seed"] = 7      # noiseless for exactness
    path = os.path.join(out_dir, "uwb.csv")
    tab = synthesise_uwb(tracks, cfg, path); tab.close()

    ok = True
    exp_rows = int(dur * cfg["rate_hz"]) * (6 if cfg["both_directions"] else 3)
    got = tab.rows
    print(f"  rows: got {got}, expected ~{exp_rows}")
    ok &= abs(got - exp_rows) <= 6

    raw = open(path).read().splitlines()
    hdr = raw[0].split(",")
    ok &= hdr == UWB_COLS
    print(f"  header {hdr} {'OK' if hdr == UWB_COLS else 'FAIL'}")
    ncols = {len(r.split(",")) for r in raw[1:]}
    ok &= ncols == {4}
    print(f"  fixed column count: {ncols} {'OK' if ncols == {4} else 'FAIL'}")
    ok &= all(all(c not in r for c in '" \t') for r in raw[1:])
    print("  no quoting/whitespace in data rows: OK")

    # values must match a hand-computed range with the antenna offset applied
    f0 = raw[1].split(",")
    t0, h, pr, r = int(f0[0]), int(float(f0[1])), int(float(f0[2])), float(f0[3])
    ph, qh = tracks[h].at(t0); pp, qp = tracks[pr].at(t0)
    ah = ph + quat_to_R(qh) @ np.array(cfg["antenna_offset"][h])
    ap = pp + quat_to_R(qp) @ np.array(cfg["antenna_offset"][pr])
    ref = float(np.linalg.norm(ap - ah))
    print(f"  range[0] = {r:.12f}, recomputed {ref:.12f}, err {abs(r-ref):.2e}")
    ok &= abs(r - ref) < 1e-12

    # the antenna offset must actually matter (this is the observability fix)
    cfg0 = dict(cfg); cfg0["antenna_offset"] = {}
    p0 = os.path.join(out_dir, "uwb_noant.csv")
    t0b = synthesise_uwb(tracks, cfg0, p0); t0b.close()
    r_noant = float(open(p0).read().splitlines()[1].split(",")[3])
    d = abs(r - r_noant)
    print(f"  antenna offset changes the range by {d*100:.2f} cm "
          f"{'OK' if d > 1e-3 else 'FAIL - offsets not applied'}")
    ok &= d > 1e-3

    # float round-trip must be exact
    v = 1.7345678901234567e9
    ok &= float(FLOAT_FMT % v) == v
    print(f"  float round-trip with {FLOAT_FMT}: "
          f"{'exact' if float(FLOAT_FMT % v) == v else 'LOSSY'}")

    # --- demonstrate the multi-schema file layout: one table per type x drone
    print("\n  file layout (one table per message type x drone, never merged):")
    demo = {}
    for i in range(3):
        ns = DEFAULT_DRONES[i]
        p_arr, q_arr = truth[i]
        od = Table(os.path.join(out_dir, f"odom_{ns}.csv"), ODOM_COLS)
        im = Table(os.path.join(out_dir, f"imu_{ns}.csv"), IMU_COLS)
        mo = Table(os.path.join(out_dir, f"motors_{ns}.csv"), MOTOR_COLS)
        for k in range(0, n, 10):
            od.add(int(t[k]), list(p_arr[k]) + list(q_arr[k]) + [0] * 6)
            im.add(int(t[k]), [0, 0, 9.81, 0, 0, 0])
            mo.add(int(t[k]), [700, 700, 700, 700])
        for tb in (od, im, mo):
            tb.close(); demo[os.path.basename(tb.path)] = (len(tb.columns), tb.rows)
    demo["uwb.csv"] = (len(UWB_COLS), tab.rows)
    for name in sorted(demo):
        c, r = demo[name]
        print(f"     {name:<26} {c:>2} cols  {r:>5} rows")
    widths = {c for c, _ in demo.values()}
    print(f"  distinct column counts across files: {sorted(widths)} "
          f"-> different schemas stay in different files")
    ok &= len(widths) > 1

    # --- cable-direction synthesis: the result must be BODY frame
    ccfg = dict(CABLE); ccfg["angle_noise_deg"] = 0.0; ccfg["rate_hz"] = 50.0
    pay = Pose6Track(t, np.tile(np.array([0.0, 0.0, 1.0]), (n, 1)),
                     np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)))
    ctabs = synthesise_cable(tracks, pay, ccfg, out_dir, DEFAULT_DRONES)
    for tb in ctabs.values():
        tb.close()
    first = open(os.path.join(out_dir, f"cable_{DEFAULT_DRONES[1]}.csv")).read().splitlines()[1]
    f = [float(x) for x in first.split(",")]
    tq, s_body = int(f[0]), np.array(f[1:4])
    print(f"\n  cable synthesis: |s| = {np.linalg.norm(s_body):.15f} "
          f"{'OK' if abs(np.linalg.norm(s_body)-1) < 1e-12 else 'FAIL'}")
    ok &= abs(np.linalg.norm(s_body) - 1.0) < 1e-12
    # rotating s back into the world must reproduce the hook -> attach direction
    pi_, qi_ = tracks[1].at(tq); pL_, qL_ = pay.at(tq)
    Ri = quat_to_R(qi_)
    e = (pL_ + quat_to_R(qL_) @ np.array(ccfg["attach_point"][1])) \
        - (pi_ + Ri @ np.array(ccfg["hook_offset"][1]))
    err = np.linalg.norm(Ri @ s_body - e / np.linalg.norm(e))
    print(f"  R_i @ s_body reproduces the world direction: err {err:.2e} "
          f"{'OK' if err < 1e-12 else 'FAIL'}")
    ok &= err < 1e-12
    # a body-frame quantity must be invariant to the drone's yaw; a world one is not
    print(f"  (body-frame check passes only if R_i^T was applied)")

    # collision guard: two topics of the same type for one drone must not merge
    owner = {}
    k1 = table_key("odom", "falcon1", "/falcon1/odometry_estimate", owner)
    k2 = table_key("odom", "falcon1", "/falcon1/odometry_gt", owner)
    print(f"  same-type topic collision -> '{k1}' vs '{k2}' "
          f"{'OK' if k1 != k2 else 'FAIL - would interleave'}")
    ok &= k1 != k2

    print("\n  SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", help="input ROS 1 .bag")
    ap.add_argument("-o", "--out", default="data/extracted", help="output directory")
    ap.add_argument("--drones", default=",".join(DEFAULT_DRONES),
                    help="comma-separated namespaces; order fixes the integer index")
    ap.add_argument("--list", action="store_true", help="print the bag inventory and exit")
    ap.add_argument("--self-test", action="store_true", help="verify synthesis + CSV, no bag needed")
    ap.add_argument("--uwb-rate", type=float, help="override UWB rate [Hz]")
    ap.add_argument("--uwb-noise", type=float, help="override UWB noise std [m]")
    ap.add_argument("--uwb-seed", type=int, help="override UWB RNG seed")
    ap.add_argument("--no-antenna-offset", action="store_true",
                    help="zero the antenna offsets (WARNING: degenerate, see ekf_plan.md section 1)")
    args = ap.parse_args()

    cfg = dict(UWB)
    cfg["antenna_offset"] = dict(UWB["antenna_offset"])
    if args.uwb_rate is not None:
        cfg["rate_hz"] = args.uwb_rate
    if args.uwb_noise is not None:
        cfg["noise_std"] = args.uwb_noise
    if args.uwb_seed is not None:
        cfg["seed"] = args.uwb_seed
    if args.no_antenna_offset:
        cfg["antenna_offset"] = {}
        print("! antenna offsets zeroed: the system is rank 44/48 in this configuration")

    if args.self_test:
        os.makedirs(args.out, exist_ok=True)
        return self_test(args.out)
    if not args.bag:
        ap.error("a bag is required unless --self-test is given")
    if args.list:
        list_bag(args.bag)
        return 0
    convert(args.bag, args.out, [d for d in args.drones.split(",") if d], cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
