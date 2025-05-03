#!/usr/bin/env python3
import argparse
import asyncio
import base64
import json
import os
import struct
import time

import numpy as np
import open3d as o3d
from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer

FLOAT32_TYPE = 7  # as defined in the foxglove.PointCloud schema :contentReference[oaicite:0]{index=0}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("pcd_dir")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--skip_n", type=int, default=0)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    # 1) Load the foxglove.PointCloud JSON Schema (you can npm-install @foxglove/schemas
    #    or copy schemas/jsonschema/PointCloud.json from https://github.com/foxglove/schemas)
    with open("visualizer/PointCloud.json") as f:
        schema_json = f.read()

    async with FoxgloveServer(
        args.host, args.port, "PCD JSON → PointCloud", supported_encodings=["json"]
    ) as server:
        chan_id = await server.add_channel(
            {
                "topic": "/pointcloud",
                "encoding": "json",
                "schemaName": "foxglove.PointCloud",
                "schema": schema_json,
                "schemaEncoding": "jsonschema",
            }
        )

        files = sorted(f for f in os.listdir(args.pcd_dir) if f.endswith(".pcd"))
        while True:
            for fname in files[args.skip_n :]:
                # -- load and axis‐transform your PCD as before --
                tpcd = o3d.t.io.read_point_cloud(os.path.join(args.pcd_dir, fname))
                pts = tpcd.point.positions.numpy()
                argmax = tpcd.point["class"].numpy().astype(np.float32).squeeze()
                pts_t = np.empty_like(pts)
                pts_t[:, 0], pts_t[:, 1], pts_t[:, 2] = pts[:, 1], pts[:, 2], pts[:, 0]

                # -- pack as <float32 x, float32 y, float32 z, float32 argmax> --
                buf = bytearray()
                for x, y, z, a in zip(pts_t[:, 0], pts_t[:, 1], pts_t[:, 2], argmax):
                    buf += struct.pack("<ffff", x, y, z, a)

                # -- build the foxglove.PointCloud message --
                ts_ns = int(time.time() * 1e9)
                msg = {
                    "timestamp": {
                        "sec": ts_ns // 1_000_000_000,
                        "nsec": ts_ns % 1_000_000_000,
                    },
                    "frame_id": "lidar",
                    "pose": {
                        "position": {"x": 0, "y": 0, "z": 0},
                        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                    },
                    "point_stride": 16,  # four floats × 4 bytes
                    "fields": [
                        {"name": "x", "offset": 0, "type": FLOAT32_TYPE},
                        {"name": "y", "offset": 4, "type": FLOAT32_TYPE},
                        {"name": "z", "offset": 8, "type": FLOAT32_TYPE},
                        {"name": "class", "offset": 12, "type": FLOAT32_TYPE},
                    ],
                    "data": base64.b64encode(buf).decode("ascii"),
                }

                await server.send_message(
                    chan_id,
                    int(time.time() * 1e3),  # publish timestamp in ms
                    json.dumps(msg).encode("utf8"),
                )
                await asyncio.sleep(1.0 / args.fps)


if __name__ == "__main__":
    run_cancellable(main())
