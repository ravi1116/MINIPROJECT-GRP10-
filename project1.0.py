#!/usr/bin/env python3
"""
traffic_passkey.py - Optimized version

Improvements:
- Use capture timestamp (frame position) instead of wall-clock time
- Adaptive occupancy check (mean + std dev thresholding)
- Hamming distance logging between endpoints
"""

import cv2
import numpy as np
import argparse
import time
import hashlib
import hmac
import sys

# -------------------------
# Config
# -------------------------
FRAME_SIZE = (320, 180)
ROI = (0, 0, FRAME_SIZE[0], FRAME_SIZE[1])
GRID = (8, 8)
SCHEMA_ID = b"traffic_schema_v1"
DEVICE_SALT = b"device-secret-EXAMPLE"
KEY_LEN = 32
MAC_LEN = 32

# -------------------------
# Crypto helpers
# -------------------------
def sha256(x: bytes) -> bytes:
    return hashlib.sha256(x).digest()

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    okm, previous, i = b"", b"", 1
    while len(okm) < length:
        data = previous + info + bytes([i])
        previous = hmac.new(prk, data, hashlib.sha256).digest()
        okm += previous
        i += 1
    return okm[:length]

def hkdf(salt: bytes, ikm: bytes, info: bytes, length: int):
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)

def compute_hmac(key: bytes, message: bytes) -> bytes:
    return hmac.new(key, message, hashlib.sha256).digest()

# -------------------------
# Image -> bits pipeline
# -------------------------
def preprocess_frame(frame_bgr):
    frame = cv2.resize(frame_bgr, FRAME_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x,y,w,h = ROI
    crop = gray[y:y+h, x:x+w]
    return crop

def cell_is_occupied(cell):
    mean = float(cell.mean())
    std = float(cell.std())
    return 1 if (mean > 25 or std > 15) else 0

def extract_occupancy_bits(gray_crop, grid=GRID):
    gh, gw = grid
    h, w = gray_crop.shape
    cell_h, cell_w = h // gh, w // gw
    bits = []
    for r in range(gh):
        for c in range(gw):
            cell = gray_crop[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            if cell.size == 0:
                bits.append(0)
            else:
                bits.append(cell_is_occupied(cell))
    # pack bits
    b, out = 0, bytearray()
    for i, bit in enumerate(bits):
        b = (b << 1) | (bit & 1)
        if (i % 8) == 7:
            out.append(b)
            b = 0
    if len(bits) % 8 != 0:
        b <<= (8 - (len(bits) % 8))
        out.append(b)
    return bytes(out)

def condition_raw(raw_bytes: bytes, timestamp_ms: int) -> bytes:
    ts_bytes = str(timestamp_ms).encode()
    return sha256(raw_bytes + SCHEMA_ID + ts_bytes)

def derive_frame_key(conditioned: bytes, device_salt: bytes, timestamp_ms: int) -> bytes:
    salt = device_salt + str(timestamp_ms).encode()
    return hkdf(salt, conditioned, b"frame-key", KEY_LEN)

# -------------------------
# Utils
# -------------------------
def bytes_hamming(a: bytes, b: bytes) -> int:
    dist = 0
    for x, y in zip(a, b):
        dist += bin(x ^ y).count("1")
    if len(a) != len(b):
        longer = a if len(a) > len(b) else b
        for x in longer[len(min(a,b)):]:
            dist += bin(x).count("1")
    return dist

# -------------------------
# Demo
# -------------------------
def run_demo(video_source=0, simulate_noise=False, frames_to_process=120):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("ERROR: cannot open video source", video_source)
        return

    idx, stats = 0, {"total":0, "match":0}
    print("Starting demo. Press Ctrl+C to stop.")

    try:
        while idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break

            # use frame timestamp if available
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if ts_ms == 0:
                ts_ms = int(time.time()*1000)

            cropA = preprocess_frame(frame)
            rawA = extract_occupancy_bits(cropA, GRID)

            if simulate_noise:
                noisy = cv2.GaussianBlur(frame, (5,5), 0)
                noisy = cv2.convertScaleAbs(noisy, alpha=1.02, beta=4)
                M = np.float32([[1, 0, 1], [0, 1, -1]])
                noisy = cv2.warpAffine(noisy, M, (frame.shape[1], frame.shape[0]))
                cropB = preprocess_frame(noisy)
            else:
                cropB = preprocess_frame(frame)
            rawB = extract_occupancy_bits(cropB, GRID)

            condA, condB = condition_raw(rawA, ts_ms), condition_raw(rawB, ts_ms)
            keyA, keyB = derive_frame_key(condA, DEVICE_SALT, ts_ms), derive_frame_key(condB, DEVICE_SALT, ts_ms)

            msg = ("frame:%d" % idx).encode()
            tagA, tagB = compute_hmac(keyA, msg), compute_hmac(keyB, msg)
            match = hmac.compare_digest(tagA, tagB)

            stats["total"] += 1
            if match: stats["match"] += 1

            hd = bytes_hamming(rawA, rawB)
            print(f"[frame {idx}] ts={ts_ms} match={match} hamming={hd} "
                  f"A_key[:8]={keyA[:8].hex()} B_key[:8]={keyB[:8].hex()}")

            disp = cv2.resize(cropA, (GRID[1]*40, GRID[0]*40), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("ROI-preview (A)", disp)
            if simulate_noise:
                dispB = cv2.resize(cropB, (GRID[1]*40, GRID[0]*40), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("ROI-preview (B-noisy)", dispB)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            idx += 1
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if stats["total"] > 0:
            print("Demo finished. Match rate: %.2f%% (%d/%d)" %
                  (100.0 * stats["match"] / stats["total"], stats["match"], stats["total"]))
        else:
            print("No frames processed.")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Traffic-frame passkey prototype (optimized)")
    parser.add_argument("--mode", choices=["demo","webcam"], default="demo")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    source = args.video if args.video else 0
    if args.mode == "demo":
        run_demo(video_source=source, simulate_noise=args.noise, frames_to_process=args.frames)
    else:
        run_demo(video_source=source, simulate_noise=False, frames_to_process=args.frames)

if __name__ == "__main__":
    main()
