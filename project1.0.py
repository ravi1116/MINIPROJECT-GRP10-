#!/usr/bin/env python3
"""
traffic_passkey.py

Minimal prototype for per-frame passkey extraction from traffic camera frames.

Usage:
    python traffic_passkey.py --mode demo            # runs a local demo simulating two endpoints from the same source
    python traffic_passkey.py --mode webcam          # uses your default webcam (single-run demo)
    python traffic_passkey.py --video path/to.mp4    # process a video file instead of webcam/mode

Notes:
 - This prototype DOES NOT include ECC/fuzzy-extractor reconciliation.
 - It demonstrates deterministic preprocessing, occupancy-grid extraction,
   conditioning (SHA-256), HKDF-based key derivation, and MAC confirmation.
 - You can toggle 'noise' to simulate viewpoint differences between endpoints.
"""

import cv2
import numpy as np
import argparse
import time
import hashlib
import hmac
import os
import sys

# -------------------------
# Config (tweakable)
# -------------------------
FRAME_SIZE = (320, 180)     # width, height after resize
ROI = (0, 0, FRAME_SIZE[0], FRAME_SIZE[1])  # x,y,w,h within resized frame
GRID = (8, 8)               # occupancy grid (rows, cols)
SCHEMA_ID = b"traffic_schema_v1"
DEVICE_SALT = b"device-secret-EXAMPLE"   # in real use, keep secret on-device
KEY_LEN = 32                # bytes (256-bit)
MAC_LEN = 32                # HMAC-SHA256

# -------------------------
# Utility crypto
# -------------------------
def sha256(x: bytes) -> bytes:
    return hashlib.sha256(x).digest()

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    # simple HKDF expand (RFC 5869)
    okm = b""
    previous = b""
    i = 1
    while len(okm) < length:
        data = previous + info + bytes([i])
        previous = hmac.new(prk, data, hashlib.sha256).digest()
        okm += previous
        i += 1
        if i > 255:
            raise Exception("HKDF expand iterations exceeded")
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
    # deterministic: resize -> grayscale -> crop ROI
    frame = cv2.resize(frame_bgr, FRAME_SIZE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x,y,w,h = ROI
    crop = gray[y:y+h, x:x+w]
    return crop

def extract_occupancy_bits(gray_crop, grid=GRID, threshold_method="mean"):
    # coarse occupancy grid: mark cell=1 if there's 'something' in the cell
    gh, gw = grid
    h, w = gray_crop.shape
    cell_h = h // gh
    cell_w = w // gw
    bits = []
    for r in range(gh):
        for c in range(gw):
            cell = gray_crop[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            if cell.size == 0:
                bits.append(0)
                continue
            if threshold_method == "mean":
                val = int(cell.mean() > 12)   # tune threshold for your camera
            else:
                # fallback
                val = int(cell.mean() > 12)
            bits.append(val)
    # pack bits into bytes
    b = 0
    out = bytearray()
    for i,bit in enumerate(bits):
        b = (b << 1) | (bit & 1)
        if (i % 8) == 7:
            out.append(b)
            b = 0
    # If leftover bits (not multiple of 8), pad on the right
    rem = len(bits) % 8
    if rem != 0:
        b <<= (8 - rem)
        out.append(b)
    return bytes(out)

def condition_raw(raw_bytes: bytes, timestamp_ms: int) -> bytes:
    # conditioning: SHA256(raw || schema || timestamp)
    ts_bytes = str(timestamp_ms).encode()
    return sha256(raw_bytes + SCHEMA_ID + ts_bytes)

def derive_frame_key(conditioned: bytes, device_salt: bytes, timestamp_ms: int) -> bytes:
    # HKDF: use conditioned as IKM, salt = device_salt || timestamp
    salt = device_salt + str(timestamp_ms).encode()
    key = hkdf(salt=salt, ikm=conditioned, info=b"frame-key", length=KEY_LEN)
    return key

# -------------------------
# Demo flow (simulate two endpoints A and B)
# -------------------------
def run_demo(video_source=0, simulate_noise=False, frames_to_process=120):
    """
    Simulate sender (A) and receiver (B) reading the same source.
    If simulate_noise=True, B will get slightly noised frames to simulate viewpoint differences.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("ERROR: cannot open video source", video_source)
        return

    idx = 0
    stats = {"total":0, "match":0}
    print("Starting demo. Press Ctrl+C to stop.")

    # running average background (for optional motion detection) - simple
    bg = None
    try:
        while idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break
            ts_ms = int(time.time()*1000)

            # Preprocess for endpoint A
            cropA = preprocess_frame(frame)
            # optionally update bg
            if bg is None:
                bg = cropA.astype(np.float32)
            else:
                cv2.accumulateWeighted(cropA.astype(np.float32), bg, 0.05)
            # Extract raw bits for A
            rawA = extract_occupancy_bits(cropA, GRID)

            # Endpoint B - either same frame or noisy version
            if simulate_noise:
                # apply slight blur + brightness shift + small translation to simulate viewpoint difference
                noisy = cv2.GaussianBlur(frame, (5,5), 0)
                noisy = cv2.convertScaleAbs(noisy, alpha=1.02, beta=4)
                # small translation:
                M = np.float32([[1, 0, 1], [0, 1, -1]])  # shift x by +1, y by -1
                noisy = cv2.warpAffine(noisy, M, (frame.shape[1], frame.shape[0]))
                cropB = preprocess_frame(noisy)
            else:
                cropB = preprocess_frame(frame)

            rawB = extract_occupancy_bits(cropB, GRID)

            # Condition -> derive per-frame keys (A & B)
            condA = condition_raw(rawA, ts_ms)
            condB = condition_raw(rawB, ts_ms)   # note: in field, ts must be synchronized / same frame index

            keyA = derive_frame_key(condA, DEVICE_SALT, ts_ms)
            keyB = derive_frame_key(condB, DEVICE_SALT, ts_ms)

            # Use MAC confirmation: A computes tag on 'frame-index' and B verifies
            msg = ("frame:%d" % idx).encode()
            tagA = compute_hmac(keyA, msg)
            tagB = compute_hmac(keyB, msg)

            match = hmac.compare_digest(tagA, tagB)
            stats["total"] += 1
            if match:
                stats["match"] += 1

            # Print small status
            print(f"[frame {idx}] ts={ts_ms} match={match} (A_key[:8]={keyA[:8].hex()} B_key[:8]={keyB[:8].hex()})")

            # Optionally display (small) to see ROI
            disp = cv2.resize(cropA, (GRID[1]*40, GRID[0]*40), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("ROI-preview (A)", disp)
            if simulate_noise:
                dispB = cv2.resize(cropB, (GRID[1]*40, GRID[0]*40), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("ROI-preview (B-noisy)", dispB)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            idx += 1
            # small sleep to mimic realtime if reading fast files
            time.sleep(0.02)

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
# Simple single-run (webcam) flow
# -------------------------
def run_single_webcam_capture(video_source=0, frames=30):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Cannot open webcam/video. Abort.")
        return
    print("Capturing %d frames from webcam..." % frames)
    for i in range(frames):
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = int(time.time()*1000)
        crop = preprocess_frame(frame)
        raw = extract_occupancy_bits(crop, GRID)
        cond = condition_raw(raw, ts_ms)
        key = derive_frame_key(cond, DEVICE_SALT, ts_ms)
        print(f"frame {i} ts={ts_ms} key={key.hex()[:64]} ...")
        time.sleep(0.05)
    cap.release()

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Traffic-frame passkey prototype")
    parser.add_argument("--mode", choices=["demo","webcam"], default="demo",
                        help="demo: simulate two endpoints; webcam: single webcam capture")
    parser.add_argument("--video", type=str, default=None, help="optional video file path instead of webcam")
    parser.add_argument("--noise", action="store_true", help="simulate viewpoint noise on endpoint B")
    parser.add_argument("--frames", type=int, default=120, help="frames to process in demo")
    args = parser.parse_args()

    source = 0
    if args.video:
        source = args.video

    if args.mode == "demo":
        run_demo(video_source=source, simulate_noise=args.noise, frames_to_process=args.frames)
    elif args.mode == "webcam":
        run_single_webcam_capture(video_source=source, frames=args.frames)
    else:
        print("Unknown mode")

if __name__ == "__main__":
    main()
    