import os
import sys
import numpy as np


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Network", "Model1"))

import airsim  # noqa: E402

DEPTH_MAX_VALUE = 50.0
DEPTH_INVALID_EPS = 1e-3


def stats(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    zeros = np.sum(arr == 0)
    le_eps = np.sum(arr <= 1e-3)
    nonfinite = np.sum(~finite)
    if np.any(finite):
        min_v = float(np.min(arr[finite]))
        max_v = float(np.max(arr[finite]))
        mean_v = float(np.mean(arr[finite]))
    else:
        min_v = max_v = mean_v = float("nan")
    print(
        f"{name}: shape={arr.shape}, min={min_v:.6f}, max={max_v:.6f}, mean={mean_v:.6f}, "
        f"zeros={int(zeros)}, <=1e-3={int(le_eps)}, nonfinite={int(nonfinite)}"
    )


def sanitize_depth(depth_img):
    depth_img = np.asarray(depth_img, dtype=np.float32)
    depth_img = np.nan_to_num(
        depth_img,
        nan=DEPTH_MAX_VALUE,
        posinf=DEPTH_MAX_VALUE,
        neginf=DEPTH_MAX_VALUE,
    )
    depth_img[depth_img <= DEPTH_INVALID_EPS] = DEPTH_MAX_VALUE
    return np.clip(depth_img, 0.0, DEPTH_MAX_VALUE)


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    vehicles = client.listVehicles()
    vehicle_name = vehicles[0] if vehicles else ""
    print(f"Vehicle: {vehicle_name!r}")

    responses = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)],
        vehicle_name=vehicle_name,
    )
    if not responses or responses[0].width <= 0:
        print("Depth response invalid")
        return

    r = responses[0]
    raw = np.array(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
    cleaned = sanitize_depth(raw)

    stats("raw_depth", raw)
    stats("cleaned_depth", cleaned)

    raw_zero = raw <= 1e-3
    if np.any(raw_zero):
        cleaned_at_raw_zero = cleaned[raw_zero]
        finite = np.isfinite(cleaned_at_raw_zero)
        if np.any(finite):
            print(
                "cleaned_at_raw_zero: "
                f"count={int(cleaned_at_raw_zero.size)}, "
                f"min={float(np.min(cleaned_at_raw_zero[finite])):.6f}, "
                f"max={float(np.max(cleaned_at_raw_zero[finite])):.6f}, "
                f"mean={float(np.mean(cleaned_at_raw_zero[finite])):.6f}"
            )
        else:
            print(f"cleaned_at_raw_zero: count={int(cleaned_at_raw_zero.size)}, all_nonfinite=1")
    else:
        print("raw_depth has no <=1e-3 pixels")


if __name__ == "__main__":
    main()
