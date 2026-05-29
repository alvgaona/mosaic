import sys
import os
import math
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


SCENES = {
    "Escena3": {"count": 2, "pattern": "Escena3/Escena3_Imagen{}.{}"},
    "Escena4": {"count": 3, "pattern": "Escena4/Escena4_imagen{}.{}"},
    "Escena5": {"count": 7, "pattern": "Escena5/Escena5_Imagen{}.{}"},
}

RATIO_THRESH = 0.76
OUTLIER_FACTOR = 3.0
OUTLIER_MAX_ITER = 5
ERROR_DISPLAY_THRESH = 2.0
GEOMETRIC_THRESH = 5.0

RANSAC_THRESH = 3.0
RANSAC_MAX_ITER = 2000
RANSAC_SAMPLE_SIZE = 4


def load_scene(scene_name, count, pattern):
    base = os.path.dirname(os.path.abspath(__file__))
    grays = []
    colors = []
    for i in range(1, count + 1):
        pgm_path = os.path.join(base, pattern.format(i, "pgm"))
        jpg_path = os.path.join(base, pattern.format(i, "jpg"))
        gray = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
        if gray is None:
            raise FileNotFoundError(f"Cannot load {pgm_path}")
        if color is None:
            raise FileNotFoundError(f"Cannot load {jpg_path}")
        print(f"  Image {i}: gray {gray.shape}, color {color.shape}")
        grays.append(gray)
        colors.append(color)
    return grays, colors


DETECTORS = {
    "SIFT": lambda: cv2.SIFT_create(),
    "ORB": lambda: cv2.ORB_create(nfeatures=10000),
    "KAZE": lambda: cv2.KAZE_create(),
    "BRISK": lambda: cv2.BRISK_create(),
}

BINARY_DESCRIPTORS = {"ORB", "BRISK"}


def detect_features(gray, detector_name="SIFT"):
    detector = DETECTORS[detector_name]()
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio_thresh=RATIO_THRESH, detector_name="SIFT"):
    norm = cv2.NORM_HAMMING if detector_name in BINARY_DESCRIPTORS else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


def _normalize_points(pts):
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = math.sqrt(2) / (mean_dist + 1e-12)
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ])
    return T


def compute_homography(pts_src, pts_dst):
    T_src = _normalize_points(pts_src)
    T_dst = _normalize_points(pts_dst)
    src_h = np.column_stack([pts_src, np.ones(len(pts_src))])
    dst_h = np.column_stack([pts_dst, np.ones(len(pts_dst))])
    norm_src = (T_src @ src_h.T).T[:, :2]
    norm_dst = (T_dst @ dst_h.T).T[:, :2]

    n = norm_src.shape[0]
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = norm_src[i]
        xp, yp = norm_dst[i]
        A[2 * i] = [x, y, 1, 0, 0, 0, -xp * x, -xp * y]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -yp * x, -yp * y]
        b[2 * i] = xp
        b[2 * i + 1] = yp
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H_norm = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ])
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H = H / H[2, 2]
    return H


def reprojection_errors(pts_src, pts_dst, H):
    n = pts_src.shape[0]
    src_h = np.column_stack([pts_src, np.ones(n)])
    projected = (H @ src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - pts_dst, axis=1)
    return errors


def filter_outliers(pts_src, pts_dst, factor=OUTLIER_FACTOR, max_iterations=OUTLIER_MAX_ITER):
    H = compute_homography(pts_src, pts_dst)
    for iteration in range(max_iterations):
        errors = reprojection_errors(pts_src, pts_dst, H)
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        print(f"    Iteration {iteration}: {len(pts_src)} points, "
              f"mean error = {mean_err:.2f} px, max error = {max_err:.2f} px")
        thresh = factor * mean_err
        mask = errors <= thresh
        if np.all(mask):
            break
        pts_src = pts_src[mask]
        pts_dst = pts_dst[mask]
        if len(pts_src) < 4:
            print("    WARNING: fewer than 4 points remaining")
            break
        H = compute_homography(pts_src, pts_dst)
    return pts_src, pts_dst, H


def filter_outliers_geometric(pts_src, pts_dst, thresh=GEOMETRIC_THRESH, max_iterations=OUTLIER_MAX_ITER):
    H = compute_homography(pts_src, pts_dst)
    for iteration in range(max_iterations):
        errors = reprojection_errors(pts_src, pts_dst, H)
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        gate = max(thresh, 3.0 * np.median(errors))
        print(f"    Iteration {iteration}: {len(pts_src)} points, "
              f"mean error = {mean_err:.2f} px, max error = {max_err:.2f} px "
              f"(gate = {gate:.2f} px, target = {thresh:.2f} px)")
        mask = errors <= gate
        if gate <= thresh and np.all(mask):
            break
        pts_src = pts_src[mask]
        pts_dst = pts_dst[mask]
        if len(pts_src) < 4:
            print("    WARNING: fewer than 4 points remaining")
            break
        H = compute_homography(pts_src, pts_dst)
    return pts_src, pts_dst, H


def filter_outliers_ransac(
    pts_src, pts_dst,
    thresh=RANSAC_THRESH, max_iter=RANSAC_MAX_ITER, sample_size=RANSAC_SAMPLE_SIZE,
):
    n = len(pts_src)
    best_inliers = None
    best_count = 0
    rng = np.random.default_rng(42)
    for _ in range(max_iter):
        idx = rng.choice(n, size=sample_size, replace=False)
        H_candidate = compute_homography(pts_src[idx], pts_dst[idx])
        errors = reprojection_errors(pts_src, pts_dst, H_candidate)
        inliers = errors <= thresh
        count = np.sum(inliers)
        if count > best_count:
            best_count = count
            best_inliers = inliers
    pts_src_f = pts_src[best_inliers]
    pts_dst_f = pts_dst[best_inliers]
    H = compute_homography(pts_src_f, pts_dst_f)
    errors = reprojection_errors(pts_src_f, pts_dst_f, H)
    print(f"    RANSAC: {n} → {len(pts_src_f)} inliers "
          f"(thresh={thresh}, iters={max_iter}), "
          f"mean error = {np.mean(errors):.2f} px, max error = {np.max(errors):.2f} px")
    return pts_src_f, pts_dst_f, H


def chain_homographies(pairwise_H, ref_idx):
    n = len(pairwise_H) + 1
    cumulative = [None] * n
    cumulative[ref_idx] = np.eye(3)
    for j in range(ref_idx - 1, -1, -1):
        cumulative[j] = cumulative[j + 1] @ pairwise_H[j]
    for j in range(ref_idx + 1, n):
        cumulative[j] = cumulative[j - 1] @ np.linalg.inv(pairwise_H[j - 1])
    return cumulative


def compute_canvas_bounds(image_shapes, cumulative_H):
    all_corners = []
    for i, (h, w) in enumerate(image_shapes):
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        corners_h = np.column_stack([corners, np.ones(4)])
        transformed = (cumulative_H[i] @ corners_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        all_corners.append(transformed)
    all_corners = np.vstack(all_corners)
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1],
    ], dtype=np.float64)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    adjusted_H = [T @ H for H in cumulative_H]
    return canvas_w, canvas_h, adjusted_H


def blend_mosaic(color_images, cumulative_H, canvas_h, canvas_w):
    accumulator = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    count = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    for i, img in enumerate(color_images):
        warped = cv2.warpPerspective(
            img.astype(np.float64), cumulative_H[i], (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        mask = cv2.warpPerspective(
            np.ones(img.shape[:2], dtype=np.float64), cumulative_H[i], (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        accumulator += warped
        count += mask
    mosaic = np.where(
        count[:, :, np.newaxis] > 0,
        accumulator / count[:, :, np.newaxis],
        0,
    ).astype(np.uint8)
    return mosaic


def crop_black_margins(mosaic):
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return mosaic
    x, y, w, h = cv2.boundingRect(coords)
    return mosaic[y:y + h, x:x + w]


def visualize_matches(gray1, gray2, kp1, kp2, matches, title="", save_path=None):
    img = cv2.drawMatches(
        gray1, kp1, gray2, kp2, matches, None,
        matchColor=(255, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_correspondences_with_error(
    gray, pts_src, pts_dst, H, error_thresh=ERROR_DISPLAY_THRESH, title="", save_path=None
):
    errors = reprojection_errors(pts_src, pts_dst, H)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    src_h = np.column_stack([pts_src, np.ones(len(pts_src))])
    projected = (H @ src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    for i in range(len(pts_src)):
        pt = tuple(pts_dst[i].astype(int))
        pt_proj = tuple(projected[i].astype(int))
        color = (0, 255, 255) if errors[i] <= error_thresh else (0, 0, 255)
        thickness = 3 if errors[i] <= error_thresh else 1
        cv2.circle(img, pt, 3, color, thickness)
        if errors[i] > error_thresh:
            cv2.line(img, pt, pt_proj, (0, 0, 255), 1)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} (mean={np.mean(errors):.2f}, max={np.max(errors):.2f})")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_scene(scene_name, count, pattern):
    print(f"\n{'=' * 60}")
    print(f"Processing {scene_name} ({count} images)")
    print(f"{'=' * 60}")

    print("\n[1] Loading images...")
    grays, colors = load_scene(scene_name, count, pattern)

    print("\n[2] Detecting SIFT features...")
    all_kp = []
    all_desc = []
    for i, gray in enumerate(grays):
        kp, desc = detect_features(gray)
        print(f"  Image {i + 1}: {len(kp)} keypoints")
        all_kp.append(kp)
        all_desc.append(desc)

    print("\n[3] Matching descriptors between adjacent pairs...")
    all_matches = []
    all_pts_src = []
    all_pts_dst = []
    for i in range(count - 1):
        matches = match_descriptors(all_desc[i], all_desc[i + 1])
        print(f"  Pair ({i + 1}, {i + 2}): {len(matches)} matches after ratio test")
        pts_src = np.float64([all_kp[i][m.queryIdx].pt for m in matches])
        pts_dst = np.float64([all_kp[i + 1][m.trainIdx].pt for m in matches])
        all_matches.append(matches)
        all_pts_src.append(pts_src)
        all_pts_dst.append(pts_dst)
        visualize_matches(
            grays[i], grays[i + 1], all_kp[i], all_kp[i + 1], matches,
            title=f"{scene_name}: Matches ({i + 1}, {i + 2})",
            save_path=f"matches_{scene_name}_{i + 1}_{i + 2}.png",
        )

    print("\n[4] Computing homographies and filtering outliers...")
    pairwise_H = []
    for i in range(count - 1):
        print(f"  Pair ({i + 1}, {i + 2}):")
        pts_src_f, pts_dst_f, H = filter_outliers(all_pts_src[i], all_pts_dst[i])
        print(f"    H =\n{H}")
        pairwise_H.append(H)
        visualize_correspondences_with_error(
            grays[i + 1], pts_src_f, pts_dst_f, H,
            title=f"{scene_name}: Correspondences ({i + 1}, {i + 2}) after filtering",
            save_path=f"correspondences_{scene_name}_{i + 1}_{i + 2}.png",
        )

    ref_idx = count // 2
    print(f"\n[5] Chaining homographies (reference = Image {ref_idx + 1})...")
    cumulative_H = chain_homographies(pairwise_H, ref_idx)

    image_shapes = [(img.shape[0], img.shape[1]) for img in colors]
    canvas_w, canvas_h, cumulative_H = compute_canvas_bounds(image_shapes, cumulative_H)
    print(f"  Canvas size: {canvas_w} x {canvas_h}")

    print("\n[6] Warping and blending...")
    mosaic = blend_mosaic(colors, cumulative_H, canvas_h, canvas_w)

    print("\n[7] Cropping and saving...")
    mosaic = crop_black_margins(mosaic)
    output_path = f"panorama_{scene_name}.jpg"
    cv2.imwrite(output_path, mosaic)
    print(f"  Saved: {output_path} ({mosaic.shape[1]} x {mosaic.shape[0]})")

    plt.figure(figsize=(20, 6))
    plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    plt.title(f"Panorama {scene_name}")
    plt.axis("off")
    plt.savefig(f"panorama_{scene_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_ablation(scene_name, count, pattern):
    print(f"\n{'=' * 60}")
    print(f"Ablation Study: {scene_name} ({count} images)")
    print(f"{'=' * 60}")

    print("\nLoading images...")
    grays, _ = load_scene(scene_name, count, pattern)

    results = []
    for det_name in DETECTORS:
        print(f"\n--- {det_name} ---")
        t0 = time.time()
        all_kp = []
        all_desc = []
        for i, gray in enumerate(grays):
            kp, desc = detect_features(gray, detector_name=det_name)
            all_kp.append(kp)
            all_desc.append(desc)
        t_detect = time.time() - t0

        total_kp = sum(len(kp) for kp in all_kp)
        print(f"  Keypoints: {[len(kp) for kp in all_kp]} (total: {total_kp})")

        if any(d is None for d in all_desc):
            print(f"  SKIPPED — no descriptors produced")
            results.append({
                "detector": det_name, "keypoints": total_kp, "matches_raw": 0,
                "matches_filtered": 0, "mean_error": float("inf"),
                "max_error": float("inf"), "time_detect": t_detect, "time_total": t_detect,
            })
            continue

        _ = time.time()
        pair_results = []
        for i in range(count - 1):
            matches = match_descriptors(
                all_desc[i], all_desc[i + 1],
                ratio_thresh=RATIO_THRESH, detector_name=det_name,
            )
            if len(matches) < 4:
                print(f"  Pair ({i+1},{i+2}): only {len(matches)} matches — too few")
                pair_results.append((len(matches), 0, float("inf"), float("inf")))
                continue
            pts_src = np.float64([all_kp[i][m.queryIdx].pt for m in matches])
            pts_dst = np.float64([all_kp[i + 1][m.trainIdx].pt for m in matches])
            n_raw = len(matches)
            pts_src_f, pts_dst_f, H = filter_outliers(pts_src, pts_dst)
            errors = reprojection_errors(pts_src_f, pts_dst_f, H)
            pair_results.append((n_raw, len(pts_src_f), np.mean(errors), np.max(errors)))
            print(f"  Pair ({i+1},{i+2}): {n_raw} raw → {len(pts_src_f)} filtered, "
                  f"mean err = {np.mean(errors):.2f} px, max err = {np.max(errors):.2f} px")
        t_total = time.time() - t0

        avg_raw = np.mean([r[0] for r in pair_results])
        avg_filtered = np.mean([r[1] for r in pair_results])
        avg_mean_err = np.mean([r[2] for r in pair_results if r[2] != float("inf")])
        avg_max_err = np.mean([r[3] for r in pair_results if r[3] != float("inf")])

        results.append({
            "detector": det_name,
            "keypoints": total_kp,
            "matches_raw": avg_raw,
            "matches_filtered": avg_filtered,
            "mean_error": avg_mean_err if not np.isnan(avg_mean_err) else float("inf"),
            "max_error": avg_max_err if not np.isnan(avg_max_err) else float("inf"),
            "time_detect": t_detect,
            "time_total": t_total,
        })

    print(f"\n{'=' * 60}")
    print(f"{'Detector':<10} {'Keypoints':>10} {'Matches':>10} {'Filtered':>10} "
          f"{'Mean Err':>10} {'Max Err':>10} {'Time (s)':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r['detector']:<10} {r['keypoints']:>10} {r['matches_raw']:>10.0f} "
              f"{r['matches_filtered']:>10.0f} {r['mean_error']:>10.2f} "
              f"{r['max_error']:>10.2f} {r['time_total']:>10.2f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    detectors = [r["detector"] for r in results]
    colors_plot = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    axes[0].bar(detectors, [r["keypoints"] for r in results], color=colors_plot[:len(results)])
    axes[0].set_title("Total Keypoints")
    axes[0].set_ylabel("Count")

    axes[1].bar(detectors, [r["matches_filtered"] for r in results], color=colors_plot[:len(results)])
    axes[1].set_title("Avg Filtered Matches")
    axes[1].set_ylabel("Count")

    valid = [r for r in results if r["mean_error"] != float("inf")]
    axes[2].bar(
        [r["detector"] for r in valid],
        [r["mean_error"] for r in valid],
        color=colors_plot[:len(valid)],
    )
    axes[2].set_title("Avg Mean Reproj. Error")
    axes[2].set_ylabel("Pixels")

    axes[3].bar(detectors, [r["time_total"] for r in results], color=colors_plot[:len(results)])
    axes[3].set_title("Total Time")
    axes[3].set_ylabel("Seconds")

    plt.suptitle(f"Ablation Study — {scene_name}", fontsize=14)
    plt.tight_layout()
    save_path = f"ablation_{scene_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved chart: {save_path}")


def run_filter_comparison(scene_name, count, pattern):
    print(f"\n{'=' * 60}")
    print(f"Filter Comparison: {scene_name} ({count} images)")
    print(f"{'=' * 60}")

    print("\nLoading images...")
    grays, _ = load_scene(scene_name, count, pattern)

    print("\nDetecting SIFT features...")
    all_kp = []
    all_desc = []
    for i, gray in enumerate(grays):
        kp, desc = detect_features(gray)
        all_kp.append(kp)
        all_desc.append(desc)

    print("\nMatching descriptors...")
    all_pts_src = []
    all_pts_dst = []
    for i in range(count - 1):
        matches = match_descriptors(all_desc[i], all_desc[i + 1])
        pts_src = np.float64([all_kp[i][m.queryIdx].pt for m in matches])
        pts_dst = np.float64([all_kp[i + 1][m.trainIdx].pt for m in matches])
        all_pts_src.append(pts_src)
        all_pts_dst.append(pts_dst)

    methods = {
        "Residual (3x mean)": lambda s, d: filter_outliers(s, d),
        "Geometric (d<=5.0 px)": lambda s, d: filter_outliers_geometric(s, d),
        "RANSAC (t=3.0, n=2000)": lambda s, d: filter_outliers_ransac(s, d),
    }

    results = []
    for method_name, method_fn in methods.items():
        print(f"\n--- {method_name} ---")
        pair_results = []
        t0 = time.time()
        for i in range(count - 1):
            print(f"  Pair ({i+1},{i+2}):")
            pts_src_f, pts_dst_f, H = method_fn(
                all_pts_src[i].copy(), all_pts_dst[i].copy(),
            )
            errors = reprojection_errors(pts_src_f, pts_dst_f, H)
            pair_results.append({
                "pair": f"({i+1},{i+2})",
                "raw": len(all_pts_src[i]),
                "filtered": len(pts_src_f),
                "mean_err": np.mean(errors),
                "max_err": np.max(errors),
            })
        elapsed = time.time() - t0
        results.append({
            "method": method_name,
            "pairs": pair_results,
            "time": elapsed,
        })

    print(f"\n{'=' * 60}")
    print(f"{'Method':<28} {'Pair':>8} {'Raw':>8} {'Filtered':>10} "
          f"{'Mean Err':>10} {'Max Err':>10}")
    print("-" * 78)
    for r in results:
        for p in r["pairs"]:
            print(f"{r['method']:<28} {p['pair']:>8} {p['raw']:>8} {p['filtered']:>10} "
                  f"{p['mean_err']:>10.2f} {p['max_err']:>10.2f}")
        avg_mean = np.mean([p["mean_err"] for p in r["pairs"]])
        avg_filtered = np.mean([p["filtered"] for p in r["pairs"]])
        print(f"{'  AVERAGE':<28} {'':>8} {'':>8} {avg_filtered:>10.0f} "
              f"{avg_mean:>10.2f} {'':>10}   ({r['time']:.2f}s)")
        print()

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    _ = [r["method"] for r in results]
    colors_plot = ["#2196F3", "#4CAF50", "#FF9800"]
    x = np.arange(count - 1)
    width = 0.25

    for idx, r in enumerate(results):
        axes[0].bar(
            x + idx * width, [p["filtered"] for p in r["pairs"]],
            width, label=r["method"], color=colors_plot[idx],
        )
    axes[0].set_title("Filtered Matches per Pair")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([p["pair"] for p in results[0]["pairs"]])
    axes[0].legend()

    for idx, r in enumerate(results):
        axes[1].bar(
            x + idx * width, [p["mean_err"] for p in r["pairs"]],
            width, label=r["method"], color=colors_plot[idx],
        )
    axes[1].set_title("Mean Reproj. Error per Pair")
    axes[1].set_ylabel("Pixels")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([p["pair"] for p in results[0]["pairs"]])
    axes[1].legend()

    short_names = [r["method"].split(" ")[0] for r in results]
    bars = axes[2].bar(short_names, [r["time"] for r in results], color=colors_plot)
    axes[2].set_title("Total Time")
    axes[2].set_ylabel("Seconds")
    for bar, r in zip(bars, results):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{r['time']:.2f}s", ha="center", va="bottom", fontsize=9,
        )

    plt.suptitle(f"Filter Comparison — {scene_name}", fontsize=14)
    plt.tight_layout()
    save_path = f"filter_comparison_{scene_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {save_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    ablation = "--ablation" in args
    filter_cmp = "--filter-comparison" in args
    names = [a for a in args if not a.startswith("--")]
    if not names:
        names = list(SCENES.keys())
    for name in names:
        if name not in SCENES:
            print(f"Unknown scene: {name}")
            continue
        scene = SCENES[name]
        if ablation:
            run_ablation(name, scene["count"], scene["pattern"])
        elif filter_cmp:
            run_filter_comparison(name, scene["count"], scene["pattern"])
        else:
            process_scene(name, scene["count"], scene["pattern"])
