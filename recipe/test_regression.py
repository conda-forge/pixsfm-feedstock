"""Regression test against the unmodified pixsfm 1.0 (COLMAP 3.8, pycolmap 0.4).

The inputs in regression_data.tar.xz were made with COLMAP 3.8 on the Sacre
Coeur images shipped in the pixsfm sources (SIFT, 2048 features per image,
exhaustive matching, incremental mapping; descriptors dropped). The reference
numbers below were produced by pixsfm 1.0.0 (conda-forge build 28, osx-arm64)
on these inputs, with the same configuration as this test.

On the same platform, keypoint adjustment and query refinement reproduce
pixsfm 1.0 exactly. Bundle adjustment converges to an equivalent solution: the
pose parametrization changed from Ceres' quaternion manifold to Eigen's (as in
COLMAP 4).

The dense features differ slightly between platforms (pixsfm 1.0 on
linux-aarch64 gives a BA initial cost of 60.189359 instead of 60.188280), so
the tolerances cover the spread of the unmodified pixsfm 1.0 across platforms.
The BA point update is weakly constrained: for pixsfm 1.0 its mean ranges from
0.066 (linux-aarch64) to 0.070 (osx-arm64), and the x86 (AVX2/FMA) builds land
higher still, so it is only checked loosely.
"""
import math
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pycolmap

from pixsfm.localization import QueryLocalizer
from pixsfm.refine_colmap import PixSfM
from pixsfm.util.colmap import read_keypoints_from_db

REFERENCE = {
    "ka": {
        "num_moved": 3208,
        "mean_shift": 0.117407369833,
        "max_shift": 5.89021698654,
    },
    "ba": {
        "initial_cost": 60.188279539,
        "final_cost": 55.4573792406,
        "mean_point_update": 0.0701497092952,
    },
    "qka": {
        "query": "93341989_396310999.jpg",
        "num_correspondences": 391,
        "mean_shift": 0.42976635116,
        "max_shift": 2.56299017498,
    },
    "qba": {
        "rotation_error_deg": 0.239258300072,
        "translation_error": 0.0141439514853,
    },
}

FEATURES = {"model": {"name": "dsift"}, "device": "cpu", "dtype": "float",
            "max_edge": 1024, "sparse": True, "use_cache": False}
SINGLE_THREAD = {"num_threads": 1}
failures = []


def check(name, value, expected, rtol=0.0, atol=0.0):
    ok = math.isclose(value, expected, rel_tol=rtol, abs_tol=atol)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {value:.12g} "
          f"(pixsfm 1.0: {expected:.12g})")
    if not ok:
        failures.append(name)


def rotation_angle(R):
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2))))


work = Path(tempfile.mkdtemp())
with tarfile.open(Path(__file__).parent / "regression_data.tar.xz") as tar:
    tar.extractall(work)
database, model = work / "database.db", work / "model"
images = Path("datasets/sacre_coeur/mapping").resolve()

sfm = PixSfM({"dense_features": FEATURES,
              "KA": {"optimizer": {"print_summary": False,
                                   "solver": SINGLE_THREAD}},
              "BA": {"optimizer": {"print_summary": False,
                                   "solver": SINGLE_THREAD},
                     "references": SINGLE_THREAD}})

# Keypoint adjustment.
before = read_keypoints_from_db(database, as_cpp_map=False)
sfm.refine_keypoints_from_db(work / "refined.db", database, images)
after = read_keypoints_from_db(work / "refined.db", as_cpp_map=False)
shift = np.concatenate([np.linalg.norm(after[n] - before[n], axis=1)
                        for n in sorted(before)])
ref = REFERENCE["ka"]
check("ka.num_moved", int((shift > 1e-9).sum()), ref["num_moved"], rtol=0.01)
check("ka.mean_shift", float(shift.mean()), ref["mean_shift"], rtol=1e-4)
check("ka.max_shift", float(shift.max()), ref["max_shift"], atol=1e-3)

# Featuremetric bundle adjustment.
rec = pycolmap.Reconstruction(str(model))
xyz0 = {i: np.array(p.xyz) for i, p in rec.points3D.items()}
rec, ba_data, _ = sfm.run_ba(rec, images)
summary = ba_data["summary"][0]
update = np.mean([np.linalg.norm(rec.points3D[i].xyz - xyz0[i]) for i in xyz0])
ref = REFERENCE["ba"]
check("ba.initial_cost", summary.initial_cost, ref["initial_cost"], rtol=1e-4)
# Different (equivalent) pose manifold: same optimum up to solver tolerance.
check("ba.final_cost", summary.final_cost, ref["final_cost"], rtol=2e-3)
check("ba.mean_point_update", float(update), ref["mean_point_update"],
      rtol=0.25)

# Query keypoint and bundle adjustment from a fixed, perturbed pose.
rec = pycolmap.Reconstruction(str(model))
query = max(rec.images.values(),
            key=lambda im: sum(p.has_point3D() for p in im.points2D))
ref = REFERENCE["qka"]
if query.name != ref["query"]:
    failures.append("qka.query")
camera = rec.cameras[query.camera_id]
keypoints = np.array([p.xy for p in query.points2D], dtype=np.float64)
p2D_idxs = [i for i, p in enumerate(query.points2D) if p.has_point3D()]
p3D_ids = [query.points2D[i].point3D_id for i in p2D_idxs]
points3D = [np.array(rec.points3D[i].xyz) for i in p3D_ids]
gt = query.cam_from_world()
R_gt, t_gt = gt.rotation.matrix(), np.array(gt.translation)
a = math.radians(0.2)
dR = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
               [0, math.sin(a), math.cos(a)]])

localizer = QueryLocalizer(
    rec, {"dense_features": FEATURES,
          "QKA": {"optimizer": {"solver": SINGLE_THREAD}},
          "QBA": {"optimizer": {"solver": SINGLE_THREAD}}},
    dense_features=None, image_dir=images)
unique_idxs = sorted(set(p2D_idxs))
query_fmaps = localizer.extractor(
    images / query.name, keypoints=keypoints[unique_idxs], as_dict=False,
    keypoint_ids=unique_idxs, overwrite_sparse=None)
points2D = keypoints[p2D_idxs]
points2D_init = points2D.copy()
refs = localizer.get_query_references(p3D_ids, query_fmaps, points2D, p2D_idxs)
localizer.query_keypoint_adjuster.refine_multilevel(
    points2D, query_fmaps, refs, point2D_idxs=p2D_idxs)
qka_shift = np.linalg.norm(points2D - points2D_init, axis=1)
check("qka.num_correspondences", len(p2D_idxs), ref["num_correspondences"])
check("qka.mean_shift", float(qka_shift.mean()), ref["mean_shift"], rtol=1e-4)
check("qka.max_shift", float(qka_shift.max()), ref["max_shift"], atol=1e-3)

cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(dR @ R_gt),
                                  t_gt + np.array([0.01, -0.005, 0.0]))
localizer.query_bundle_adjuster.refine_multilevel(
    cam_from_world, camera, points3D, query_fmaps, refs, point2D_idxs=p2D_idxs)
ref = REFERENCE["qba"]
check("qba.rotation_error_deg",
      rotation_angle(cam_from_world.rotation.matrix() @ R_gt.T),
      ref["rotation_error_deg"], atol=1e-3)
check("qba.translation_error",
      float(np.linalg.norm(np.array(cam_from_world.translation) - t_gt)),
      ref["translation_error"], atol=1e-4)

shutil.rmtree(work)
if failures:
    sys.exit(f"Regression against pixsfm 1.0 failed: {', '.join(failures)}")
print("OK: results match pixsfm 1.0")
