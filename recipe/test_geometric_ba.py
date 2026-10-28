"""Synthetic check of pixsfm's geometric BA against COLMAP 4 reconstructions."""
import copy
import sys

import numpy as np
import pycolmap

from pixsfm.bundle_adjustment import GeometricBundleAdjuster
from pixsfm._pixsfm import _bundle_adjustment as ba


def synthesize(num_cameras_per_rig=1, num_frames=8, seed=0):
    pycolmap.set_random_seed(seed)
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = 1
    opts.num_cameras_per_rig = num_cameras_per_rig
    opts.num_frames_per_rig = num_frames
    opts.num_points3D = 200
    return pycolmap.synthesize_dataset(opts)


def mean_reproj_error(rec):
    errs = []
    for image in rec.images.values():
        if not image.has_pose:
            continue
        cam = rec.cameras[image.camera_id]
        cam_from_world = image.cam_from_world()
        for p2D in image.points2D:
            if not p2D.has_point3D():
                continue
            xc = cam_from_world * rec.points3D[p2D.point3D_id].xyz
            errs.append(np.linalg.norm(cam.img_from_cam(xc) - p2D.xy))
    return float(np.mean(errs))


def perturb(rec, rng, skip_frame_ids):
    for p3D in rec.points3D.values():
        p3D.xyz = p3D.xyz + rng.normal(0, 0.05, 3)
    for frame_id, frame in rec.frames.items():
        if frame_id in skip_frame_ids:
            continue
        T = frame.rig_from_world
        T.translation = T.translation + rng.normal(0, 0.02, 3)
        frame.rig_from_world = T


def main():
    rng = np.random.default_rng(1)
    gt = synthesize()
    assert all(im.is_ref_in_frame() for im in gt.images.values())
    print("gt error", mean_reproj_error(gt))

    rec = copy.deepcopy(gt)
    reg = sorted(rec.reg_image_ids())
    const_img, tvec_img = reg[0], reg[1]
    perturb(rec, rng, skip_frame_ids={rec.images[const_img].frame_id})
    before = mean_reproj_error(rec)
    pose_const = rec.images[const_img].cam_from_world().params.copy()
    t0_before = rec.images[tvec_img].cam_from_world().translation[0]

    setup = ba.BundleAdjustmentSetup()
    setup.add_images(set(reg))
    setup.set_constant_pose(const_img)
    setup.set_constant_tvec(tvec_img, [0])
    adjuster = GeometricBundleAdjuster({"optimizer": {
        "refine_focal_length": False, "refine_extra_params": False,
        "loss": {"name": "trivial", "params": []},
        "solver": {"max_num_iterations": 200}}})
    out = adjuster.refine(rec, problem_setup=setup)
    after = mean_reproj_error(rec)
    print(f"reprojection error: before {before:.4f}px, after {after:.6f}px")
    print(out["summary"].BriefReport())

    assert before > 1.0, before
    assert after < 1e-3, after
    # Constant pose stays untouched, constant tvec index stays untouched.
    np.testing.assert_allclose(
        rec.images[const_img].cam_from_world().params, pose_const, atol=1e-12)
    t0_after = rec.images[tvec_img].cam_from_world().translation[0]
    assert abs(t0_after - t0_before) < 1e-12, (t0_before, t0_after)
    # Quaternions remain normalized.
    for frame in rec.frames.values():
        q = frame.rig_from_world.rotation.quat
        assert abs(np.linalg.norm(q) - 1) < 1e-9

    # Non-trivial rigs must be rejected with a clear error.
    rig_rec = synthesize(num_cameras_per_rig=2, num_frames=4)
    try:
        GeometricBundleAdjuster({}).refine(rig_rec)
    except ValueError as e:
        assert "trivial rigs" in str(e), e
        print("rig rejected:", str(e).splitlines()[0])
    else:
        raise AssertionError("non-trivial rig was not rejected")
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
