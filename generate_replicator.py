#!/usr/bin/env python3
"""Small 3x3 tray SDG example; run rendering with Isaac Sim's python.sh."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def read_intrinsics(path, width, height):
    import cv2

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise ValueError(f"cannot read calibration: {path}")
        source_width = int(storage.getNode("image_width").real())
        source_height = int(storage.getNode("image_height").real())
        matrix = storage.getNode("camera_matrix").mat()
        distortion = storage.getNode("distortion_coefficients").mat()
    finally:
        storage.release()
    if (source_width <= 0 or source_height <= 0 or matrix is None or matrix.shape != (3, 3)
            or distortion is None or distortion.size != 5
            or not np.isfinite(matrix).all() or not np.isfinite(distortion).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1]) or matrix[0, 1] != 0 or matrix[1, 0] != 0
            or not (0 <= matrix[0, 2] < source_width and 0 <= matrix[1, 2] < source_height)):
        raise ValueError("expected finite OpenCV K and five distortion coefficients")
    if height is None:
        height = round(width * source_height / source_width)
    matrix[0] *= width / source_width
    matrix[1] *= height / source_height
    return matrix, distortion.ravel(), height


def project_points(points, view, projection, width, height, intrinsics=None):
    """USD matrices use row vectors; image coordinates refer to pixel centers."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("expected finite Nx3 world points")
    camera_points = np.column_stack((points, np.ones(len(points)))) @ view
    if intrinsics is not None:
        import cv2

        camera_points = camera_points[:, :3] * (1, -1, -1)
        if not np.isfinite(camera_points).all() or (camera_points[:, 2] < .01).any() or (camera_points[:, 2] > 10).any():
            raise ValueError("point outside camera depth range")
        pixels, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), *intrinsics)
        pixels = pixels.reshape(-1, 2)
        if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels >= (width, height)).any():
            raise ValueError("point outside distorted image; reduce camera variation")
        return pixels
    clip = camera_points @ projection
    if not np.isfinite(clip).all() or (clip[:, 3] <= 0).any():
        raise ValueError("point behind camera or invalid projection")
    ndc = clip[:, :3] / clip[:, 3:]
    if (np.abs(ndc) > 1).any():
        raise ValueError("point outside camera frustum; reduce camera variation")
    return np.column_stack(((ndc[:, 0] + 1) * width / 2 - .5,
                            (1 - ndc[:, 1]) * height / 2 - .5))


def frame_labels(name, slots, view, projection, width, height, intrinsics=None):
    records, shapes = [], []
    for slot_id, (x, y, center, occupied) in enumerate(slots):
        # shortcut: geometric cell ROIs, use production detector boxes for later realism checks.
        corners = [(x + dx, y + dy, z) for dx in (-.046, .046)
                   for dy in (-.046, .046) for z in (0, .035)]
        pixels = project_points([center, *corners], view, projection, width, height, intrinsics)
        cx, cy = pixels[0].tolist()
        low = np.maximum(pixels[1:].min(axis=0), (0, 0))
        high = np.minimum(pixels[1:].max(axis=0), (width, height))
        if not (low[0] <= cx < high[0] and low[1] <= cy < high[1]):
            raise ValueError("center outside cell ROI")
        occupancy = "occupied" if occupied else "empty"
        records.append(dict(image=f"labelme/{name}", source_image=name,
                            split_group="replicator-simplified", split="train",
                            label_state="localizable", occupancy=occupancy,
                            center_x=cx, center_y=cy, bbox=[*low.tolist(), *high.tolist()],
                            slot_id=slot_id, synthetic=True, human_reviewed=False,
                            label_source="projected_geometry"))
        shapes.extend([
            dict(label="bbox", shape_type="rectangle", group_id=slot_id,
                 points=[low.tolist(), high.tolist()], flags={}),
            dict(label=occupancy, shape_type="circle", group_id=slot_id,
                 points=[[cx, cy], [cx + (50 if cx + 50 < width else -50), cy]], flags={}),
        ])
    annotation = dict(version="5.5.0", imagePath=name, imageData=None,
                      imageWidth=width, imageHeight=height, shapes=shapes,
                      flags=dict(synthetic=True, auto_generated=True, needs_review=True))
    return records, annotation


def rebase_records(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or not any(r.get("split") == "val" for r in records):
        raise ValueError("baseline needs a real validation split")
    for record in records:
        image = (path.parent / record["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        if record.get("split_group") == "replicator-simplified" or record.get("synthetic"):
            raise ValueError("use an original real-data baseline, not a previous synthetic merge")
        record["image"] = str(image)
    return records


def generate(args):
    baseline = rebase_records(args.baseline) if args.baseline else []
    intrinsics = None
    if args.intrinsics:
        matrix, distortion, args.height = read_intrinsics(args.intrinsics, args.width, args.height)
        intrinsics = (matrix, distortion)
    else:
        args.height = args.height or args.width
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "PathTracing",
                         "width": args.width, "height": args.height, "multi_gpu": False},
                        experience=str(Path(__file__).with_name("replicator.kit")))
    try:
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        args.output.mkdir(parents=True, exist_ok=False)
        if args.intrinsics:
            shutil.copyfile(args.intrinsics, args.output / "camera_intrinsics.yaml")
        images = args.output / "labelme"
        images.mkdir()
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        rep.orchestrator.set_capture_on_play(False)
        rep.settings.set_render_pathtraced(samples_per_pixel=64)
        rng = np.random.default_rng(args.seed)

        def material(name, color, metallic, roughness):
            mat = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
            shader = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat, shader

        tray_mat, tray_shader = material("Tray", (.35, .35, .35), .5, .4)
        dark_mat, dark_shader = material("Hole", (.02, .02, .02), 0., .8)
        part_mat, part_shader = material("Part", (.65, .65, .65), .8, .25)

        def bind(geom, mat):
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)

        base = UsdGeom.Cube.Define(stage, "/World/Tray")
        base.CreateSizeAttr(1.)
        base.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.008))
        base.AddScaleOp().Set(Gf.Vec3f(.34, .34, .016))
        bind(base, tray_mat)
        parts = []
        for slot_id in range(9):
            x, y = (slot_id % 3 - 1) * .1, (slot_id // 3 - 1) * .1
            # shortcut: dark discs stand in for cavities; replace with CAD before claiming realism.
            disc = UsdGeom.Cylinder.Define(stage, f"/World/Hole{slot_id}")
            disc.CreateAxisAttr("Z")
            disc.CreateRadiusAttr(.038)
            disc.CreateHeightAttr(.002)
            disc.AddTranslateOp().Set(Gf.Vec3d(x, y, .001))
            bind(disc, dark_mat)
            part = UsdGeom.Cylinder.Define(stage, f"/World/Part{slot_id}")
            part.CreateAxisAttr("Z")
            part.CreateRadiusAttr(.027)
            part.CreateHeightAttr(.022)
            position = part.AddTranslateOp()
            bind(part, part_mat)
            parts.append((x, y, part, position))

        dome = UsdLux.DomeLight.Define(stage, "/World/Ambient")
        dome.CreateIntensityAttr(300.)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(2000.)
        sun.CreateAngleAttr(15.)
        sun_rotation = sun.AddRotateXYZOp()
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(35.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(36. * args.height / args.width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
        if intrinsics is not None:
            prim = camera.GetPrim()
            if not prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI"):
                raise RuntimeError("OpenCV lens schema unavailable; use a compatible Isaac Sim release")
            prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
            prefix = "omni:lensdistortion:opencvPinhole:"
            prim.CreateAttribute(prefix + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(args.width, args.height))
            values = dict(fx=matrix[0, 0], fy=matrix[1, 1], cx=matrix[0, 2], cy=matrix[1, 2],
                          **dict(zip(("k1", "k2", "p1", "p2", "k3"), distortion)))
            for key, value in values.items():
                prim.CreateAttribute(prefix + key, Sdf.ValueTypeNames.Float).Set(float(value))
        camera_pose = camera.AddTransformOp()
        product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
        rgb = rep.annotators.get("rgb")
        rgb.attach(product)
        all_records = []
        frame_info = []
        for frame in range(args.frames):
            slots = []
            for x, y, part, position in parts:
                occupied = bool(rng.random() < .5)
                dx, dy = rng.uniform(-.003, .003, 2)
                position.Set(Gf.Vec3d(x + dx, y + dy, .013))
                part.CreateVisibilityAttr().Set("inherited" if occupied else "invisible")
                center = (x + dx, y + dy, .024) if occupied else (x, y, .002)
                slots.append((x, y, center, occupied))
            eye = Gf.Vec3d(float(rng.uniform(-.12, .12)), float(rng.uniform(-.12, .12)),
                          float(rng.uniform(.65, .85)))
            view = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(0, 0, 0), Gf.Vec3d(0, 1, 0))
            camera_pose.Set(view.GetInverse())
            sun_rotation.Set(Gf.Vec3f(float(rng.uniform(-40, 40)), float(rng.uniform(-40, 40)), 0))
            sun.GetIntensityAttr().Set(float(rng.uniform(800, 3500)))
            dome.GetIntensityAttr().Set(float(rng.uniform(150, 600)))
            for shader in (tray_shader, part_shader):
                shade = float(rng.uniform(.2, .75))
                shader.GetInput("diffuseColor").Set(Gf.Vec3f(shade, shade, shade))
                shader.GetInput("roughness").Set(float(rng.uniform(.15, .7)))
            dark = float(rng.uniform(.01, .06))
            dark_shader.GetInput("diffuseColor").Set(Gf.Vec3f(dark, dark, dark))
            rep.orchestrator.step(rt_subframes=8, delta_time=0.0)
            frustum = camera.GetCamera().frustum
            view_matrix = np.array(frustum.ComputeViewMatrix())
            projection = np.array(frustum.ComputeProjectionMatrix())
            name = f"frame_{frame:05d}.png"
            records, annotation = frame_labels(name, slots, view_matrix, projection, args.width, args.height, intrinsics)
            pixels = np.asarray(rgb.get_data())
            if pixels.shape != (args.height, args.width, 4) or pixels.dtype != np.uint8:
                raise ValueError(f"unexpected RGB data: {pixels.shape}, {pixels.dtype}")
            Image.fromarray(pixels[:, :, :3]).save(images / name)
            (images / Path(name).with_suffix(".json")).write_text(json.dumps(annotation, indent=2) + "\n")
            all_records.extend(records)
            frame_info.append(dict(image=name, view=view_matrix.tolist(), projection=projection.tolist(),
                                   slots=slots))
            print(f"frame={frame + 1}/{args.frames}", flush=True)
        rep.orchestrator.wait_until_complete()
        rgb.detach()
        product.destroy()
        text = "".join(json.dumps(r) + "\n" for r in all_records)
        (args.output / "synthetic.jsonl").write_text(text, encoding="utf-8")
        if baseline:
            old = "".join(json.dumps(r) + "\n" for r in baseline)
            (args.output / "labels.jsonl").write_text(old + text, encoding="utf-8")
        summary = dict(seed=args.seed, frames=args.frames, samples=len(all_records),
                       occupancy=dict(Counter(r["occupancy"] for r in all_records)),
                       baseline=str(args.baseline.resolve()) if args.baseline else None,
                       baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest() if baseline else None,
                       real_validation_samples=sum(r.get("split") == "val" for r in baseline),
                       simplified_geometry=True, calibrated_intrinsics=intrinsics is not None,
                       calibrated_extrinsics=False,
                       intrinsics_sha256=hashlib.sha256(args.intrinsics.read_bytes()).hexdigest() if args.intrinsics else None,
                       frames_metadata=frame_info)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {len(all_records)} training samples to {args.output.resolve()}")
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, help="real JSONL; preserves val and writes merged labels.jsonl")
    parser.add_argument("--intrinsics", type=Path, help="OpenCV YAML K/D; also models lens distortion")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, help="default: calibration aspect ratio, otherwise same as width")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output is None or args.frames <= 0 or args.width < 128 or (args.height is not None and args.height < 128):
        parser.error("use --output, positive --frames, and dimensions >= 128")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    generate(args)


if __name__ == "__main__":
    main()
