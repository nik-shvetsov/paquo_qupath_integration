import argparse
import warnings
import math
from pathlib import Path

from shapely.geometry import Polygon
from paquo.projects import QuPathProject
from paquo.images import QuPathImageType
from paquo.classes import QuPathPathClass
from paquo.java import LogManager

import torch
import numpy as np
import cv2
import pyvips
import polars as pl

from inference_dataset import VipsImagePatchedDataset
from model_processor import CellPoseProcessor, post_proc_pipeline_single
import config

pl.Config.set_tbl_hide_dataframe_shape(True)
warnings.filterwarnings("ignore")
LogManager.setWarn()


def is_valid_polygon(polygon, annotation):
    return annotation.contains(polygon)


def setup_project_classes(project, class_colors, verbose=False):
    if verbose:
        print(f"Setting up project classes: {project}")
    with QuPathProject(project, mode="a") as qp:
        updated_classes = []
        for class_name, class_color in class_colors:
            updated_classes.append(QuPathPathClass(name=class_name, color=class_color))
        qp.path_classes = updated_classes
        ###
        for image in qp.images:
            image.save()
        qp.save()


def clear_all_detections(project, verbose=False):
    if verbose:
        print(f"Clearing all detections: {project}")
    with QuPathProject(project, mode="a") as qp:
        for image in qp.images:
            detections = image.hierarchy.detections
            detections.clear()


def add_cell_detections(project, qp_annotations):
    with QuPathProject(project, mode="a") as qp:
        cell_id = 0
        for image in qp.images:
            for key in qp_annotations[image.image_name]["global_polygons"]:
                for cell_type, polygons in key.items():
                    for polygon in polygons:
                        cell_id += 1
                        # cell = image.hierarchy.add_annotation(roi=polygon)
                        cell = image.hierarchy.add_cell(
                            roi=polygon, nucleus_roi=polygon
                        )
                        # cell.name = f"{cell_id}"
                        cell.update_path_class(qp.path_classes[cell_type - 1])


def infer_cell_detections(project, model, patch_size, batch_size, autocrop=True):
    with QuPathProject(project, mode="a") as qp:
        qp_annotations = {}
        for image in qp.images:
            qp_annotations[image.image_name] = {}
            if image.image_name.lower().endswith(".svs"):
                qp_annotations[image.image_name]["vips_slide"] = (
                    pyvips.Image.new_from_file(
                        Path(image.uri.replace("file:", "")), level=0, autocrop=autocrop
                    ).extract_band(0, n=3)
                )
            else:
                # print(f"Loading non-SVS image: {image.image_name}")
                qp_annotations[image.image_name]["vips_slide"] = (
                    pyvips.Image.new_from_file(
                        Path(image.uri.replace("file:", "")), level=0, autocrop=autocrop
                    ).extract_band(0, n=3)
                )

            #### print size of the image
            print(
                f"Processing image: {image.image_name}, size: {qp_annotations[image.image_name]['vips_slide'].width}x{qp_annotations[image.image_name]['vips_slide'].height}"
            )

            qp_annotations[image.image_name]["global_polygons"] = []
            with image.hierarchy.no_autoflush():
                if len(image.hierarchy.annotations) == 0:
                    print(
                        f"No annotations found for image: {image.image_name}, skipping..."
                    )
                    continue

                unprocessed_annotations = [
                    ann for ann in image.hierarchy.annotations if ann.roi is not None
                ]
                processed_polygons = []

                ### combine annotations if they overlap
                for ann in unprocessed_annotations:
                    ann_polygon = ann.roi
                    merged = False
                    for idx, proc_polygon in enumerate(processed_polygons):
                        if ann_polygon.intersects(proc_polygon):
                            # Merge the polygons
                            processed_polygons[idx] = proc_polygon.union(ann_polygon)
                            merged = True
                            break
                    if not merged:
                        processed_polygons.append(ann_polygon)

                for ann_polygon in processed_polygons:
                    print(
                        f"Processing {image.image_name} - Polygon area [{ann_polygon.area}]"
                    )

                    # global_annotation_rectangle = ann_polygon.roi.minimum_rotated_rectangle
                    global_annotation_rectangle = ann_polygon.envelope
                    # image.hierarchy.add_annotation(roi=global_annotation_rectangle)

                    global_annotation_coords = ann_polygon.exterior.coords.xy
                    global_annotation_rectangle_coords = (
                        global_annotation_rectangle.exterior.coords.xy
                    )
                    zipped_global_annotation_coords = list(
                        zip(global_annotation_coords[0], global_annotation_coords[1])
                    )
                    zipped_global_annotation_rectangle_coords = list(
                        zip(
                            global_annotation_rectangle_coords[0],
                            global_annotation_rectangle_coords[1],
                        )
                    )
                    # get top left coord from the list, it is not ordered
                    global_top_left = min(zipped_global_annotation_rectangle_coords)
                    zipped_local_annotation_coords = [
                        (
                            z_g_a_c[0] - global_top_left[0],
                            z_g_a_c[1] - global_top_left[1],
                        )
                        for z_g_a_c in zipped_global_annotation_coords
                    ]
                    annotation_shapely_polygon = Polygon(zipped_local_annotation_coords)

                    # get width and height
                    width = math.sqrt(
                        (
                            zipped_global_annotation_rectangle_coords[1][0]
                            - zipped_global_annotation_rectangle_coords[0][0]
                        )
                        ** 2
                        + (
                            zipped_global_annotation_rectangle_coords[1][1]
                            - zipped_global_annotation_rectangle_coords[0][1]
                        )
                        ** 2
                    )
                    height = math.sqrt(
                        (
                            zipped_global_annotation_rectangle_coords[3][0]
                            - zipped_global_annotation_rectangle_coords[0][0]
                        )
                        ** 2
                        + (
                            zipped_global_annotation_rectangle_coords[3][1]
                            - zipped_global_annotation_rectangle_coords[0][1]
                        )
                        ** 2
                    )

                    # Extract the region
                    x, y = int(global_top_left[0]), int(global_top_left[1])
                    width, height = int(width), int(height)

                    ### IF NON-OVERLAPPING
                    ### extend width and height to the nearest multiple of 224
                    width = (
                        width + (patch_size - width % patch_size)
                        if width % patch_size != 0
                        else width
                    )
                    height = (
                        height + (patch_size - height % patch_size)
                        if height % patch_size != 0
                        else height
                    )

                    vips_region = qp_annotations[image.image_name][
                        "vips_slide"
                    ].extract_area(x, y, width, height)

                    # dataset = VipsImagePatchedDataset(vips_region, patch_size, patch_size, 0)
                    dataset = VipsImagePatchedDataset(
                        vips_region,
                        patch_size,
                        patch_size - (patch_size // 2),
                        patch_size // 6,
                    )
                    dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, shuffle=False, num_workers=0
                    )

                    polygon_types = {
                        1: [],  # neoplastic
                        2: [],  # lymphocyte
                        3: [],  # connective
                        4: [],  # dead
                        5: [],  # epithelial
                        6: [],  # macrophage
                        7: [],  # neutrophil
                    }

                    maps_list = []
                    for batch_idx, (patch_ids, batch_patches) in enumerate(dataloader):
                        batch_maps = model(
                            batch_patches
                        )  # dict with 'cellpose' and 'type' as tensors
                        maps_list.append(batch_maps)

                    result_map = dataset.reconstruct_maps(maps_list)
                    del maps_list
                    out = post_proc_pipeline_single(result_map)
                    model_infer_results = np.stack((out["inst"], out["type"]), axis=0)

                    segmentation_map = model_infer_results[0]
                    cell_types_map = model_infer_results[1]

                    patch_top_left_xy = (
                        dataset.dframe.filter(pl.col("patch-id") == 0)[
                            "patch-top-left-coord"
                        ]
                    )[0]
                    patch_top_left_x = patch_top_left_xy[0]
                    patch_top_left_y = patch_top_left_xy[1]

                    labels = np.unique(segmentation_map)
                    labels = labels[labels != 0]

                    all_polygons = []

                    for label in labels:
                        mask = np.array(segmentation_map == label, dtype=np.uint8)
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        # non_adjusted_polygons = [Polygon(cnt.reshape(-1, 2)) for cnt in contours]
                        non_adjusted_polygons = [
                            Polygon(cnt.reshape(-1, 2))
                            for cnt in contours
                            if cnt.reshape(-1, 2).shape[0] >= 4
                        ]
                        polygons = [
                            Polygon([(x, y) for x, y in poly.exterior.coords])
                            for poly in non_adjusted_polygons
                        ]

                        ### Ensure that it is in a selected region
                        updated_polygons = []
                        for poly in polygons:
                            if is_valid_polygon(poly, annotation_shapely_polygon):
                                updated_polygons.append(poly)
                        polygons = updated_polygons

                        # Get the type of the current cell from cell_types_map (the same position as in mask)
                        cell_type = cell_types_map[mask == 1][0]
                        polygon_types[cell_type].extend(polygons)

                    global_polygon_types = {
                        1: [],  # neoplastic
                        2: [],  # lymphocyte
                        3: [],  # connective
                        4: [],  # dead
                        5: [],  # epithelial
                        6: [],  # macrophage
                        7: [],  # neutrophil
                    }

                    for cell_type, polygons in polygon_types.items():
                        for polygon in polygons:
                            # transform polygon coords to global image coordinates
                            updated_polygon = Polygon(
                                [
                                    (x + global_top_left[0], y + global_top_left[1])
                                    for x, y in polygon.exterior.coords
                                ]
                            )
                            global_polygon_types[cell_type].append(updated_polygon)
                    qp_annotations[image.image_name]["global_polygons"].append(
                        global_polygon_types
                    )
    return qp_annotations


def calculate_cell_counts(qp_annotations, class_ids):
    project_counts = {}
    for image_name, annotations in qp_annotations.items():
        data = {}
        cell_type_counts = {}
        for key in annotations["global_polygons"]:
            for cell_type, polygons in key.items():
                if cell_type not in cell_type_counts:
                    cell_type_counts[cell_type] = 0
                cell_type_counts[cell_type] += len(polygons)

        data = [
            {"cell_type": class_ids[cell_type], "count": count}
            for cell_type, count in cell_type_counts.items()
        ]
        project_counts[image_name] = pl.DataFrame(data)

    for image_name, counts in project_counts.items():
        if counts.shape[0] == 0:
            print(f"Image: {image_name} - No cells detected")
            continue
        print("-" * 50)
        print(f"Image: {image_name}")
        print(counts)
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QuPath Cell Detection and Classification"
    )
    parser.add_argument(
        "--project", type=str, required=True, metavar="P", help="QuPath project path"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="M",
        help="Model path (NB accelerator device)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="B",
        help="Batch size for inference",
    )
    parser.add_argument(
        "--sampling-patch-size",
        type=int,
        default=224,
        metavar="S",
        help="Patch size for inference",
    )

    parser.add_argument(
        "--use_autocrop",
        type=bool,
        default=True,
        metavar="C",
        help="Auto-crop images when loading with pyvips. In QuPath images are auto-cropped by default",
    )

    args = parser.parse_args()

    project = Path(args.project)
    if not project.exists():
        raise FileNotFoundError(f"Project file not found: {project}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = CellPoseProcessor(
        model_path,
        resize_output_to=(args.sampling_patch_size, args.sampling_patch_size),
    )

    class_colors = config.CLASS_COLORS
    class_ids = config.CLASS_IDs

    setup_project_classes(project, class_colors)
    clear_all_detections(project)

    qp_annotations = infer_cell_detections(
        project, model, args.sampling_patch_size, args.batch_size, autocrop=args.use_autocrop
    )
    add_cell_detections(project, qp_annotations)
    calculate_cell_counts(qp_annotations, class_ids)
