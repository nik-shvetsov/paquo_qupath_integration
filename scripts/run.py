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
from model_processor import CellPoseProcessor
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


def infer_cell_detections(project, model, patch_size, batch_size):
    with QuPathProject(project, mode="a") as qp:
        qp_annotations = {}
        for image in qp.images:
            qp_annotations[image.image_name] = {}
            qp_annotations[image.image_name]["vips_slide"] = pyvips.Image.new_from_file(
                Path(image.uri.replace("file:", "")), level=0
            ).extract_band(0, n=3)
            qp_annotations[image.image_name]["global_polygons"] = []
            with image.hierarchy.no_autoflush():
                for image_global_annotation in image.hierarchy.annotations:
                    print(
                        f"Processing {image.image_name} - Polygon area [{image_global_annotation.roi.area}]"
                    )

                    # global_annotation_rectangle = image_global_annotation.roi.minimum_rotated_rectangle
                    global_annotation_rectangle = image_global_annotation.roi.envelope
                    # image.hierarchy.add_annotation(roi=global_annotation_rectangle)

                    global_annotation_coords = (
                        image_global_annotation.roi.exterior.coords.xy
                    )
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

                    # Prepare the dataset for inference
                    dataset = VipsImagePatchedDataset(vips_region, patch_size)
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

                    model_infer_results = {}
                    for batch_idx, (patch_ids, batch_patches) in enumerate(dataloader):
                        out = model(batch_patches)
                        for i, patch_id in enumerate(patch_ids):
                            # Stacking the inst and type maps along a new axis to form a 2x224x224 numpy array
                            model_infer_results[int(patch_id)] = np.stack(
                                (out["inst"][i], out["type"][i]), axis=0
                            )

                    for patch_id, rmap in model_infer_results.items():
                        patch_top_left_xy = (
                            dataset.dframe.filter(pl.col("patch-id") == patch_id)[
                                "patch-top-left-coord"
                            ]
                        )[0]
                        patch_top_left_x = patch_top_left_xy[0]
                        patch_top_left_y = patch_top_left_xy[1]

                        segmentation_map = rmap[0]
                        cell_types_map = rmap[1]

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

                            # Adjust polygons to patch coordinates
                            polygons = [
                                Polygon(
                                    [
                                        (x + patch_top_left_x, y + patch_top_left_y)
                                        for x, y in poly.exterior.coords
                                    ]
                                )
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
        default=8,
        metavar="B",
        help="Batch size for inference",
    )
    parser.add_argument(
        "--sampling-patch-size",
        type=int,
        default=384,
        metavar="S",
        help="Patch size for inference",
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
        project, model, args.sampling_patch_size, args.batch_size
    )
    add_cell_detections(project, qp_annotations)
    calculate_cell_counts(qp_annotations, class_ids)
