"""Basic example of cell tracking with HOCT."""

from pathlib import Path

import napari
import torch
import tracksdata as td
from dask.array.image import imread

from hoct import predict
from hoct.features import normalize_image
from hoct.tracking import ILPSolverConfig


def main():
    # Load data
    images = imread("/hpc/reference/group.royer/CTC/training/Fluo-C3DL-MDA231/02/*.tif")
    labels = imread("/hpc/reference/group.royer/CTC/training/Fluo-C3DL-MDA231/02_ERR_SEG/*.tif")

    assert images.ndim == 4
    assert labels.ndim == 4

    images = images.rechunk((1, *images.shape[1:]))  # (one chunk per time point)
    images = images.map_blocks(normalize_image)

    # Load model
    model_path = Path(__file__).parent.parent / "weights" / "2026_01_30_09_23_41_job_26961657.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path, map_location=device)

    # optionally provide the ilp solver config, it could be None for default config
    solver_config = ILPSolverConfig(
        appearance_weight=0.5,
        delta_t_weight=0.5,
        disappearance_weight=0.25,
        division_weight=0.25,
        edge_bias=0.5,
        node_weight=-10.0,
        tracklet_solver=True,
    )

    # Run tracking
    graph = predict(
        model=model,
        labels=labels,
        images=images,
        solver_config=solver_config,
        distance_threshold=300.0,
        n_neighbors=5,
        max_delta_t=3,
        # this is only required for large volumes where tiled prediction is needed
        tiling_scheme=td.functional.TilingScheme(
            tile_shape=(1, 32, 256, 256),
            overlap_shape=(2, 16, 120, 120),
        ),
        test_time_augs=5,  # optional, takes longer but it improves performance
    )

    # Visualize
    tracks_df, track_graph, track_labels = td.functional.to_napari_format(graph, mask_key="mask")
    viewer = napari.Viewer()
    viewer.add_image(images, name="images")
    viewer.add_labels(track_labels, name="labels")
    viewer.add_tracks(tracks_df, graph=track_graph, name="tracks")
    napari.run()


if __name__ == "__main__":
    main()
