# HOCT

HOCT reconstructs cell trajectories and divisions from time-lapse microscopy
images and instance segmentation masks. It scores candidate links with a
Higher-Order Cell Tracking Transformer and selects a consistent lineage with
an integer linear programming (ILP) solver.

This package provides pretrained-model inference, graph construction, tracking,
and incremental correction. It supports 2D and 3D time series and exports
[GEFF](https://github.com/live-image-tracking-tools/geff) graphs or
[Cell Tracking Challenge (CTC)](https://celltrackingchallenge.net/) results.

## Start here

- [Install HOCT](installation.md) and choose the dependencies for your workflow.
- [Track cells](tracking.md) from images and masks using the CLI or Python.
- [Correct links and fine-tune](correction.md) in the Napari demo, or connect your own UI.
- [Look up the Python API](api.md) for graph, prediction, and correction parameters.
- [Develop and build documentation](development.md) from a checkout.

## What you need

Supply an instance mask for every image frame: each cell has a distinct positive
integer label within a frame, and background is zero. Labels do not need to be
consistent across time; discovering those links is the tracking task. HOCT
expects spatially aligned images and masks with matching shapes.

Pretrained weights download on first use. CUDA speeds up inference, while CPU
also works. Tracking requires an available ILP backend; see
[installation](installation.md#tracking-solver).

For the model and correction method, see the
[HOCT paper](https://arxiv.org/abs/2607.11754). The source code and examples are
on [GitHub](https://github.com/royerlab/hoct).
