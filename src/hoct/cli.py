"""Command-line interface for hoct."""

import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tracksdata as td
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from tracksdata.functional import TilingScheme

from hoct import __version__
from hoct._api import predict as predict_from_graph
from hoct._io import load_array
from hoct._models import DEFAULT_MODEL, resolve_model
from hoct.data import FrameDataset
from hoct.features import REGIONPROPS, create_graph
from hoct.inference import model_predict
from hoct.tracking import ILPSolverConfig


def _fix_inertia_tensor(graph: td.graph.BaseGraph) -> None:
    # temporary workaround to fix inertia tensor dtype
    graph._node_attr_schemas()["inertia_tensor"] = td.utils._dtypes.AttrSchema(
        "inertia_tensor",
        pl.Array(pl.Float32, (3, 3)),
        np.zeros((3, 3), dtype=np.float32),
    )


def _resolve_device(device: str) -> str:
    """Return ``device`` if available, falling back to CPU with a console warning."""
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]Warning: CUDA not available, falling back to CPU[/yellow]")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        console.print("[yellow]Warning: MPS not available, falling back to CPU[/yellow]")
        return "cpu"
    return device


def _load_solver_config(config_path: Path | None) -> ILPSolverConfig:
    """Load an ILP solver config from YAML, or return defaults."""
    if config_path is None:
        console.print("Using default solver configuration")
        return ILPSolverConfig.default()

    console.print(f"Loading solver config from: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return ILPSolverConfig(**config_dict)


def _load_jit_model(model: Path | str | None, device: str) -> torch.jit.ScriptModule:
    """Resolve a model (path, registered name, or default) and load it in eval mode."""
    if model is None:
        console.print(f"No model given; using default pre-trained model: {DEFAULT_MODEL}")
    model_path = resolve_model(model)
    console.print(f"\nLoading model from: {model_path}")
    module = torch.jit.load(model_path, map_location=device).to(device)
    module.eval()
    console.print(f"Model loaded on device: {device}")
    return module


class OutputFormat(str, Enum):
    """Output formats supported by the CLI."""

    GEFF = "geff"
    CTC = "ctc"


class TileMode(str, Enum):
    """Tiling decision modes for ``track``."""

    AUTO = "auto"
    ON = "on"
    OFF = "off"


# Default spatio-temporal tile when tiling is enabled, applied along (t, z, y, x).
# ``create_graph`` always produces a 4D graph (Z=1 for 2D+t inputs), so a
# 4-axis tile works in both regimes.
_DEFAULT_TILE_SHAPE: tuple[int, int, int, int] = (1, 64, 256, 256)
_DEFAULT_OVERLAP_SHAPE: tuple[int, int, int, int] = (2, 24, 64, 64)
# When ``--tile auto``, enable tiling above this candidate-edge density.
_AUTO_TILE_EDGE_DENSITY: float = 2_500.0


def _maybe_build_tiling_scheme(
    candidate_graph: td.graph.BaseGraph,
    mode: TileMode,
) -> TilingScheme | None:
    """Decide whether to tile inference and return the scheme (or None).

    For ``auto`` mode, tiling kicks in when the candidate graph has more than
    ``_AUTO_TILE_EDGE_DENSITY`` edges per timepoint, which empirically
    correlates with running out of GPU memory on a single-tile pass.
    """
    if mode is TileMode.OFF:
        return None

    n_edges = candidate_graph.num_edges()
    n_time = max(len(candidate_graph.time_points()), 1)
    edge_density = n_edges / n_time

    if mode is TileMode.AUTO:
        if edge_density <= _AUTO_TILE_EDGE_DENSITY:
            console.print(f"Auto-tiling: disabled (edges/time = {edge_density:.0f} ≤ {_AUTO_TILE_EDGE_DENSITY:.0f}).")
            return None
        console.print(
            f"Auto-tiling: enabled (edges/time = {edge_density:.0f} > {_AUTO_TILE_EDGE_DENSITY:.0f}); "
            f"using tile_shape={_DEFAULT_TILE_SHAPE}, overlap_shape={_DEFAULT_OVERLAP_SHAPE} along (t, z, y, x)."
        )
    else:  # TileMode.ON
        console.print(
            f"Tiling: forced on; using tile_shape={_DEFAULT_TILE_SHAPE}, "
            f"overlap_shape={_DEFAULT_OVERLAP_SHAPE} along (t, z, y, x)."
        )

    return TilingScheme(tile_shape=_DEFAULT_TILE_SHAPE, overlap_shape=_DEFAULT_OVERLAP_SHAPE)


def _save_graph(
    graph: td.graph.BaseGraph,
    output: Path,
    overwrite: bool,
    output_format: OutputFormat = OutputFormat.GEFF,
    shape: tuple[int, ...] | None = None,
    *,
    is_solution: bool = True,
) -> None:
    """Write ``graph`` to ``output`` in the requested format.

    Parameters
    ----------
    graph
        Graph to serialize.
    output
        Destination directory. Removed first when ``overwrite`` is True.
    overwrite
        Replace ``output`` if it already exists; otherwise abort.
    output_format
        ``geff`` (default) or ``ctc``. CTC writes a Cell Tracking Challenge
        ground-truth folder (label TIFFs + ``man_track.txt``) using the
        ``tracklet_id`` node attribute.
    shape
        Volume shape ``(T, [Z,] Y, X)`` used by the CTC writer to rasterize
        masks. Ignored for GEFF; required for CTC unless the graph metadata
        already carries a ``shape`` entry.
    is_solution
        Whether ``graph`` is a solved tracking graph (one parent per node).
        Tracklet ids are only assigned for solution graphs; the full candidate
        graph is not a lineage and would fail ``assign_tracklet_ids``.
    """
    if output.exists():
        if not overwrite:
            console.print(f"[red]Output directory {output} already exists. Use --overwrite to overwrite.[/red]")
            raise typer.Exit(code=1)
        shutil.rmtree(output)

    if is_solution:
        graph.assign_tracklet_ids()

    if output_format is OutputFormat.GEFF:
        graph.to_geff(str(output))
    elif output_format is OutputFormat.CTC:
        graph.to_ctc(output_dir=output, shape=shape, overwrite=True)
    else:  # pragma: no cover - guarded by Enum
        raise ValueError(f"Unknown output format: {output_format}")

    console.print("[bold green]✓ Results saved successfully![/bold green]")


app = typer.Typer(
    name="hoct",
    help="Inference CLI for Higher-Order Cell Tracking Transformer (HOCT) model",
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"hoct version: {__version__}")
        raise typer.Exit()


@app.command()
def predict(
    geff_path: Path = typer.Argument(..., help="Path to GEFF directory", exists=True, dir_okay=True, file_okay=False),
    model_path: Path | None = typer.Argument(
        None,
        help=f"Checkpoint path or registered model name. Default: '{DEFAULT_MODEL}' (auto-downloaded).",
        dir_okay=False,
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    solution: bool = typer.Option(
        False, "--solution", "-s", help="Save the solution graph rather than the full graph with probabilities"
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to ILP solver config YAML file", exists=True, dir_okay=False
    ),
    overwrite: bool = typer.Option(False, "--overwrite", "-ow", help="Overwrite output directory"),
    window_size: int = typer.Option(5, "--window", "-w", help="Temporal window size for frame dataset"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device to use: 'cuda', 'mps', or 'cpu'"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.GEFF,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format: 'geff' (default) or 'ctc' (Cell Tracking Challenge folder).",
    ),
):
    """
    Run model prediction and tracking on a GEFF directory.

    This command:
    1. Loads the GEFF graph and model
    2. Runs edge prediction with the model
    3. Solves tracking using ILP solver
    4. Saves the result with solution attributes

    Example:
        hoct predict data.geff model.pt --config solver_config.yaml
    """
    console.print(Panel.fit("HOCT - Model Prediction", style="bold blue"))

    if output.exists() and not overwrite:
        console.print(f"[red]Output directory {output} already exists. Use --overwrite to overwrite.[/red]")
        raise typer.Exit(code=1)

    solver_config = _load_solver_config(config_path)
    console.print(f"Solver config: {solver_config.model_dump()}")

    console.print(f"\nLoading GEFF from: {geff_path}")
    graph, _ = td.graph.InMemoryGraph.from_geff(str(geff_path))
    # Same feature set (and order) used to build the candidate graph, so the
    # model receives node features consistent with training.
    properties = list(REGIONPROPS)

    _fix_inertia_tensor(graph)

    # Load dataset
    ds = FrameDataset(
        graph=graph,
        properties=properties,
        min_window_size=window_size,
    )
    console.print(f"Dataset: {ds.graph.num_nodes()} nodes, {ds.graph.num_edges()} edges")

    time_points = ds.graph.time_points()
    console.print(f"Time frames: {min(time_points)} - {max(time_points)}")

    device = _resolve_device(device)
    model = _load_jit_model(model_path, device)

    # Run prediction
    console.print("\n[bold green]Running prediction and tracking...[/bold green]")
    solution_graph = model_predict(model, ds, solver_config=solver_config)

    console.print(f"\nSaving results to: {output}")
    if output_format is OutputFormat.CTC and not solution:
        console.print("CTC export uses the solution graph (overriding --solution).")
    is_solution = solution or output_format is OutputFormat.CTC
    graph_to_save = solution_graph if is_solution else ds.graph
    shape = ds.graph.metadata.get("shape")
    _save_graph(graph_to_save, output, overwrite, output_format=output_format, shape=shape, is_solution=is_solution)


@app.command()
def track(
    image_path: Path = typer.Argument(..., help="Path to image (file or folder of frames)", exists=True),
    segm_path: Path = typer.Argument(..., help="Path to segmentation labels (same format as image)", exists=True),
    model_path: Path | None = typer.Option(
        None,
        "--model",
        "-m",
        help=f"Checkpoint path or registered model name. Default: '{DEFAULT_MODEL}' (auto-downloaded).",
        dir_okay=False,
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output GEFF directory"),
    overwrite: bool = typer.Option(False, "--overwrite", "-ow", help="Overwrite output directory"),
    full_graph: bool = typer.Option(
        False,
        "--full-graph",
        help="Save the full candidate graph with predicted attributes instead of just the solution",
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to ILP solver config YAML file", exists=True, dir_okay=False
    ),
    device: str = typer.Option("cuda", "--device", "-d", help="Device to use: 'cuda', 'mps', or 'cpu'"),
    window_size: int = typer.Option(5, "--window", "-w", help="Temporal window size for frame dataset"),
    distance_threshold: float = typer.Option(
        300.0, "--max-distance", help="Maximum spatial distance for candidate edges"
    ),
    n_neighbors: int = typer.Option(5, "--neighbors", help="Maximum number of neighbors per node"),
    max_delta_t: int = typer.Option(3, "--max-dt", help="Maximum temporal gap for candidate edges"),
    scale: list[float] | None = typer.Option(
        None,
        "--scale",
        help="Physical voxel size as 't y x' (2D+t) or 't z y x' (3D+t). Repeat the flag for each value.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.GEFF,
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format: 'geff' (default) or 'ctc' (Cell Tracking Challenge folder).",
    ),
    tile: TileMode = typer.Option(
        TileMode.AUTO,
        "--tile",
        case_sensitive=False,
        help=(
            "Tiled inference: 'auto' (default; on if edges/time > 2500), 'on' (force on), "
            "or 'off' (force off). Tile shape (t, z, y, x) = (1, 64, 256, 256), overlap (2, 24, 64, 64)."
        ),
    ),
):
    """
    Run end-to-end tracking from raw images and segmentation labels.

    Reads the images and segmentation, builds the candidate graph, runs the
    model, solves tracking, and writes a GEFF directory.

    Both inputs accept either a single file (whole time series) or a folder
    of single-frame files (sorted alphabetically). They must use the same
    layout — file with file, or folder with folder.

    Example:
        hoct track images.tif segmentation.tif -m model.pt -o tracks.geff
    """
    console.print(Panel.fit("HOCT - Track from Images", style="bold blue"))

    if output.exists() and not overwrite:
        console.print(f"[red]Output directory {output} already exists. Use --overwrite to overwrite.[/red]")
        raise typer.Exit(code=1)

    if image_path.is_dir() != segm_path.is_dir():
        console.print("[red]Image and segmentation paths must use the same layout (both files or both folders).[/red]")
        raise typer.Exit(code=1)

    if output_format is OutputFormat.CTC and full_graph:
        console.print("[red]CTC export is only valid for the solution graph; drop --full-graph.[/red]")
        raise typer.Exit(code=1)

    solver_config = _load_solver_config(config_path)
    console.print(f"Solver config: {solver_config.model_dump()}")

    console.print(f"\nLoading images from: {image_path}")
    images = load_array(image_path)
    console.print(f"Images shape: {images.shape}, dtype: {images.dtype}")

    console.print(f"\nLoading segmentation from: {segm_path}")
    labels = load_array(segm_path)
    console.print(f"Segmentation shape: {labels.shape}, dtype: {labels.dtype}")

    if images.shape != labels.shape:
        console.print(f"[red]Image shape {images.shape} does not match segmentation shape {labels.shape}.[/red]")
        raise typer.Exit(code=1)

    device = _resolve_device(device)
    model = _load_jit_model(model_path, device)

    console.print("\nBuilding candidate tracking graph...")
    candidate_graph = create_graph(
        labels=labels,
        images=images,
        distance_threshold=distance_threshold,
        n_neighbors=n_neighbors,
        delta_t=max_delta_t,
        scale=tuple(scale) if scale else None,
    )
    console.print(f"Candidate graph: {candidate_graph.num_nodes()} nodes, {candidate_graph.num_edges()} edges")

    tiling_scheme = _maybe_build_tiling_scheme(candidate_graph, tile)

    console.print("\n[bold green]Running prediction and tracking...[/bold green]")
    solution_graph = predict_from_graph(
        model,
        graph=candidate_graph,
        solver_config=solver_config,
        window_size=window_size,
        tiling_scheme=tiling_scheme,
        return_solution=True,
    )

    if solution_graph is None:
        console.print("[red]Prediction returned no solution graph.[/red]")
        raise typer.Exit(code=1)

    console.print(f"\nSaving results to: {output}")
    graph_to_save = candidate_graph if full_graph else solution_graph
    # ``create_graph`` records a 4D (T, Z, Y, X) shape in metadata even for 2D+t
    # inputs, so we prefer that over ``labels.shape`` for the CTC writer.
    shape = candidate_graph.metadata.get("shape", labels.shape)
    _save_graph(graph_to_save, output, overwrite, output_format=output_format, shape=shape, is_solution=not full_graph)


@app.command()
def init_config(
    output: Path = typer.Option("solver_config.yaml", "--output", "-o", help="Output YAML path"),
):
    """
    Generate a template ILP solver configuration YAML file.

    Example:
        hoct init-config --output my_config.yaml
    """
    console.print(Panel.fit("Generating ILP Solver Config Template", style="bold blue"))

    # Create default config and export to dict
    default_config = ILPSolverConfig.default()
    config_dict = default_config.model_dump()

    # Write YAML with comments
    with open(output, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"[bold green]✓ Config template saved to: {output}[/bold green]")
    console.print("\nEdit this file to customize solver parameters.")


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version and exit"
    ),
):
    """HOCT - Command-line interface for Higher-Order Cell Tracking Transformer model inference."""
    pass


if __name__ == "__main__":
    app()
