"""Command-line interface for hoct-inference."""

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tracksdata as td
import typer
import yaml
from hoct_features.graph import create_graph
from rich.console import Console
from rich.panel import Panel

from hoct_inference import __version__
from hoct_inference._api import predict as predict_from_graph
from hoct_inference._io import load_array
from hoct_inference.data import FrameDataset
from hoct_inference.inference import model_predict
from hoct_inference.tracking import ILPSolverConfig


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


def _load_jit_model(model_path: Path, device: str) -> torch.jit.ScriptModule:
    """Load a JIT-compiled model on ``device`` and put it in eval mode."""
    console.print(f"\nLoading model from: {model_path}")
    model = torch.jit.load(model_path, map_location=device).to(device)
    model.eval()
    console.print(f"Model loaded on device: {device}")
    return model


def _save_graph(graph: td.graph.BaseGraph, output: Path, overwrite: bool) -> None:
    """Write ``graph`` to ``output`` as a GEFF, deleting an existing path if ``overwrite``."""
    if output.exists():
        if not overwrite:
            console.print(f"[red]Output directory {output} already exists. Use --overwrite to overwrite.[/red]")
            raise typer.Exit(code=1)
        shutil.rmtree(output)
    graph.to_geff(str(output))
    console.print("[bold green]✓ Results saved successfully![/bold green]")


app = typer.Typer(
    name="hoct-inference",
    help="Inference CLI for Higher-Order Cell Tracking Transformer (HOCT) model",
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"hoct-inference version: {__version__}")
        raise typer.Exit()


@app.command()
def predict(
    geff_path: Path = typer.Argument(..., help="Path to GEFF directory", exists=True, dir_okay=True, file_okay=False),
    model_path: Path = typer.Argument(..., help="Path to PyTorch model checkpoint", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "-o", help="Output GEFF directory (default: overwrite input)"),
    solution: bool = typer.Option(
        False, "--solution", "-s", help="Save solution graph rather the full graph with probabilities"
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Path to ILP solver config YAML file", exists=True, dir_okay=False
    ),
    overwrite: bool = typer.Option(False, "--overwrite", "-ow", help="Overwrite output directory"),
    window_size: int = typer.Option(5, "--window", "-w", help="Temporal window size for frame dataset"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device to use: 'cuda', 'mps', or 'cpu'"),
):
    """
    Run model prediction and tracking on a GEFF directory.

    This command:
    1. Loads the GEFF graph and model
    2. Runs edge prediction with the model
    3. Solves tracking using ILP solver
    4. Saves the result with solution attributes

    Example:
        hoct-inference predict data.geff model.pt --config solver_config.yaml
    """
    console.print(Panel.fit("HOCT Inference - Model Prediction", style="bold blue"))

    if output.exists() and not overwrite:
        console.print(f"[red]Output directory {output} already exists. Use --overwrite to overwrite.[/red]")
        raise typer.Exit()

    solver_config = _load_solver_config(config_path)
    console.print(f"Solver config: {solver_config.model_dump()}")

    console.print(f"\nLoading GEFF from: {geff_path}")
    graph, _ = td.graph.InMemoryGraph.from_geff(str(geff_path))
    properties = [
        "equivalent_diameter_area",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
        "intensity_std",
        "inertia_tensor",
        "border_dist",
    ]

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
    _save_graph(solution_graph if solution else ds.graph, output, overwrite)


@app.command()
def track(
    image_path: Path = typer.Argument(..., help="Path to image (file or folder of frames)", exists=True),
    segm_path: Path = typer.Argument(..., help="Path to segmentation labels (same format as image)", exists=True),
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to JIT-compiled model checkpoint", exists=True, dir_okay=False
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
):
    """
    Run end-to-end tracking from raw images and segmentation labels.

    Reads the images and segmentation, builds the candidate graph, runs the
    model, solves tracking, and writes a GEFF directory.

    Both inputs accept either a single file (whole time series) or a folder
    of single-frame files (sorted alphabetically). They must use the same
    layout — file with file, or folder with folder.

    Example:
        hoct-inference track images.tif segmentation.tif -m model.pt -o tracks.geff
    """
    console.print(Panel.fit("HOCT Inference - Track from Images", style="bold blue"))

    if output.exists() and not overwrite:
        console.print(f"[red]Output directory {output} already exists. Use --overwrite to overwrite.[/red]")
        raise typer.Exit(code=1)

    if image_path.is_dir() != segm_path.is_dir():
        console.print("[red]Image and segmentation paths must use the same layout (both files or both folders).[/red]")
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

    console.print("\n[bold green]Running prediction and tracking...[/bold green]")
    solution_graph = predict_from_graph(
        model,
        graph=candidate_graph,
        solver_config=solver_config,
        window_size=window_size,
        return_solution=True,
    )

    if solution_graph is None:
        console.print("[red]Prediction returned no solution graph.[/red]")
        raise typer.Exit(code=1)

    console.print(f"\nSaving results to: {output}")
    _save_graph(candidate_graph if full_graph else solution_graph, output, overwrite)


@app.command()
def init_config(
    output: Path = typer.Option("solver_config.yaml", "--output", "-o", help="Output YAML path"),
):
    """
    Generate a template ILP solver configuration YAML file.

    Example:
        hoct-inference init-config --output my_config.yaml
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
    """HOCT Inference - Command-line interface for Higher-Order Cell Tracking Transformer model inference."""
    pass


if __name__ == "__main__":
    app()
