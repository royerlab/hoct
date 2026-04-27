"""Command-line interface for hoct-inference."""

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tracksdata as td
import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from hoct_inference import __version__
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

    # Load ILP solver config
    if config_path:
        console.print(f"Loading solver config from: {config_path}")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        solver_config = ILPSolverConfig(**config_dict)
    else:
        console.print("Using default solver configuration")
        solver_config = ILPSolverConfig.default()

    console.print(f"Solver config: {solver_config.model_dump()}")

    console.print(f"\nLoading GEFF from: {geff_path}")
    graph, geff_metadata = td.graph.InMemoryGraph.from_geff(str(geff_path))
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

    # Load model
    console.print(f"\nLoading model from: {model_path}")

    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]Warning: CUDA not available, falling back to CPU[/yellow]")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        console.print("[yellow]Warning: MPS not available, falling back to CPU[/yellow]")
        device = "cpu"

    model = torch.jit.load(model_path, map_location=device)
    model = model.to(device)  # Explicitly move model to device
    model.eval()
    console.print(f"Model loaded on device: {device}")

    # Run prediction
    console.print("\n[bold green]Running prediction and tracking...[/bold green]")
    solution_graph = model_predict(model, ds, solver_config=solver_config)

    # Save output
    console.print(f"\nSaving results to: {output}")

    # Remove existing directory if overwrite is enabled
    if output.exists() and overwrite:
        shutil.rmtree(output)

    if solution:
        solution_graph.to_geff(output)
    else:
        ds.graph.to_geff(output)

    console.print("[bold green]✓ Results saved successfully![/bold green]")


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
