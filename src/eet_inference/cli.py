"""Command-line interface for eet-inference."""

from pathlib import Path
from typing import Optional

import tracksdata as td
import torch
import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from eet_inference import __version__
from eet_inference.data import FrameDataset
from eet_inference.inference import model_predict
from eet_inference.tracking import ILPSolverConfig

app = typer.Typer(
    name="eet-inference",
    help="Inference CLI for Edge Embedding Tracking (EET) model",
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"eet-inference version: {__version__}")
        raise typer.Exit()


@app.command()
def predict(
    geff_path: Path = typer.Argument(..., help="Path to GEFF directory", exists=True, dir_okay=True, file_okay=False),
    model_path: Path = typer.Argument(..., help="Path to PyTorch model checkpoint", exists=True, dir_okay=False),
    output: Path = typer.Option(None, "--output", "-o", help="Output GEFF directory (default: overwrite input)"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to ILP solver config YAML file", exists=True, dir_okay=False
    ),
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
        eet-inference predict data.geff model.pt --config solver_config.yaml
    """
    console.print(Panel.fit("EET Inference - Model Prediction", style="bold blue"))

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

    # Load dataset
    ds = FrameDataset(
        graph=graph,
        properties=properties,
        min_window_size=window_size,
    )
    console.print(f"Dataset: {ds.graph.num_nodes()} nodes, {ds.graph.num_edges()} edges")
    console.print(f"Time frames: {ds.graph.node_attrs(attr_keys=['t'])['t'].min()} - {ds.graph.node_attrs(attr_keys=['t'])['t'].max()}")

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
    model.eval()
    console.print(f"Model loaded on device: {device}")

    # Run prediction
    console.print("\n[bold green]Running prediction and tracking...[/bold green]")
    model_predict(model, ds, solver_config=solver_config)

    # Save output
    console.print(f"\nSaving results to: {output}")
    ds.graph.to_geff(output)
    console.print("[bold green]✓ Results saved successfully![/bold green]")


@app.command()
def init_config(
    output: Path = typer.Option("solver_config.yaml", "--output", "-o", help="Output YAML path"),
):
    """
    Generate a template ILP solver configuration YAML file.

    Example:
        eet-inference init-config --output my_config.yaml
    """
    console.print(Panel.fit("Generating ILP Solver Config Template", style="bold blue"))

    # Create default config and export to dict
    default_config = ILPSolverConfig.default()
    config_dict = default_config.model_dump()

    # Add comments as a separate structure
    config_with_comments = {
        "# Configuration for ILP tracking solver": None,
        "# Weight for appearance edges (nodes appearing or orphans)": None,
        "appearance_weight": config_dict["appearance_weight"],
        "# Weight for disappearance edges": None,
        "disappearance_weight": config_dict["disappearance_weight"],
        "# Weight for cell division edges (set to 1e6 to disable divisions)": None,
        "division_weight": config_dict["division_weight"],
        "# Weight for node selection": None,
        "node_weight": config_dict["node_weight"],
        "# Penalty for edges spanning multiple frames": None,
        "delta_t_weight": config_dict["delta_t_weight"],
        "# Bias added to edge weights": None,
        "edge_bias": config_dict["edge_bias"],
        "# Solver timeout in seconds": None,
        "timeout": config_dict["timeout"],
        "# Use two-pass tracklet solver": None,
        "tracklet_solver": config_dict["tracklet_solver"],
    }

    # Write YAML with comments
    with open(output, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    console.print(f"[bold green]✓ Config template saved to: {output}[/bold green]")
    console.print("\nEdit this file to customize solver parameters.")


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version and exit"
    ),
):
    """EET Inference - Command-line interface for Edge Embedding Tracking model inference."""
    pass


if __name__ == "__main__":
    app()
