"""Command-line interface for eet-inference."""

import argparse

from eet_inference import __version__


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Inference CLI for Edge Embedding Tracking (EET) model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"eet-inference {__version__}",
    )

    # Add subcommands here as needed

    args = parser.parse_args()

    # Placeholder for inference logic
    print("EET Inference CLI - Model inference interface")


if __name__ == "__main__":
    main()
