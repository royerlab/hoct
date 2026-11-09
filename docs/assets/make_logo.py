"""Generate the HOCT logo in light and dark variants.

Writes ``logo-light.svg`` (for light backgrounds) and ``logo-dark.svg`` (for
dark backgrounds) next to this script. Both have transparent backgrounds.

Run with ``uv run --no-project docs/assets/make_logo.py``.
"""

from pathlib import Path

PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "ink": "#1d2b53",
        "ink2": "#141f3d",
        "teal": "#009688",
        "teal2": "#4db6ac",
        "glass": "#b2dfdb",
        "coral": "#ff5a5f",
        "coral2": "#d94449",
        "star": "#f2b632",
    },
    "dark": {
        "ink": "#f4f1ea",
        "ink2": "#d6d1c4",
        "teal": "#4dd0c4",
        "teal2": "#80cbc4",
        "glass": "#1d4f4a",
        "coral": "#ff6b70",
        "coral2": "#e0484e",
        "star": "#f2c14e",
    },
}


# Color names used in TEMPLATE: ink/ink2 (letters, rocket body and its shaded side),
# teal/teal2 (lineage graph, nucleus, porthole ring), glass (porthole),
# coral/coral2 (nose, fins, flame), star.
TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="600" height="240"
     role="img" aria-labelledby="title desc">
  <title id="title">HOCT</title>
  <desc id="desc">The word HOCT with a rocket lifting off, trailing a cell lineage graph.</desc>
  <defs>
    <linearGradient id="trail" gradientUnits="userSpaceOnUse" x1="24" y1="0" x2="300" y2="0">
      <stop offset="0" stop-color="{teal}" stop-opacity="0.15"/>
      <stop offset="1" stop-color="{teal}"/>
    </linearGradient>
    <linearGradient id="flame" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffb347"/>
      <stop offset="1" stop-color="{coral}" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <!-- lineage graph: one track that divides, fading into the past -->
  <g fill="none" stroke="url(#trail)" stroke-width="3" stroke-linecap="round">
    <path d="M24,222 L300,214"/>
    <path d="M300,214 C360,212 405,200 446,182"/>
    <path d="M300,214 C365,222 420,222 470,206"/>
  </g>
  <g fill="{teal}">
    <circle cx="24"  cy="222" r="3" opacity="0.25"/>
    <circle cx="116" cy="219" r="3.5" opacity="0.45"/>
    <circle cx="208" cy="217" r="4" opacity="0.7"/>
    <circle cx="300" cy="214" r="6"/>
    <circle cx="380" cy="203" r="4.5"/>
    <circle cx="390" cy="219" r="4.5"/>
  </g>

  <!-- wordmark -->
  <g font-family="'Avenir Next', 'Montserrat', 'Futura', 'Helvetica Neue', 'DejaVu Sans', sans-serif"
     font-weight="700" font-size="140" fill="{ink}">
    <text x="20" y="180">H</text>
    <text x="262" y="180">C</text>
    <text x="378" y="180">T</text>
  </g>

  <!-- O as a cell: membrane and an off-center nucleus -->
  <circle cx="196" cy="130" r="43" fill="none" stroke="{ink}" stroke-width="19"/>
  <circle cx="186" cy="123" r="13" fill="{teal}"/>

  <!-- rocket -->
  <g transform="translate(534,102) rotate(40) scale(1.15)">
    <path d="M-9,36 C-10,58 -3,76 0,92 C3,76 10,58 9,36 Z" fill="url(#flame)"/>
    <path d="M-4,36 C-4,48 -1,56 0,64 C1,56 4,48 4,36 Z" fill="#ffe08a"/>
    <path d="M-15,6 C-28,14 -32,28 -30,42 L-15,32 Z" fill="{coral}"/>
    <path d="M15,6 C28,14 32,28 30,42 L15,32 Z" fill="{coral2}"/>
    <path d="M0,-62 C18,-44 22,-8 15,36 L-15,36 C-22,-8 -18,-44 0,-62 Z" fill="{ink}"/>
    <path d="M0,-62 C18,-44 22,-8 15,36 L0,36 Z" fill="{ink2}"/>
    <path d="M0,-62 C9,-54 14,-44 17,-34 Q0,-29 -17,-34 C-14,-44 -9,-54 0,-62 Z" fill="{coral}"/>
    <circle cx="0" cy="-8" r="10" fill="{glass}" stroke="{teal2}" stroke-width="3"/>
    <circle cx="-2" cy="-9" r="4" fill="{teal}"/>
  </g>

  <!-- two quiet stars -->
  <path d="M478,34 l1.6,4.4 4.4,1.6 -4.4,1.6 -1.6,4.4 -1.6,-4.4 -4.4,-1.6 4.4,-1.6 Z" fill="{star}"/>
  <path d="M590,170 l1.2,3.3 3.3,1.2 -3.3,1.2 -1.2,3.3 -1.2,-3.3 -3.3,-1.2 3.3,-1.2 Z" fill="{star}"/>
</svg>
"""


def render(palette: dict[str, str]) -> str:
    """Fill the SVG template with a color palette.

    Parameters
    ----------
    palette : dict[str, str]
        Mapping from template color names to hex colors.

    Returns
    -------
    str
        The SVG document.
    """
    return TEMPLATE.format(**palette)


def main() -> None:
    """Write every palette variant next to this script."""
    out_dir = Path(__file__).parent
    for name, palette in PALETTES.items():
        (out_dir / f"logo-{name}.svg").write_text(render(palette))


if __name__ == "__main__":
    main()
