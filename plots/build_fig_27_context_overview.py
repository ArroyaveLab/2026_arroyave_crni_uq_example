"""Crop the full Cr-Ni overview and mark the uncertainty-panel window.

The phase diagram itself is reused from the figure of the two fitted points
(chapter/fig_two_modes.py). Requires pdfcrop (TeX Live), pypdf and reportlab. The
outlined box is only a viewport, not a phase boundary. Figure 27's uncertainty
bands remain in their independently topology-checked temperature interval.
"""
from io import BytesIO
from pathlib import Path
import subprocess

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "chapter/figures/fig_two_modes.pdf"
OUTPUT = ROOT / "chapter/figures/fig_27_full_phase_diagram_context.pdf"
CROPPED = OUTPUT.with_name(".fig_27_full_phase_diagram_context_cropped.pdf")

# Crop the upper-left full-diagram panel, in source PDF points.
subprocess.run(
    ["pdfcrop", "--bbox", "0 188 260 357", str(SOURCE), str(CROPPED)],
    check=True,
)
page = PdfReader(str(CROPPED)).pages[0]
width, height = float(page.mediabox.width), float(page.mediabox.height)
source_page = PdfReader(str(SOURCE)).pages[0]
source_height = float(source_page.mediabox.height)

# Remove the original panel letter so the combined figure can label both panels.
# Then mark x_Ni=0..0.70 and T=1350..1550 K, the exact viewport of panel (b).
source_x0, source_x1 = 23.431211, 247.745128
x0, x1 = source_x0, source_x0 + 0.70 * (source_x1 - source_x0)

def cropped_y(temperature_k: float) -> float:
    y_top = 6.741308 + (2200.0 - temperature_k) / (2200.0 - 800.0) * (
        143.597025 - 6.741308
    )
    return source_height - y_top - 188.0

box_y0, box_y1 = sorted((cropped_y(1350.0), cropped_y(1550.0)))
assert width == 260.0 and height == 169.0
assert 0 < x0 < x1 < width and 0 < box_y0 < box_y1 < height

buffer = BytesIO()
overlay_canvas = canvas.Canvas(buffer, pagesize=(width, height))
overlay_canvas.setFillColorRGB(1, 1, 1)
overlay_canvas.rect(0, height - 10, 8, 10, stroke=0, fill=1)
overlay_canvas.setStrokeColorRGB(0.12, 0.35, 0.55)
overlay_canvas.setLineWidth(1.25)
overlay_canvas.setDash(4, 2)
overlay_canvas.rect(x0, box_y0, x1 - x0, box_y1 - box_y0, stroke=1, fill=0)
overlay_canvas.save()
buffer.seek(0)
page.merge_page(PdfReader(buffer).pages[0])
writer = PdfWriter()
writer.add_page(page)
with OUTPUT.open("wb") as output_file:
    writer.write(output_file)
CROPPED.unlink()
print(f"Wrote {OUTPUT}")
