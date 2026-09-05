#!/usr/bin/env python3
"""Build the grouped parameterized generic sparse-QSVT compiler diagram."""

from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "outputs/generic_sparse_qsvt_compiler/generic_sparse_qsvt_compiler_diagram.pdf"


NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6B9A")
TEAL = colors.HexColor("#2A8C82")
GOLD = colors.HexColor("#C6922D")
RED = colors.HexColor("#A7483F")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_TEAL = colors.HexColor("#E8F5F2")
LIGHT_GOLD = colors.HexColor("#FBF3DF")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
MID_GRAY = colors.HexColor("#66737F")


def _font_setup() -> tuple[str, str]:
    regular = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("DiagramArial", str(regular)))
        pdfmetrics.registerFont(TTFont("DiagramArialBold", str(bold)))
        return "DiagramArial", "DiagramArialBold"
    return "Helvetica", "Helvetica-Bold"


def _paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    font: str,
    size: float = 8.2,
    color=colors.black,
    leading: float | None = None,
    align: int = TA_CENTER,
) -> None:
    style = ParagraphStyle(
        "diagram",
        fontName=font,
        fontSize=size,
        leading=leading or size * 1.22,
        textColor=color,
        alignment=align,
        spaceAfter=0,
        spaceBefore=0,
    )
    paragraph = Paragraph(text, style)
    _, used = paragraph.wrap(width, height)
    paragraph.drawOn(pdf, x, y + (height - used) / 2)


def _box(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    fill,
    stroke,
    regular: str,
    bold: str,
    title_size: float = 8.8,
    body_size: float = 7.5,
) -> None:
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(1.0)
    pdf.roundRect(x, y, width, height, 3 * mm, stroke=1, fill=1)
    _paragraph(
        pdf,
        title,
        x + 2 * mm,
        y + height - 10 * mm,
        width - 4 * mm,
        8 * mm,
        font=bold,
        size=title_size,
        color=stroke,
    )
    _paragraph(
        pdf,
        body,
        x + 2.3 * mm,
        y + 2.0 * mm,
        width - 4.6 * mm,
        height - 12 * mm,
        font=regular,
        size=body_size,
        color=NAVY,
    )


def _arrow(pdf: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=NAVY) -> None:
    pdf.setStrokeColor(color)
    pdf.setFillColor(color)
    pdf.setLineWidth(1.4)
    pdf.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    for offset in (0.48, -0.48):
        pdf.line(
            x2,
            y2,
            x2 - 3.2 * mm * math.cos(angle + offset),
            y2 - 3.2 * mm * math.sin(angle + offset),
        )


def build(output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    regular, bold = _font_setup()
    page_width, page_height = landscape(A3)
    pdf = canvas.Canvas(str(output), pagesize=(page_width, page_height), pageCompression=1)
    pdf.setTitle("Generic sparse-QSVT selected-output compiler")
    pdf.setAuthor("Generic sparse-QSVT compiler evidence generator")

    margin = 14 * mm
    pdf.setFillColor(NAVY)
    pdf.setFont(bold, 19)
    pdf.drawString(margin, page_height - 19 * mm, "Generic sparse-QSVT selected-output compiler")
    pdf.setFont(regular, 9.5)
    pdf.setFillColor(MID_GRAY)
    pdf.drawRightString(
        page_width - margin,
        page_height - 18 * mm,
        "Construction is separated from statevector, sampling, and resource execution",
    )

    # Structured public inputs.
    group_top = page_height - 30 * mm
    input_height = 31 * mm
    gap = 3 * mm
    available = page_width - 2 * margin
    input_width = (available - 6 * gap) / 7
    input_specs = [
        ("MatrixSpec", "real values; shape; source;<br/>normalization orientation"),
        ("SupportSpec", "coordinates; slots or rule;<br/>transpose convention"),
        ("QuantizationSpec", "magnitude bits <i>b</i><sub>v</sub>;<br/>sign; scale; rounding"),
        ("QSVTSpec", "alpha or lambda; beta; <i>C</i>;<br/>polynomial; phases; degree <i>d</i>"),
        ("ResidualSpec", "vector <i>r</i>; split; source;<br/>state-preparation convention"),
        ("FunctionalSpec", "physical vectors; primary;<br/>postselection and readout"),
        ("ExecutionSpec", "basis; optimization; seeds;<br/>shot budgets; simulator"),
    ]
    for index, (title, body) in enumerate(input_specs):
        x = margin + index * (input_width + gap)
        _box(
            pdf,
            x,
            group_top - input_height,
            input_width,
            input_height,
            title,
            body,
            fill=LIGHT_BLUE,
            stroke=BLUE,
            regular=regular,
            bold=bold,
            title_size=8.2,
            body_size=6.9,
        )

    validator_y = group_top - input_height - 17 * mm
    pdf.setFillColor(LIGHT_GRAY)
    pdf.setStrokeColor(MID_GRAY)
    pdf.roundRect(margin, validator_y, available, 11 * mm, 2 * mm, stroke=1, fill=1)
    _paragraph(
        pdf,
        "<b>Typed validation and structured failures</b>  |  shape and orientation  |  support uniqueness and bounds  |  slot feasibility  |  finite real values  |  beta/lambda consistency  |  degree/parity and phase count  |  residual and functional dimensions  |  uncomputation  |  register collisions",
        margin + 4 * mm,
        validator_y + 1 * mm,
        available - 8 * mm,
        9 * mm,
        font=regular,
        size=7.2,
        color=NAVY,
    )

    # Main integrated path.
    pipeline_y = validator_y - 50 * mm
    pipeline_h = 38 * mm
    stage_gap = 4 * mm
    stage_widths = [42, 70, 55, 51, 45, 52]
    stages = [
        ("Residual preparation", "normalize <i>r</i><br/>controlled state preparation", LIGHT_GOLD, GOLD),
        ("Sparse access", "index lookup + value/sign lookup<br/>slot permutations + controlled rotations<br/><i>n, s, b</i><sub>v</sub> are parameters", LIGHT_TEAL, TEAL),
        ("Sparse block encoding", "top block <i>H</i><sub>q</sub><sup>T</sup>/beta<br/>inverse wrapper and clean work path", LIGHT_TEAL, TEAL),
        ("QSVT sequence", "<i>d</i> signal calls; <i>d</i>+1 phases<br/>one convention conversion", LIGHT_BLUE, BLUE),
        ("Postselection", "flag = 0 iff sparse work = 0<br/>distinct direct companion circuit", LIGHT_GOLD, GOLD),
        ("Signed selected-output readout", "real interference branch<br/>(<i>N</i><sub>00</sub>-<i>N</i><sub>10</sub>)/shots<br/>physical recovery factor", LIGHT_GOLD, GOLD),
    ]
    widths = [value * mm for value in stage_widths]
    total = sum(widths) + stage_gap * (len(stages) - 1)
    x = (page_width - total) / 2
    centers: list[float] = []
    for width, (title, body, fill, stroke) in zip(widths, stages, strict=True):
        _box(
            pdf,
            x,
            pipeline_y,
            width,
            pipeline_h,
            title,
            body,
            fill=fill,
            stroke=stroke,
            regular=regular,
            bold=bold,
            title_size=9.2,
            body_size=7.3,
        )
        centers.append(x + width / 2)
        x += width + stage_gap
    x = (page_width - total) / 2
    for width in widths[:-1]:
        start = x + width
        _arrow(pdf, start + 0.7 * mm, pipeline_y + pipeline_h / 2, start + stage_gap - 0.7 * mm, pipeline_y + pipeline_h / 2)
        x += width + stage_gap

    # Compiler output and execution separation.
    output_y = pipeline_y - 43 * mm
    output_h = 29 * mm
    left_x = margin
    middle_x = margin + 116 * mm
    right_x = margin + 244 * mm
    _box(
        pdf,
        left_x,
        output_y,
        103 * mm,
        output_h,
        "Structured compiler result",
        "validated metadata; padded dimensions; registers; assignment and lookup tables; rotations; permutations; wrapper; QSVT sequence; final measured circuits; recovery factors; stable component hashes",
        fill=LIGHT_BLUE,
        stroke=BLUE,
        regular=regular,
        bold=bold,
        title_size=9.5,
        body_size=7.1,
    )
    _box(
        pdf,
        middle_x,
        output_y,
        115 * mm,
        output_h,
        "Execution layer: same stored source circuit",
        "exact joint statevector distribution  |  Aer fixed-seed counts  |  transpiled resources  |  source/transpiled QPY archives  |  no per-functional phase refit  |  no direct output-state preparation",
        fill=LIGHT_TEAL,
        stroke=TEAL,
        regular=regular,
        bold=bold,
        title_size=9.5,
        body_size=7.1,
    )
    _box(
        pdf,
        right_x,
        output_y,
        page_width - margin - right_x,
        output_h,
        "Evidence products",
        "lookup, block, unitarity, action, polynomial, quantization, and support errors kept separate; shot rows and seed summaries; register and resource ledgers; dimension/slot/precision/degree scaling",
        fill=LIGHT_GOLD,
        stroke=GOLD,
        regular=regular,
        bold=bold,
        title_size=9.5,
        body_size=7.0,
    )
    _arrow(pdf, centers[-1], pipeline_y - 1 * mm, centers[-1], output_y + output_h + 2 * mm, color=MID_GRAY)

    scaling_y = 67 * mm
    scaling_h = 28 * mm
    scaling_gap = 4 * mm
    scaling_width = (available - 3 * scaling_gap) / 4
    scaling_specs = [
        ("Dimension <i>n</i>", "4, 8, 16<br/>fixed <i>s</i>=3, <i>b</i><sub>v</sub>=6, <i>d</i>=31"),
        ("Slots <i>s</i>", "2, 3, 4<br/>infeasible support rows retained"),
        ("Precision <i>b</i><sub>v</sub>", "4, 6, 8<br/>resources + quantization error"),
        ("Degree <i>d</i>", "15, 31, 63<br/>only valid phase protocols compile"),
    ]
    for index, (title, body) in enumerate(scaling_specs):
        _box(
            pdf,
            margin + index * (scaling_width + scaling_gap),
            scaling_y,
            scaling_width,
            scaling_h,
            title,
            body,
            fill=LIGHT_GRAY,
            stroke=MID_GRAY,
            regular=regular,
            bold=bold,
            title_size=8.8,
            body_size=7.0,
        )
    pdf.setFont(bold, 8.5)
    pdf.setFillColor(MID_GRAY)
    pdf.drawString(margin, scaling_y + scaling_h + 3 * mm, "COMPILED ONE-FACTOR-AT-A-TIME EVIDENCE GRID")

    boundary_y = 38 * mm
    pdf.setFillColor(colors.HexColor("#FBEAEA"))
    pdf.setStrokeColor(RED)
    pdf.roundRect(margin, boundary_y, available, 16 * mm, 2 * mm, stroke=1, fill=1)
    _paragraph(
        pdf,
        "<b>Verified API boundary and claim boundary:</b> real square power-of-two matrices only; rectangular inputs are rejected explicitly. Current values are direct-multiplexed rotation angles, not a scalable value-memory oracle. Evidence is small-scale classical simulation and transpilation only: no dense fallback, hardware execution, fault-tolerant estimate, scalable IEEE-size access, speedup, advantage, or practical-competitiveness claim.",
        margin + 4 * mm,
        boundary_y + 1.5 * mm,
        available - 8 * mm,
        13 * mm,
        font=regular,
        size=7.5,
        color=RED,
    )

    pdf.setFont(regular, 7.5)
    pdf.setFillColor(MID_GRAY)
    pdf.drawRightString(page_width - margin, 7 * mm, "generic_sparse_qsvt_compiler_v1 | parameterized architecture diagram")
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    build()
