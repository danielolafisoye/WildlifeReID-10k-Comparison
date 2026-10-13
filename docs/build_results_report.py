"""Build the final WildlifeReID comparison report PDF."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
RESULTS_PATH = REPO_ROOT / "src" / "models" / "comparison_results.json"
FIGURES_DIR = DOCS_DIR / "figures"
OUTPUT_PATH = DOCS_DIR / "WildlifeReID_Final_Report.pdf"

GREEN = colors.HexColor("#2E7D4F")
PALE_GREEN = colors.HexColor("#EAF2ED")
TEXT = colors.HexColor("#1F2A24")
MUTED = colors.HexColor("#66736B")
BORDER = colors.HexColor("#D9DED8")
LIGHT = colors.HexColor("#F6F7F4")


def format_percent(value):
    return f"{value * 100:.2f}%"


def page_number(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "WildlifeReID-10k model comparison")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def scaled_image(path, width):
    image = Image(str(path))
    image.drawHeight = image.imageHeight * width / image.imageWidth
    image.drawWidth = width
    return image


def build_report():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    convnext = results["models"]["convnext"]
    miewid = results["models"]["miewid"]
    protocol = results["protocol"]

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=TEXT,
            alignment=TA_LEFT,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=14,
            leading=20,
            textColor=MUTED,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=24,
            textColor=TEXT,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=GREEN,
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCopy",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=TEXT,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=21,
            textColor=GREEN,
            backColor=PALE_GREEN,
            borderColor=GREEN,
            borderWidth=0.7,
            borderPadding=12,
            spaceBefore=8,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=10,
        )
    )

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="WildlifeReID-10k Model Comparison",
        author="WildlifeReID-10k Comparison Project",
    )

    story = [
        Spacer(1, 16 * mm),
        Paragraph("WildlifeReID-10k<br/>Model Comparison", styles["ReportTitle"]),
        Paragraph(
            "Final evaluation of ConvNeXt Base and MiewID on a shared open-set animal re-identification protocol",
            styles["Subtitle"],
        ),
        Spacer(1, 8 * mm),
        Paragraph(
            "MiewID is the recommended model: it achieved the strongest Top-1 accuracy and open-set AUROC on the same 606 test queries.",
            styles["Callout"],
        ),
    ]

    overview = Table(
        [
            ["Known identities", "Known queries", "Unknown queries", "Total queries"],
            [
                str(protocol["known_identities"]),
                str(convnext["known_queries"]),
                str(convnext["unknown_queries"]),
                str(protocol["query_count"]),
            ],
        ],
        colWidths=[41 * mm] * 4,
        rowHeights=[12 * mm, 16 * mm],
    )
    overview.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("TEXTCOLOR", (0, 1), (-1, 1), GREEN),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 17),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
            ]
        )
    )
    story.extend(
        [
            overview,
            Spacer(1, 20 * mm),
            Paragraph("Evaluation scope", styles["Subsection"]),
            Paragraph(
                "The experiment uses the existing trained checkpoints. It does not retrain either model. Both checkpoints contain the same ordered set of 100 identities selected from the WildlifeReID-10k training split with random_state=42.",
                styles["BodyCopy"],
            ),
            PageBreak(),
            Paragraph("Evaluation protocol", styles["Section"]),
            Paragraph(
                "Known queries are test-split images of the selected training identities. Unknown queries come from 20 identities that never appear in the complete training split. Top-1 and Top-5 use the 238 known queries. Open-set AUROC uses all 606 queries.",
                styles["BodyCopy"],
            ),
            Paragraph("Different score types", styles["Subsection"]),
            Paragraph(
                "ConvNeXt produces softmax class probabilities. MiewID generates an embedding and retrieves identities using cosine similarity. These scores do not share a common scale, so the comparison uses threshold-independent AUROC rather than treating the same numeric confidence threshold as equivalent for both models.",
                styles["BodyCopy"],
            ),
        ]
    )

    result_rows = [
        ["Model", "Approach", "Top-1", "Top-5", "AUROC", "Mean time"],
        [
            "ConvNeXt Base",
            "Softmax classification",
            format_percent(convnext["top_1"]),
            format_percent(convnext["top_5"]),
            format_percent(convnext["open_set_auroc"]),
            f"{convnext['mean_inference_ms']:.1f} ms",
        ],
        [
            "MiewID",
            "Cosine retrieval",
            format_percent(miewid["top_1"]),
            format_percent(miewid["top_5"]),
            format_percent(miewid["open_set_auroc"]),
            f"{miewid['mean_inference_ms']:.1f} ms",
        ],
    ]
    results_table = Table(
        result_rows,
        colWidths=[29 * mm, 40 * mm, 24 * mm, 24 * mm, 25 * mm, 27 * mm],
        rowHeights=[12 * mm, 15 * mm, 15 * mm],
    )
    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEXT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 2), (-1, 2), PALE_GREEN),
                ("TEXTCOLOR", (0, 2), (-1, 2), GREEN),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph("Results", styles["Section"]),
            results_table,
            Spacer(1, 8 * mm),
            Paragraph(
                "MiewID leads Top-1 by 5.47 percentage points and open-set AUROC by 5.05 percentage points. ConvNeXt leads Top-5 by 2.10 percentage points and has the lower mean inference time.",
                styles["BodyCopy"],
            ),
            PageBreak(),
            Paragraph("Identification performance", styles["Section"]),
            scaled_image(FIGURES_DIR / "metrics_comparison.png", 166 * mm),
            Paragraph(
                "Figure 1. Closed-set identification metrics use known queries only; open-set AUROC uses known and unknown queries.",
                styles["Caption"],
            ),
            Paragraph(
                "MiewID makes the strongest first-choice predictions and separates unseen animals more reliably. ConvNeXt's high Top-5 score shows that the correct identity often remains among its leading candidates.",
                styles["BodyCopy"],
            ),
            PageBreak(),
            Paragraph("Unknown-animal detection", styles["Section"]),
            scaled_image(FIGURES_DIR / "open_set_roc.png", 140 * mm),
            Paragraph(
                "Figure 2. ROC curves treat a larger value of 1 minus the model's maximum score as stronger evidence that a query is unknown.",
                styles["Caption"],
            ),
            Spacer(1, 3 * mm),
            Paragraph(
                "MiewID's AUROC of 0.959 indicates better score separation across possible operating thresholds. ConvNeXt also performs strongly at 0.908, but its curve remains below MiewID across most of the low false-positive region.",
                styles["BodyCopy"],
            ),
            PageBreak(),
            Paragraph("Efficiency and deployment", styles["Section"]),
            scaled_image(FIGURES_DIR / "inference_time.png", 145 * mm),
            Paragraph(
                "Figure 3. Mean per-image inference time on the same Apple Metal device. Model loading and image download time are excluded.",
                styles["Caption"],
            ),
            Paragraph("Deployment decision", styles["Subsection"]),
            Paragraph(
                "MiewID is the recommended primary model because Top-1 and open-set AUROC best represent the application's goal: identify one known animal while recognizing when the animal is outside the known set. ConvNeXt remains a useful faster baseline and comparison option.",
                styles["BodyCopy"],
            ),
            Paragraph("Limitations", styles["Subsection"]),
            Paragraph(
                "The experiment compares complete pipelines rather than identical architectures. It covers a reproducible 100-identity subset, and the web application's UNKNOWN threshold has not been calibrated separately for each score type. Timing may also change on a CUDA GPU or production server.",
                styles["BodyCopy"],
            ),
            Paragraph("Reproduce the comparison", styles["Subsection"]),
            Paragraph(
                "Run <font name='Courier'>python src/models/evaluate_comparison.py --device auto</font>. The command verifies the shared identity list and writes both aggregate JSON metrics and a per-query CSV without retraining either model.",
                styles["BodyCopy"],
            ),
        ]
    )

    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
