"""
report_generator.py
--------------------
Builds a downloadable PDF summary of one analysis run: company snapshot,
live prediction, model performance (single split + cross-validation),
and backtest results. Used by the "Download PDF Report" button in app.py.
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ACCENT = colors.HexColor("#4472C4")


def _styled_table(data, col_widths):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
    ]))
    return t


def generate_pdf_report(symbol: str, info: dict, latest_date, horizon: int,
                         rf_pred: int, rf_confidence: float,
                         svm_pred: int, svm_confidence: float,
                         predicted_return: float,
                         rf_metrics: dict, svm_metrics: dict, linreg_metrics: dict,
                         backtest_perf: dict, cv_results: dict = None) -> BytesIO:
    """Builds the PDF in-memory and returns a BytesIO buffer ready for st.download_button."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=40, bottomMargin=40,
                             leftMargin=40, rightMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18, spaceAfter=2)
    story.append(Paragraph(f"{symbol} — Stock Analysis &amp; ML Prediction Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
        ParagraphStyle("sub", parent=styles["Normal"], textColor=colors.grey, fontSize=9)
    ))
    story.append(Spacer(1, 14))

    # --- Company overview ---
    story.append(Paragraph("Company Overview", styles["Heading2"]))
    info_data = [
        ["Company", str(info.get("name", "N/A"))],
        ["Sector", str(info.get("sector", "N/A"))],
        ["Current Price (Rs.)", str(info.get("current_price", "N/A"))],
        ["P/E Ratio", str(info.get("pe_ratio", "N/A"))],
    ]
    story.append(_styled_table(info_data, [160, 300]))
    story.append(Spacer(1, 14))

    # --- Live prediction ---
    story.append(Paragraph("Live Prediction", styles["Heading2"]))
    story.append(Paragraph(
        f"Based on data as of {latest_date}, forecasting {horizon} day(s) ahead.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 6))
    pred_data = [
        ["Model", "Prediction", "Confidence"],
        ["Random Forest", "UP" if rf_pred == 1 else "DOWN", f"{rf_confidence*100:.1f}%"],
        ["SVM", "UP" if svm_pred == 1 else "DOWN", f"{svm_confidence*100:.1f}%"],
        ["Linear Regression", f"{predicted_return:+.2f}% predicted return", "—"],
    ]
    story.append(_styled_table(pred_data, [160, 200, 100]))
    story.append(Spacer(1, 14))

    # --- Model performance (single split) ---
    story.append(Paragraph("Model Performance — Held-Out Test Set", styles["Heading2"]))
    perf_data = [
        ["Metric", "Random Forest", "SVM"],
        ["Accuracy", f"{rf_metrics['accuracy']*100:.2f}%", f"{svm_metrics['accuracy']*100:.2f}%"],
        ["Precision", f"{rf_metrics['precision']*100:.2f}%", f"{svm_metrics['precision']*100:.2f}%"],
        ["Recall", f"{rf_metrics['recall']*100:.2f}%", f"{svm_metrics['recall']*100:.2f}%"],
        ["F1 Score", f"{rf_metrics['f1_score']*100:.2f}%", f"{svm_metrics['f1_score']*100:.2f}%"],
    ]
    story.append(_styled_table(perf_data, [160, 150, 150]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Regression Model — % Return Magnitude", styles["Heading2"]))
    reg_data = [
        ["RMSE", "MAE", "R-squared"],
        [f"{linreg_metrics['rmse']:.3f}", f"{linreg_metrics['mae']:.3f}", f"{linreg_metrics['r2_score']:.3f}"],
    ]
    story.append(_styled_table(reg_data, [150, 150, 150]))
    story.append(Spacer(1, 14))

    # --- Cross-validation (more reliable estimate) ---
    if cv_results:
        story.append(Paragraph(
            f"Cross-Validation — {cv_results['n_splits']}-Fold Walk-Forward (more reliable than a single split)",
            styles["Heading2"]
        ))
        cv_data = [
            ["Model", "Mean Accuracy", "Std Dev (fold-to-fold variance)"],
            ["Random Forest", f"{cv_results['rf_mean_accuracy']*100:.2f}%", f"±{cv_results['rf_std']*100:.2f}%"],
            ["SVM", f"{cv_results['svm_mean_accuracy']*100:.2f}%", f"±{cv_results['svm_std']*100:.2f}%"],
        ]
        story.append(_styled_table(cv_data, [130, 150, 220]))
        story.append(Spacer(1, 14))

    # --- Backtest ---
    story.append(Paragraph("Backtest Results (Random Forest Signal vs Buy &amp; Hold)", styles["Heading2"]))
    bt_data = [
        ["Metric", "Value"],
        ["Strategy Return", f"{backtest_perf['total_strategy_return_%']}%"],
        ["Buy & Hold Return", f"{backtest_perf['total_buyhold_return_%']}%"],
        ["Sharpe Ratio", str(backtest_perf["sharpe_ratio"])],
        ["Max Drawdown", f"{backtest_perf['max_drawdown_%']}%"],
        ["Win Rate", f"{backtest_perf['win_rate_%']}%"],
        ["Trades Taken", str(backtest_perf["num_trades"])],
    ]
    story.append(_styled_table(bt_data, [220, 220]))
    story.append(Spacer(1, 20))

    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    story.append(Paragraph(
        "Disclaimer: This report is generated by an educational machine-learning pipeline for "
        "portfolio/demonstration purposes. Accuracy in the 45–58% range is realistic and expected "
        "for short-term equity direction prediction. This is not financial advice and should not "
        "be used to make real trading decisions.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer