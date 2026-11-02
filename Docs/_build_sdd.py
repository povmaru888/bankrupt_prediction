from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "Docs"
ASSETS = DOCS / "_sdd_assets"
OUTPUT = DOCS / "11_企業破產預測_SDD.docx"
DATE_TEXT = "2026年10月22日"

NAVY = "17365D"
TEAL = "1F6F78"
BLUE = "DCEAF5"
PALE = "F2F6F8"
PALE_BLUE = "EAF3F8"
GRAY = "D9D9D9"
DARK_GRAY = "3F4A54"
WHITE = "FFFFFF"
BLACK = "000000"
RED = "9C2F2F"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msjh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msjhbd.ttc")


def pc(value):
    """Convert a six-digit OOXML color to a Pillow color."""
    if isinstance(value, str) and len(value) == 6 and not value.startswith("#"):
        return "#" + value
    return value


def pil_font(size: int, bold: bool = False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        current = ""
        for ch in raw:
            trial = current + ch
            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def draw_centered_text(draw, box, text, font, fill=BLACK, line_gap=8):
    x1, y1, x2, y2 = box
    lines = wrap_text(draw, text, font, int(x2 - x1 - 34))
    heights = [draw.textbbox((0, 0), line or " ", font=font)[3] for line in lines]
    total = sum(heights) + line_gap * max(0, len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2
    for line, h in zip(lines, heights):
        w = draw.textbbox((0, 0), line, font=font)[2]
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=font, fill=pc(fill))
        y += h + line_gap


def rounded_box(draw, box, text, fill, outline=TEAL, font=None, radius=22, text_fill=BLACK):
    draw.rounded_rectangle(box, radius=radius, fill=pc(fill), outline=pc(outline), width=4)
    draw_centered_text(draw, box, text, font or pil_font(28), text_fill)


def arrow(draw, start, end, color=DARK_GRAY, width=5):
    draw.line([start, end], fill=pc(color), width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for delta in (2.6, -2.6):
        p = (end[0] + 18 * math.cos(angle + delta), end[1] + 18 * math.sin(angle + delta))
        draw.line([end, p], fill=pc(color), width=width)


def base_canvas(title: str, subtitle: str = "", size=(1800, 1050)):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 42), title, font=pil_font(42, True), fill=pc(NAVY))
    if subtitle:
        draw.text((72, 100), subtitle, font=pil_font(24), fill=pc(DARK_GRAY))
    return img, draw


def make_figures():
    ASSETS.mkdir(parents=True, exist_ok=True)

    img, d = base_canvas("系統容器與執行邊界", "全部服務只在本機執行並綁定 127.0.0.1")
    rounded_box(d, (90, 220, 420, 430), "分析人員", PALE_BLUE, font=pil_font(34, True))
    rounded_box(d, (570, 180, 1030, 430), "Streamlit\n四分頁工作台", BLUE, font=pil_font(32, True))
    rounded_box(d, (1210, 180, 1690, 430), "FastAPI\n驗證 評分 解釋 匯出", PALE_BLUE, font=pil_font(30, True))
    rounded_box(d, (1210, 650, 1690, 885), "版本化模型目錄\nactive model 指標", PALE, font=pil_font(28, True))
    rounded_box(d, (570, 650, 1030, 885), "離線 CLI\ntrain activate rollback", PALE, font=pil_font(28, True))
    arrow(d, (420, 325), (570, 325))
    arrow(d, (1030, 325), (1210, 325))
    arrow(d, (1450, 430), (1450, 650))
    arrow(d, (1030, 765), (1210, 765))
    d.text((610, 485), "HTTP 同步呼叫", font=pil_font(24), fill=pc(DARK_GRAY))
    d.text((1110, 920), "不保存上傳檔與評分歷史", font=pil_font(24), fill=pc(RED))
    img.save(ASSETS / "fig01_context.png", dpi=(180, 180))

    img, d = base_canvas("FastAPI 元件責任", "資料驗證與模型推論採單一責任元件並共享錯誤格式")
    xs = [80, 360, 650, 940, 1230, 1510]
    labels = [
        "API 路由\n與檔案限制", "Schema 驗證\n列級品質檢查", "特徵前處理\n補值與欄位選擇", "模型與校準器\n信心分數", "排序與解釋\nSHAP 因素", "回應與匯出\nJSON CSV"
    ]
    fills = [PALE, PALE_BLUE, BLUE, PALE_BLUE, BLUE, PALE]
    for x, label, fill in zip(xs, labels, fills):
        rounded_box(d, (x, 300, x + 220, 650), label, fill, font=pil_font(25, True))
    for i in range(len(xs) - 1):
        arrow(d, (xs[i] + 220, 475), (xs[i + 1], 475))
    rounded_box(d, (560, 760, 1240, 930), "共用服務\nModelRegistry · ErrorMapper\nAuditLogger · TempFileManager", PALE, font=pil_font(22, True))
    arrow(d, (900, 760), (900, 650))
    img.save(ASSETS / "fig02_components.png", dpi=(180, 180))

    img, d = base_canvas("模型訓練與封存流程", "所有可學習步驟只在訓練折內估計")
    steps = [
        ("1", "欄名正規化\n資料稽核"), ("2", "64 16 20\n分層切分"), ("3", "5 fold CV\n每模型 30 組"),
        ("4", "Sigmoid\n交叉驗證校準"), ("5", "validation AP\n模型勝出與 F2 門檻"), ("6", "final test\n只評估一次"),
        ("7", "建立候選模型包\n人工 activate")
    ]
    coords = [(80, 250), (430, 250), (780, 250), (1130, 250), (1130, 650), (780, 650), (430, 650)]
    for (n, label), (x, y) in zip(steps, coords):
        rounded_box(d, (x, y, x + 270, y + 190), f"{n}\n{label}", PALE_BLUE if int(n) % 2 else BLUE, font=pil_font(25, True))
    for i in range(3):
        arrow(d, (coords[i][0] + 270, coords[i][1] + 95), (coords[i + 1][0], coords[i + 1][1] + 95))
    arrow(d, (1265, 440), (1265, 650))
    arrow(d, (1130, 745), (1050, 745))
    arrow(d, (780, 745), (700, 745))
    d.text((80, 920), "禁止 test set 參與欄位處理 調參 校準 模型選擇或門檻選擇", font=pil_font(26, True), fill=pc(RED))
    img.save(ASSETS / "fig03_training.png", dpi=(180, 180))

    img, d = base_canvas("批次評分序列", "檔案級錯誤整批拒絕 列級錯誤允許部分成功")
    actors = [(140, "分析人員"), (500, "Streamlit"), (880, "FastAPI"), (1280, "Active model")]
    top = 190
    bottom = 930
    for x, label in actors:
        rounded_box(d, (x - 90, 145, x + 90, 230), label, PALE_BLUE, font=pil_font(22, True), radius=12)
        d.line((x, 230, x, bottom), fill=pc(GRAY), width=3)
    messages = [
        (140, 500, 310, "1 上傳 CSV 與門檻"), (500, 880, 400, "2 POST score"),
        (880, 880, 490, "3 Schema 與列級驗證"), (880, 1280, 590, "4 前處理 評分 SHAP"),
        (1280, 880, 690, "5 分數與貢獻"), (880, 500, 790, "6 JSON 或 CSV 結果"),
        (500, 140, 875, "7 檢視與下載")
    ]
    for x1, x2, y, label in messages:
        arrow(d, (x1, y), (x2, y), color=TEAL, width=4)
        d.text((min(x1, x2) + 12, y - 36), label, font=pil_font(22), fill=pc(DARK_GRAY))
    img.save(ASSETS / "fig04_scoring.png", dpi=(180, 180))

    img, d = base_canvas("模型生命週期", "候選模型不會自動取代 active model")
    nodes = [
        (110, 350, "訓練資料與設定"), (450, 350, "候選模型包"), (790, 350, "人工審查"),
        (1130, 350, "activate"), (1470, 350, "active model")
    ]
    for i, (x, y, label) in enumerate(nodes):
        rounded_box(d, (x, y, x + 230, y + 190), label, PALE_BLUE if i % 2 else BLUE, font=pil_font(27, True))
        if i:
            arrow(d, (nodes[i - 1][0] + 230, y + 95), (x, y + 95))
    rounded_box(d, (1130, 700, 1700, 900), "rollback 指定既有版本\n校驗通過後更新 active 指標", PALE, font=pil_font(27, True))
    arrow(d, (1585, 540), (1585, 700))
    img.save(ASSETS / "fig05_lifecycle.png", dpi=(180, 180))

    img, d = base_canvas("本機部署拓樸", "兩個服務程序與一個唯讀模型目錄")
    rounded_box(d, (100, 250, 500, 790), "Windows 11 主機\n\n瀏覽器\nhttp://127.0.0.1:8501\n\nStreamlit 程序\nport 8501", PALE_BLUE, font=pil_font(26, True))
    rounded_box(d, (680, 250, 1120, 790), "FastAPI 程序\nport 8000\n\nSchema Validator\nScoring Service\nExplanation Service\nTempFileManager", BLUE, font=pil_font(26, True))
    rounded_box(d, (1300, 250, 1700, 790), "本機檔案系統\n\nmodels versions\nactive.json\nlogs\n臨時目錄", PALE, font=pil_font(26, True))
    arrow(d, (500, 520), (680, 520))
    arrow(d, (1120, 520), (1300, 520))
    d.text((520, 465), "HTTP", font=pil_font(24, True), fill=pc(TEAL))
    d.text((1160, 465), "唯讀載入", font=pil_font(24, True), fill=pc(TEAL))
    img.save(ASSETS / "fig06_deployment.png", dpi=(180, 180))

    img, d = base_canvas("Streamlit 四分頁低擬真線框", "相同 active model 與當次批次狀態在分頁間共用", size=(1800, 1200))
    cards = [
        (70, 170, 870, 620, "總覽與模型資訊", ["使用限制與 active version", "主要測試指標", "模型建立時間與預設門檻"]),
        (930, 170, 1730, 620, "批次評分", ["CSV 上傳與驗證摘要", "門檻滑桿與分數分布", "高風險名單與下載"]),
        (70, 680, 870, 1130, "個案解釋", ["record_id 選擇", "原始值與前五項正負因素", "非因果解釋聲明"]),
        (930, 680, 1730, 1130, "模型效能", ["PR 與校準曲線", "混淆矩陣與 Recall@Top10%", "資料切分與模型限制"])
    ]
    for x1, y1, x2, y2, title, items in cards:
        d.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=pc("FFFFFF"), outline=pc(TEAL), width=4)
        d.rectangle((x1, y1, x2, y1 + 70), fill=pc(NAVY))
        d.text((x1 + 24, y1 + 16), title, font=pil_font(28, True), fill=pc(WHITE))
        y = y1 + 100
        for item in items:
            d.rounded_rectangle((x1 + 28, y, x2 - 28, y + 82), radius=10, fill=pc(PALE_BLUE), outline=pc(GRAY), width=2)
            d.text((x1 + 50, y + 23), item, font=pil_font(23), fill=pc(BLACK))
            y += 100
    img.save(ASSETS / "fig07_wireframes.png", dpi=(180, 180))


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color=GRAY, size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), size)
        el.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_row_cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    tr_pr.append(cant_split)


def set_keep_with_next(paragraph, value=True):
    p_pr = paragraph._p.get_or_add_pPr()
    el = p_pr.find(qn("w:keepNext"))
    if value and el is None:
        el = OxmlElement("w:keepNext")
        p_pr.append(el)
    elif not value and el is not None:
        p_pr.remove(el)


def set_keep_together(paragraph, value=True):
    p_pr = paragraph._p.get_or_add_pPr()
    el = p_pr.find(qn("w:keepLines"))
    if value and el is None:
        el = OxmlElement("w:keepLines")
        p_pr.append(el)


def set_font(run, name="Microsoft JhengHei", size=None, bold=None, color=None):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    set_font(run, size=9, color=DARK_GRAY)


def set_picture_alt(inline_shape, title, description):
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("name", title)
    doc_pr.set("title", title)
    doc_pr.set("descr", description)


class SDDBuilder:
    def __init__(self, toc_pages=None):
        self.doc = Document()
        self.toc_pages = toc_pages or {}
        self.figure_number = 0
        self.table_number = 0
        self.configure_document()

    def configure_document(self):
        sec = self.doc.sections[0]
        sec.page_width = Cm(21.0)
        sec.page_height = Cm(29.7)
        sec.top_margin = Cm(1.65)
        sec.bottom_margin = Cm(1.6)
        sec.left_margin = Cm(1.75)
        sec.right_margin = Cm(1.75)
        sec.header_distance = Cm(0.7)
        sec.footer_distance = Cm(0.7)
        sec.different_first_page_header_footer = True

        styles = self.doc.styles
        normal = styles["Normal"]
        normal.font.name = "Microsoft JhengHei"
        normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft JhengHei")
        normal.font.size = Pt(10.5)
        normal.paragraph_format.space_after = Pt(5)
        normal.paragraph_format.line_spacing = 1.15

        title = styles["Title"]
        title.font.name = "Microsoft JhengHei"
        title._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft JhengHei")
        title.font.size = Pt(26)
        title.font.bold = True
        title.font.color.rgb = RGBColor(0, 0, 0)
        title_ppr = title._element.get_or_add_pPr()
        title_border = title_ppr.find(qn("w:pBdr"))
        if title_border is not None:
            title_ppr.remove(title_border)

        for name, size, before, after in (("Heading 1", 17, 12, 7), ("Heading 2", 13.5, 9, 4), ("Heading 3", 11.5, 7, 3)):
            style = styles[name]
            style.font.name = "Microsoft JhengHei"
            style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft JhengHei")
            style.font.size = Pt(size)
            style.font.bold = True
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.paragraph_format.space_before = Pt(before)
            style.paragraph_format.space_after = Pt(after)
            style.paragraph_format.keep_with_next = True

        for style_name in ("List Bullet", "List Number"):
            s = styles[style_name]
            s.font.name = "Microsoft JhengHei"
            s._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft JhengHei")
            s.font.size = Pt(10.2)
            s.paragraph_format.left_indent = Cm(0.65)
            s.paragraph_format.first_line_indent = Cm(-0.35)
            s.paragraph_format.space_after = Pt(3)

        if "Caption Custom" not in styles:
            cap = styles.add_style("Caption Custom", WD_STYLE_TYPE.PARAGRAPH)
            cap.font.name = "Microsoft JhengHei"
            cap._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft JhengHei")
            cap.font.size = Pt(9)
            cap.font.color.rgb = RGBColor.from_string(DARK_GRAY)
            cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_before = Pt(3)
            cap.paragraph_format.space_after = Pt(8)
            cap.paragraph_format.keep_with_next = False

        header = sec.header
        hp = header.paragraphs[0]
        hp.text = "企業破產預測系統 軟體設計說明書"
        hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for r in hp.runs:
            set_font(r, size=8.5, color=DARK_GRAY)
        footer = sec.footer
        fp = footer.paragraphs[0]
        add_page_number(fp)

        settings = self.doc.settings._element
        update_fields = OxmlElement("w:updateFields")
        update_fields.set(qn("w:val"), "true")
        settings.append(update_fields)

        self.doc.core_properties.title = "企業破產預測系統 軟體設計說明書"
        self.doc.core_properties.author = "專題小組"
        self.doc.core_properties.subject = "企業破產風險辨識 MVP 軟體設計"
        self.doc.core_properties.keywords = "SDD, IEEE 1016, FastAPI, Streamlit, XGBoost, Logistic Regression"

    def add_heading(self, text, level=1):
        p = self.doc.add_paragraph(text, style=f"Heading {level}")
        set_keep_with_next(p)
        return p

    def add_para(self, text="", bold_lead=None):
        p = self.doc.add_paragraph()
        if bold_lead:
            r = p.add_run(bold_lead)
            set_font(r, bold=True)
        r = p.add_run(text)
        set_font(r)
        set_keep_together(p)
        return p

    def add_bullets(self, items):
        for item in items:
            p = self.doc.add_paragraph(style="List Bullet")
            r = p.add_run(item)
            set_font(r, size=10.2)
            set_keep_together(p)

    def add_table(self, headers, rows, widths=None, font_size=8.8, caption=None):
        if caption:
            self.table_number += 1
            p = self.doc.add_paragraph(f"表 {self.table_number}  {caption}", style="Caption Custom")
            set_keep_with_next(p)
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        header_row = table.rows[0]
        set_repeat_table_header(header_row)
        set_row_cant_split(header_row)
        for idx, h in enumerate(headers):
            cell = header_row.cells[idx]
            set_cell_shading(cell, NAVY)
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(h))
            set_font(r, size=font_size, bold=True, color=WHITE)
        for ridx, row in enumerate(rows):
            added_row = table.add_row()
            set_row_cant_split(added_row)
            cells = added_row.cells
            for idx, value in enumerate(row):
                cell = cells[idx]
                set_cell_shading(cell, PALE_BLUE if ridx % 2 else WHITE)
                set_cell_border(cell)
                set_cell_margins(cell)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if len(str(value)) > 18 else WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(str(value))
                set_font(r, size=font_size)
        if widths:
            for row in table.rows:
                for idx, width in enumerate(widths):
                    row.cells[idx].width = Cm(width)
        for row in table.rows:
            row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        self.doc.add_paragraph().paragraph_format.space_after = Pt(1)
        return table

    def add_figure(self, path, caption, width_cm=16.5, description=None):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run()
        shape = run.add_picture(str(path), width=Cm(width_cm))
        self.figure_number += 1
        set_picture_alt(shape, f"圖 {self.figure_number} {caption}", description or caption)
        cap = self.doc.add_paragraph(f"圖 {self.figure_number}  {caption}", style="Caption Custom")
        return shape

    def new_page(self):
        self.doc.add_page_break()

    def cover(self):
        for _ in range(5):
            self.doc.add_paragraph()
        p = self.doc.add_paragraph(style="Title")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("企業破產預測系統")
        set_font(r, size=26, bold=True, color=BLACK)
        p2 = self.doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p2.add_run("軟體設計說明書")
        set_font(r, size=22, bold=True, color=BLACK)
        p2.paragraph_format.space_after = Pt(28)
        p3 = self.doc.add_paragraph()
        p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p3.add_run("Software Design Description")
        set_font(r, size=13, color=DARK_GRAY)
        for _ in range(7):
            self.doc.add_paragraph()
        meta = [
            ("文件版本", "Draft 0.1"), ("作者", "專題小組"), ("文件日期", DATE_TEXT),
            ("適用範圍", "本機單機 Web MVP"), ("文件狀態", "設計基線")
        ]
        t = self.doc.add_table(rows=0, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for k, v in meta:
            cells = t.add_row().cells
            cells[0].width = Cm(4.0)
            cells[1].width = Cm(8.0)
            for c in cells:
                set_cell_border(c, color=WHITE, size="0")
                set_cell_margins(c, top=110, bottom=110)
            p = cells[0].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = p.add_run(k)
            set_font(r, size=10.5, bold=True, color=DARK_GRAY)
            p = cells[1].paragraphs[0]
            r = p.add_run(v)
            set_font(r, size=10.5)
        self.new_page()

    def control_and_toc(self):
        self.add_heading("文件控制", 1)
        self.add_table(
            ["項目", "內容"],
            [["需求基線", "10_企業破產預測_AI核心與MVP方案.docx"],
             ["設計方法", "IEEE 1016 精簡版"], ["目標讀者", "負責實作與驗收的工程人員"],
             ["系統版本", "MVP Draft 0.1"], ["機密等級", "專題內部使用"]],
            widths=[4.2, 12.3], font_size=9.0, caption="文件識別資訊")
        self.add_heading("目錄", 1)
        toc = [
            "1 文件目的與範圍", "2 系統架構設計", "3 元件與責任", "4 資料設計",
            "5 模型設計", "6 主要流程設計", "7 公開介面設計", "8 使用者介面設計",
            "9 模型包與部署設計", "10 非功能與錯誤處理", "11 測試與驗收設計",
            "12 需求追溯矩陣", "附錄 A 名詞與參考資料"
        ]
        toc_table = self.doc.add_table(rows=0, cols=2)
        toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        toc_table.autofit = False
        for item in toc:
            cells = toc_table.add_row().cells
            cells[0].width = Cm(15.4)
            cells[1].width = Cm(1.1)
            for c in cells:
                set_cell_border(c, color=WHITE, size="0")
                set_cell_margins(c, top=20, bottom=20, start=0, end=0)
            p = cells[0].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(item)
            set_font(r, size=10.2)
            p = cells[1].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(self.toc_pages.get(item, "-")))
            set_font(r, size=10.2)
        self.add_para("本目錄以最終渲染頁面校準。章節標題使用 Word 標題樣式，後續修訂時應同步更新目錄頁碼。")
        self.new_page()

    def section1(self):
        self.add_heading("1 文件目的與範圍", 1)
        self.add_heading("1 1 文件目的", 2)
        self.add_para("本文件定義企業破產預測系統 MVP 的軟體結構、資料契約、模型生命週期、公開介面、錯誤處理與驗收條件。工程人員應能依本文件完成實作，而不需要重新決定系統邊界或模型治理規則。")
        self.add_heading("1 2 系統目標", 2)
        self.add_para("系統使用既有 95 個財務特徵辨識並排序高風險樣本，協助分析人員優先檢視。系統輸出只代表現有資料標籤定義下的模型信心分數，不取代授信、投資或法律判斷。")
        self.add_heading("1 3 使用者與使用情境", 2)
        self.add_table(["角色", "責任", "主要操作"], [
            ["分析人員", "檢查資料、調整情境門檻並解讀結果", "上傳 CSV、檢視排序、個案解釋與下載"],
            ["模型維護者", "訓練、檢查與啟用模型", "執行 train、activate、rollback CLI"],
            ["系統維護者", "啟動本機服務與處理失敗", "管理 Python 環境、程序與日誌"]
        ], widths=[3.1, 6.2, 7.2], caption="角色與責任")
        self.add_heading("1 4 範圍界線", 2)
        self.add_table(["納入 MVP", "排除 MVP"], [[
            "CSV 批次評分、信心分數、百分位、高風險分類、Top 10% 名單、個案解釋、結果下載、模型報告",
            "原始財報解析、95 指標自動計算、雲端部署、多人帳號、資料庫歷史、SMOTE、第三種模型、自動決策"
        ]], widths=[8.2, 8.2], caption="系統範圍")
        self.add_heading("1 5 設計原則", 2)
        self.add_bullets([
            "測試集隔離。test set 不參與清理規則估計、調參、校準、模型選擇或門檻選擇。",
            "可重現。資料雜湊、切分、亂數種子、套件版本、模型參數與門檻都必須保存。",
            "人工治理。候選模型不會自動成為 active model，且必須能回滾。",
            "資料最小化。上傳檔與評分結果不保存，日誌不記錄完整財務特徵。",
            "清楚表達限制。介面只使用模型信心分數，並揭露資料缺少公司、年度與預測窗口。"
        ])
        self.new_page()

    def section2(self):
        self.add_heading("2 系統架構設計", 1)
        self.add_heading("2 1 架構概述", 2)
        self.add_para("MVP 採本機兩程序架構。Streamlit 提供四分頁工作台，透過 HTTP 呼叫 FastAPI。FastAPI 是唯一可讀取 active model 的線上程序，統一負責驗證、評分、解釋、排序與匯出。模型訓練由離線 CLI 執行，不與 Web 請求共用程序。")
        self.add_figure(ASSETS / "fig01_context.png", "系統容器與執行邊界", width_cm=16.4,
                        description="分析人員經由 Streamlit 呼叫本機 FastAPI，FastAPI 讀取 active model；離線 CLI 建立與啟用版本化模型。")
        self.add_heading("2 2 信任邊界", 2)
        self.add_bullets([
            "Streamlit 與 FastAPI 只綁定 127.0.0.1，不接受外部網路連線。",
            "MVP 不設登入，信任邊界為已登入 Windows 的單一受信任使用者。",
            "模型包只允許來自本機訓練流程，載入前驗證 manifest 與檔案校驗值。",
            "評分暫存目錄須使用隨機工作目錄，回應完成或失敗後都要清理。"
        ])
        self.add_heading("2 3 技術棧", 2)
        self.add_table(["層次", "技術", "設計責任"], [
            ["使用者介面", "Streamlit", "分頁、上傳、圖表、門檻情境與下載"],
            ["應用服務", "FastAPI Pydantic", "API、schema 驗證、錯誤格式與同步評分"],
            ["資料與模型", "pandas scikit-learn XGBoost SHAP", "前處理、訓練、校準、評估與解釋"],
            ["保存", "joblib JSON HTML PNG", "模型包、manifest、指標、報告與圖表"],
            ["執行環境", "Windows 11 CPU Python venv", "兩個本機服務程序與離線 CLI"]
        ], widths=[3.0, 5.0, 8.5], caption="技術棧與責任")
        self.new_page()

    def section3(self):
        self.add_heading("3 元件與責任", 1)
        self.add_figure(ASSETS / "fig02_components.png", "FastAPI 元件與資料流", width_cm=16.4)
        self.add_table(["元件", "輸入", "責任與輸出"], [
            ["SchemaValidator", "上傳 CSV", "正規化欄名、驗證 95 特徵、record_id 與檔案限制"],
            ["RowQualityValidator", "逐列特徵值", "標示非數字、無限值、缺值超標與 out_of_range"],
            ["FeaturePipeline", "合法列", "依模型包執行欄位選擇、中位數補值與必要縮放"],
            ["ScoringService", "前處理矩陣、門檻", "產生信心分數、分類、批次百分位與 Top 10%"],
            ["ExplanationService", "模型與合法列", "產生 SHAP 正負向貢獻，提供 UI 前五項與匯出前三項"],
            ["ModelRegistry", "active.json、模型包", "校驗版本、載入 active model、提供模型摘要"],
            ["TempFileManager", "上傳串流", "建立受控暫存目錄並在成功或失敗後刪除"],
            ["AuditLogger", "請求摘要", "只記錄版本、筆數、耗時與錯誤碼統計"]
        ], widths=[3.2, 4.0, 9.3], font_size=8.5, caption="線上元件責任")
        self.add_heading("3 1 元件依賴規則", 2)
        self.add_para("UI 不得直接載入 joblib 或呼叫 SHAP。FastAPI 不得修改 active model，也不得在請求期間訓練。訓練 CLI 與線上服務共用同一份 schema 定義，但以版本化檔案交換，不透過程序內共享狀態。")
        self.new_page()

    def section4(self):
        self.add_heading("4 資料設計", 1)
        self.add_heading("4 1 資料集概況", 2)
        self.add_para("現有資料包含 6,819 列、95 個特徵與 1 個標籤，其中破產樣本 220 列，比例約 3.23%。原始特徵名稱均有前置空白；資料沒有公司識別碼、會計年度或事件日期，因此系統不得將輸出描述為指定預測窗口的破產機率。")
        self.add_heading("4 2 推論 CSV 契約", 2)
        self.add_table(["規則", "處理"], [
            ["欄名", "去除前後空白後，以區分大小寫方式精確比對 95 個特徵"],
            ["可選識別欄", "只允許 record_id；必須非空且唯一，未提供時產生 row_000001 格式"],
            ["禁止欄位", "Bankrupt? 不得出現在推論檔"],
            ["檔案級錯誤", "缺少特徵、未知欄位、編碼失敗、空檔、超過 10 MB 或超過 10,000 列"],
            ["列級錯誤", "非數字、無限值或缺少 10 個以上特徵；其他合法列仍可評分"],
            ["缺值", "每列最多 9 個缺值，以訓練集欄位中位數補值並加入警告"],
            ["極端值", "有限值即接受；超出訓練最小最大值時標示 out_of_range，不截尾"]
        ], widths=[4.0, 12.5], caption="推論資料契約")
        self.add_heading("4 3 特徵清理", 2)
        self.add_table(["類型", "處理決策", "保存紀錄"], [
            ["常數欄", "移除 Net Income Flag", "manifest.removed_features"],
            ["完全重複", "保留首次欄位；移除 Current Liability to Liability 與 Current Liability to Equity", "feature_schema.json"],
            ["高度相關", "只列入稽核報告，不自動刪除", "model_report.html"],
            ["缺值統計", "訓練資料內計算中位數與觀察範圍", "model.joblib 與 manifest"],
            ["欄位順序", "保存正規化後 95 欄順序；推論時重排", "feature_schema.json"]
        ], widths=[3.0, 8.0, 5.5], font_size=8.6, caption="特徵清理規則")
        self.new_page()
        self.add_heading("4 4 評分結果契約", 2)
        fields = [
            ["識別與版本", "record_id model_version"],
            ["分數與分類", "confidence_score batch_percentile threshold_used risk_class top_10_percent_flag"],
            ["品質狀態", "row_status warning_codes error_code error_message"],
            ["解釋", "前三個正向與前三個負向因素名稱及貢獻值"]
        ]
        self.add_table(["群組", "欄位"], fields, widths=[3.5, 13.0], caption="評分輸出欄位")

    def section5(self):
        self.add_heading("5 模型設計", 1)
        self.add_heading("5 1 資料切分", 2)
        self.add_para("系統使用固定亂數種子 42 進行分層切分。訓練集占 64%，validation set 占 16%，final test set 占 20%。標籤比例在三個資料集維持一致，切分索引或其雜湊必須保存。")
        self.add_heading("5 2 候選模型與搜尋空間", 2)
        self.add_table(["模型", "Pipeline 與搜尋參數", "不平衡處理"], [
            ["Logistic Regression", "MedianImputer；StandardScaler 或 RobustScaler；L1 或 L2；C 採對數尺度搜尋；max_iter 足以收斂", "class_weight 候選以 balanced 為主"],
            ["XGBoost", "MedianImputer；樹深、學習率、樹數、subsample、colsample_bytree、min_child_weight、reg_alpha、reg_lambda", "scale_pos_weight 依訓練折比例建立候選"],
        ], widths=[3.4, 9.0, 4.2], font_size=8.5, caption="候選模型設計")
        self.add_para("兩個模型各執行 30 組 RandomizedSearchCV，內層採 Stratified 5-fold，scoring 固定為 average_precision。禁止 SMOTE 與任何在切分前估計的縮放或特徵處理。")
        self.add_heading("5 3 校準與模型選擇", 2)
        self.add_bullets([
            "調參完成後，對各候選模型使用 5-fold Sigmoid calibration。",
            "使用 validation set 比較校準後 Average Precision；差距小於 0.01 時選 Logistic Regression。",
            "在勝出模型的 validation 分數上選擇 F2 最大的門檻，保存為 default_threshold。",
            "UI 可調整當次 threshold，但不得寫回 manifest 或 active model。",
            "final test set 只在模型與門檻凍結後使用一次。"
        ])
        self.add_heading("5 4 評估與通過門檻", 2)
        self.add_table(["類型", "指標與條件"], [
            ["主要排序", "Average Precision 與 PR 曲線"],
            ["分類", "Recall Precision F1 混淆矩陣與誤報數"],
            ["資源限制", "Recall@Top10% 及批次 Top 10% 名單"],
            ["可靠性", "Brier Score 與校準曲線"],
            ["MVP 通過", "test AP 至少為 test 破產比例基準的 2 倍，且 Recall@Top10% 至少 50%"]
        ], widths=[3.5, 13.0], caption="模型評估條件")
        self.add_heading("5 5 模型解釋", 2)
        self.add_para("Logistic Regression 使用 Linear SHAP，XGBoost 使用 Tree SHAP。UI 顯示選定個案前五個正向及負向因素；CSV 保存前三個正向及負向因素。畫面同時顯示原始特徵值與貢獻方向，並說明貢獻反映模型使用資訊的方式，不代表因果關係。")
        self.add_figure(ASSETS / "fig03_training.png", "模型訓練 評估與封存流程", width_cm=16.3)

    def section6(self):
        self.add_heading("6 主要流程設計", 1)
        self.add_heading("6 1 離線訓練流程", 2)
        self.add_table(["步驟", "處理", "失敗行為"], [
            ["1 資料載入", "驗證標籤、95 特徵、列數與數值型態", "終止並輸出資料稽核錯誤"],
            ["2 固定切分", "以 seed 42 建立 64 16 20 分層切分", "不得改用隨機替代切分"],
            ["3 交叉驗證", "兩模型各 30 組、5-fold、AP scoring", "保存所有失敗組合與原因"],
            ["4 校準比較", "Sigmoid calibration、validation AP 與勝出規則", "候選未達品質條件仍可保存但不可標記通過"],
            ["5 凍結評估", "F2 門檻與 final test 一次性評估", "測試結果不得觸發重新選模"],
            ["6 候選封存", "輸出模型包、HTML 報告與 checksum", "不自動修改 active.json"]
        ], widths=[3.0, 8.0, 5.5], font_size=8.4, caption="訓練流程控制")
        self.add_heading("6 2 批次評分流程", 2)
        self.add_figure(ASSETS / "fig04_scoring.png", "批次評分同步序列", width_cm=16.2)
        self.add_para("若發生檔案級錯誤，FastAPI 不執行任何列的評分。若只有列級錯誤，合法列完成評分，錯誤列保留 record_id、row_status 與錯誤訊息。批次百分位與 Top 10% 只使用合法列計算；Top 10% 取 ceil 有效列數乘以 10%，同分時維持原始列順序。")
        self.add_heading("6 3 門檻情境", 2)
        self.add_para("未提供 threshold 時使用 active model 的 default_threshold。提供 threshold 時必須位於 0 到 1，僅影響 risk_class；confidence_score、batch_percentile、Top 10% 與 SHAP 貢獻不變。回應與下載檔均記錄 threshold_used。")
        self.new_page()

    def section7(self):
        self.add_heading("7 公開介面設計", 1)
        self.add_heading("7 1 命令列介面", 2)
        self.add_table(["命令", "必要輸入", "輸出與約束"], [
            ["train", "data path config path", "建立版本化候選模型包；不更新 active model"],
            ["activate", "version id", "驗證 manifest 與 checksum 後原子更新 active.json"],
            ["rollback", "既有 version id", "確認版本完整後切換；保留操作紀錄"],
        ], widths=[3.2, 5.0, 8.3], caption="CLI 契約")
        self.add_heading("7 2 HTTP API", 2)
        self.add_table(["方法與路徑", "請求", "成功回應"], [
            ["GET /health", "無", "服務狀態、active_model_loaded、model_version"],
            ["GET /api/v1/model", "無", "版本、建立時間、預設門檻、特徵數與主要 test 指標"],
            ["POST /api/v1/score", "multipart file；可選 threshold；format=json 或 csv", "批次摘要、列級結果、警告與實際門檻"]
        ], widths=[4.5, 5.5, 6.5], caption="FastAPI 端點")
        self.add_heading("7 3 Score 回應摘要", 2)
        self.add_table(["欄位", "型別", "說明"], [
            ["request_id", "string", "單次請求識別，用於日誌關聯"],
            ["model_version", "string", "active model 版本"],
            ["threshold_used", "number", "預設或使用者指定門檻"],
            ["total_rows", "integer", "輸入列數"],
            ["succeeded_rows", "integer", "成功評分列數"],
            ["failed_rows", "integer", "列級失敗數"],
            ["warnings", "array", "批次級警告代碼與摘要"],
            ["rows", "array", "JSON 格式時的列級結果；CSV 格式改為下載內容"]
        ], widths=[4.0, 3.0, 9.5], font_size=8.7, caption="Score 回應欄位")
        self.add_heading("7 4 錯誤格式與狀態碼", 2)
        self.add_table(["狀態碼", "情境", "回應要求"], [
            ["200", "全部成功或部分列失敗", "摘要列出成功與失敗數；每列保留 row_status"],
            ["413", "檔案超過 10 MB", "FILE_TOO_LARGE 與上限"],
            ["422", "檔案 schema 或 threshold 不合法", "error_code message details remediation"],
            ["503", "沒有可用 active model", "MODEL_UNAVAILABLE，不接受評分"],
            ["500", "未預期錯誤", "不回傳堆疊或檔案路徑；使用 request_id 追蹤"]
        ], widths=[2.5, 6.0, 8.0], font_size=8.6, caption="HTTP 錯誤處理")
        self.add_para("JSON 錯誤物件固定包含 error_code、message、details 與 remediation。CSV 匯出前須處理 record_id 的公式注入；以等號、加號、減號或 @ 開頭的文字值要以前置單引號轉義。")

    def section8(self):
        self.add_heading("8 使用者介面設計", 1)
        self.add_para("Streamlit 使用四分頁工作台。分頁共用 active model 摘要與當次批次結果，不保存跨瀏覽器工作階段的歷史。若沒有 active model，所有評分控制項停用並顯示 503 對應提示。")
        self.add_figure(ASSETS / "fig07_wireframes.png", "Streamlit 四分頁線框", width_cm=14.8)
        self.add_table(["分頁", "必要元件", "主要驗收行為"], [
            ["總覽與模型資訊", "限制聲明、版本、日期、預設門檻、AP、Recall@Top10%", "不使用破產機率字樣"],
            ["批次評分", "上傳、驗證摘要、門檻滑桿、分布圖、名單、下載", "調整門檻只改分類，不改 Top 10%"],
            ["個案解釋", "record_id 選擇、分數、原始值、前五正負因素", "顯示非因果解釋聲明"],
            ["模型效能", "PR、校準、混淆矩陣、切分與限制", "只顯示凍結模型的保存結果"]
        ], widths=[3.5, 8.0, 5.0], font_size=8.5, caption="UI 行為契約")
        self.add_heading("8 1 介面用語", 2)
        self.add_bullets([
            "0 到 1 的輸出一律標示為模型信心分數。",
            "分類使用高風險與未達高風險門檻，不使用將會破產與不會破產。",
            "每個結果頁面顯示僅供分析優先順序使用，不得作為自動拒絕或法律處置唯一依據。",
            "out_of_range 與補值警告必須能在摘要及個案列中查看。"
        ])
        self.new_page()

    def section9(self):
        self.add_heading("9 模型包與部署設計", 1)
        self.add_heading("9 1 模型包結構", 2)
        self.add_table(["檔案", "內容", "載入要求"], [
            ["model.joblib", "完整 Pipeline、校準器與推論所需物件", "只載入受信任本機產物"],
            ["manifest.json", "版本、資料雜湊、seed、特徵、套件、參數、門檻與 checksum", "schema 版本相容且 checksum 正確"],
            ["feature_schema.json", "95 欄順序、型別、移除欄、補值與觀察範圍", "與 manifest 版本一致"],
            ["metrics.json", "CV、validation 與 final test 指標", "必要指標齊全"],
            ["model_report.html", "資料稽核、選模、曲線、混淆矩陣、限制與解釋", "可離線開啟"],
            ["charts", "PR、校準、混淆矩陣與重要性圖", "報告引用路徑有效"]
        ], widths=[3.6, 8.0, 4.9], font_size=8.4, caption="模型包內容")
        self.add_figure(ASSETS / "fig05_lifecycle.png", "模型建立 啟用與回滾生命週期", width_cm=16.4)
        self.add_heading("9 2 部署拓樸", 2)
        self.add_figure(ASSETS / "fig06_deployment.png", "Windows 本機部署拓樸", width_cm=16.4)
        self.add_para("啟動順序為先檢查 models 目錄及 active.json，再啟動 FastAPI，最後啟動 Streamlit。FastAPI 啟動時完成模型包校驗並只讀載入。若模型損壞或不相容，服務維持健康檢查端點，但評分端點回傳 503。")

    def section10(self):
        self.add_heading("10 非功能與錯誤處理", 1)
        self.add_heading("10 1 非功能需求", 2)
        self.add_table(["類別", "要求", "驗收方法"], [
            ["效能", "10,000 列含驗證、評分、前三項解釋與序列化在 10 秒內完成", "在驗收機記錄 CPU、RAM、套件版本與三次執行結果"],
            ["可重現", "固定 seed 42；保存資料雜湊、切分、參數、門檻與依賴版本", "相同資料與設定重建相同切分及等價指標"],
            ["可用性", "部分列錯誤不阻擋合法列；沒有模型時明確顯示不可評分", "整合測試與 UI 驗收"],
            ["隱私", "不保存上傳檔、結果或特徵值日誌", "暫存清理與日誌內容檢查"],
            ["安全", "localhost only；模型 checksum；路徑與 CSV 公式注入防護", "安全測試清單"],
            ["可維護", "元件責任分離、統一錯誤碼、模型 schema 版本化", "程式審查與介面測試"]
        ], widths=[2.8, 8.5, 5.2], font_size=8.5, caption="非功能需求")
        self.add_heading("10 2 錯誤分類", 2)
        self.add_table(["層級", "範例", "處理"], [
            ["檔案級", "欄位缺漏、未知欄、Bankrupt?、編碼錯誤、空檔", "整批拒絕，不建立部分結果"],
            ["列級", "非數字、無限值、缺值 10 欄以上", "合法列繼續；錯誤列保存狀態與訊息"],
            ["警告級", "補值、out_of_range", "照常評分，在摘要與列級輸出標示"],
            ["模型級", "無 active model、checksum 失敗、schema 不相容", "評分回傳 503；健康端點說明原因"],
            ["系統級", "記憶體不足、未預期例外", "回傳 request_id，清理暫存，不暴露內部資訊"]
        ], widths=[3.0, 7.0, 6.5], caption="錯誤層級與處理")
        self.add_heading("10 3 日誌與資料保留", 2)
        self.add_para("評分日誌只保存時間、request_id、模型版本、總列數、成功數、失敗數、處理耗時與錯誤碼統計。不得保存檔名中的敏感資訊、record_id、完整輸入列、信心分數或 SHAP 貢獻。上傳內容存於隨機暫存目錄，使用 try finally 清理。")
        self.new_page()

    def section11(self):
        self.add_heading("11 測試與驗收設計", 1)
        self.add_heading("11 1 測試層次", 2)
        self.add_table(["層次", "重點", "通過條件"], [
            ["單元測試", "欄名、schema、補值、極端值、欄位移除、指標、排序、分類", "所有邊界案例有明確預期值"],
            ["防洩漏測試", "Imputer、Scaler、selector、權重、校準與門檻", "test set 不出現在任何 fit 或選擇流程"],
            ["模型測試", "固定切分、30 組搜尋、勝出規則、一次性 test", "指標可重現且達 MVP 門檻"],
            ["整合測試", "CLI、模型包、activate、rollback、API、Streamlit", "端到端結果與保存指標一致"],
            ["安全韌性", "檔案超限、路徑穿越、公式注入、損壞模型、暫存清理", "無敏感資訊外洩且錯誤可復原"],
            ["效能測試", "10,000 列同步評分", "完整處理在 10 秒內完成"]
        ], widths=[3.1, 8.0, 5.4], caption="測試層次")
        self.add_heading("11 2 核心驗收案例", 2)
        cases = [
            ["T01", "95 特徵欄名前後含空白", "正規化後接受並依 canonical order 重排"],
            ["T02", "缺少任一特徵或含未知欄", "422 且整批不評分"],
            ["T03", "一列缺 9 欄", "成功評分，使用訓練中位數並產生警告"],
            ["T04", "一列缺 10 欄", "該列失敗，其他合法列成功"],
            ["T05", "有限值超出訓練範圍", "成功評分並標示 out_of_range"],
            ["T06", "調整 threshold", "只改 risk_class，分數、百分位與 Top 10% 不變"],
            ["T07", "分數同分跨越 Top 10%", "依原始列順序穩定取 ceil 10%"],
            ["T08", "勝出模型 AP 差小於 0.01", "選 Logistic Regression"],
            ["T09", "候選模型通過門檻", "仍需人工 activate，active.json 才更新"],
            ["T10", "模型 checksum 失敗", "評分 503，健康端點指出模型不可用"],
            ["T11", "record_id 以等號開頭", "CSV 匯出前轉義，避免公式執行"],
            ["T12", "10,000 列有效資料", "驗證、評分、解釋與序列化在 10 秒內"]
        ]
        self.add_table(["案例", "輸入或條件", "預期結果"], cases, widths=[2.0, 7.1, 7.4], font_size=8.2, caption="核心驗收案例")
        self.add_heading("11 3 文件驗收", 2)
        self.add_bullets([
            "標題階層與靜態目錄一致，頁碼與最終渲染相符。",
            "所有表格有可見邊框、重複表頭及足夠內距，無截字或固定列高。",
            "所有圖以 inline 方式插入，附圖名與替代文字，無浮動物件。",
            "逐頁檢查繁中文字型、圖表清晰度、頁首頁尾與分頁。",
            "文件不得含工具標記、待填內容或未確認的功能承諾。"
        ])

    def section12(self):
        self.add_heading("12 需求追溯矩陣", 1)
        rows = [
            ["F01", "資料稽核", "SchemaValidator AuditLogger model report", "GET model；train", "T01 T02 T03 T04 T05"],
            ["F02", "分層 train validation test", "TrainingOrchestrator SplitManager", "train", "防洩漏與固定切分測試"],
            ["F03", "缺值 縮放 特徵處理 Pipeline", "FeaturePipeline", "train；POST score", "T03 T04 與 Pipeline fit 範圍測試"],
            ["F04", "Logistic Regression 與 XGBoost", "ModelSearchService", "train", "30 組搜尋與勝出規則測試"],
            ["F05", "類別權重", "ModelSearchService", "train", "權重只依訓練折估計"],
            ["F06", "門檻選擇與情境調整", "ThresholdSelector ScoringService", "POST score threshold", "T06 與 F2 門檻測試"],
            ["F07", "分數 百分位 排序", "ScoringService", "POST score", "T06 T07"],
            ["F08", "係數 SHAP 個案解釋", "ExplanationService", "POST score；個案解釋頁", "解釋數量 方向與非因果聲明"],
            ["F09", "批次結果與匯出", "ResponseFormatter ExportService", "POST score format", "部分成功 CSV JSON 一致性 T11"],
            ["F10", "模型版本 指標 限制與報告", "ModelRegistry ReportBuilder", "GET model；model_report.html", "模型包完整性與報告欄位測試"]
        ]
        self.add_table(["需求", "需求摘要", "設計元件", "介面", "驗收"], rows,
                       widths=[1.4, 3.3, 4.1, 3.3, 4.4], font_size=7.9, caption="F01 至 F10 追溯矩陣")
        self.add_heading("12 1 完成定義", 2)
        self.add_para("MVP 完成表示 F01 至 F10 都有可執行元件、介面與測試證據；active model 通過 final test 門檻；Streamlit 與 FastAPI 在 localhost 完成端到端評分；模型包、HTML 報告、下載檔與限制聲明一致。單有模型指標或畫面 Demo 不構成完成。")
        self.add_heading("12 2 設計變更控制", 2)
        self.add_para("任何改變資料欄位、標籤定義、預測窗口、模型家族、外部連線、認證方式或保存政策的需求，都需要更新本 SDD、manifest schema 及追溯矩陣。修改模型門檻或調參空間時，必須建立新模型版本，不得覆寫既有版本。")
        self.new_page()

    def appendix(self):
        self.add_heading("附錄 A 名詞與參考資料", 1)
        self.add_heading("A 1 名詞", 2)
        self.add_table(["名詞", "定義"], [
            ["模型信心分數", "經 Sigmoid 校準後的 0 到 1 輸出；只反映現有標籤定義，不代表指定未來窗口的真實破產機率"],
            ["Average Precision", "PR 曲線的排序品質摘要，作為主要模型選擇指標"],
            ["Recall Top 10 percent", "批次最高 10% 信心分數名單涵蓋的破產樣本比例"],
            ["F2", "Recall 權重高於 Precision 的 F score，用於選擇預設門檻"],
            ["active model", "通過人工 activate 且由線上服務載入的模型版本"],
            ["候選模型包", "完成訓練與評估但尚未被啟用的版本化產物"],
            ["部分成功", "檔案 schema 合法時，合法列評分、錯誤列回傳狀態的批次處理方式"]
        ], widths=[4.2, 12.3], font_size=8.7, caption="主要名詞")
        self.add_heading("A 2 參考資料", 2)
        self.add_bullets([
            "專題小組 10_企業破產預測_AI核心與MVP方案.docx 2026。",
            "bankrupt_data.csv 現有訓練資料 6,819 列與 96 欄。",
            "IEEE 1016 Systems and Software Engineering Design Description。",
            "scikit-learn User Guide Model selection Calibration Pipeline and Metrics。",
            "XGBoost Documentation Python Package。",
            "SHAP Documentation LinearExplainer and TreeExplainer。"
        ])
        self.add_heading("A 3 文件限制", 2)
        self.add_para("本文件描述 MVP 的軟體設計，不宣稱模型已完成訓練或達成效能門檻。實作完成後，最終指標、硬體量測與模型版本應由 model_report.html、metrics.json 與驗收報告提供。")

    def build(self):
        self.cover()
        self.control_and_toc()
        self.section1()
        self.section2()
        self.section3()
        self.section4()
        self.section5()
        self.section6()
        self.section7()
        self.section8()
        self.section9()
        self.section10()
        self.section11()
        self.section12()
        self.appendix()
        self.doc.save(OUTPUT)
        return OUTPUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toc-json")
    args = parser.parse_args()
    make_figures()
    toc_pages = {}
    if args.toc_json:
        toc_pages = json.loads(Path(args.toc_json).read_text(encoding="utf-8"))
    out = SDDBuilder(toc_pages=toc_pages).build()
    print(out)


if __name__ == "__main__":
    main()
