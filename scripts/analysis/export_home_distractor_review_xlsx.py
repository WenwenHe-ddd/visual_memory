"""Export the frozen Home candidate queue as an editable Excel review workbook."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/home_all_questions_distractor_review_20261031"
OUTPUT = SOURCE / "Home_52题_干扰图人工复核.xlsx"
MIN_ORDER = 1


def main() -> None:
    with (SOURCE / "review_candidates.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = [row for row in csv.DictReader(handle)
                   if int(row["add_order"]) >= MIN_ORDER]
    if not records or len(records) % 104:
        raise ValueError("Expected equal candidate counts for both groups of 52 questions")
    candidates_per_group = len(records) // 104
    wb = Workbook()
    summary = wb.active
    summary.title = "题目总览"
    review = wb.create_sheet("候选复核")
    info = wb.create_sheet("复核说明")

    headers = ["题目ID", "组别", "加入顺序", "原问题", "正确记忆ID", "候选记忆ID",
               "CLIP图像相似度", "字面重合词提示", "候选原图", "图文复核页", "候选caption",
               "候选用户对话", "候选助手对话", "旧试点已通过", "人工接受", "共有问题线索",
               "可能也是有效证据", "caption与图一致", "备注"]
    review.append(headers)
    for item in records:
        page_name = item["question_id"].split("/")[-1] + ".html"
        review.append([
            item["question_id"], item["condition"], int(item["add_order"]),
            item["question"], item["gold_memory_ids"], item["candidate_memory_id"],
            float(item["clip_visual_cosine_to_nearest_gold"]),
            item["exact_question_word_overlap_hint"], item["candidate_image_path"],
            f"questions/{page_name}", item["candidate_caption"],
            item["candidate_user_dialogue"], item["candidate_assistant_dialogue"],
            item["previous_small_pilot_passed"], "", "", "", "", "",
        ])
        cell = review.cell(review.max_row, 10)
        cell.hyperlink = f"questions/{page_name}"
        cell.style = "Hyperlink"
        review.cell(review.max_row, 7).number_format = "0.0000"
    review.freeze_panes = "D2"
    review.auto_filter.ref = f"A1:S{review.max_row}"
    widths = {"A": 42, "B": 15, "C": 12, "D": 75, "E": 60, "F": 47,
              "G": 18, "H": 25, "I": 85, "J": 22, "K": 90, "L": 75,
              "M": 75, "N": 16, "O": 16, "P": 55, "Q": 23,
              "R": 21, "S": 65}
    for column, width in widths.items():
        review.column_dimensions[column].width = width
    for row in review.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        review.row_dimensions[row[0].row].height = 42
    for column in ("O", "Q", "R"):
        validation = DataValidation(type="list", formula1='"是,否,待定"', allow_blank=True)
        validation.error = "请选择 是、否 或 待定"
        validation.errorTitle = "无效复核值"
        validation.showErrorMessage = True
        review.add_data_validation(validation)
        validation.add(f"{column}2:{column}{review.max_row}")
    for cell in review[1]:
        cell.fill = PatternFill("solid", fgColor="254E70")
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    review.row_dimensions[1].height = 36

    summary.append(["题目ID", "相似候选", "无关候选", "相似已接受", "无关已接受", "图文复核页"])
    question_ids = list(dict.fromkeys(row["question_id"] for row in records))
    last_row = review.max_row
    for index, qid in enumerate(question_ids, 2):
        summary.append([
            qid,
            f'=COUNTIFS(候选复核!$A$2:$A${last_row},A{index},候选复核!$B$2:$B${last_row},"similar")',
            f'=COUNTIFS(候选复核!$A$2:$A${last_row},A{index},候选复核!$B$2:$B${last_row},"unrelated")',
            f'=COUNTIFS(候选复核!$A$2:$A${last_row},A{index},候选复核!$B$2:$B${last_row},"similar",候选复核!$O$2:$O${last_row},"是")',
            f'=COUNTIFS(候选复核!$A$2:$A${last_row},A{index},候选复核!$B$2:$B${last_row},"unrelated",候选复核!$O$2:$O${last_row},"是")',
            f"questions/{qid.split('/')[-1]}.html",
        ])
        summary.cell(index, 6).hyperlink = f"questions/{qid.split('/')[-1]}.html"
        summary.cell(index, 6).style = "Hyperlink"
    summary.freeze_panes = "B2"
    summary.auto_filter.ref = f"A1:F{summary.max_row}"
    for column, width in {"A": 44, "B": 17, "C": 17, "D": 19, "E": 19, "F": 25}.items():
        summary.column_dimensions[column].width = width
    for cell in summary[1]:
        cell.fill = PatternFill("solid", fgColor="254E70")
        cell.font = Font(color="FFFFFF", bold=True)

    instructions = [
        f"本工作簿覆盖 Home 领域全部 52 道题，每题两组各 {candidates_per_group} 张候选图；原始候选顺序从 {MIN_ORDER} 开始。",
        "请打开题目总览中的图文复核页，查看目标图、候选原图、原 caption 和对话。",
        "similar 和 unrelated 分组由目标图间 CLIP 图像相似度定义。最终加入顺序另按问题对候选的文本或图像相似度排序。",
        "候选复核的 O 列填写人工接受；P 列写共享的问题定位线索；Q 列判断候选图是否可能也是题目的有效证据；R 列判断 caption 是否忠实于原图。",
        "若候选可能也是有效证据，请在 O 列填否；未通过的候选需要替换并重新计算相应曲线。",
        "旧试点通过只表示该候选曾被整体确认可用；不自动填入本次 O 列。",
        "此工作簿不含答案与选项，也尚未运行答题测试。",
    ]
    info.column_dimensions["A"].width = 120
    for line in instructions:
        info.append([line])
        info.cell(info.max_row, 1).alignment = Alignment(wrap_text=True, vertical="top")
        info.row_dimensions[info.max_row].height = 35
    info["A1"].font = Font(bold=True, color="254E70")
    wb.save(OUTPUT)
    print(f"{OUTPUT} · {len(question_ids)} questions · {len(records)} candidates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--min-order", type=int, default=MIN_ORDER)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    MIN_ORDER = args.min_order
    OUTPUT = args.output.resolve() if args.output else SOURCE / "Home_52题_干扰图人工复核.xlsx"
    main()
