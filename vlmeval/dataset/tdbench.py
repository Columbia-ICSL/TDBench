"""TDBench (https://github.com/Columbia-ICSL/TDBench): VLMs on top-down images, evaluated with RotationalEval.

    TDBench              9 multiple-choice categories, each question under 4 rotations (7,200 rows)
    TDBench_Grounding    visual grounding, each question under 4 rotations (800 rows)
    TDBench_CS_*         case studies (zoom, height, integrity, depth)

Rows of TDBench and TDBench_Grounding carry `qid` (the question) and `rotation` (degrees). VE is the accuracy at
0 degrees; RE counts a question as correct only if all four rotations are answered correctly. When both datasets
are evaluated in one run, the 10-category table is written to <model>_TDBench_main_REresult.csv.
"""
import os
import os.path as osp

import pandas as pd
from tabulate import tabulate

from ..smp import LMUDataRoot, decode_base64_to_image_file, dump, get_intermediate_file_path, load, read_ok
from .image_mcq import ImageMCQDataset
from .image_vqa import ImageVQADataset
from .utils.tdbench import evaluate_bbox, extract_bbox_from_string

RESULT_SUFFIX = '_REresult'
PAPER_TABLE = 'TDBench_main'  # 9 multiple-choice categories + visual grounding


def load_local_tsv(dataset_name):
    path = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    if not osp.exists(path):
        raise FileNotFoundError(f'{path} not found; build it with `tdbench build` '
                                '(see https://github.com/Columbia-ICSL/TDBench#data)')
    return load(path)


def dump_jpeg(self, line):
    """Cache images as JPEG, as in the paper's experiments (newer VLMEvalKit versions cache PNG)."""
    os.makedirs(self.img_root, exist_ok=True)
    tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
    if not read_ok(tgt_path):
        decode_base64_to_image_file(line['image'], tgt_path)
    return [tgt_path]


def rotational_table(data):
    """Per-category VE, RE, accuracy averaged over rotations and share of questions missed under all four
    rotations. `data` needs columns qid, rotation, category, hit (1 = correct)."""
    data = data.assign(hit=data['hit'].astype(float), rotation=data['rotation'].astype(int))
    per_q = data.groupby(['category', 'qid']).agg(
        all_hit=('hit', lambda x: float((x == 1).all())), all_miss=('hit', lambda x: float((x == 0).all())))
    table = pd.DataFrame({
        'total_questions': per_q.groupby('category').size(),
        'VE': data[data['rotation'] == 0].groupby('category')['hit'].mean(),
        'RE': per_q.groupby('category')['all_hit'].mean(),
        'VE_mean_rot': data.groupby('category')['hit'].mean(),
        'miss_all': per_q.groupby('category')['all_miss'].mean(),
    })
    return with_average(table)


def with_average(table):
    table = table.drop(index='Average', errors='ignore')
    avg = table.mean()
    avg['total_questions'] = table['total_questions'].sum()
    table.loc['Average'] = avg
    table['total_questions'] = table['total_questions'].astype(int)
    return table.round(3)


def write_result(table, eval_file, dataset_name):
    path = get_intermediate_file_path(eval_file, RESULT_SUFFIX, 'csv')
    dump(table.reset_index().rename(columns={'index': 'category'}), path)
    print(f'{dataset_name} (VE: 0 degrees, RE: all four rotations correct)')
    print(tabulate(table, headers='keys'))
    combine_paper_table(eval_file, dataset_name)


def combine_paper_table(eval_file, dataset_name):
    """Merge the TDBench and TDBench_Grounding results of the same run into the 10-category table."""
    other = {'TDBench': 'TDBench_Grounding', 'TDBench_Grounding': 'TDBench'}[dataset_name]
    sibling = eval_file.replace(f'_{dataset_name}.', f'_{other}.')
    paths = [get_intermediate_file_path(f, RESULT_SUFFIX, 'csv') for f in (eval_file, sibling)]
    if sibling == eval_file or not all(osp.exists(p) for p in paths):
        return
    parts = [load(p).set_index('category').drop(index='Average', errors='ignore') for p in paths]
    table = with_average(pd.concat(parts).sort_index())
    out = get_intermediate_file_path(eval_file.replace(f'_{dataset_name}.', f'_{PAPER_TABLE}.'), RESULT_SUFFIX, 'csv')
    dump(table.reset_index().rename(columns={'index': 'category'}), out)
    print(f'TDBench, 10 categories -> {out}')
    print(tabulate(table[['total_questions', 'VE', 'RE']], headers='keys'))


class TDBench(ImageMCQDataset):
    """TDBench multiple-choice questions (main benchmark and case studies)."""

    DATASET_URL = {
        'TDBench': None,
        'TDBench_CS_Zoom': None,
        'TDBench_CS_Height': None,
        'TDBench_CS_Integrity': None,
        'TDBench_CS_Depth': None,
    }
    DATASET_MD5 = {}
    DEFAULT_JUDGE_MODEL = 'gpt-4o'
    dump_image = dump_jpeg

    def prepare_tsv(self, url, file_md5=None):
        self.data_path = osp.join(LMUDataRoot(), f'{self.dataset_name}.tsv')
        return load_local_tsv(self.dataset_name)

    def evaluate(self, eval_file, **judge_kwargs):
        acc = super().evaluate(eval_file, **judge_kwargs)
        if self.dataset_name != 'TDBench':
            return acc  # case studies use a single orientation
        model = judge_kwargs.get('model', 'exact_matching')
        name_str = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}.get(model, model)
        judged = load(get_intermediate_file_path(eval_file, f'_{name_str}_result'))
        meta = self.data.set_index('index')
        judged['qid'] = [meta.at[i, 'qid'] for i in judged['index']]
        judged['rotation'] = [meta.at[i, 'rotation'] for i in judged['index']]
        table = rotational_table(judged)
        write_result(table, eval_file, self.dataset_name)
        return table


class TDBenchGrounding(ImageVQADataset):
    """TDBench visual grounding, scored by whether the predicted box centre falls in the ground-truth box."""

    DATASET_URL = {'TDBench_Grounding': None}
    DATASET_MD5 = {}
    DEFAULT_JUDGE_MODEL = 'centroid'
    dump_image = dump_jpeg

    def prepare_tsv(self, url, file_md5=None):
        self.data_path = osp.join(LMUDataRoot(), f'{self.dataset_name}.tsv')
        return load_local_tsv(self.dataset_name)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        obj = line['question']
        question = f'\nPlease output the coordinates of the {obj} in the image in the format [x1, y1, x2, y2]. Do not include any additional text. Respond with relative coordinates between 0 and 1, with top left corner (0, 0), top right (1, 0) and bottom right (1, 1).'  # noqa: E501
        msgs = [dict(type='image', value=p) for p in self.dump_image(line)]
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        # --judge may name the multiple-choice answer extractor; grounding is scored without an LLM
        method = 'iou' if judge_kwargs.get('model') == 'iou' else 'centroid'
        data = load(eval_file).sort_values(by='index')
        meta = self.data.set_index('index')
        scores = []
        for _, row in data.iterrows():
            try:
                pred = extract_bbox_from_string(str(row['prediction']))
                scores.append(evaluate_bbox(pred, extract_bbox_from_string(str(row['answer'])), method))
            except Exception:
                scores.append(0.0)
        data['hit'] = scores
        data['category'] = 'visual_grounding'
        data['qid'] = [meta.at[i, 'qid'] for i in data['index']]
        data['rotation'] = [meta.at[i, 'rotation'] for i in data['index']]
        dump(data, get_intermediate_file_path(eval_file, f'_{method}_result'))
        if method == 'iou':
            table = with_average(pd.DataFrame({'total_questions': [len(data)], 'mean_IoU': [data['hit'].mean()]},
                                              index=['visual_grounding']))
            dump(table.reset_index(), get_intermediate_file_path(eval_file, '_iou', 'csv'))
            return table
        table = rotational_table(data)
        write_result(table, eval_file, self.dataset_name)
        return table
