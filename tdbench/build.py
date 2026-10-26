"""Assemble the TDBench TSVs in $LMUData from the Hugging Face release and the ERA and Semantic Drone downloads."""
import base64
import json
import os
import os.path as osp
from collections import defaultdict

import pandas as pd

from . import HF_REPO, HF_REVISION, SPLITS
from .render import render
from .sources import SOURCES

RELEASE_DIR = 'tdbench'


def lmudata_root():
    root = os.environ.get('LMUData', osp.expanduser('~/LMUData'))
    os.makedirs(root, exist_ok=True)
    return root


def fetch_release(revision=HF_REVISION):
    from huggingface_hub import snapshot_download
    path = snapshot_download(HF_REPO, repo_type='dataset', revision=revision, allow_patterns=[f'{RELEASE_DIR}/*'])
    return osp.join(path, RELEASE_DIR)


def _order(recipe):
    # recipes that share a source image are rendered one after another
    p = recipe['patches'][0]
    return p['file'], json.dumps(p, sort_keys=True), bool(recipe.get('encode_base')), recipe['index']


def build(archives, release=None, out=None, splits=None, revision=HF_REVISION):
    """archives: {'era': path, 'sdd': path}; release: local copy of the HF release folder."""
    release = release or fetch_release(revision)
    out = out or lmudata_root()
    recipes = defaultdict(list)
    with open(osp.join(release, 'rebuild.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            recipes[r['split']].append(r)

    splits = splits or list(SPLITS)
    needed = {p['source'] for s in splits for r in recipes[s] for p in r['patches']}
    missing = sorted(needed - {k for k, v in archives.items() if v})
    if missing:
        raise SystemExit('missing ' + ', '.join(f'--{m}' for m in missing) + '; see the README for the downloads')
    sources = {k: SOURCES[k](v) for k, v in archives.items() if v and k in needed}

    for split in splits:
        df = pd.read_csv(osp.join(release, f'{split}.tsv'), sep='\t', dtype=str, keep_default_na=False)
        row = {idx: i for i, idx in enumerate(df['index'])}
        for r in sorted(recipes[split], key=_order):
            df.at[row[str(r['index'])], 'image'] = base64.b64encode(render(r, sources)).decode()
        path = osp.join(out, f'{SPLITS[split]}.tsv')
        df.to_csv(path, sep='\t', index=False)
        print(f'{SPLITS[split]:<22} {len(df):>5} rows -> {path}')
