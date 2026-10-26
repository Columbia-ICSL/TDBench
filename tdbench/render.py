"""Render TDBench tiles from rebuild recipes.

Recipe fields:
    pipeline     'main', 'grounding' or 'case_study'
    rotation     counter-clockwise rotation in degrees (main, grounding)
    patches      list of {source, file, crop: [x1, y1, x2, y2] (relative), size: [w, h], pos: [x, y]}
    canvas       optional base64 PNG holding the hosted part of a composed tile
    divider      whether the divider of dynamic-temporal tiles is drawn
    encode_base  case_study: whether the base image was stored as JPEG before cropping
    box          case_study: pixel box [left, top, right, bottom] cropped from the base image
"""
import base64
import io
import json

import numpy as np
from PIL import Image

TILE = 512


def to_jpeg(img, **kwargs):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', **kwargs)
    return buf.getvalue()


def from_jpeg(data):
    return Image.open(io.BytesIO(data))


def _divider():
    d = np.zeros((6, TILE, 3), dtype=np.uint8)
    d[3:] = 255
    return Image.fromarray(d)


def _patch(p, sources):
    im = sources[p['source']].open(p['file'])
    if 'crop' in p:
        x1, y1, x2, y2 = p['crop']
        im = im.crop((x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height))
    if 'size' in p:
        im = im.resize(tuple(p['size']))
    return im, tuple(p.get('pos', (0, 0)))


def compose(recipe, sources):
    patches = [_patch(p, sources) for p in recipe['patches']]
    if recipe.get('canvas') is None and len(patches) == 1 and patches[0][0].size == (TILE, TILE):
        return patches[0][0]
    if recipe.get('canvas'):
        img = Image.open(io.BytesIO(base64.b64decode(recipe['canvas']))).convert('RGB')
    else:
        img = Image.new('RGB', (TILE, TILE))
    for im, pos in patches:
        img.paste(im, pos)
    if recipe.get('divider'):
        img.paste(_divider(), (0, TILE // 2 - 2))
    return img


_base_cache = {}


def _base(recipe, sources):
    # case-study tiles that share a base image are rendered one after another
    key = (id(sources), json.dumps(recipe['patches'][0], sort_keys=True), bool(recipe.get('encode_base')))
    if key not in _base_cache:
        _base_cache.clear()
        img, _ = _patch(recipe['patches'][0], sources)
        _base_cache[key] = from_jpeg(to_jpeg(img)) if recipe.get('encode_base') else img
    return _base_cache[key]


def render(recipe, sources):
    """JPEG bytes of the tile described by ``recipe``."""
    pipeline = recipe['pipeline']
    if pipeline == 'case_study':
        img = from_jpeg(to_jpeg(_base(recipe, sources).crop(tuple(recipe['box'])), quality=95))
        return to_jpeg(img.resize((TILE, TILE)))
    img = compose(recipe, sources)
    if pipeline == 'main':
        img = from_jpeg(to_jpeg(img)).rotate(recipe['rotation'], expand=True)
    elif pipeline == 'grounding':
        img = img.rotate(recipe['rotation'])
    else:
        raise ValueError(f'unknown pipeline {pipeline!r}')
    return to_jpeg(img)
