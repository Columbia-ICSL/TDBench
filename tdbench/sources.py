"""Readers for the ERA and Semantic Drone Dataset downloads (the zip as downloaded, or its extracted folder)."""
import io
import os
import re
import zipfile
from functools import lru_cache

from PIL import Image


class _Release:
    marker = None  # a path that identifies the release root

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None
        self.root = self._find_root()

    def _find_root(self):
        if self.zip:
            for name in self.zip.namelist():
                head, sep, _ = name.partition(self.marker)
                if sep:
                    return head
        else:
            for dirpath, dirnames, _ in os.walk(self.path):
                if os.path.basename(self.marker.rstrip('/')) in dirnames:
                    return os.path.relpath(dirpath, self.path)
        raise ValueError(f'{self.path} is not the {self.name} download (no {self.marker!r} found)')

    def read(self, file):
        if self.zip:
            return self.zip.read(self.root + file)
        return open(os.path.join(self.path, self.root, file), 'rb').read()

    def open(self, file):
        return self._open(file)


class ERA(_Release):
    """ERA (Mou et al., 2020): ERA_Dataset.zip from https://lcmou.github.io/ERA_Dataset/."""
    name = 'ERA'
    marker = 'SingleFrames/'

    @lru_cache(maxsize=64)
    def _open(self, file):
        # TDBench used quality-80 JPEG copies of the frames
        png = Image.open(io.BytesIO(self.read(file))).convert('RGB')
        buf = io.BytesIO()
        png.save(buf, format='JPEG', quality=80)
        return Image.open(io.BytesIO(buf.getvalue()))


class SemanticDrone(_Release):
    """Semantic Drone Dataset (TU Graz), https://ivc.tugraz.at/research-project/semantic-drone-dataset/."""
    name = 'Semantic Drone Dataset'
    image = re.compile(r'\d{3}\.jpg')

    def _find_root(self):
        if self.zip:
            names = self.zip.namelist()
        else:
            names = [os.path.relpath(os.path.join(d, f), self.path) for d, _, fs in os.walk(self.path) for f in fs]
        roots = {n[:-7] for n in names if self.image.fullmatch(os.path.basename(n)) and n[:-7].count('label') == 0}
        if len(roots) != 1:
            raise ValueError(f'{self.path} is not the {self.name} download (expected one folder of NNN.jpg images)')
        root = roots.pop()
        return root if self.zip else root.rstrip('/') or '.'

    @lru_cache(maxsize=4)  # 6000x4000 images; recipes are rendered grouped by file
    def _open(self, file):
        return Image.open(io.BytesIO(self.read(file)))


SOURCES = {'era': ERA, 'sdd': SemanticDrone}
