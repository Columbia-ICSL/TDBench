"""tdbench command line.

    tdbench build --era ERA_Dataset.zip --sdd semantic_drone_dataset_semantics_v1.1.zip
    tdbench run main --model Qwen2.5-VL-7B-Instruct [run.py options]
"""
import argparse
import os.path as osp
import subprocess
import sys

from . import SPLITS

GROUPS = {
    'main': ['TDBench', 'TDBench_Grounding'],
    'case_study': ['TDBench_CS_Zoom', 'TDBench_CS_Height', 'TDBench_CS_Integrity', 'TDBench_CS_Depth'],
}
GROUPS['all'] = GROUPS['main'] + GROUPS['case_study']


def main(argv=None):
    parser = argparse.ArgumentParser(prog='tdbench')
    sub = parser.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='assemble the benchmark TSVs in $LMUData')
    b.add_argument('--era', help='ERA_Dataset.zip, or the folder it was extracted to')
    b.add_argument('--sdd', help='Semantic Drone Dataset download, or the folder it was extracted to')
    b.add_argument('--release', help='local copy of the Hugging Face release folder (default: download)')
    b.add_argument('--out', help='output folder (default: $LMUData)')
    b.add_argument('--splits', nargs='+', choices=list(SPLITS), help='default: all')

    r = sub.add_parser('run', allow_abbrev=False,
                       help='evaluate models with VLMEvalKit; other options are passed to run.py')
    r.add_argument('group', choices=list(GROUPS))
    r.add_argument('--no-grounding', action='store_true', help='leave out visual grounding (TDBench_Grounding)')

    args, rest = parser.parse_known_args(argv)
    if args.cmd == 'build':
        from .build import build
        build({'era': args.era, 'sdd': args.sdd}, release=args.release, out=args.out, splits=args.splits)
    else:
        data = [d for d in GROUPS[args.group] if not (args.no_grounding and d == 'TDBench_Grounding')]
        run_py = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'run.py')
        sys.exit(subprocess.call([sys.executable, run_py, '--data', *data, *rest]))


if __name__ == '__main__':
    main()
