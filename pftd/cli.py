import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='PFTD core implementation')
    sub = parser.add_subparsers(dest='command', required=True)
    compile_cmd = sub.add_parser('compile', help='compile curated training roots into separated public/private assets')
    compile_cmd.add_argument('--roots', required=True)
    compile_cmd.add_argument('--output', required=True)
    train_cmd = sub.add_parser('train', help='single-GPU assembly; requires original adapters, curated roots and tool setup')
    train_cmd.add_argument('--roots', required=True)
    train_cmd.add_argument('--compiled', required=True)
    train_cmd.add_argument('--adapter', required=True)
    train_cmd.add_argument('--output', required=True)
    train_cmd.add_argument('--budget', type=int, default=2_000_000)
    train_cmd.add_argument('--sampler', choices=['vllm', 'transformers'], default='vllm')
    train_cmd.add_argument('--tools', choices=['none', 'wikipedia'], default='wikipedia')
    train_cmd.add_argument('--variant', choices=['full', 'no_merge', 'fixed_schedule', 'outcome_only'], default='full')
    train_cmd.add_argument('--reward', choices=['joint', 'local', 'downstream'], default='joint')
    sub.add_parser('config', help='print paper settings')
    args = parser.parse_args()
    if args.command == 'compile':
        from pftd.compiler import Compiler, save_units
        from pftd.sandbox import Sandbox
        roots = [json.loads(line) for line in Path(args.roots).read_text(encoding='utf-8').splitlines() if line.strip()]
        units, rejected = Compiler(Sandbox()).compile_roots(roots)
        save_units(units, args.output)
        Path(args.output, 'compilation_report.json').write_text(json.dumps({'accepted_units': len(units), 'rejected': rejected}, indent=2), encoding='utf-8')
        print(f'Compiled {len(units)} verified units; rejected {len(rejected)} candidate spans. All training roots retain full-task eligibility.')
    elif args.command == 'train':
        from pftd.train_entry import train
        print(json.dumps(train(args), indent=2))
    else:
        print((Path(__file__).resolve().parents[1] / 'configs' / 'pftd.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
