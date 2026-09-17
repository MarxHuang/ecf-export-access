"""Relocate an existing frozen case; dry-run unless --execute is requested."""
from pathlib import Path
import argparse,json,sys

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'reproduction'
ARCHIVE=WORK/'outputs/ees_transition/archive'
FROZEN=ARCHIVE/'151_FROZEN_SCENARIO_EVALUATION_20260907'

def relocate(value):
    normalized=value.replace('\\','/')
    marker='/r4r_rebuild/'
    if marker in normalized:
        relative=normalized.split(marker,1)[1]
        path=(WORK/relative).resolve()
        if not path.is_relative_to(WORK.resolve()):raise ValueError(value)
        return str(path)
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true')
    p.add_argument('--case',default='primary')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--limit-days',type=int)
    args=p.parse_args()
    protocol=json.loads((FROZEN/'EVALUATION_PROTOCOL.json').read_text('utf-8'))
    tasks={v['id']:v for v in protocol['tasks']}
    if args.list:
        print('\n'.join(tasks));return 0
    if args.case not in tasks:p.error('Unknown case; use --list')
    if args.workers<1:p.error('workers must be positive')
    if args.limit_days is not None and args.limit_days<1:p.error('limit-days must be positive')
    argv=[relocate(v) for v in tasks[args.case]['arguments']]
    output=WORK/'reruns'/args.case
    if args.limit_days:output=output.with_name(output.name+f'_smoke_{args.limit_days}d')
    argv[argv.index('--output-dir')+1]=str(output)
    argv+=['--workers',str(args.workers)]
    if args.limit_days:argv+=['--limit-days',str(args.limit_days)]
    required=[]
    for flag in ['--package','--mapping-file','--dss-root','--protocol','--load-scale-calibration',
                 '--authorization-margin-freeze','--sensitivity-protocol','--eligible-dates-file']:
        if flag in argv:required.append(Path(argv[argv.index(flag)+1]))
    package=Path(argv[argv.index('--package')+1])
    for name in ['AUSGRID_TRAIN_2010_2011.npz','AUSGRID_DEVELOPMENT_2011_2012.npz',
                 'AUSGRID_EVALUATION_2012_2013.npz']:
        required.append(package/name)
    missing=[str(v.relative_to(WORK)) for v in required if not v.exists()]
    print(json.dumps({'case':args.case,'execute':args.execute,'arguments':argv,
                      'missing_inputs':missing,'expected_days':tasks[args.case]['expected_days']},indent=2))
    if not args.execute:return 0
    if missing:p.error('Acquire/build the missing original inputs; see docs/DATA_SOURCES.md')
    if output.exists():p.error('Output already exists; do not overwrite a prior run')
    source=FROZEN/'scientific_source'
    sys.path.insert(0,str(source/'src'))
    sys.path.insert(0,str(source/'tools'))
    import run_ees_australian_external_replay as runner
    runner.REPOSITORY_ROOT=WORK
    runner.SCIENTIFIC_CORE_MODULES['environment_specification']=ARCHIVE/'093_EES_COMPUTATIONAL_ENVIRONMENT.yaml'
    sys.argv=[str(source/'tools/run_ees_australian_external_replay.py')]+argv
    return runner.main()

if __name__=='__main__':raise SystemExit(main())
