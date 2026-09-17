"""Check deposited bytes and saved-result arithmetic without running simulations."""
from pathlib import Path
import argparse, csv, gzip, hashlib, io, json, math

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'reproduction/outputs/ees_transition/archive'

def read_bytes(path):
    path=Path(path)
    if path.is_file(): return path.read_bytes()
    return gzip.decompress(path.with_name(path.name+'.gz').read_bytes())

def read_json(path):
    return json.loads(read_bytes(path).decode('utf-8-sig'))

def rows(path):
    return list(csv.DictReader(io.StringIO(read_bytes(path).decode('utf-8-sig'))))

def close(a,b,atol=1e-8):
    if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=atol):
        raise AssertionError(f'{a} != {b}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    manifest=read_json(ROOT/'provenance/SOURCE_MANIFEST.json')
    for item in manifest['files']:
        path=ROOT/item['path']
        assert path.resolve().is_relative_to(ROOT.resolve()),item['path']
        raw=path.read_bytes()
        assert len(raw)==item['bytes'],item['path']
        assert hashlib.sha256(raw).hexdigest()==item['sha256'],item['path']
        original=gzip.decompress(raw) if item['encoding']=='gzip' else raw
        assert len(original)==item['source_bytes'],item['path']
        assert hashlib.sha256(original).hexdigest()==item['source_sha256'],item['path']
    family=ARCHIVE/'151_FROZEN_SCENARIO_EVALUATION_20260907'
    cases=read_json(family/'VERIFIED_RESULTS.json')
    count=0
    for case,expected in cases['results'].items():
        directory=family/'runs'/case
        summary=read_json(directory/'AU_EXTERNAL_RUN_SUMMARY.json')
        daily=rows(directory/'AU_EXTERNAL_DAY_SUMMARY.csv')
        intervals=rows(directory/'AU_EXTERNAL_INTERVAL_RESULTS.csv')
        assert len(intervals)==expected['records'],case
        count+=len(intervals)
        for control,totals in summary['aggregate'].items():
            selected=[r for r in daily if r['control']==control]
            assert len(selected)==totals['trajectory_count'],(case,control)
            for key in ['delivered_export_mwh','authorized_export_mwh','idle_authorization_mwh',
                        'available_export_mwh','curtailed_available_export_mwh']:
                close(math.fsum(float(r[key]) for r in selected),totals[key])
            close(totals['authorized_export_mwh']-totals['delivered_export_mwh'],
                  totals['idle_authorization_mwh'])
            close(totals['available_export_mwh']-totals['delivered_export_mwh'],
                  totals['curtailed_available_export_mwh'])
    assert count==cases['records']
    primary=read_json(family/'runs/primary/AU_EXTERNAL_RUN_SUMMARY.json')['aggregate']
    ecf=primary['ECF']
    comparisons={}
    for r in rows(ROOT/'figure_data/australian_main/Fig9b_energy.csv'):
        comparator=primary[r['comparator']]
        mapping={'delivery_MWh':'delivered_export_mwh',
                 'authorization_MWh':'authorized_export_mwh',
                 'unused_authorization_MWh':'idle_authorization_mwh'}
        comparisons[r['comparator']]={}
        for plotted,key in mapping.items():
            difference=ecf[key]-comparator[key]
            close(difference,r[plotted])
            comparisons[r['comparator']][plotted]=difference
    report={'passed':True,'files_verified':len(manifest['files']),
            'australian_cases':len(cases['results']),'australian_interval_rows':count,
            'primary_days':ecf['trajectory_count'],'primary_comparisons':comparisons,
            'primary_qualification_count':ecf['gate_count'],
            'primary_assessed_intervals':ecf['trajectory_count']*48,
            'scope':'Saved-file integrity and aggregate arithmetic; no new simulation or AC solution.'}
    print(json.dumps(report,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
