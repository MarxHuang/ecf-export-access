"""Losslessly expand compressed originals to a separate tree."""
from pathlib import Path
import gzip,hashlib,json
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    manifest=json.loads((ROOT/'provenance/SOURCE_MANIFEST.json').read_text('utf-8'))
    count=0
    for item in manifest['files']:
        if item['encoding']!='gzip':continue
        target=ROOT/'expanded'/Path(item['path']).with_suffix('')
        assert target.resolve().is_relative_to((ROOT/'expanded').resolve())
        raw=gzip.decompress((ROOT/item['path']).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==item['source_sha256']
        if target.exists():
            assert target.read_bytes()==raw,f'Existing expanded file differs: {target}'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
        count+=1
    print(f'Verified {count} expanded files under {ROOT / "expanded"}')
