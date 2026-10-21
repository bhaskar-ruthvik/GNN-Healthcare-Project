"""Read-only extraction of the source's existing diagnosis/category assignments.
No patient identifiers are read into the mapping. Run from Implementation.
"""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import json
from pathlib import Path
import zipfile
from lxml import etree as ET

ROOT = workspace()
SOURCE = ROOT / 'data/Base_DataSet _version-09-30-2025_Bhaskar.xlsx'
OUT = ROOT / 'prepared_data/three_node_v1/disease_mapping_PRIVATE.json'
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    assignments = set()
    wanted = set()
    def token(cell):
        v = cell.find(NS+'v')
        if cell.get('t') == 'inlineStr':
            return ('v', ''.join(cell.itertext()))
        return ('s' if cell.get('t') == 's' else 'n', v.text if v is not None else '')
    with zipfile.ZipFile(SOURCE) as z:
        with z.open('xl/worksheets/sheet4.xml') as f:
            for _, row in ET.iterparse(f, events=('end',), tag=NS+'row'):
                vals = {c.get('r','').rstrip('0123456789'): token(c) for c in row
                        if c.get('r','').rstrip('0123456789') in {'E','CO','DM'}}
                if row.get('r') != '1':
                    entry = tuple(vals.get(k, ('v','')) for k in ('CO','E','DM'))
                    assignments.add(entry)
                    wanted.update(int(t[1]) for t in entry if t[0]=='s')
                row.clear()
                while row.getprevious() is not None:
                    del row.getparent()[0]
        strings = {}
        with z.open('xl/sharedStrings.xml') as f:
            for i, (_, el) in enumerate(ET.iterparse(f, events=('end',), tag=NS+'si')):
                if i in wanted:
                    strings[i] = ''.join(el.itertext())
                el.clear()
                while el.getprevious() is not None:
                    del el.getparent()[0]
                if len(strings) == len(wanted):
                    break
    def decode(t):
        if t[0] == 'n' and t[1]:
            # Match the notebook's Excel numeric-cell conversion before string dtype.
            x = float(t[1])
            return str(int(x)) if x.is_integer() else str(x)
        return (strings[int(t[1])] if t[0]=='s' else t[1]).strip()
    mapping = {}
    for l,d,c in assignments:
        if decode(l) != '1':
            continue
        dx, category = decode(d), decode(c)
        if not dx:
            continue
        if dx in mapping and mapping[dx] != category:
            raise ValueError('Conflicting diagnosis-to-category assignments')
        mapping[dx] = category
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(json.dumps({'diagnosis_assignments':len(mapping), 'nonblank_categories':len(set(mapping.values())-{''})}))

if __name__ == '__main__':
    main()
