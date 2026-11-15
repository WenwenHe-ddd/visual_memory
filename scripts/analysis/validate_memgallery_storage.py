"""Independently recount exported storage and read all official SimpleMem DBs."""
import csv
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/memgallery_storage_20261103'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-only',action='store_true',help='Recount files and validate the workbook without reopening unchanged SimpleMem databases.')
    args=parser.parse_args()
    if not args.storage_only:
        sys.path.insert(0,str(ROOT/'.vendor/simplemem_official_runtime'))
        import lancedb
    audit=json.loads((OUT/'storage_audit.json').read_text())
    summary=json.loads((OUT/'storage_summary.json').read_text())
    database_checks=[]
    for i,row in enumerate([] if args.storage_only else audit['simplemem_databases'],1):
        table=lancedb.connect(row['path']).open_table('memory_entries')
        count=table.count_rows();assert count==row['entries'],row['path']
        vector=table.schema.field('vector');assert vector.type.list_size==384
        ids=table.to_arrow().column('entry_id').to_pylist()
        assert len(ids)==len(set(ids)) and set(ids)==set(row['entry_ids'])
        database_checks.append(dict(model=row['model'],task=row['task'],entries=count,
            vector_type=str(vector.type),source_ids_match_snapshot=True))
        if i%20==0:print('Verified DBs',i,flush=True)
    if not args.storage_only:
        (OUT/'simplemem_database_validation.json').write_text(json.dumps(dict(status='passed',databases=database_checks),indent=2)+'\n')
    verified={}
    for path,expected in audit['source_and_measured_files'].items():
        p=Path(path)
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected['sha256'] and p.stat().st_size==expected['bytes'],path
        verified[path]=p.stat().st_size
    totals=defaultdict(int);seen=defaultdict(set)
    with (OUT/'storage_files.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key=(row['model'],row['method']);assert row['path'] not in seen[key]
            seen[key].add(row['path']);assert int(row['bytes'])==verified[row['path']]
            totals[key]+=verified[row['path']]
    book=load_workbook(next(OUT.glob('*.xlsx')),data_only=True)
    sheet=book['Storage汇总'];models=[sheet.cell(1,i).value for i in range(2,5)]
    rows={(r['model'],r['method']):r for r in summary['rows']}
    assert len(rows)==len(totals)==24
    for key,r in rows.items():assert totals[key]==r['storage_bytes']
    for i in range(2,sheet.max_row+1):
        method=sheet.cell(i,1).value
        for j,model in enumerate(models,2):assert abs(sheet.cell(i,j).value-totals[model,method]/1048576)<1e-10
    proof=dict(status='passed',combinations=len(rows),unique_files_rehashed=len(verified),
        storage_csv_independently_summed=True,workbook_values_match_bytes=True,
        simplemem_databases_read=len(database_checks),
        simplemem_database_read_skipped=args.storage_only,
        prior_database_validation=str(OUT/'simplemem_database_validation.json') if args.storage_only else None,
        simplemem_entries_by_model={m:sum(r['entries'] for r in database_checks if r['model']==m) for m in models} if database_checks else None,
        source_files_unchanged=True)
    (OUT/'independent_validation.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(proof,ensure_ascii=False))

if __name__=='__main__':main()
