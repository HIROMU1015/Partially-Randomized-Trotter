"""Finalize this static audit and validate transfer/provenance only (stdlib)."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/AbeHiromu/projects/partially-randomized-trotter')

def read(name):return json.loads((ROOT/name).read_text())
def csvread(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def write(name,x):(ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

manifest=read('acquisition_manifest.json')
bykey={(r['commit'],r['path']):r for r in manifest}
hashes=[]
for r in manifest:
    copybytes=(ROOT/r['local_path']).read_bytes()
    objectbytes=subprocess.check_output(['git','-C',str(REPO),'show',r['commit']+':'+r['path']])
    assert copybytes==objectbytes
    assert hashlib.sha256(copybytes).hexdigest()==r['sha256']
    assert hashlib.sha1(b'blob '+str(len(copybytes)).encode()+b'\0'+copybytes).hexdigest()==r['git_blob_sha1']
    hashes.append({'commit':r['commit'],'path':r['path'],'copy_and_fixed_blob_unchanged':True,'sha256':r['sha256']})

remote=[]
for r in read('github_receipts.json'):
    local=bykey[r['commit'],r['path']]
    assert r['remote_ok'] and local['git_blob_sha1']==r['git_blob_sha1']
    remote.append(dict(commit=r['commit'],path=r['path'],blob_SHA_match=True,remote_content_partial=r['content_partial']))
linked=read('linked_input_hash_checks.json');assert all(r['hash_match'] for r in linked)
shot=read('shot_static_checks.json');assert len(shot)==3 and all(r['AST_equal'] for r in shot)

rows=csvread('evidence_rows_75.csv');matrix=csvread('comparison_ABC_15.csv');aux=csvread('B0_B0I_auxiliary_60.csv')
assert len(rows)==75 and len(matrix)==15 and len(aux)==60
assert len({(r['comparison'],r['system'],r['epsilon'],r['formula']) for r in rows})==75
assert sum(r['cost_exact']=='null' for r in aux)==16
assert all(r['eligibility']=='ELIGIBLE' for r in rows)
transfer=[]
for r in rows:
    p=ROOT/'sources'/r['result_commit']/r['result_path']
    if p.suffix=='.csv':
        with p.open(newline='') as f:rr=list(csv.DictReader(f))
        n=int(re.search(r'CSV data row (\d+)',r['record_locator']).group(1))-1
        original=rr[n];literal=original['replacement_G_exact']
    else:
        original=json.loads(p.read_text())
        for part in r['record_locator'].strip('/').split('/'):
            original=original[int(part)] if isinstance(original,list) else original[part]
        literal=original.get('G_RZ_exact')
        if literal is None:
            v=original['total_RZ_exact'];literal=str(v['numerator'])+'/'+str(v['denominator'])
    assert Fraction(literal)==Fraction(r['saved_cost_exact'])
    assert original.get('cell_id',original.get('candidate_id'))==r['candidate_id']
    if 'eligible' in original:assert original['eligible']
    transfer.append(dict(system=r['system'],epsilon=r['epsilon'],formula=r['formula'],comparison=r['comparison'],original_saved_total_identical=True,candidate_id=r['candidate_id']))

private=read('private_metadata_audit.json')
for key,hashkey in [('old_spec_locator_from_public_binding','old_spec_sha256'),('old_runtime_manifest_path','old_runtime_manifest_sha256')]:
    assert hashlib.sha256(Path(private[key]).read_bytes()).hexdigest()==private[hashkey]
assert all(r['match'] for r in private['provider_file_comparisons'])

roles={
'5d536985':'旧7系列図・表示/技術生成SOURCE',
'ee160395':'高次PR図・表示/技術生成SOURCE',
'8613b3ec':'保存費用結果・報告（科学実行SOURCEではない）',
'9a77d2b0':'保存費用の科学評価SOURCE',
'cf99e445':'旧H2 D取得/保存会計SOURCE',
'de72e5d4':'旧H2 signal取得SOURCE',
'84db3909':'旧H4 D取得/保存会計SOURCE',
'4ec26921':'H2 outer32取得SOURCE',
'5a448d94':'H4/H6 outer32取得SOURCE',
'217c971d':'凍結provider SOURCE',
'3eca4614':'PR独立cohort確認の固定結果・報告',
'495496c4':'legacy new_2の固定報告',
'427480e4':'理論/compiledの別研究dossier',
'3983fe4f':'旧PF単価置換の保存結果',
'7430a386':'旧PF単価政策が参照する固定証拠',
'a05f4ed1':'H2 outer32取得の固定公開結果',
'605ca07a':'H4/H6 outer32取得の固定公開結果'}
lines=['# SOURCE・取得状態の全索引','','全行の取得状態は `ACQUIRED_LOCAL_FIXED_GIT_BLOB`。各パスは元repository内の位置、ローカルコピーはリンク先。科学実行・SOURCE importはしていない。GitHubによる別取得の有無はreceipt列に記載する。','','| full commit | 元パス／ローカルコピー | 種別 | GitHub取得 | SHA256 |','|---|---|---|---|---|']
remote_keys={(r['commit'],r['path']):r for r in remote}
for r in manifest:
    k=r['commit'],r['path'];rr=remote_keys.get(k)
    remote_label='全文＋blob SHA一致' if rr and not rr['remote_content_partial'] else '部分内容＋blob SHA一致' if rr else 'ローカル固定blob取得'
    role=roles.get(r['commit'][:8],'リンク元で固定された入力/補助証拠')
    lines.append('| `'+r['commit']+'` | ['+r['path']+']('+r['local_path']+') / [GitHub]('+r['url']+') | '+role+' | '+remote_label+' | `'+r['sha256']+'` |')
lines+=['','## private metadataと未取得資料','','| 資料 | 取得状態 | 根拠・用途 |','|---|---|---|',
'| 旧H2 D private spec | `READ_ONLY_METADATA_ACQUIRED_HASH_MATCH` | 公開bindingにあるexact locatorを使用。原本をコピー/実行せず、safeなhash・provider対応だけprivate_metadata_audit.jsonへ記録 |',
'| 旧H2 runtime manifest | `READ_ONLY_METADATA_ACQUIRED_HASH_MATCH` | 封印digest一致。記載された全runtime binaryを再hashしたとの主張ではない |',
'| 新outer32取得時の完全private config/runtime/provider/guard受入record | `NOT_ACQUIRED_FULL_CLOSURE` | 公開summary・input_provenanceと科学SOURCEは取得済み。原本manifestのexact locatorは未取得。広域検索せず不足として保持 |',
'| H6旧Full PFのbias/logB signal実体 | `NOT_ACQUIRED_IN_THIS_AUDIT` | 元保存会計rowと保持単価は取得済み。headroomから科学数値を再構成しない |',
'| m5上流PF repositoryの20d64c2全文 | `NOT_FETCHED_UPSTREAM_REPOSITORY` | 固定reference内の関数SOURCE/coefficientsと本repoの実使用scheduleを取得。上流全文取得とは称しない |',
'| 会議bento_slides.pdf・10/22発表用構成案原本 | `NOT_ACQUIRED` | ユーザー提示の対応情報とF7/FH固定図資料を利用。PDF目視/実ページ一致未確認 |',
'| 科学array NPZ/QPY・seed一覧・有効grant原本 | `NOT_LOADED_OR_EXPORTED` | 本監査は保存集計/metadata/SOURCEのみ。private旧specを実行せず、科学arrayを読まない |']
(ROOT/'SOURCE_INDEX.md').write_text('\n'.join(lines)+'\n')

template=(ROOT/'REPORT.template.md').read_text()
assert template.count('{{EVIDENCE_TABLE}}')==1
table=(ROOT/'TABLES.md').read_text().split('\n',2)[2]
(ROOT/'REPORT.md').write_text(template.replace('{{EVIDENCE_TABLE}}',table.rstrip()))
text=(ROOT/'REPORT.md').read_text()
assert '{{' not in text
link_targets=[target for target in re.findall(r'\]\(([^)]+)\)',text) if not target.startswith(('https://','http://','#'))]
for target in link_targets:
    if target!='verification.json':assert (ROOT/target).is_file(),target

checks=dict(status='STATIC_EVIDENCE_AUDIT_PASS_MANDATORY_STOP',date_JST='2026-10-23',scope='FIXED_SAVED_VALUES_AND_STATIC_METADATA_ONLY',scientific_reversal_C='NOT_CERTIFIED',primary_agent_only=True,independent_reviewer=False,selected_records=75,selected_total_transfer_checks=transfer,comparison_conditions=15,reversal_candidates=[dict(system=r['system'],epsilon=r['epsilon']) for r in matrix if r['classification']=='REVERSAL_CANDIDATE'],same_direction_count=sum(r['classification']=='SAME_DIRECTION' for r in matrix),auxiliary_slots=60,preserved_missing_auxiliary_slots=16,github_blob_checks=remote,linked_manifest_hash_checks=len(linked),shot_AST_checks=shot,acquired_fixed_blob_count=len(manifest),acquired_blob_hash_checks=hashes,private_old_metadata_hash_unchanged=True,private_old_provider_matches=len(private['provider_file_comparisons']),report_relative_links_exist=True,originals_modified_by_this_task=0,scientific_module_imports=0,scientific_calls=0,simulation=0,sampling=0,circuit_build=0,compile=0,new_shot_calculation=0,new_cost_accounting=0,new_candidate_ranking=0,coefficient_evaluation_or_optimization=0,grant_issued=0,grant_reused=0,commit=0,push=0,mandatory_STOP=True)
write('verification.json',checks)
assert all((ROOT/target).is_file() for target in link_targets)
write('completion_status.json',dict(status=checks['status'],scientific_claim_not_adopted=True,next_actor='USER',next_action='Decide scope and allowance policy; optionally supply exact missing metadata locators',next_scientific_stage_authorized=False,mandatory_STOP=True))
print(json.dumps({k:checks[k] for k in ['status','selected_records','comparison_conditions','reversal_candidates','same_direction_count','acquired_fixed_blob_count','linked_manifest_hash_checks','private_old_provider_matches','scientific_calls','commit','push']},ensure_ascii=False,indent=2))
