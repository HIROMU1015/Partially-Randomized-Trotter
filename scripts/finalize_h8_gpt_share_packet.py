"""Finish local H8 attachments, validate redaction and make a ZIP. No network/science."""
import csv
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20'
C=B/'candidate';A=B/'audit';L=B/'local_only'
ZIP_NAME='H8_GPT_SHARE_PACKET_20261020.zip'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def require(ok,message):
    if not ok:raise ValueError(message)

README='''# H8 GPTレビュー用・共有候補パケット

2026-10-20 JST。ローカル作成済み・未送信。実際のGPT添付、外部upload、GitHub pushは承認されていない。科学レビューと最終evidence freezeも未開始。このpacketはH8部分の証拠確認を準備するもので、研究全体の科学的妥当性を判定したものではない。

## 条件と保存結果

linear H8、隣接1 Å、STO-3G、tol-only1e-8 DF28、16 spin modes、sector4900、T=.8、P=0、二次partial PF、r4/K4。通常signed bias・allowance倍率1、ordinary ancilla、full-Fock scalar一回、reference不回転。結果は保存値の転記であり再計算していない。

|scope / PR候補|L_D / q / δ|n / 軸数|保存mean exact / paired SE|固定PF / 保存G|
|---|---|---|---|---|
|H8_CONFIRM_P8_Q2 / B2_p8_q2_2nd_r4_K4|8 / 2 / .4|48 / 96|9840910801/8 / 3175066.353534034|B0I_p11_q1_2nd / 1267321444|
|H8_CONFIRM_P7_Q16 / B2_p7_q16_2nd_r4_K4|7 / 16 / .05|48 / 96|546493203592571/6 / 144211830223.71762|B0I_p15_q4_4th / 94193016652860|

両driverの保存native receiptはexit0/reaped。継続runはp8で47親・94子、p7で48親・101子。p8は同じ新確認cohortで完了した元SOURCEの1軌道2軸を再利用し、探索n8・旧確認n32・条件付きpilotは混合していない。旧STOP/未完了予約も保持する。296 native nodeは旧driver・旧未完了workerを含むため、すべて成功したとの意味ではない。

結果commit：`e6eea5387a6e84d1e39a9adb3b61975acc74e200`。
実行SOURCE：`745a674546f2cb6258526fbfde1d9a00601689f8`。
seed導出とp8再利用1軌道の測定SOURCE：`1549b6f891a15ff48379996c34b344118ba2606c`。
前回回収local commit：`253356942845dddf05c2bb6f4582b026e0bce942`。
公開索引commit：`9f33ad0b11b4febe48b802000c38c5a0e3fd267c`。

## 読む順序と検証できる範囲

1. `01_results_and_source.json`と`public/`：固定主副評価と192軸projection。p8副評価の固定PF不適格/nullを維持する。
2. `02_execution_receipts.json`、`03_worker_receipts.json`：実行時field、SOURCE/technical digest、native exit/reap、匿名化した親子関係。`04_grant_ledger_marker_stop.json`：新旧grantの選択field、承認receiptのscope、非返却台帳、marker、旧STOP、terminal。
3. `05_seed_cohort_audit.json`、`06_task_axis_links.json`：固定96登録・cohort管理の証拠、軸ごとのSOURCE・event digest・fingerprint・整数metricとnative記録。registryの`actual_RTE_event_sampling=0`はregistry作成段階の宣言であり、後の科学実行全体のsampling数ではない。
4. `07_input_runtime_bindings.json`：input65/58件、provider各931件、runtime34,273件のhash結合・前回の原本検証条件。今回は新たなruntime初期化や科学array復元をしていない。
5. `08_provenance_and_redactions.json`：E番号、原本SHA/bytes、原bytesを収録した監査JSONと匿名化projectionの区別。`10_source_conditions.md`と`11_public_sources.json`：固定公開コードの検査条件・URL・SHA。
6. `09_missing_evidence.csv`と`09_limits.md`：原profile・保存PR回路の欠測、匿名化で失われる独立再検証能力、結論への影響。

terminal・STOP・one-shot marker・reservationは安全な許可fieldだけの原JSON bytesを収録しており、その原SHAを再計算できる。他の原SHAは、非共有原本への来歴linkである。匿名化したnative記録・grant等から、原fileそのもののSHAや実PID/starttime/PPID/UID、元承認チャットの真正性を再認証することはできない。

seed値、除外値、raw event列、認証情報、内部絶対path、private科学raw、非共有原本索引は含めない。公開コードと組み合わせるとseedを復元できる導出用digestも除外した。ここに残るseed-plan/registry原fileのSHAは原bytesへの来歴linkであり、その導出用digestとは異なる。seed非衝突・除外の固定導出は原本で前回確認済みだが、受領者はこのpacketだけから再実行できない。seedの非衝突は数学的な確率独立性の証明ではない。

## オフライン構造検査

展開したdirectoryで `python3 -B verify_packet.py --self-test` を実行できる。標準libraryのみを使い、packet SHA、安全な原監査JSONのSHA、192軸projection、親子関係、scope/grant/marker/台帳の対応、欠測表示を検査する。科学計算・統計解析・sampling・compile・費用再計算は呼ばない。これは記録の構造照合であり、科学的妥当性や原本の真正性を証明する検査ではない。

## 解釈上の境界

原compiler profile file bytesと保存PR回路は未取得のままである。保存fingerprintは入力wrapper側の記録であり、元回路から再導出していない。旧compiler出力のbit同一性・final角度列・非Clifford/T数を認定しない。p8の`saved_exact_export_match=false`、p7のtrueを保持する。

標本点費用の再現という保存結論から、未観測rare eventを含む母平均優位性、厳密bias、最終energy/RPE総費用の保証へ拡張しない。CI/p値は保存null。選択済みPF監査をPR実行監査に代用しない。新規科学計算・統計解析・費用再計算・再ランキング0。

**MANDATORY STOP：共有候補作成まで。** 次担当はユーザー。内容・残る開示リスクを確認し、具体的な添付とレビュー開始を別途判断する。
'''

LIMITS='''# 不足証拠・匿名化による制限

`09_missing_evidence.csv`は「原本で確認済み／公開資料のみ確認／未取得／確認不能」を区別する。確認済みはローカル原本照合の範囲を意味し、受領者が全項目を独立再計算できるとの意味ではない。

- 原compiler profileの現物は未取得。継承identifierと現在のcompiler設定・runtimeの一致から、原profile bytesや旧出力の完全同一性を補わない。
- 保存PR wrapper/final circuitは未取得。event digest・準備hash・SOURCE・metricの来歴は残るが、元回路からのfingerprint再導出やfinal角度/T費用の検証はできない。新規回路再生成・compileは今回の範囲外。
- native identityをaliasにし、grantの内部path・CPU割当・承認本文を除いた。親子graph、exit/reap、technical digestとscopeの一致は確認できる。実process identityと元承認本文・署名の真正性は再認証できない。
- seed値、raw event列、および公開コードからseed復元を可能にする導出用digestを共有しない。旧除外4,136、A128を含む統合除外5,000、新96登録、旧探索592/確認192行のcoverageの原本監査結果と、その検証条件は保持する。具体的なcollision検査・seed階層・event digest計算を受領者が独立に再実行することはできない。
- input/provider/runtimeの原bytesを共有しない。expected hash・役割・件数・前回検査条件を保持する。原bytes照合や科学arrayの意味的正当性は受領者がこのpacketだけで再検証できない。
- 元raw結果全体を共有しない。公開済みscalar192軸と保存比較表、必要な監査field、原SHAのlinkを保持する。平均/SD/SE/費用は再集計せず保存文字列を転記した。
- JSONとSHAの自己整合性は、独立した署名・第三者認証・immutable CIの代わりにならない。SOURCE中のguardと実行receiptを区別する。公開PF receiptは選択済みPFに限定し、PR workerの証拠に置換しない。

完全な原本再検証を求める場合、対象file・受領者・共有先を固定した追加承認が必要。seed原本・private raw・内部pathを含む原本索引・未加工grant/receiptは添付しない。詳細科学レビュー、最終evidence freeze、追加科学計算は未開始。
'''

PURPOSE={
'00_README_GPT.md':('読み方・条件・保存結論・監査範囲とSTOP','公開情報＋匿名化した運用概要','運用構造と証拠不足を開示'),
'01_results_and_source.json':('二条件の主副評価の保存tokenとSOURCE対応','公開結果中心','公開済み数値と測定SOURCEの対応を開示'),
'02_execution_receipts.json':('startup/terminalとseal・結果SHAの結合','原監査の選択項目','未公開原本SHA/bytesと実行完了関係を開示'),
'03_worker_receipts.json':('296 native nodeのexit/reapと匿名化親子関係','匿名化監査projection','worker構造・失敗履歴・原SHAの相関が残る'),
'04_grant_ledger_marker_stop.json':('新旧grant/scopeと原bytesを保持した安全な台帳・marker・STOP','原監査field＋許可済みfieldだけの監査JSON','原SHA・消費予約・STOP理由を開示。承認本文やprocess identityは除外'),
'05_seed_cohort_audit.json':('登録taskと除外coverage・固定導出検査の条件','seed値を含まない監査projection','registry全体SHAと件数を開示。実seed検査の再実行は不能'),
'06_task_axis_links.json':('96軌道192軸のSOURCE/native/event digest/fingerprint/公開metric対応','公開metric＋監査hash','非公開event/circuitを識別するhash相関が残る'),
'07_input_runtime_bindings.json':('input/provider/runtime/準備hashと前回検証条件','原bytesを含まない閉包projection','原input SHA/bytes/役割、内部閉包の件数を開示'),
'08_provenance_and_redactions.json':('E番号の原SHA/bytesと匿名化・再検証限界','非公開原本への相関ID','未公開原本の相関hashとbyte長を開示。内部pathは除外'),
'09_missing_evidence.csv':('四区分status・受領者の再検証可否・結論への影響','不足一覧','証拠の欠測と監査限界を開示'),
'09_limits.md':('匿名化で検証不能になる範囲と追加承認事項','不足の説明','欠測状況を開示'),
'10_source_conditions.md':('固定公開SOURCEの条件抜粋と行番号・SHA','既存公開コード','公開済み仕様のみ。コードの存在と実行PASSを区別'),
'11_public_sources.json':('固定commit・公開18資料と添付公開fileのSHA','既存公開資料の取得metadata','公開済み識別子のみ'),
'public/comparisons.csv':('公開済み主副評価4行の原bytes','既存公開結果','既存公開と同じ'),
'public/paired_metrics.csv':('公開済み192軸projectionの原bytes','既存公開結果','既存公開と同じ'),
'public/result_report.md':('固定結果報告の原bytes','既存公開報告','既存公開と同じ'),
'verify_packet.py':('標準libraryのみのoffline構造・SHA・privacy検査','検査コード','科学moduleや外部通信を呼ばない'),
}

def main():
    A.mkdir(exist_ok=True)
    (C/'00_README_GPT.md').write_text(README)
    (C/'09_limits.md').write_text(LIMITS)
    rows=[
        dict(item='ORIGINAL_COMPILER_PROFILE',local_status='未取得',recipient_recheck='不可',impact='原profile bytesと旧compiler出力bit同一性を認定できない',additional_approval='既存原fileの所在提示・read-only取得。外部共有は別承認'),
        dict(item='SAVED_PR_CIRCUIT',local_status='未取得',recipient_recheck='不可',impact='元回路からfingerprint再導出・final角度/T数は検証不能',additional_approval='既存回路の所在提示。再生成/compileは別の科学実行承認が必要'),
        dict(item='ORIGINAL_CIRCUIT_FINGERPRINT_REDERIVATION',local_status='確認不能',recipient_recheck='不可',impact='保存hashの対応まで。回路実体の同一性の追加認定なし',additional_approval='保存回路がある場合の開示範囲を別承認'),
        dict(item='MATHEMATICAL_STOCHASTIC_INDEPENDENCE',local_status='確認不能',recipient_recheck='証明なし',impact='非衝突やcohort分離を確率独立性の証明と扱えない',additional_approval='今回のpacketから新しい科学的証明や統計解析へ自動進行しない'),
        dict(item='SEED_AND_EXCLUSION_REPLAY',local_status='原本で確認済み',recipient_recheck='匿名化により不可',impact='登録・除外coverageの選択fieldと前回検査条件まで確認可能',additional_approval='seed原本を必要とする再検証は対象fileと安全な共有先の追加承認'),
        dict(item='NATIVE_PROCESS_IDENTITY',local_status='原本で確認済み',recipient_recheck='匿名化graphだけ照合可能',impact='実PID/starttime/PPID/UIDの再照合・OS状態再認証は不能',additional_approval='未加工native receiptの必要箇所と受領者を別承認'),
        dict(item='INPUT_PROVIDER_RUNTIME_ORIGINAL_BYTES',local_status='原本で確認済み',recipient_recheck='hash/役割/件数と前回検査条件まで',impact='原bytes自体の再照合や科学arrayの意味的正当性の再認定はできない',additional_approval='必要な原fileだけを選定して追加承認'),
        dict(item='GRANT_APPROVAL_CHAT_AUTHENTICITY',local_status='確認不能',recipient_recheck='ローカル宣言fieldとscope結合まで',impact='元チャットの独立認証・署名ではない',additional_approval='承認本文を共有する場合は個別に範囲と共有先を承認'),
        dict(item='RAW_RESULT_AND_PUBLIC_PROJECTION',local_status='原本で確認済み',recipient_recheck='公開192軸と保存比較tokenの対応まで',impact='private rawやseed/event列を共有せず、平均/SE/費用を再計算していない',additional_approval='原rawの提供や科学的再集計は別承認'),
        dict(item='SELECTED_FIXED_PF_NATIVE_AUDIT',local_status='公開資料のみ確認',recipient_recheck='固定公開receiptを参照可能',impact='選択済みPFの監査。後日のPR native監査の代用不可',additional_approval='private PF native原本の開示が必要なら別承認'),
    ]
    with (C/'09_missing_evidence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    shutil.copyfile(ROOT/'scripts/verify_h8_gpt_share_packet.py',C/'verify_packet.py')
    payload=sorted(str(p.relative_to(C)) for p in C.rglob('*') if p.is_file() and p.name!='packet_manifest.json')
    require(set(payload)==set(PURPOSE),'EXACT_CANDIDATE_ALLOWLIST')
    manifest=dict(schema='h8_gpt_share_candidate_v1',client_date='2026-10-20',timezone='Asia/Tokyo',publication_status='LOCAL_CANDIDATE_NOT_SENT',self_excluded=True,not_an_evidence_freeze=True,external_send_upload_push_authorized=False,files=[dict(filename=name,bytes=(C/name).stat().st_size,sha256=sha(C/name),purpose=PURPOSE[name][0],confidentiality=PURPOSE[name][1],remaining_disclosure_risk=PURPOSE[name][2],recommendation='CONDITIONAL_AFTER_USER_CONTENT_AND_DESTINATION_APPROVAL') for name in payload])
    save(C/'packet_manifest.json',manifest)
    spec=importlib.util.spec_from_file_location('packet_verify',C/'verify_packet.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checked,data=module.validate(C);checked['mutation_checks']=module.self_test(data)
    # Compare candidate numeric tokens with known confidential seed values locally.
    registry_path=REGISTRY_FILE
    registry=json.loads(registry_path.read_text());secret={str(v) for v in registry['excluded']}|{str(t['seed']) for t in registry['tasks']}
    derivation_secret=registry['seed_plan_digest']
    plan_raw=(registry_path.parent/'seed_plan.json').read_bytes()
    require(hashlib.sha256(plan_raw).hexdigest()!=derivation_secret,'RAW_PLAN_HASH_MUST_NOT_EQUAL_DERIVATION_INPUT')
    for p in C.rglob('*'):
        if p.is_file():
            text=p.read_text();tokens=set(re.findall(r'(?<![A-Za-z0-9_])\d{9,}(?![A-Za-z0-9_])',text))
            require(not (tokens&secret),'KNOWN_PRIVATE_SEED_LEAK')
            require(derivation_secret not in text,'SEED_DERIVATION_INPUT_LEAK')
            module.scan_text(text)
    # Preserve only the candidate directory in the ZIP; local_only is never selected.
    with zipfile.ZipFile(B/ZIP_NAME,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(C.rglob('*')):
            if p.is_file():
                info=zipfile.ZipInfo('H8_GPT_SHARE_PACKET_20261020/'+str(p.relative_to(C)),date_time=(2026,10,20,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(B/ZIP_NAME) as z:
        expected={'H8_GPT_SHARE_PACKET_20261020/'+str(p.relative_to(C)) for p in C.rglob('*') if p.is_file()}
        require(set(z.namelist())==expected and len(z.namelist())==len(expected),'ZIP_EXACT_ALLOWLIST')
        for name in z.namelist():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'ZIP_PATH_TRAVERSAL')
            inner=name.split('/',1)[1];raw=z.read(name)
            require(raw==(C/inner).read_bytes(),'ZIP_CONTENT_SHA')
            module.scan_text(raw.decode())
        require(z.testzip() is None,'ZIP_CRC')
    local=json.loads((L/'NONSHARE_original_index_PRIVATE.json').read_text())
    for row in local['observed_originals']:
        s=Path(row['path']).stat();require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==row['stat'],'ORIGINAL_STAT_CHANGED')
    previous=json.loads((PRIOR/'evidence_inventory_PRIVATE.json').read_text())
    for row in previous['files']:
        s=Path(row['path']).stat();require([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==row['stat'],'PREVIOUS_ORIGINAL_STAT_CHANGED')
    audit=dict(status='PASS_PRE_SHARING_AUDIT_STRUCTURE_ONLY',structure=checked,privacy=dict(method='allowlist extraction; recursive forbidden-field checks; private path/credential patterns; exact local comparison with known seed tokens and seed-derivation input; archive content allowlist',candidate_files_checked=len(manifest['files'])+1,known_seed_values_compared=len(secret),known_seed_values_found=0,seed_derivation_input_found=0,seed_derivation_input_withheld=True,private_absolute_paths_found=0,credential_patterns_found=0,private_raw_fields_found=0,limit='Pattern scanning cannot prove absence of every possible secret. Selected private hashes and operational structure remain disclosed.'),preservation=dict(originals_rehashed_during_extraction=local['input_original_files_rehashed'],originals_stat_stable_after_extraction=len(local['observed_originals']),previous_known_original_files_stat_stable=len(previous['files']),existing_scientific_originals_written=0,prior_outputs_written=0),archive=dict(filename=ZIP_NAME,sha256=sha(B/ZIP_NAME),bytes=(B/ZIP_NAME).stat().st_size,entries=len(expected),candidate_only=True,nonshare_index_included=False,roundtrip_bytes_and_CRC=True),network_calls=0,external_send_upload_push_performed=False,new_scientific_calls=0,new_statistics=0,result_recalculation=False,scientific_validity_review_started=False,final_evidence_freeze=False,mandatory_STOP=True)
    save(A/'PRE_SHARING_AUDIT.json',audit)
    (A/'ZIP_SHA256.txt').write_text(sha(B/ZIP_NAME)+'  '+ZIP_NAME+'\n')
    catalog=['# H8共有対象一覧（ローカル保存）\n\n作成のみ承認済み。実際の送信・外部upload・pushは未承認。推奨判定は、ユーザーが内容と送付先を確認して承認した後の添付候補を意味する。\n\n推奨する添付候補は **`'+ZIP_NAME+'` 1個**。ZIP内はcandidateだけで、非共有原本索引・builder・監査用ローカル所在情報を含まない。\n\n|file|目的|SHA256|機密性・残る開示リスク|判定|\n|---|---|---|---|---|\n']
    for row in manifest['files']:
        catalog.append('|'+row['filename']+'|'+row['purpose']+'|`'+row['sha256']+'`|'+row['confidentiality']+'：'+row['remaining_disclosure_risk']+'|内容確認・共有承認後に推奨|\n')
    catalog.append('|packet_manifest.json|添付payloadの全SHA/目的/開示リスク（自身はhash対象外）|`'+sha(C/'packet_manifest.json')+'`|公開metadata＋選択原本の相関hash|内容確認・共有承認後に推奨|\n')
    catalog.append('|'+ZIP_NAME+'|candidate全fileをまとめた推奨添付候補|`'+sha(B/ZIP_NAME)+'`|上記の開示リスクの合計。自動送信なし|内容確認・共有承認後に1個だけ添付|\n')
    catalog.append('|local_only/NONSHARE_original_index_PRIVATE.json|E番号→原本の所在・field対応|`'+sha(L/'NONSHARE_original_index_PRIVATE.json')+'`|内部絶対pathを含む|**添付禁止・別途具体的開示承認が必要**|\n')
    catalog.append('|audit/PRE_SHARING_AUDIT.json|構造/SHA/privacy/保全のローカル監査|`'+sha(A/'PRE_SHARING_AUDIT.json')+'`|件数・検査条件と残余リスク。原pathなし|必要時のみ内容確認・共有承認後に追加添付|\n')
    catalog.append('\nこの一覧自体は添付対象ではない。自身のSHAは自己参照を避けて記録しない。ZIP SHAはaudit/ZIP_SHA256.txtにも保存。seed原本・private raw・未加工grant/native receipt・非共有索引・branch全体は添付しない。GPTレビュー開始は未承認。MANDATORY STOP。\n')
    (B/'H8_GPT_SHARE_FILES_20261020.md').write_text(''.join(catalog))
    print(json.dumps(dict(status=audit['status'],packet_files=len(manifest['files'])+1,zip_bytes=audit['archive']['bytes'],zip_sha256=audit['archive']['sha256'],mutation_checks=len(checked['mutation_checks']),new_scientific_calls=0,actual_external_send=False),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-dossier',type=Path,required=True)
    parser.add_argument('--registry-file',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,default=B)
    args=parser.parse_args()
    PRIOR,REGISTRY_FILE=args.prior_dossier,args.registry_file
    B=args.output_directory
    C,A,L=B/'candidate',B/'audit',B/'local_only'
    main()
