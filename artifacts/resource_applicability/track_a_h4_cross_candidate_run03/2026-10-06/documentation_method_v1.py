import json
from pathlib import Path

ROOT=Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-cross-candidate-run03-20261006')
BUNDLE=ROOT/'artifacts/resource_applicability/track_a_h4_cross_candidate_run03/2026-10-06'
DOC='docs/research/track_a_h4_cross_candidate_run03.md'
source='79a552d517ee4d391629b1929ab90d2ba1dae451'
check=json.loads((BUNDLE/'binding_checks_v1.json').read_bytes())
body=f'''# H4 run03：候補間compile投入・12 worker再実行

利用者の「その修正を入れて、ワーカー数を増やして再実行して」を認可根拠とする。
source実装と人工job検査はlocal PASS。productionの成功・科学的結論は未確定。
実起動は別controlのfresh検査PASS後だけで、完了後はMAP_COMPLETE_STOP。

## 修正と不変条件

旧run02は確認時点ですでにowned pool failed STOPで終了していた。旧source・全worktree・
6 NPZ・generation-freeze・journal・2 completed records・1 signal・4消費invocationsを保持する。
旧partial metrics/cacheを新sourceの結果へ再利用せず、明示認可された新run03を一度だけ実行する。
入力の再生成・SCF/DF/state生成はしない。旧source commitで作られた入力を、独立の
input_reuse_and_prior_budget_v1.jsonとplan/auth/reviewのdigestへ結合してread-only再利用する。
新generation-freezeの作成や、旧freezeのsource/run IDの書換えはしない。

compile queueは候補間をまたぐ。6 baseline候補の2軸ずつを最初の12 jobとして投入できる。
同時未完了invocationsはadmitted workers以下、circuitはlazy生成し待機全回路を保存しない。
候補をまたいでもcache reuse範囲は従来のgeometry/candidate/axis/numerical identityのまま。
候補の公開順と候補内のtrajectory/cosine/sine順は固定順を維持し、完了順へ依存しない。
workerは前jobのcircuit参照・compiler循環参照を次IPC読み取り前に解放する。
最初のpool failure内容をdriver例外へ含める。失敗時はowned childrenだけを停止しretry/resumeしない。

H4 linear neutral singlet/STO-3G、4 spatial/8 system+ancilla1、DF12。
6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、T=0.8、二次DF-prefix PF/canonical finite-RTE。
218 templates/距離、L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1。
登録済みr/K・32 paired trajectories・master_seed=20261006・seed導出algorithm・compiler条件は不変。
seed identityは従来どおりSOURCE_COMMITを含むため、今回の新source identityで数値seedを再結合する。
circuits.py/inputs.py/signal.py/identity.pyのbytesとcompiler options/dependenciesは旧固定sourceと一致する。
source変更前の物理random結果は得られておらず、旧partial結果を新runへ混合しない。

## 固定identity・CPU・予算

- branch：track-a-h4-cross-candidate-run03-20261006
- SOURCE_COMMIT：{source}
- 固定checkout：{ROOT}
- plan SHA-256：{check['plan_sha256']}
- plan fingerprint：{check['plan_fingerprint']}
- authorization digest：{check['authorization_digest']}
- approved review digest：{check['review_digest']}
- new run ID：track-a-h4-geometry-v2-20261006-run03
- CPU：[3,5,6,7,8,9,10,11,12,13,14,15]、12 owned workers、own新規driverだけにtaskset mask0xffe8。
- thread各1、driver/worker AS/RSS各8 GiB、headroom16 GiB、fresh available memory要求120 GiB。
- 旧累積bytes165161600を新journalへcarryし、新journalの128-byte carry rowも課金する。
- 旧compile invocations4件を含めactual invocations合計74784以下。新attemptの残上限74780。
- 旧wallの保守的上界4903.371601050114 s（確認までのdowntimeも含む）を引継ぎ、合計72h以下。
- 全campaign charge cap10 GiBを維持し、入力再利用の新規stage空き条件は3.5 GiB・560000 inodes。
- 出力：signal1308件、logical wrappers74784件。old journal/ledgerのreset・予約・上書きなし。

同時publisherのplanning余裕を8から14へ増やした差分3 MiBを容量見積りへ加えた。
保守的小計3576791040 bytesは上方丸めた3.5 GiB以内。全72時間の監視出力も引き続き含める。
実際の出力size・所要時間・production成功をこの見積りから断定しない。

## 検査と実行条件

49 local tests PASS、errors/failures/skips0、今回の人工/実transpile追加0。
1 worker/12 workersの同じ人工jobでorder・metrics・seeds・axis組を照合した。
先行47 PASS/1 FAILは以前のsource監査pathへ固定されたassertionだけで、新固定先へ更新して48 PASS。
workerの循環参照解放testを追加し最終49 PASS。失敗履歴を隠さずbundleへ保存する。

承認reviewは実際の利用者指示の転記であり、外部reviewerが承認したとは主張しない。
旧false draftは保持する。新CPU12件のonline/cpuset/physical cores/NUMA/受動SMT load・CPU quota、
memory120 GiB・PSI0/OOM変化なし・source19/dependency45/compiler・freeze/6 NPZ streaming hashes・
quota/user available disk3.5 GiB/inodes560000・旧PID/worker終了・新output不在を起動直前に再確認する。
欠測、不合格、既存new outputなら科学runnerを起動せずSTOP。
GPU query/use、共有環境変更、install/upgrade、他job affinity/signal操作を行わない。
H6/Track B/追加trajectory/anchor/長RPE/総costへは進まない。

実際の起動記録・ログ入口：
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/executions/h4-signal-compile-run03-revision/README.md`。
native runnerはこのcheckoutのscripts/resource_applicability/run_h4_geometry_signal_compile.py。
plan/auth/reviewは本bundleのsignal_compile_plan_v1.json、authorization_user_approved_v1.json、
stage_review_user_approved_v1.jsonにabsolute pathで固定し、fresh preflight後に一度execする。

[bundle](../../artifacts/resource_applicability/track_a_h4_cross_candidate_run03/2026-10-06/README.md)。
科学NPZ・実runtime・checkpoint・cacheはcommit対象に含めない。
'''
with (ROOT/DOC).open('x') as f:f.write(body)
with (BUNDLE/'README.md').open('x') as f:
    f.write('# H4 cross-candidate run03\n\n[資料入口](../../../../docs/research/track_a_h4_cross_candidate_run03.md)。\n\n')
    f.write('利用者認可：その修正を入れて、ワーカー数を増やして再実行して。SOURCE '+source+'。\n\n')
    f.write('49 local metadata/artificial-job tests PASS、追加transpile0。旧run02全証跡保持。6凍結入力をsource lineage付きで再利用し入力を再生成しない。CPU12件・12 workers。累積10GiB/72h/74784 invocationsをresetしない。fresh preflight PASS後にのみ新runを一度起動しMAP_COMPLETE_STOP。\n\n')
    f.write('source_freeze_v1.json / input_reuse_and_prior_budget_v1.json / signal_compile_plan_v1.json / authorization_user_approved_v1.json / stage_review_user_approved_v1.json / binding_checks_v1.json / stage_storage_estimate_v1.json / test_attempt_2,3.log/xmlを一組として参照。production完了や科学的結論は未検証。\n')
updates={
 'PROJECT_MAP.md':'docs/research/track_a_h4_cross_candidate_run03.md',
 'docs/README.md':'research/track_a_h4_cross_candidate_run03.md',
 'docs/research/README.md':'track_a_h4_cross_candidate_run03.md',
 'scripts/README.md':'../docs/research/track_a_h4_cross_candidate_run03.md',
 'src/trotterlib/README.md':'../../docs/research/track_a_h4_cross_candidate_run03.md',
 'docs/research/研究概要・現状.md':'track_a_h4_cross_candidate_run03.md',
 'docs/research/研究ノート/README.md':'../track_a_h4_cross_candidate_run03.md',
 'docs/research/研究ノート/2026-10-06.md':'../track_a_h4_cross_candidate_run03.md'}
for p,link in updates.items():
    with (ROOT/p).open('a') as f:
        f.write('\n\n## H4候補間compile投入・12 worker明示再実行\n\n')
        f.write('利用者の増員再実行指示により[run03 source・認可・検査記録]('+link+')を追加した。')
        f.write('候補内2回路の完了待ちで4 workerがidleとなる問題を、候補間bounded queueで修正。')
        f.write('旧run02はworker failure STOPで全証跡を保持し、旧6入力を再生成せず利用する。')
        f.write('49人工job/metadata testsはlocal PASSで科学的結果ではない。')
        f.write('12 workerのfresh CPU/memory/容量/hash検査後だけ一度起動しMAP_COMPLETE_STOP。')
        f.write('旧累積bytes/wall/actual invocationsを引継ぎ、科学条件・compiler・上限は不変。\n')
manifest_path=ROOT/'artifacts/validation_manifest.json';m=json.loads(manifest_path.read_bytes())
m['artifact_inventory']['present'].append({'path':'artifacts/resource_applicability/track_a_h4_cross_candidate_run03/2026-10-06','kind':'directory','supports':['h4_server_execution_implementation'],'reason':'User-approved cross-candidate scheduler run03 source/binding and49 artificial/metadata local tests; production science completion pending; six NPZ/runtime/checkpoint/cache excluded from Git.'})
m['h4_cross_candidate_run03']={'status':'implementation_locally_validated_production_completion_pending','source_commit':source,'tests':49,'failures':0,'errors':0,'skipped':0,'new_test_transpiles':0,'requested_workers':12,'input_generation_source_commit':'049e69919af16ad29a67a217dc7a407d6b1754a6','production_inputs_regenerated':False,'mandatory_completion_stop':'MAP_COMPLETE_STOP','entry':DOC,'historical_numerical_claims_unchanged':True}
manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
with (BUNDLE/'documentation_method_v1.py').open('xb') as f:f.write(Path(__file__).read_bytes())
print('Documentation,8 indexes/history entries and manifest updated')
