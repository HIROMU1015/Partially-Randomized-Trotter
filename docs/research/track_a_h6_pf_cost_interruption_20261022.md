# H6 metadata取得完了・共通PF費用取得の外部中断

2026-10-22 JST。**不足63件のmetadata、新102 signal、1,380 endpointは正常終了。COSTは187候補の完成pairを残してプロセスが消失し、残る22候補とH6全域比較はUNRESOLVED。** 使用済みgrantを再開せず、新たなSTOP観測記録を保存した。

専用branchは `pf-common-h6-metadata-execution-20261022`。科学SOURCEはmetadataが `cba2a82021b17cba0eec31b03fb19d80f35225f6`、共通PFが `b7662487185086aeaa1b298f82089e194b7270db`。各SOURCEの実行対象ファイルを監査bundleへ保存した。今回の中断監査・実行方式修正は別SOURCEであり、科学SOURCEを上書きしていない。

## 固定条件と完了範囲

linear H6、隣接1 Å、STO-3G、tol-only DF threshold 1e-8、原保存rank19。T=.8、P=0、δ=T/q=.8/.4/.2/.1。S2・Yoshida4・m5_bestの元binary64係数とsigned schedule、B0/B0I prefix L_D=5..15、Full prefix19、q=1/2/4/8の276候補を維持した。Full cosine/sine Hadamard wrapper、symmetric-directional control、元scalarを一回適用。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、元native/runtime系列を維持。H4 DF12、q16、追加候補は含まない。

原H6 snapshot SHAは `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、Hamiltonian hashは `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、state hashは `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3`。今回は原snapshotを復元し、新SCF・DF・state・full-H reference生成は行っていない。

| 処理 | 実績 | 終了状態 |
|---|---:|---|
| 別予算metadata | 63観測、旧52 signal照合、新11 raw signal | PASS、63 worker exit0/reap |
| 共通SIGNAL | 新102＝基礎51＋overlay51、独立forward51 | PASS、全276 signalが利用可能 |
| 共通QUALIFY | 1,380 endpoint、239候補が少なくとも1精度で適格 | PASS、適格性不明0 |
| 共通COST | 完成187 pair／374軸、compile予約394軸 | 外部中断、成功receiptなし |
| 保存価格 | 同native価格30適格pair | identity確認済み、未会計 |
| 未取得価格 | 22 pair | UNRESOLVED |
| 共通ACCOUNT・588候補最終比較 | 未実行 | UNRESOLVED |

H6では全精度不適格37候補を含む254候補で取得要件が揃うが、COST stage全体の正常終了や固定総費用評価完了を意味しない。H2 84・H4 228の既存完了結果は再計算せず、旧H6記録を含む合計75ファイルを基準commitからbyte照合した。3系588候補が最終評価済みとは主張しない。[旧H2/H4比較](track_a_pf_common_domain_results_20261022.md)をそのまま参照する。

## metadataの方法と検証

B0Iについて元のnative corrected/raw recurrenceと別forward経路を実行し、元と同じ作用順序・scalar位相・中間norm観測を保持した。単にB0値を転用していない。`raw_corrected_vector_closure`は別々に保存したraw/correctedベクトルから計算した。今回の値0は観測値であり、欠損の0補完ではない。

旧signalを持つ52候補はcorrected/reference/signed biasの元保存値をそのまま維持し、新観測との誤差を照合した。旧raw未回収11候補の新signalは新規取得として区別した。最大corrected差5.9338e-14、reference差2.7104e-14、signed bias差6.2974e-14、別forwardベクトル差2.3597e-12、中間norm最大1.0000000000001967。全て固定gateを満たした。新計算を旧原観測の復元とは呼ばない。

[metadata終了監査](../../artifacts/resource_applicability/track_a_h6_pf_metadata_completion/2026-10-22/ending_audit.json)、[grant・792 ledger](../../artifacts/resource_applicability/track_a_h6_pf_metadata_completion/2026-10-22/grant_and_ledger.json)、[63ベクトルarchive目録](../../artifacts/resource_applicability/track_a_h6_pf_metadata_completion/2026-10-22/archive_inventory.json)。

## 中断の観測と完成pairの監査

再接続時、COST driver・workerと後続driverは存在せず、COST completion・worker exit/reap receiptもなかった。対象プロセスの不在を2回確認し、観測時刻を新記録へ保存した。**元の停止時刻・exit code・原因は未確定。** チャット切断やサーバー再起動が原因だったと断定しない。前回の「バックグラウンド継続」案内後、予定した後処理は完了しなかった。

187件の完成resultについて、SOURCE・入力・preparation・compiler・元native gateを照合し、6 archive/pair、計1,122 archiveの圧縮SHAと原byte SHAを復元確認した。保存されたbefore/afterベクトルの差とHadamard期待値も別途読み取り照合した。最大compiled vector差1.6045e-9（gate1e-7）、独立occupation差2.0649e-12、保存signal差3.4694e-13（各gate1e-9）。これは保存値監査で、新しいnative作用・compileではない。

22未完了候補のうち12には片軸等の38ファイルが残るが、完全なpaired resultと元archive digest receiptがないため価格に採用しない。残る10候補は出力directoryだけが残る。完成187pairも、元parentのjoin/reap証明があるとは表現しない。中断stageの成功receiptを補作せず、grant・ledger・marker・全partialを保持した。

| COST非返却予約 | 実消費 | 元固定上限 |
|---|---:|---:|
| native evolution／独立occupation／native low／high | 各209 | 各224 |
| signed S2 piece | 3,334 | 3,625 |
| wrapper build | 408 | 448 |
| uncompiled action | 397 | 448 |
| wrapper compile | 394 | 448 |
| compiled action | 390 | 448 |

snapshot/reference/oracle/full preparation各1、prefix preparation10も消費済み。完了pair数から消費を逆算せず、台帳予約を全て非返却で計上した。元127 worker、BLAS/compiler各1、利用時の空きCPU/RAM/disk観測を保存した。共有設定変更は0。

## 再取得に必要な契約差分

22候補を**元と同じpaired取得手法**で新SOURCE・新grantへ分離する場合の具体案を[未承認契約](../../artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/fresh_completion_contract_UNAPPROVED.json)へ保存した。これは承認済みgrantではなく、科学再起動は行っていない。

| call | 新取得案 | 消費済みを含む合計 | 固定上限超過 |
|---|---:|---:|---:|
| native evolution／独立occupation／low／high | 各22 | 各231 | 各7 |
| native signed S2 piece | 1,520 | 4,854 | 1,229 |
| wrapper build | 44 | 452 | 4 |
| wrapper compile | 44 | 438 | 0 |
| uncompiled／compiled action | 各44 | 441／434 | 0 |

新contextはsnapshot/reference/oracle/full preparation各1、prefix preparation7。signal・適格性・PR samplingの追加0。compile軸は上限内でも、内部call上限を超えるので現承認では実行できない。partial軸を利用する別方式なら削減の可能性があるが、必要なidentity・観測証明が未完成であり、これを最小必要予算の数学的下限とは呼ばない。条件や予算を推測で変更しない。

## 中断に強い起動方式の実装準備

`scripts/run_detached_local_job.py`を追加した。自分のuser managerの一時serviceへ起動を分離し、元CPU affinityと明示されたBLAS環境を保持する。request、heartbeat、外部exit/reap receiptを保存し、既存root再使用と自動retryを禁止する。科学的成功は元stage receiptで別確認し、launcherのexit0だけで認定しない。共有serviceや他ユーザーの設定は変更しない。

人工3試験PASS。さらに3秒だけの人工タスクを自分の一時user serviceで実行し、起動元processが終了した後に完了・exit0・reap receiptが保存されることを確認した。[人工実動作検証](../../artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/detached_artificial_validation.json)。実際のチャット切断を再現する試験や、中断原因の証明ではない。新科学stageをこの方式で起動していない。

## 保存と限界

[中断監査summary](../../artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/summary.json)、[全276状態](../../artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/candidate_status_276.json)、[未完22候補](../../artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/unresolved_candidates_22.json)、[STOP](../../artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/COST_STOP.json)、[旧結果保全](../../artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22/old_results_preserved.json)。private側に元raw QPY/NPZ・grant・ledger・marker・旧STOPを保持し、公開bundleには可搬な監査・signal・適格性・価格証跡を保存した。

通常signed bias、5精度、経験的allowance、凍結binary64 shot算術、compiled RZ×analytic shot、P=0の政策は不変。経験的allowanceは厳密certificateではない。保存PRの有限標本・探索選択効果・未観測rare eventを維持し、母平均順位や全energy/QPE/RPE総費用最適化を保証しない。H6の全域最良・PR改善率・588候補最終比較は欠測のためUNRESOLVED。

新規科学call・追加grant・同grant retry/resume・新PR sampling・GitHub pushは0。監査・実行方式修正・成果をローカルcommit後、mandatory STOP。元の数値手法で22候補を再取得するには、上記の内部call予算差分に対する明示承認が必要。
