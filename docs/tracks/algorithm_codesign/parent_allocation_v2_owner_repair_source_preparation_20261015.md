# Parent allocation v2：owner PID受渡しの限定修正・実行source準備

2026-10-15 JST。**READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION**。
今回の利用者指示は、直前に提示した「新sourceでの限定修正準備」への承認として扱った。
登録比較の再実行承認としては扱わず、新しいscience runner・実worker・合成を起動していない。

## 修正と保護

v1は[固定停止報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b7b723d99805996a10eca680548a740dd4cb4f/docs/tracks/algorithm_codesign/parent_allocation_v1_failed_execution_and_gpt_handoff_20261015.md)のとおり、
初回key jobで`parent_pid`の二重kwargsにより停止した。source S1=`8dfec590a8d760eb15932d862698b270649e9bcf`、
authorization A1=`66d8c026210a56108f2c443695c6d9f52f59fe0b`、result=`87b7b723d99805996a10eca680548a740dd4cb4f`。
v1のmarker SHA256=`257112f17667e42550b434ced6aea4cb926ebad939adbbba0e0c806d4e01ad7c`は消費済みのまま保持する。

新branchは`track-b-parent-allocation-owner-repair-source-preparation-20261015`。
基点は公開確認commit `5cb68ee8b688081ab3391cefe9fd9684ddd11add`。旧実行branchは変更していない。
旧contract／承認JSON／manifest／全結果／guard receipts／audit／terminal／STOPのbytesを保護する。
改訂した旧3 source pathはS1のGit blob bytes/SHA256を別に検証し、旧manifestを現在の改訂コード用manifestとして再利用しない。

実行コードの修正は[acquire](../../../src/trottertracks/algorithm_codesign/parent_experiment/pipeline.py)のjob生成方式のみ。

```python
job = dict(identity, key=key['key'], ratio=key['ratio'],
           parent_pid=os.getpid(), job_index=index)
```

これにより入力identityを変更せず、外側owner PIDを持つledger identityから、keyを所有するledger PIDへjobの`parent_pid`を置き換える。
outer→ledger／ledger→keyのそれぞれのppid検査を維持する。所有検査、marker/inventory/source bindingは緩めない。
候補生成、freeze、proposal、誤差、十分N、価格計数、guard、worker backend、科学的分類規則には変更なし。

## Contractとnamespace

[execution contract v2](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v2/execution_contract_v2.json)の
科学条件／精度／capsはv1とJSONで完全一致する。変更は`execution`内のauthority・manifest・protected ledger・result・receiptのpathだけ。
`kind`のV1は科学契約schemaの版であり、execution namespaceのv2と区別する。
[契約差分証拠](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v2/contract_and_source_delta.json)を参照。

| 項目 | 維持した条件 |
|---|---|
| 入力・target | 既知3-qubit distinct basis、p=(1/5,3/10,1/2)、x=5/7、状態000、σ+1、finite P5、r1 |
| 比較 | 中心ordinary／closed P5、参照partial-return／P3+tail／matched CTS |
| 候補・proposal | common／全頂点／KKT有理候補／halfway、canonical／適用可能global IS／正則化3条件 |
| 比較規模 | 最大15構成、75 law。新価格前の候補freeze、全方式へ同じ機会 |
| 精度・confidence | primitive1e-6、K256/H160、rho/eta1e-12、axis1/200、alpha_axis1/40、guard100dps |
| calls | 新root key最大18・各一回、旧unique guard最大36、全primitive guard最大54、root matrix診断最大72 |
| 実行資源 | 2 worker、numeric threads1、key CPU120s、総CPU3600s、key RSS512 MiB／ledger2048 MiB／aggregate3072 MiB、output64 MiB、log各64 KiB |
| wall・STOP | total/phase wall上限なし、retry0、全結果mandatory STOP |

新authorityは`artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v2/authorization.json`、pending／execution false／source_commit null。
新resultは`artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v2`。準備中に作成しない。
v1のprefix候補／価格／guardをv2科学結果として流用・再採点しない。将来の承認runでは同じ固定入力から契約済み手順を一回実施する。
旧markerを解除・再利用しない。

## 技術検証

[追加回帰テスト](../../../tests/tracks/algorithm_codesign/test_parent_experiment_owner_repair.py)は、取得関数を丸ごとmockせず、実際の`acquire`を非空inventoryで通す。
job JSON・process receiptを人工temporary namespaceへ保存し、`Popen`とwait4だけをmockする。
実worker/backendは起動せず、登録済みfixture・sequenceを評価しない。

新7件の確認事項：

- 既存outer PIDを持つidentityから初回key job作成・fake完了まで到達する。
- 2 keyの各job_index・ledger parent・source／inventory／candidate hashを保持する。
- 実際のworker gateがledger pp idを受理し、outer PIDを所有者とする誤ったppidを拒否する。
- 失敗workerのraw receiptを一回保存し、再試行しない。
- worker resultのowner metadata改ざんを拒否する。
- peer failure時に残るprocessをterminateし、終了receiptを保存する。
- 元の重複kwargsを通常Pythonで再現し、置換後にinput identity／markerを変更しない。

新7＋既存experiment45＋native bridge26＝**78 PASS、FAIL/ERROR/SKIP0**。
[実行記録](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v2/focused_tests_execution.json)は固定runtime、約1.38s wall、peak69,432 KiB。
local evidenceでありCI／外部再現ではない。fake receiptの`synthesis_calls=1`はmock完了を示し、実合成は0。

[zero-compute preflight](../../../scripts/tracks/algorithm_codesign/preflight_parent_allocation_experiment.py)はv2契約の実pathをlaunch定数から取得するよう修正した。
v2 source manifest、全旧保護hash、旧source blobs、固定runtime、pending authority、新namespace未存在を確認する。
既存v1 sourceのstartup入口を科学的に再実行していない。数値的なcandidate／angle／N／ランキング／registered operator評価は今回0。

## 固定sourceと次の境界

固定source S2と公開確認HEADは、後から保存する`v2/publication_receipt.json`で区別する。
将来実行するならS2のauthorization-only直接子A2が必要で、A1や公開確認HEADからは実行しない。
A2の許可pathはv2 authorization JSONとcontract指定receiptのみ。
runnerは引き続き `python scripts/tracks/algorithm_codesign/run_parent_allocation_one_shot.py --source <full S2>`。
今回この入口を起動していない。

数学・科学契約を変える必要がある問題は今回見つかっていない。
実key合成・登録root診断・全law会計・caps内完遂はまだ未検証であり、mockテストの成功から実科学実行の成功は保証しない。
このPID修正のために追加GPT科学レビューを要求する具体的な論点は現在ない。
次の境界は、利用者による**固定S2の別途one-shot実行承認**。

準備資料のcommit／push／GitHub再取得後に**mandatory STOP**。
科学的採否・新規性・研究継続は判断せず、新しい実行やレビューを自動開始しない。
