# Parent allocation v3：JSON保存の限定修正・固定source準備

2026-10-15 JST。93登録外tests PASS、人工I/O比較PASS。
利用者の原文「次の作業に進んで大丈夫　修正が終わったらその先もできるなら行って」を、
科学条件を維持した技術修正と、その検証が通った後の新sourceでの一回実行への承認として記録する。
このsource準備中は登録係数・角度・operator・N・価格を評価せず、科学workerを起動していない。

## 修正と科学条件の保持

基点はv2公開確認 `808453725dfb6501b3d5bc8719701387f3d26b89`。
[v2停止報告](parent_allocation_v2_results_and_gpt_handoff_20261015.md)のとおり、総CPU上限3600秒で停止した。
旧v1/v2の結果・source・authorization・marker・STOPは保持する。

共通の`rte_development/evidence.py`、Guard、pipeline、candidate、law、bias、N、価格計数、backendは変更しない。
新しい[Parent専用writer](../../../src/trottertracks/algorithm_codesign/parent_experiment/evidence.py)が、
従来と同じJSONEncoderのfragmentを固定16 KiB bufferへ集約する。
UTF-8の境界でflushし、各blockの書込み前に従来のguardと全output-tree予算確認を行う。
encoderの順序・Fraction文字列・改行・exclusive `.partial`/link・fsync・512 KiB review chunkは維持する。
periodic SIGALRM 0.1秒、outerの監視、hard CPU、RSS、byte/log/call上限は変更しない。

既存コード変更はrunnerのwriter接続とlaunchのv3 path/hash・歴史的source保護だけ。
新helperの一時bufferは16 KiB＋従来16384文字encode slice以内（UTF-8最大約80 KiB）に限定する。
中断したpartialの位置はI/O単位の変更で異なり得るが、完成したJSONとchunkのbytesは同一。

[v3 contract](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/execution_contract_v3.json)は、
`execution`以外の全科学条件・capsがv2と完全一致する。
`execution`はv3 namespaceと`json_write_block_bytes=16384`のみを更新した。
[契約・source差分](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/contract_and_source_delta.json)を参照。

known 3-qubit distinct-basis、p=(1/5,3/10,1/2)、x=5/7、state000、sigma+1、finite P5、r1。
ordinary/closed P5とpartial-return/P3+tail/matched CTS、最大15構成/75 law、new key18・各一回、old guard36、root診断72を維持。
primitive1e-6、K256/H160、rho/eta1e-12、strict phase 100dps、axis1/200、alpha_axis1/40。
2 worker・numeric threads1、総CPU3600秒、key CPU120秒、RSS各上限、output64 MiB、wall上限なし、retry0。
旧v2の61 prefix行や価格を新runの結果として流用しない。固定D1 input/cacheのみを契約どおり使う。

## 技術検証と性能

[新15 tests](../../../tests/tracks/algorithm_codesign/test_parent_experiment_buffered_evidence.py)は人工データのみ。
旧writerとのexact byte/hash一致、UTF-8/escape/block/chunk境界、大きな有理数、外部fileを含む予算、
CPU guard中断、partial保全、exclusive再起動拒否、改ざん、fsync失敗、非有限値、固定bufferを検証する。
新15＋owner regression7＋既存experiment45＋native26＝**93 PASS、FAIL/ERROR/SKIP0**。
[実行記録](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/focused_tests_execution.json)は約1.62秒wall、peak69,392 KiB。
local evidenceであり、CIや外部再現ではない。

[人工I/O比較](../../../artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/artificial_writer_benchmark.json)は、
48 artificial record＋200人工receipt file、各実装一回、実Guardを使った比較。
出力20,574 bytesのSHA256は両者で一致。

| 指標 | 旧writer | 新buffered writer |
|---|---:|---:|
| wall秒 | 8.968810223974288 | 0.012745537795126438 |
| CPU秒 | 8.955824618 | 0.008115593999999504 |
| writerからのguard calls | 5474 | 3 |

本番の速度改善率・完遂をこの人工比較から保証しない。
hostは32 CPUs、memory約62.5 GiB。I/O処理はCPUで行う。
GPU inventory commandはdriver通信失敗で、GPUを計算に使用していない。
固定2 workers/threads1を維持し、他の並列条件・数学的精度を変えていない。

## Source固定と次の実行

478旧ファイル、v1の3 source Git blob、S2のlaunch/runnerの2 blobを保護する。
新manifestとzero-compute preflightでsource、契約、runtime、旧marker、新namespace未存在を確認する。
S3は本準備を囲む固定commitで、公開確認HEADとは区別する。
新しい実行はS3のauthorization-only直接子A3から開始し、v3 namespaceのみを使用する。
今回の継続承認は、source整合・guard・テスト・remote確認が通り、科学条件を変える問題が残らない場合の一回に限定する。
実行HEADへ移る前に原文と固定S3を承認JSON・receiptへ記録し、commit/pushする。

技術修正のために追加のGPT科学レビューを必要とする具体的な論点は現在ない。
科学条件や比較保証を変更する必要が生じればSTOPする。
新v3 run終了後は結果にかかわらずmandatory STOP。追加retryや次研究stage・GPTレビューを自動開始しない。
