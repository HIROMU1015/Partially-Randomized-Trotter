# Controlled Hadamard全18round接続：統合runnerの技術準備

2026-11-04開始、2026-11-05 JST完了 / LOCAL / 専用branch。
**有限parameter・support差・controlled lowering・18round/36軸strict判定を一つのrunnerへ統合した。
人工147件PASS。実H4の新しい数値評価・包含結果の再解析は0件。**
H4での接続成立はまだ判定していない。
RPE_tolerance_certified=false、scientific_status=UNDETERMINED、
U_virtual_IID_upper=null、U_exec_upper=nullを維持する。

## SOURCEと再利用

基点は57b4792728c40acb337d69d0120dd0417b257bbd。
前回の[数学的解析](../rpe_categorical_circuit_connection_20261104/CONNECTION_ANALYSIS.md)と
[計画](../rpe_categorical_circuit_connection_20261104/SCIENTIFIC_VALIDATION_PLAN.md)を変更せず再利用した。
その計画にあったSCIENTIFIC_RUNNER_NOT_READYは当時の記録であり、
本成果で技術実装は完了した。H4科学実行の許可を意味しない。

| 新規ファイル | 必要な差分と再利用先 |
|---|---|
| [run_connection.py](run_connection.py) | defaultのbyte/schema/literal validator、明示的な--run、固定SOURCE確認、H4 loader、資源観測。既存run_round_enclosures.metadata、run_local_probes.verify_preparation/worker_environment、保存H loader、prepare_df_partial_s2、support登録を呼ぶ |
| [integration_audit.py](integration_audit.py) | Operatorを追加評価しないdescriptor export、全component coverage、primitive payload照合、有限control template監査、strict classifier |
| [connection_pipeline.py](connection_pipeline.py) | 前回connection_adapterのscalar_gap/block_gap/boundary_bound/one_particle/component_gap/total_transferを接続。既存round_adapter.decodeを使い、並列workerのMatrixBallを半径込みで復元 |
| [SOURCE_INPUTS.json](SOURCE_INPUTS.json) | 旧FIXED_BINDINGS.jsonの固定commit/blob/SHA256へ接続。追加依存SOURCEとQiskit関連6ファイルを固定 |
| [test_connection_runner.py](test_connection_runner.py) | 人工fixtureによる同じpipelineのend-to-end、失敗・cache・parallel・strict比較のテスト |

数学的な誤差伝播やcategorical平均を新しく実装し直していない。
新規実装が必要だったのは、旧runnerには全selected descriptorのexport、全supportの包含、
boundary差と保存36軸を結ぶclassifierがなかったため。
旧export_adapter.export_basisはlocal Operatorを数値評価するので、人工fixtureの作成だけに使用した。
新しい実行側exportは既存metadata/runtime gateのclass・wire・parameter・canonical hashを読み、
保存full exportのlocal matrix欄を再計算せずに照合する。

GitHubで前回解析/計画、C run02、全18categorical報告、C SOURCEのrepeated builder、
categorical SOURCEのround_adapter、m17報告の公開blobを確認した。
C SOURCE7279f700a、C結果82b2df6af、categorical SOURCEf4905fff5、結果d2e7a7db6、
m17結果34840ce9bのprovenanceを区別した。新branchへ他研究の結果をmergeしていない。

## 実装された接続

1. 保存4 half primitiveとblock0 full primitiveをproductionと同じgeneratorで生成する。
   保存half payloadと2つのraw scalarはhex一致を要求する。
2. scalar gをproductionと同じ「−δ×q×(constant＋extracted)」の順で生成し、
   生成済みbinary64をexact rationalへ写してd_sを求める。
   block差d_bと合わせ、Δ_boundary=min(2,d_s+(q−1)d_b)。
3. full11＋selected最大218 basisのruntime class/descriptorを確認する。
   RZとJW隣接XXPlusYYだけを認め、一粒子8×8のAcb96-bit包含を作る。
   両MatrixBallのradiusを含むcolumn差からΔ_Sを計算する。
4. RZ/RZZ.control(1)の有限template、parameter doubling/halving、wire順、closed control、
   global phaseを確認する。CZ/PhaseのSOURCEも固定する。
   全必要条件の確認時だけΔ_lowering=0。証拠不足はUNKNOWN/STOP。
5. 各mに対して
   Δ_extra=min(2,Δ_boundary+3qr max Δ_S+Δ_lowering)。
   保存U_cat＋Δ_extra＋保存shot半幅上端 < 保存εをFractionでstrict比較する。
   保存I_catも±Δ_extraへ拡張する。
   **既存I_catに含まれるevent/product π64を再加算しない。**

Δ_lowering=0は、RZ/RZZからCRZ/CCX/CXへの固定templateと、標準CZ/Phaseを
exact logical gateとして扱う範囲の主張である。CCX等の再帰的native分解・transpile、
binary64 Statevector/to_matrixの算術誤差を0と認定するものではない。
basis inverseは固定RZ/XXPlusYY SOURCEのexact parameter符号反転に基づく。
測定wrapperのRe/+Imとbit0=+1は前回の代数・SOURCE解析を再利用し、実H4 wrapperを新buildしない。

保存36軸のinterval/shot幅/U_catとN/α/β/εは読出し・照合だけ行った。
今回classifierを実H4保存値へ適用していない。
signal/model τの取り違えを拒否し、θ/order lawは保存hex/CDF/massの一致を確認する。
τからatanやCDFを作り直さない。
normalized ray/sector/endianのidentityは既存B/run02のbyte provenanceと保存metadataを再利用する。
状態・norm²の再計算を追加しない。

## Cache・並列化・部分結果

同じbasis hash/parameter列は1回だけ評価する。aliasesの内容差は拒否する。
独立basisをspawn processで並列化し、各workerではAcb96 bit・FLINT1 threadを使う。
worker間はMatrixBallのexact center/radius/normをserialized形式で渡し、
既存decodeが復元時の不確かさも保持する。計算順を変えた行列積への置換はしない。

科学起動時にCPU affinity・CPU idle・MemAvailable・親process RSSを観測してworker数を選ぶ。
BLAS環境変数は当該processのspawn期間だけ設定し、共有venv/OS/他jobを変更しない。
恣意的なCPU/RAM/worker/wall-time capは置かない。
小さいAcb行列にGPUは選ばず、cacheと独立basis並列を使う。
人工直列/2-workerの結果一致を確認したが、H4 speedup倍率は主張しない。

fresh directoryだけにRESULT.json、DESCRIPTORS.json、basis_enclosures.jsonlを保存する。
stage/call開始数、完了済みcomponent/round、失敗理由を残す。
basis workerは共有stop通知後の未開始taskをskipし、稼働済みtaskを回収する。
workerの異常終了等で開始数が確定しなければ、その不確定性を記録する。
必要な前提の不一致は途中STOP。strict条件だけを満たさない場合は、事前固定した36比較を
1 passで完了してUNDETERMINEDを報告する。追加precision、retry/resume、別候補へ移らない。
既存出力への上書きを拒否する。

## 人工テスト結果

新規33件＋前回114件＝147件PASS。
新規は人工2-mode preparationとmock保存軸を用い、18round・36軸の同一pipelineを通した。
人工rは1/2/4/8/16/32、qは2^mだが、q-sized circuitやmany-body evolutionは作っていない。
保存軸のmock値から得たPASSを、人工物理モデルの科学的成功保証ともH4の結果とも扱わない。

確認内容はSOURCE/blob/hash不一致、q/r/K誤結合、class/wire/parameter/hash不一致、
非隣接XXPlusYY、support coverage、half payload不一致、lowering不足、
subnormal/非有限値、丸め順序とscalar相殺、negative/missing bound、
equal/近接εのexact strict比較、π64二重計上防止、部分結果STOP、出力保全、
cache・serial/parallel一致。
人工の相殺fixtureはmock scalarを注入したもので、productionのpreparation整合性検査を緩めていない。

~~~sh
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python -m pytest -q \
  docs/research/rpe_controlled_connection_runner_20261104/test_connection_runner.py \
  docs/research/rpe_categorical_circuit_connection_20261104/test_connection_adapter.py \
  docs/research/rpe_controlled_hadamard_preparation_20261104/test_circuit_semantics.py \
  docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/test_mean_enclosure.py \
  docs/research/rpe_controlled_hadamard_runner_v2_20261104/test_local_probes.py
~~~

実行結果は147 passed in 17.48s。local技術テストでありimmutable CI/独立科学レビューではない。
科学入口run_science/h4_contextは起動していない。
default実行ではPREPARED_NOT_SCIENTIFICALLY_EXECUTED、
18round・36軸・new_H4_scientific_calls=0を返した。承認証明・grant機構は追加していない。
正式共通運用指示2026-10-25の§8.1、§8.4、§10を読み、旧SOURCE/raw/run01/run02/STOPを保全した。

## 残る評価と次段判断

技術SOURCEはREADY_FOR_SEPARATE_SCIENTIFIC_PERMISSION。
実H4のd_s/d_b、selected descriptor適合性、全Δ_S、lowering有限audit、
18個のΔ_extra、36軸strict比較は全て未実施。
operator一様上界が広く、実行後もUNDETERMINEDとなる可能性は残る。
既存12局所probeのPASSを全trajectoryへ拡張していない。

推奨は[固定科学実行計画](SCIENTIFIC_EXECUTION_PLAN.md)の有限auditを1 pass実施すること。
ユーザーが本commitのSOURCE・scopeへ別途明示許可した後、Codexがその1 passだけを実行する。
PCG64 IID、Float64仮想測定、compiled/noisy、状態準備、Dのα/decoder/E0、
全体RPE成功保証は接続完了後も別の未保証事項であり、本runnerは評価しない。

本成果のcommit/push・remote SHA/公開blob照合後はmandatory STOP。
