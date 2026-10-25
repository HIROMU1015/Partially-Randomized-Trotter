# 研究B B1：IID full-wrapper期待native費用

## 固定実行契約

本SOURCEのB1は一回の固定IID Monte Carloである。科学SOURCEは本実装commitで固定し、結果公開commitと区別する。実行前の本節は契約であり、結果を主張しない。

基礎は[B0完了](research_b_outer_b0_recovery_20261025.md)の科学SOURCE `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e`、作業起点 `0db9b3ed189eb6cd9d66e070993dbd0e72d29104`。[既存B1設計](research_b_outer_numeric_uncertainty_native_cost_design_20261025.md)を引き継ぐ。旧A0・B0・22行analytic結果を再解析・再実行しない。

Synthetic 2-mode one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、n_D=1、T=±0.18、scalar／identity分離とfinite-RTE lawを維持する。分子geometry／AO basis／分子DF rank探索は非該当。

| ID | Outer | D内部 | q/f | r/K |
|---|---|---|---|---|
| B3 | Yoshida4 | A2 | q2 | 2/2 |
| D1 | Yoshida4 | A3 | q2 | 2/2 |
| R2 | Yoshida4 | A3 | q2 | 4/2 |
| R4 | Yoshida4 | A3 | q2 | 2/4 |
| B4 | SABA2 | A2 | q2 | 2/2 |
| F1 | Refined-S2 | A2 | q2,f8 | 2/2 |

12 case×256 IID trajectory＝3,072、paired X/Y＝6,144 full-wrapper build／compile要求。科学実行開始後のretry／resume／別outputでの追加attemptは禁止する。中断時にも部分標本・該当回路・数値診断・STOPを保存し、費用順位を確定しない。標本追加、shots、A1、Stage 2、全RPE・FT費用は対象外。

## 再利用と最小拡張

入口 [`run_b2d_outer_b1.py`](../../scripts/run_b2d_outer_b1.py)、接続 [`b1_native.py`](../../src/trottertracks/b2d_evaluator/b1_native.py)、人工test [`test_b2d_b1_native.py`](../../tests/management/test_b2d_b1_native.py)。B0 renderer／独立数学列にsampled wordsを渡すkeywordと、既計算operatorをexpectationに渡すkeywordのみを追加した。既定B0挙動は維持する。EvidenceはB1専用出力schema／容量を受け取れるようにし、既定の旧出力policyは変更しない。

`rte.iter_sample_rte_events`、PCG64、既存finite signed lawを利用する。Seed規約は `sha256-b1-v1/master41/candidate/T/sample/occurrence/step/PCG64`。規約名、41、candidate ID、canonical FractionのT、sample index、occurrence、short-stepのJSON配列をSHA256へ渡し、先頭16byteをbig-endian整数にする。各short-stepは独立subseedで一件抽出し、B0のevent生成と同一かを照合する。X/Yには同じ古典trajectoryを用いる。Python hashやB0代表の流用はしない。全seed・event index／order／probability／angle／phase／normalizationをJSONL保存する。

CompilerはB0 v5そのもの：Qiskit1.3.0、`rz/sx/x/cx`、opt1、seed17、approximation1.0、`qubits_initially_zero=False`、topologyなし、`official_definitions_angle_preserving_native_v5`。微小角・離散phaseの扱いを変更しない。Wrapperはmode0準備、ancilla準備、controlled full evolution、X/Y終端、measurement各1を含む。

## Scope・correctness・統計

Metadata scopeは3,072 sample／receive、6,144 build／compile要求、18,432 operator gate、12,288 expectation、6,144 cost要求、51,200 sampled short-step、24統計群。Workerのfixture／DF preparationは一回ずつ再利用する。有限のnative primitive境界と実測呼出数も保存し、旧B0の代表・analytic mean scopeは実行しない。

Complex128、operator spectral 2-norm absolute atol=1e-12、rtol=0、phase fitなし。固定snapshotとHamiltonian／DF／state／split identityを照合する。各trajectoryについてsystem4×4、controlled8×8、両axisのraw／compiled8×8とHadamard expectationを独立参照に照合し、両axis全gateがPASSした後にその費用を記録する。最終期待費用・比較は全3,072件PASSした場合のみ作成する。Realized trajectory correctnessとensemble meanの数学的保証を区別する。

`CompiledMetricAccumulator(weighted=False)`を薄いadapterで再利用し、canonical sample順の等重みWelford集計でmean／unbiased variance／SE／min／max／countを保存する。Event probabilityを標本へ再乗算しない。全24群でRZ／CX count、RZ／CX depth、total depth、size、measurement数を記録する。Relative SE 2%は診断目標、mean±1.96SEは参考normal区間でcoverage保証ではない。区間重複と各R occurrenceの理論次数確率・観測数・component数を保存する。K4最大次数未観測なら `INCONCLUSIVE_RARE_EVENT_COVERAGE` とし、未観測費用をゼロとせず、追加標本を起動しない。

## 実行資源・保存・停止

Affinity CPUと実MemAvailableを確認してprocess並列化し、子processのBLAS/OpenMP/Rayon threadを各1にする。空RAMの70%を計画用とし、worker baseline512MiB＋実回路trajectory64MiBの見積りでLRUサイズを決める。これは実測メモリ保証ではないため実RSSを記録し、空RAMが開始時の10%未満になる場合は自jobを停止する。固定科学標本数を資源上限と解釈しない。Global設定・他jobへ干渉しない。

同一物理trajectoryは同じworkerへ集める。Cache identityはcase、signed RTE words、入力identity、SOURCE、control／準備／axis、compiler profileを含む。LRUは実QC・operator・参照・metricsを一体で保持し、hit時にもoperatorとexpectationを再検査する。Metricsだけのhitをcorrectness根拠にしない。Logical要求数とactual transpile／metric query、hit／miss、wall／CPU／RSSを区別する。

Fresh output `artifacts/research_b_outer_b1/20261025/iid256-once/` にtrajectory／sampleをexclusive streaming保存し、Evidence総容量128MiB、JSON単体8MiB、STOP reserve512KiBを用いる。失敗標本はgzip+base64 QPYとoperator matricesを保存する。依存version・Python・platform・SOURCE・snapshot provenanceを記録する。`--inspect`はmetadataのみ、`--execute`のみ科学入口であり承認証明ではない。承認packet等は不要。

新SOURCEのclean detached checkoutでread-only preflight後、次の一回だけを実行する（SOURCEとcheckoutは固定後に記録）。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-b1-science-20261025/scripts/run_b2d_outer_b1.py \
  --execute --source-commit <B1_SOURCE> \
  --output /tmp/prt-research-b-outer-b1-science-20261025/artifacts/research_b_outer_b1/20261025/iid256-once
```

## 実行結果

人工45件PASS（B1 9、B0 opaque 11、followup 10、Stage1 launcher 15）。B1最終変更後の9件も再PASS。Metadata inspectは科学importなしで3,072／6,144／51,200を確認。SOURCE固定・preflight前であり、人工テストと科学的PASSを区別し、科学結果は本節へ追記する。Gamma／u_num／shots／shot-weighted Gはnullのまま。B0の代表費用をIID期待費用へ混ぜず、RZ単価だけで科学的GO・同精度総資源優位を認定しない。実行・保存・公開照合後はmandatory STOP。次の科学計算の採否は研究Bのユーザーが判断する。
