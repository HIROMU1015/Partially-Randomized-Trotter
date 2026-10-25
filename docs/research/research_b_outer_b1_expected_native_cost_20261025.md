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

固定SOURCEのclean detached checkoutでread-only preflight後、次のコマンドを一回実行した。使用済みoutputで再起動しない。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-b1-science-20261025/scripts/run_b2d_outer_b1.py \
  --execute --source-commit 11e2532a62d54a65f8fad5489be6b135f826011b \
  --output /tmp/prt-research-b-outer-b1-science-20261025/artifacts/research_b_outer_b1/20261025/iid256-once
```

## 実行結果

科学SOURCE [`11e2532a62d54a65f8fad5489be6b135f826011b`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/11e2532a62d54a65f8fad5489be6b135f826011b)。最終状態 **INCONCLUSIVE**、runtime `COMPLETED_DESCRIPTIVE_MANDATORY_STOP`。人工45件PASS（B1 9、B0 opaque 11、followup 10、Stage1 launcher 15）、read-only preflight PASS。新しい独立branch `codex/research-b-outer-b1-iid256-20261025` のclean detached checkoutで一回だけ実行した。SOURCE変更・retry・resume・追加attempt・追加標本は0件。

原本 [iid256-once/](../../artifacts/research_b_outer_b1/20261025/iid256-once/)、[全完了標本CSV](../../artifacts/research_b_outer_b1/20261025/SAMPLES.csv)、[実行要約](../../artifacts/research_b_outer_b1/20261025/SUMMARY.json)、[process記録](../../artifacts/research_b_outer_b1/20261025/execution_logs/)、[人工test記録](../../artifacts/research_b_outer_b1/20261025/artificial_tests/)。保存trajectoryは3,072件、correctness全通過して費用保存した標本は3072／3,072件（X/Y費用6144／6,144件）。

### Correctnessと完了範囲

完了標本の最大残差：`{"compiled_wrapper": 4.550538909283757e-13, "controlled": 3.8985950626951644e-14, "raw_wrapper": 3.644847141100376e-14, "system": 1.493683375107909e-13, "raw_signal": 3.26405569239796e-14, "compiled_signal": 2.871036741680655e-13}`。Operator2-norm tol1e-12／rtol0／phase fitなし。Input gateはparentとworkerで実行し、入力identity・DF preparationを固定snapshotと照合した。

公開用raw全10fileは科学checkout原本とbyte一致を確認した。Sample／trajectory ID全3,072件と登録scopeが一致し、公開CSVは費用6,144行・統計168行。保存rawから等重みmean／variance／SEを独立に照合した（新しいoperator評価・build／compileなし）。旧科学結果・AGENTS.md・共通運用指示への変更はない。

固定標本を全件完了した。18,432 operator gate／12,288 expectationと6,144 logical費用要求が通過し、24統計群を生成した。Raw／compiled QCを保持するcache hitも各sampleの検査を省略しない。実現trajectoryの経験的correctnessであり、保証付きgammaではない。

実環境はPython3.11.9、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Qiskit Nature0.7.2、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0。詳細platform／executableとsnapshot blobはbinding.jsonに記録した。Input gateはparent1＋worker32の計33回PASSした。

### Candidate別期待費用と標本SE

各値はmean ± SE、256 IID／case、full prepared wrapper／axis。T別・X/Y別に記載する。

| ID | T | Axis | Expected RZ ± SE | Expected CX ± SE | Relative SE RZ / CX |
|---|---|---|---|---|---|
| B3 | +0.18 | X | 4096.89453 ± 1.59814 | 2788.84375 ± 1.26439 | 0.039009% / 0.045337% |
| B3 | +0.18 | Y | 4096.89453 ± 1.59814 | 2788.84375 ± 1.26439 | 0.039009% / 0.045337% |
| B3 | −0.18 | X | 4098.81641 ± 1.70473 | 2790.48437 ± 1.35475 | 0.041591% / 0.048549% |
| B3 | −0.18 | Y | 4098.81641 ± 1.70473 | 2790.48437 ± 1.35475 | 0.041591% / 0.048549% |
| D1 | +0.18 | X | 27423.63281 ± 1.65513 | 18772.04688 ± 1.29393 | 0.006035% / 0.006893% |
| D1 | +0.18 | Y | 27423.63281 ± 1.65513 | 18772.04688 ± 1.29393 | 0.006035% / 0.006893% |
| D1 | −0.18 | X | 27423.23828 ± 1.67576 | 18771.53125 ± 1.31642 | 0.006111% / 0.007013% |
| D1 | −0.18 | Y | 27423.23828 ± 1.67576 | 18771.53125 ± 1.31642 | 0.006111% / 0.007013% |
| R2 | +0.18 | X | 27548.05078 ± 2.32664 | 18868.60937 ± 1.80706 | 0.008446% / 0.009577% |
| R2 | +0.18 | Y | 27548.05078 ± 2.32664 | 18868.60937 ± 1.80706 | 0.008446% / 0.009577% |
| R2 | −0.18 | X | 27549.84766 ± 2.38316 | 18870.20312 ± 1.84333 | 0.008650% / 0.009768% |
| R2 | −0.18 | Y | 27549.84766 ± 2.38316 | 18870.20312 ± 1.84333 | 0.008650% / 0.009768% |
| R4 | +0.18 | X | 27424.37109 ± 1.70766 | 18772.65625 ± 1.34921 | 0.006227% / 0.007187% |
| R4 | +0.18 | Y | 27424.37109 ± 1.70766 | 18772.65625 ± 1.34921 | 0.006227% / 0.007187% |
| R4 | −0.18 | X | 27423.45312 ± 1.44781 | 18771.62500 ± 1.15785 | 0.005279% / 0.006168% |
| R4 | −0.18 | Y | 27423.45312 ± 1.44781 | 18771.62500 ± 1.15785 | 0.005279% / 0.006168% |
| B4 | +0.18 | X | 2087.82422 ± 1.42353 | 1416.09375 ± 1.13304 | 0.068182% / 0.080012% |
| B4 | +0.18 | Y | 2087.82422 ± 1.42353 | 1416.09375 ± 1.13304 | 0.068182% / 0.080012% |
| B4 | −0.18 | X | 2089.28125 ± 1.35072 | 1417.17187 ± 1.06539 | 0.064650% / 0.075177% |
| B4 | −0.18 | Y | 2089.28125 ± 1.35072 | 1417.17187 ± 1.06539 | 0.064650% / 0.075177% |
| F1 | +0.18 | X | 10896.96484 ± 2.53092 | 7429.21875 ± 2.02413 | 0.023226% / 0.027246% |
| F1 | +0.18 | Y | 10896.96484 ± 2.53092 | 7429.21875 ± 2.02413 | 0.023226% / 0.027246% |
| F1 | −0.18 | X | 10894.18359 ± 2.66979 | 7427.01563 ± 2.10633 | 0.024507% / 0.028360% |
| F1 | −0.18 | Y | 10894.18359 ± 2.66979 | 7427.01563 ± 2.10633 | 0.024507% / 0.028360% |

全metricのvariance／min／max／count／参考95%区間は[STATISTICS.csv](../../artifacts/research_b_outer_b1/20261025/STATISTICS.csv)と[result.json](../../artifacts/research_b_outer_b1/20261025/iid256-once/result.json)に保存した。Normal95は厳密coverageではなく、多重比較補正もない。RZ/CXの参考区間重複は8／120比較：[SUMMARY.json](../../artifacts/research_b_outer_b1/20261025/SUMMARY.json)。区間が分離しても同精度総資源優位を意味しない。

RZ/CXの参考区間では、両T・両axisとも `B4 < B3 < F1 < {D1,R4} < R2` の群間差が分離した。D1対R4の8区間比較はすべて重なり、順位を識別できない。全24群でRZ／CX relative SEの診断目標2%を満たし、最大はB4/+TのCXで約0.0800%。これは観測標本のSEであり、未観測rare eventの寄与や厳密coverageを保証しない。X/YのRZ／CX記録は各sampleで同一だったが、独立標本数を512へ増やした扱いにはしない。

### Rare event coverage

| Case | K | Observed orders | Max p(order=K) across occurrences | Order K observed |
|---|---|---|---|---|
| B3:-9/50 | 2 | {'0': 3072} | 2.0234067e-05 | False |
| B3:9/50 | 2 | {'0': 3072} | 2.0234067e-05 | False |
| B4:-9/50 | 2 | {'0': 2048} | 1.7454515e-06 | False |
| B4:9/50 | 2 | {'0': 2048} | 1.7454515e-06 | False |
| D1:-9/50 | 2 | {'0': 3072} | 2.0234067e-05 | False |
| D1:9/50 | 2 | {'0': 3072} | 2.0234067e-05 | False |
| F1:-9/50 | 2 | {'0': 8192} | 1.0909106e-07 | False |
| F1:9/50 | 2 | {'0': 8192} | 1.0909106e-07 | False |
| R2:-9/50 | 2 | {'0': 6144} | 5.0586617e-06 | False |
| R2:9/50 | 2 | {'0': 6144} | 5.0586617e-06 | False |
| R4:-9/50 | 4 | {'0': 3072} | 6.8238752e-11 | False |
| R4:9/50 | 4 | {'0': 3072} | 6.8238752e-11 | False |

51,200 sampled short-stepはすべてorder 0で、order 2とorder 4の観測数は0だった。これは指定lawから得た固定IID標本の結果であり、次数を強制置換したものではない。B1 correctnessの実績は実際に抽出されたtrajectoryに限定され、高次数trajectoryを今回新たに検証したとはいえない。K2側もrare-event coverageが不足し、R4のorder-2確率も正である。各occurrenceの全次数確率はresult.json、case別確率範囲と明示的な0観測数はSUMMARY.jsonへ保存した。

R4の最大次数未観測は費用ゼロの根拠にならない。観測標本に基づく小さいSEやnormal区間は未観測tailの影響を保証しない。全次数の理論probabilityとoccurrence別観測をraw resultへ残し、固定標本を増加・層別化しない。

### Compile効率と資源

32 process、worker内thread各1、RAM由来LRU10 trajectory／worker。集約cache／compiler記録：`{"actual_metric_queries": 6078, "compiler_cpu_seconds": 34401.567642493006, "compiler_wall_seconds": 34851.591293300036, "completed_samples": 3072, "evictions": 2719, "hits": 66, "misses": 6078}`。Logical要求と実transpile／metric queryを区別する。Worker CPU合計92819.032秒、worker単体最大RSS 1,120,720KiB。各worker peak RSSの単純和は同時aggregate peakではない。Whole-job wall／CPU／RSSは[time.txt](../../artifacts/research_b_outer_b1/20261025/execution_logs/time.txt)を参照する。速度向上率の対照測定は行っていない。

Whole-job wallは **50分46.95秒**、CPU合計92,876.33秒（30.48コア相当、CPU3048%）、exit0、swap0、保存失敗0。Logical compile要求6,144に対しactual transpile6,078、実回路cache hit66。Workerは各95–97標本を担当し、wall範囲2,835–3,028秒。Compiler時間34,851.59秒wall／34,401.57秒CPUはworker分の累積で、whole-job経過時間ではない。実行後のSTOPは2026-10-25 20:25 JST（原本UTC11:25）に保存された。

### 確認できたことと次の判断

B0は決定論的代表の限定費用であり、今回のIID期待費用標本に混ぜていない。A0／B0／22行の保存数値を再解析せず、旧SOURCE・raw・grant・STOPを保全した。Gamma／u_num／shots／shot-weighted Gはnull。RZ単価、観測bias、A0のEMPIRICAL_CONSISTENCYから同精度達成・科学的GO・総資源優位を認定しない。ローカル固定SOURCE実行でありCIや外部再現ではない。

次の担当は研究Bのユーザー。Rare-event対応の追加／層別MC、shot／gamma／保証付き数値包含のどれを優先するかを判断する。Codexは採択・別許可後にだけその固定作業を行う。今回は公開内容照合後mandatory STOPし、追加科学計算へ進まない。
