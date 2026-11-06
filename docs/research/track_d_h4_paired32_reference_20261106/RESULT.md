# Track D Stage A：保存32 trajectoryのReference paired検証結果

2026-11-06 JST。**保存32本すべてで、Referenceは強いsymmetric-directional＋既存共有・融合よりCX数・depthを削減した。RZ数は全32本で増加し、qubitsは9→17。** 全32 pathのprecompile検査と128 full-wrapperのcompile後照合はPASS、retry/resumeは0。`TECHNICAL_PASS_MANDATORY_STOP`。これは固定有限cohortの結果であり、全sampling lawの母平均、必要shot数、同じenergy精度・成功確率での全RPE費用の優位は未評価である。

## 固定入力とSOURCE

[実行前の固定PLAN](PLAN.md)、[静的調査](../track_d_pr_rpe_static_survey_20261106/REPORT.md)、[保存word02の先行検証](../track_d_h4_reference_validation_20261106/RESULT.md)を基点とした。対象はlinear H4・隣接1.00 Å・STO-3G・Nα=Nβ=2、number-CDF8・元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=0.1の**1 h-step**。全pathともcomplete event16直後の同じcutを使った。

| 役割 | 固定identity |
|---|---|
| 今回の科学実行SOURCE | `2dd012dc77dcc877d5918c7d212819eb79b97557`。adapter・runner・人工回帰・PLANを実行前にcommitし公開 |
| 参照静的調査／先行実H4結果 | `523efe89fde52a14a8e0a4eba4ea6b70e3f512e1`／`7d0dc55199c49a3f49a9ba93323843d8a336cce1` |
| 再利用C科学SOURCE | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。143 SOURCE file・16保存入力、および保存lawを照合 |
| 保存結果のcarrier | `3649b75af191916daf44abb91e14d34c43b83e8f` の `artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/C/`。sample00..31のevents、独立full256 CAR reference、ordinary raw QPYの計96 bindingを保存。carrierは今回の実行SOURCEではない |
| Cohort | `near_practical_PR_r32_fixed32_v1`、master2026110304、32 events/path、PCG64。元sampling policyのpaired=false／adaptive=falseを保持。今回の「paired」は同じ保存path上の方式間比較を指す |
| Cohort SOURCE manifest SHA256 | `103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739` |
| CDF endpoint J/U SHA256 | `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| 元h/g/scalar packet SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |

完全なpath/hash/package bindingは[SOURCE.json](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/SOURCE.json)。SOURCE・入力・保存結果は実行前後で一致した。別cohort、別endpoint、別cutへの置換はない。元Hamiltonianの厳密基底状態を仮定し、状態準備費を除外した。保存own-DF ground vectorは使用せず、新規ground solve、DF再fit、samplingはすべて0。

## 最小拡張・位相・強いbaseline

新[adapter](../../../src/trottertracks/track_d_h4_paired_reference.py)と[runner](../../../scripts/run_track_d_h4_paired_reference.py)は、既存Track D H4 adapter、correctness harness、P0 compile/postcheckを再利用した。既存科学SOURCE・関数・テストは変更していない。共通head/providerをworker内で再利用し、保存eventsの入口だけを拡張した。

symmetric-directionalは前半headをuncontrolled、後半逆順headをdirectional、中央tailをordinary controlledとし、元のrun grouping、fsum、解析位相補正付きrelative Gaussian、tailの連続同basis共有を維持した。head relative fusion=true、cross-head/tail relative fusion=falseは両方式で共通。Gaussian全制御や共有を外した弱いbaselineへ変更していない。

ReferenceのBは前半head＋先頭16 events、Aは後半16 events＋逆順head＋scalar/identity一回。A†は実際の`A.inverse()`を保存した。scalar=2.29310124732、extracted identity=0.041217410629575235、scalar phase=−0.23343186579495756。pathごとのχはscalar/identity、event.phase、signed event rotationを含む解析metadataから求め、範囲は**[−0.24905823568816152, −0.20842967396583126] rad**。固定word02のχを全pathへ流用していない。真空ray、split位相、actual inverse、ABと保存ordered pathを照合した。λ_R=0.5000442435909978と各event normalizationを保存し、normalization／attenuationをunitaryやobservableへ乗算していない。

固定cutの両端でbasis keyが一致するpathは**14/32**。symmetricは全tailを共有する一方、Referenceは2 halfを別々に最適化するので、この14本ではcutが連続同basis共有を分断する。下記価格にはその損失を含む。別cutや融合有無の追加compileは行っておらず、損失の独立した因果的cost内訳は未評価である。

## Correctness

実行前の[既存人工回帰7件](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/technical/existing_synthetic.txt)と[今回の人工回帰2件](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/technical/paired_synthetic.txt)はPASS。今回の2件は符号依存χ／observableと、異なるendpoint・cohort・SOURCE・sampleの拒否を対象とした。人工回帰は分子sampling・compileを実行していない。

全32 pathについて、既存22項目/path（計704項目）に加え、raw X/Yの方式間paired observableを照合し、**全precompile PASS後にだけcompileを開始した**。χ省略の負対照は全32×X/Yで位相欠落を検出した。全128 wrapperのpostcheckとcompiled X/Yのpaired observableもPASSした。

| 保存checkからの最大値 | 観測値 |
|---|---:|
| Precompile既存704項目の最大成分差 | 1.59872e−14 |
| Postcompile全576項目の最大成分差 | 1.57147e−12 |
| Compiled vs CAR observable：最大成分／数値Frobenius差 | 4.73177e−13／7.38851e−12 |
| Compiled方式間paired observable：最大成分／数値Frobenius差 | 4.84168e−13／7.50794e−12 |
| Compiled Referenceのreference-lane leakage：最大成分／入力列L2 | 1.43933e−13／1.66972e−13 |

Referenceはreference/ancilla=0として全256 system入力basis列・各131072出力振幅を既存CPU Aer doubleのstate-actionで評価した。raw complex coherence、X/Y observable、compiled isometryとobservableを比較した。direct対照は全512 operatorで位相を含めてcompile前後を比較した。17-qubit全Operatorは構成していない。この検証は元H groundを含むsystem入力全体の誘導測定演算子を対象とし、任意のreference初期状態の保証ではない。

既存のcomplex128/binary64、component・unitarity abs1e−10、observable数値Frobenius／signal allowance2e−10、rtol0、global phase fitなしを維持した。失敗後の閾値緩和はない。これはlocal数値照合であり、丸め誤差の厳密包囲やimmutable CI evidenceではない。

## 128 full-wrapperの資源比較

全方式・全pathでP0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initial-zero false、topologyなし。既存`compile_checked`→MPFR160 declared-phase accumulation→位相保持unrollを使用した。MPFR160はphase蓄積の精度であり、全数値照合の精度ではない。H→core→YのみSdg→H→ancilla測定を含み、状態準備費を除外する。2方式×X/Y×32をそれぞれ一回、**計128 compile**した。

下表は**各軸32 pathの有限cohort単純平均**。X/Yは別々にcompileし、7資源指標は各pathで同じだった。X/Yを64独立pathとして集計していない。Δ=Reference−symmetric、SDは32本自身の変動（分母32）であり、母平均の標準誤差／CIではない。

| 指標 | symmetric平均 | Reference平均 | Δ平均 | Δ最小..最大 | Δ cohort SD | Reference低／同／高 |
|---|---:|---:|---:|---:|---:|---:|
| RZ count | 5531.09375 | 5814.15625 | +283.0625 | +202..+425 | 91.83135 | 0／0／32 |
| CX count | 2089 | 2042.9375 | −46.0625 | −72..−4 | 28.26431 | 32／0／0 |
| RZ depth | 1322.75 | 739.15625 | −583.59375 | −823..−395 | 98.29257 | 32／0／0 |
| CX depth | 1008.71875 | 493.5625 | −515.15625 | −639..−421 | 51.88997 | 32／0／0 |
| Total depth | 3071.71875 | 1649.53125 | −1422.1875 | −1939..−1011 | 214.57202 | 32／0／0 |
| Qubits | 9 | 17 | +8 | +8..+8 | 0 | 0／0／32 |
| Circuit size | 11247.65625 | 11623.40625 | +375.75 | +210..+659 | 184.83185 | 0／0／32 |

[trajectory別paired CSV](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/paired_resources.csv)／[集計JSON](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/finite_cohort_summary.json)に全64 axis/path行と集計を保存した。CX削減が4に留まるpathもあり、固定word02だけで見えた値を一律に当てはめることはできない。

Referenceの価格は追加8-qubit register、前後2 CSWAP cascade（各8対）、P(χ)、cutの共有損失を含む実際のfull wrapperである。depth減少とRZ／qubit増加を区別する。topology-freeのためrouting／接続制約は含まず、RZのfault-tolerant合成精度・T費用、reference registerの維持／reset費も未評価。単一の総価格へ換算していない。

## 保存物と実行記録

[run01/result.json](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/result.json)、[STOP.json](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/STOP.json)、[元出力inventory](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/inventory.json)を保存した。sample00..31の原出力はlocalで保全し、公開用には[32 archiveとmember index](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/run01/payload_archives.json)にまとめた。archive計212,460,247 bytes。全memberのbytes／SHA256を元inventoryと照合し、展開しても同一bytesである。

- QPY計352：raw wrapper128、compiled wrapper128、A/B/actual A†各32。
- NPZ計160：保存CAR／raw observable等のreferences32、compiled observable128。
- 各pathのconditions、fusion、precompile、4 compile／postcheck、result等のJSON計384。rootのSOURCE・paired比較・result・STOP等は別に保存。

archiveは`run01`で展開すると`sampleXX/`を復元する。private環境pathやtask cacheは公開対象に含めない。入力96件はSOURCE.jsonに元commit/path/hashを保持し、既存carrierのbytesを再利用した。

科学実行は一回、wall=6596.801秒。64 CPUと開始時MemAvailable=485612707840 bytesを観測し、16 path worker×4 numerical threads、小行列BLAS1を使用した。postcheck workerの最大RSS記録は8245824 KiB、parent peak RSS=27648 KiB。aggregate peakやchild合算CPUは測定していない。CPU/RAM/worker/wallの人工cap、共有環境設定・他jobの変更はない。実測対照のないspeedupは主張しない。[Python3.12.3とpackage一覧](../../../artifacts/track_d_h4_paired32_reference/2026-11-06/technical/packages.txt)、固定10 pinsを記録した。

実行入口は以下（変数は今回のtask専用復元入力／venvに対応し、再実行指示ではない）。

```bash
"$TRACK_D_PYTHON" -B scripts/run_track_d_h4_paired_reference.py \
  --origin "$TRACK_D_ORIGIN" --saved "$TRACK_D_SAVED" \
  --output artifacts/track_d_h4_paired32_reference/2026-11-06/run01 --execute
```

新sampling・DF refit・ground solve・量子Born shots・RPE shot評価・RPE round評価・retry/resumeは0。Aer state snapshotの内部`shots=1`は量子shot実験ではない。既存grant・ledger・marker・STOPは変更・再消費していない。

## 終了と未解決事項

本結果は「同一endpoint・同PR条件の保存32 pathでもCX／depth削減が維持されるか」というStage Aに対し、上記P0・固定cutの範囲で肯定的な有限cohort証拠を与える。全lawの期待価格・母平均CI、元H ground上のRPE信号と必要shot数、同energy精度・成功確率の全round費用、routingや合成精度を含む最終資源指標は未解決である。

専用branch `codex/track-d-h4-paired32-reference-20261106` へSOURCEと本成果物をcommit/pushし、remote SHA・公開Git tree/blobとlocal bytesを照合して終了する。旧科学結果・固定条件・STOPは維持する。

**mandatory STOP。次担当はユーザー／GPT**：CX／depth減少とRZ／qubit増加を踏まえ、主資源指標と継続の要否を判断する。Codexは次の明示許可なしにB段階（元HのRPE統計・必要shot数）、C段階（全round総費用）、追加条件・新sampling・詳細科学レビューへ進まない。
