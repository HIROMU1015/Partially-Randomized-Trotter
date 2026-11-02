# 研究C：PR-aware DF Stage B2・native費用校正の限定実行

2026-10-27 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**技術的中断を修正したfresh run02が652/652 COMPLETE、correctness FAIL0で完了した。64 kernelのfull512・raw/compiled/reference・actual adjoint・X/YがすべてPASS。** RZの標本平均はBaseline5198.3125／Proposed4765.875で、今回の固定32標本/armではProposedが約8.3188%小さかった。これは有限bit校正の記述統計であり、ideal law母平均・保証付き費用優位・CA・shots・総資源の資格ではない。

## SOURCE・条件・技術的再実行

| 項目 | identity |
|---|---|
| 元B2科学SOURCE | `e74226a817e65918c0e00a735498e5bada77146d` |
| 元B2数値実装 | `4d7d9dbd3b521e288afd7e47aaecaf16facd6d3c` |
| run02科学SOURCE | `a1cf0046967c206f414dbad568a75884ee076ebf` |
| 修正数値実装 | `3e1e8d1446de04b9ebec5c0bcf455e7e7a47d17a` |
| [技術再実行契約](research_c_pr_aware_df_stage_b2_compat_contract_20261027.json) SHA256 | `23763d603c242f59001239e132c604f3f4f3b3cb7c71efc520bd30a65e09f7ce` |
| [新SOURCE50件](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/compat_technical/SOURCE.json) SHA256 | `e6facbf59838f93dcdd61258c0589da7b3dadb2116a041209d7a08fb8f471d28` |
| sampling用SOURCE digest（元のまま） | `cdf79ed569127f5cb6c7ee017b76bce6422166a656bb3ae866de6e86917c45bf` |
| 元H4 input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| Stage A最終Baseline J/U SHA256 | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` |
| Stage A最終Proposed J/U SHA256 | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` |

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8の2 arm、Nα=Nβ=2／sector36、system8＋ancilla1／full controlled512。元frame順prefix LD2、元g補正κ、核scalar一度、partial symmetric S2、δ=.1／s4／h1/40／r16／K0。Own-DF ground-state oracle、状態準備費用除外、CA=.00159／DF reserveCA/8。共通exact spin-lift κ基底・保存full-frame U・PrecisionBuilder・head内部relative fusion・既存tail隣接同一basis共有、singleton／cross-head-tail一般相対融合なし。P0 Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／qubits_initially_zero=false、MPFR160 declared phase、phase fittingなしを維持した。

ユーザーは実行と予期しない中断の修正・再実行を明示許可した。run01は429 units（428 COMPLETE、kernel_build proposed_17のRuntimeError `Already borrowed` 1件）で技術STOP。数値correctness FAILは0、部分18 metricsから候補平均を作らなかった。旧run01／SOURCE／契約／STOPはcommit abc06e759で保全した。[技術原因・修正・20人工/回帰PASS](research_c_pr_aware_df_stage_b2_repair_20261027.md)。

Qiskit内部のSessionEquivalenceLibraryを使うkernel build／P0 compile／actual-adjoint回路構築だけを同一process-local mutexで保護した。旧executorは不変で、CAR／Operator decode／行列・X/Y検証は並列のまま。新SOURCEとsampling SOURCEを分離し、fresh run02で**元と同じ64標本を最初から評価**した。旧50 trajectory記録と新run02の該当記録が完全一致した。これは114独立標本ではなく、集計対象はrun02の64標本だけ。旧成功prefix／cacheのresume、標本追加・探索はしていない。

## 正しさ・検証範囲

SOURCE/input/evidence gate、両armのbase＋8frame full CAR対保存sector、law/identity照合、Gaussian18／relative8、共通Hadamard head/X/Y shell3がPASS。保存B1のDF予算・own-ground／alias・13roundのPF/RTE/Aを参照し、これらの誤差やgroundを再計算していない。

全64 trajectoryで以下がPASSした。

- full512の全入力列／出力成分でraw対独立CAR reference、compiled対reference、compiled対raw、raw／compiled unitarity。
- Rawとcompiledのactual circuit inverse各64（計128）をbuild/decodeし、reference adjointとidentityへの復帰を照合。
- 保存own-DF groundをfull256へ埋め込み、verified shellを用いたraw／compiled X+iYを各trajectoryの独立reference expectationへ照合。

Gaussian gate単体のsector射影や有限state probesで代用せず、control0 identity／control1 evolutionを全operatorで比較した。Scalar／tail identity／Gaussian方向／chronological順序／相対位相を含む。単一unitary trajectoryのX/Yにはattenuationを掛けない。これはB1のlaw平均信号や量子measurement sampleではない。

| 最大観測残差 | Baseline | Proposed |
|---|---:|---:|
| raw/reference max-entry | 4.55255430e-15 | 6.78160008e-15 |
| compiled/reference max-entry | 7.42160756e-13 | 6.62153033e-13 |
| compiled/raw max-entry | 7.40939032e-13 | 6.60931542e-13 |
| compiled unitarity max-entry | 1.13131726e-13 | 1.00697228e-13 |
| compiled adjoint/reference max-entry | 7.43454281e-13 | 6.63424284e-13 |
| compiled adjoint return max-entry | 1.06283155e-13 | 9.80411885e-14 |
| raw complex X/Y error | 3.78347923e-15 | 8.77631839e-15 |
| compiled complex X/Y error | 1.01244948e-13 | 9.24815790e-14 |

Matrix allowance1e−10、complex signal2e−10、rtol0を事後変更していない。Compiled/reference Frobeniusの観測最大はBaseline8.6951549351e−12／Proposed7.3537771266e−12。これらは64標本内のbinary64残差で、全sampling lawのglobal kernel operator上界や厳密rounding enclosureではない。未観測trajectoryへの資格はnullのまま。

## Complete kernel費用・標本不確かさ

各arm32独立PRNG substreams、1 sampleはh1/40のcomplete controlled kernel。K0でも各sampleに16 signed events、run02合計1024 microsteps。各kernelを一体P0 compileし、正しさgate後にmetricsを記録した。Head/tail/eventごとの個別compile価格を加算したproxyではない。

| Metric | Baseline mean ± SE | Proposed mean ± SE | Proposed/Baseline mean |
|---|---:|---:|---:|
| RZ count | 5198.31250 ± 66.71460 | 4765.87500 ± 80.38451 | 0.916811946 |
| CX count | 2605.56250 ± 20.64069 | 2472.87500 ± 24.71955 | 0.949075296 |
| RZ depth | 2064.53125 ± 19.68000 | 1935.78125 ± 21.91832 | 0.937637176 |
| CX depth | 2019.03125 ± 10.24287 | 1951.90625 ± 11.46544 | 0.966753858 |
| Total depth | 4505.28125 ± 42.80037 | 4224.03125 ± 47.82859 | 0.937573265 |
| Circuit size | 10105.81250 ± 135.06424 | 9226.25000 ± 162.25757 | 0.912964692 |

SEは標本平均の記述的standard errorで、保証付きconfidence intervalではない。RZ観測範囲はBaseline4462〜5990／Proposed3860〜5677。全6指標でProposedの標本平均が小さかったが、32標本は母平均順位・rare-event捕捉・統計精度を保証しない。

[全64 kernel costs・seed・残差CSV](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/kernel_costs64.csv)／[6指標のmean・variance・SE・観測範囲CSV](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/calibration6.csv)。Armごとに異なるfinite-bit lawを使い、paired variance reductionやideal lawへの厳密転送を仮定していない。Sample min/maxを全support boundsにしない。

共通ancilla-only shell費用は別記録した。Hadamard headはRZ2／CX0／RZ depth2／total depth3／size3、X readoutはRZ2／CX0／RZ depth2／total depth4／size4／measurement1、Y readoutはRZ1／CX0／RZ depth1／total depth3／size3／measurement1。これらは決定論DF prefix headの費用ではなく、state preparationも含まない。主表のkernel平均には加算していない。

Kernel価格には決定論prefixとrandom tailを一体で含める。このcohortではProposedのwhole-kernel価格増は観測されなかったが、deterministic head／Gaussian／tailの個別寄与を因果分離したわけではない。B1のmass・A・補正RTE改善と同じ2分解でnative単価低下も観測したという範囲に限る。必要shots、反復全roundの費用、CAや総資源の判定は行っていない。

## 科学call・資源・保全

Run02は登録LIMITSどおり**652/652 COMPLETE／FAIL0**。内訳：input/evidence/context/fit/model7、law/ground/reference/sector/raw factor28、Gaussian/relative26、共通shell build/compile/check/metrics12、64×sample/build/reference/raw gate/compile/compiled gate/adjoint pair/X-Y pair/metrics576、calibration2＋比較1で合計652。

旧run01の429と合わせ累積1081 units。技術的再起動1回、run02内retry0／resume0。元run01のSTOPのretry/resume0は保全し、科学的FAILの再試行はしていない。新fit・optimizer・rank探索・DF/PF/RTE/attenuation誤差再計算・eigensolve・量子shots・CA／総資源評価は0。

Run02は64 raw kernels＋3 shells、67 P0 compiles、128補助actual adjoints、QPY262、system reference NPZ64、raw728 filesを保存した。科学callとQPY/build数を混同しない。全call journalと部分／最終STOPを両runに保持する。

開始時はaffinity32CPU、空きRAM約52.7GiB、pressure avg10/60/300=0。既存resource_planが32 workersを選び、小行列BLASを自process内1 threadに協調した。CPU/RAM/worker/wallの人工上限や共有環境変更は0。Run02 wall613.552104s（約10.23分）、CPU10783.1541s、peak RSS2616404KiB。共有Qiskit lockの合計wait14.8293s（worker待ち時間の和、wallとは異なる）。最初のrunは途中停止なので速度向上率を算定しない。

[postflight](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/execution_postflight.json)でSOURCE50・元SOURCE47・保存input22・旧run01・元inputと契約を照合した。固定前の旧628ファイルが不変。共有venv/cache、OS設定、他job、H2/H4/H6旧証拠・grant/ledger/marker/STOPは変更していない。承認証明書・承認原文保存・新grantは作成していない。

- [run02全証拠・QPY・reference・call journal・STOP](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run02/)
- [run01技術STOPの部分証拠](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run01/)
- [実行summary](../../artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/execution_summary.json)

## 残る資格と次の判断

B2で取得できたのは、固定cohortでのcomplete native kernelの数値的正しさと有限bit費用点推定である。Ideal-law期待費用、implemented-law差、全support費用範囲、全law合成上界、丸めenclosure、全13roundのstatistical room／必要shots、共通CA・shots加重総RZ/CX/depth、新規性は未資格でnullを維持する。

次は**研究C／ユーザー**が、B1/B2の観測を基にStage Cの技術設計を依頼するか判断する。**Codex**は具体的指示後に保存B1 error/Aと今回kernel/shell価格・不確かさを再利用し、保証付き結論と条件付き点推定を分け、未知の全law項を0にしない最小のStage C接続案を準備する。必要shotsと単価の交換条件はまだ評価していない。単価低下だけでPR-aware DF全体の成功や総資源優位を認定しない。

追加sampling・全law資格付け・shots・CA・総費用・重要科学レビューには自動進行しない。結果・旧STOP・修正SOURCEをcommit/pushしremote照合後、**mandatory STOP**する。
