# 新A Q2：保存回路へのAdaptive Feedback Readout接続

2026-11-10 JST／SERVER 2。**固定16構成で、Q1 readoutを保存B′／Reference coreへ接続したraw／native期待値は指定のcorrectness判定を満たした。** この有限fixtureでは、native価格は保存Y回路と同じ。保存Xに比べRZとsizeが各1増え、CXは不変、depthはB′で不変・Referenceで1増えた。340 shotsの実装保証、全support・m17の価格、全18roundの総コスト削減は未資格のままである。

基準はQ1完了 [`f3705321515f35c08c702e377739b2c3e6e9df13`][Q1]。同じbranch `codex/algo-improve-new-a-total-cost-20261109`を継続し、Q2専用namespaceへ追加した。旧科学SOURCE・input・raw・結果・STOP・KEEP契約、Q0／Q1、V6、CT、他Trackに変更はない。

## 1. 固定scope・入力・再利用

承認された[Q1計画§2〜3][PLAN]と今回のQ2指示に従った。Linear H4／1.00 Å／STO-3G／DF12、元Hの厳密ground供給・準備費除外、Partial S2／LD3／h=.02／K2、q1/r1/j0の保存rare0/1のみ。Referenceはm0の旧vacuum phase経路であり、m17 B1 phase接続の検証ではない。

| path | 保存seed | ordered event | vacuum phase hex |
|---|---:|---|---|
| rare0 | 4216572573356262833 | order0、`df-fragment-5:Z3Z7`、productsなし | `0x1.8fd112eb44b80p+2` |
| rare1 | 4841711708811669773 | order2、`df-fragment-3:Z0Z4`、products `Z2Z3→Z1Z4`（同fragment） | `0x1.8b84c9c9f67a8p+1` |

両event SHA256、PREDECLARED、保存raw metadata、cache圧縮・pickle hash、expectedのseed／phase、再構成後の全event JSON hashが一致。cacheは固定Q1計画のcompressed SHA256 `85d7e21a…`／pickle `68269aa0…`。完全値・Git commit・入力file hashは[SOURCE.json](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run02/SOURCE.json)に保存した。新Hamiltonian・DF・ground solve・trajectory生成は0件。

角度は固定binary64の `0.0`、`1.5707963267948966`、`0.213`、`−0.719`。2path×2arm×4角＝16回路。数学的なπ/2とそのbinary近似を区別し、後者での期待式と旧Yの保存coherence成分の双方を検算した。実m16履歴やE0から角度を作ったものではない。

主な再利用は次のとおり。

- Q1 `feedback_readout` をそのまま使用。旧CT1 `sample_rows`／`instructions`／native-only `matrices_for(ct=False)`・`apply_batch`・`evaluate`をASTで再利用した。CT合成moduleやrunnerはimport／実行しない。CT helperが返す未使用の会計placeholderは破棄し、T価格には使用しない。
- B′ `build_saved`／`reconstructed`、transfer `assemble`、pilot `reference_core`／`ground` を変更せず使用。4-CX制御、signed products、scalar、F/G、actual inverse、vacuum補正、sampling lawの意味を保持した。
- cacheのstateを1回読込み、既存 `ground()` の8-bit endian写像を再利用。再正規化しない。4 raw coreの状態作用を各1回だけ計算し、同じpost-core stateに実wrapper末尾のP/Hを作用させて16 raw readoutを検証した。
- 旧CT1のnative QPY 8本と保存価格・profileを再利用。旧Compiler A SOURCEのblobと現SOURCE、Qiskit1.3.0、basis／pass profileが一致した。baseline再compileは0件。

## 2. 実行・correctness

[実行SOURCE](../../../src/trottertracks/pr_reference_feedback_q2.py)は既存未測定coreへ `H→core→P(−φbin)→H→Measure` を接続し、固定16 contextだけCompiler Aでcompileした。Reference reset8個はcompile後にprefixし、物理実行ではresetが最初。ancillaはB′=8、Reference=16、測定bit0を+1とする。Reference registerは初期vacuumなので、既存native状態作用kernelでresetをskipできる前提をlayout検査で確認した。

| 比較（各arm 8構成） | B′最大絶対差 | Reference最大絶対差 | 判定 |
|---|---:|---:|---|
| rawと保存coherenceの投影 `Re(e^(−iφbin) z_saved)` | 8.89e−16未満 | 9.05e−15未満 | PASS |
| nativeと同じ保存投影 | 6.03e−14未満 | 6.40e−14未満 | PASS |
| rawとnative | 5.94e−14未満 | 6.33e−14未満 | PASS |

指定の旧CT1数値規約 `1e−10` を使用し、phase fit、normalization補修、tolerance変更は行っていない。正負一般角、φ=0のX、binary π/2近傍のY、両armのobservable符号が一致した。common global phaseはそのまま保持して状態作用に入れ、branch relative phaseをfitで消していない。

これは**保存ground上の有限数値検証**である。任意入力のoperator norm、全support、全角度の一様位相保証ではない。raw／native期待値は全[RESULTS.json](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run02/RESULTS.json)に保存した。`cores.actual_inverse_SOURCE` fieldはReference構成の参照先を示すもので、B′がReference builderを使ったという意味ではない（CHECKSにも明記）。

## 3. Native価格

下表は**各行の4固定角すべてで同じ実価格**。平均価格・sampling lawの期待価格ではない。reset・measurement込みの旧metric規約を保持した。

| path／arm | RZ | CX | size | DAG depth | qubits |
|---|---:|---:|---:|---:|---:|
| rare0／B′ | 7,876 | 1,278 | 11,653 | 2,662 | 9 |
| rare1／B′ | 7,857 | 1,278 | 11,630 | 2,665 | 9 |
| rare0／Reference | 8,047 | 1,298 | 11,884 | 1,451 | 17 |
| rare1／Reference | 8,015 | 1,294 | 11,836 | 1,448 | 17 |

| 同じpath・armの保存対照との差 | ΔRZ | ΔCX | Δsize | Δdepth | Δqubits |
|---|---:|---:|---:|---:|---:|
| B′：旧X比 | +1 | 0 | +1 | 0 | 0 |
| Reference：旧X比 | +1 | 0 | +1 | +1 | 0 |
| 両arm：旧Y比 | 0 | 0 | 0 | 0 | 0 |

全16 contextのpass traceとnative QPYで、readout部分はancilla上の `RZ(−φbin)` として残り、後続Hのloweringと融合・取消されていないことを確認した。φ=0でもRZ(0)が残る。今回の有限16構成では「旧Y相当のnative価格」という結果だが、任意角・任意contextで常に1RZとなる保証や、一定T-priceではない。

RZ／CX／size／DAG depth／peak qubits／classical overheadを相互に換算・合算していない。Referenceの追加8qubitは既存構成に由来し、新readoutが追加したregisterではない。T-count／T-depthは未取得。

## 4. 検算・技術的な起動修正・保全

[checker](../../../scripts/check_pr_reference_feedback_q2.py)で、全16 QPYのcount／depth／size／qubits／reset／measure／readout RZ、保存X/Yとの差、source/input hash、unique context、全correctness fieldを照合した。[CHECKS.json](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run02/CHECKS.json)はPASS。checkerは新simulation・compileを行わない。状態作用kernel・readout符号・reset・保存sample選択の[人工4テスト](../../../tests/test_pr_reference_feedback_q2.py)もPASS（[log](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run02/TESTS.txt)）。いずれも著者Codexによる検査で、独立科学レビューではない。

科学core生成前に次の技術問題を修正した。

1. Q1で使ったruntimeにnumbaがなかったため、保存CT1と同じ既存runtimeへ切替えた。packageや共有環境は変更していない。
2. 最初の起動でAST adapterの`json` namespace不足が見つかり、cache読込後・core build前に停止した。[run01 SOURCE](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run01/SOURCE.json)／[実行時SOURCE原本](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run01/EXECUTED_SOURCE.py)／[STOP](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run01/STOP.json)は不変更で保全した。§10で許可された純粋な実装修正と人工regression testの後、同じ固定入力をrun02で処理した。科学条件は変更していない。

従ってcache読込は技術起動を含め2回、**科学回路の16構成はrun02で各1回だけbuild／compile／native状態作用**。run02ではcache1回・raw core4本／状態作用4回。16workerを実際の64 CPU affinity内で使い、forkで読取cacheを共有、JITを人工入力で一度準備した。完了所要は約13.15秒（性能比較ではない）。起動時の空きmemory約491GiB、CPU／memory PSI avg10=0を観測し、共有設定・他processは変更していない。

生成nativeは[QPY archive](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run02/native_qpy.tar.xz)として公開。各contextの同内容journalと展開QPYはlocalでも保持し、重複公開だけを.gitignoreで避けた。完全RESULTSは全16 journalの内容を含む。

## 5. 前進した点・未資格・mandatory STOP

Q0の理想測定方向を表すQ1 readoutが、**両方式の保存科学coreに有限correctnessを保って接続できることと、その局所native追加費**が確認できた。次はこの結果を研究判断の材料として使える。

ただし、次は未資格である。

- 実m16 countからのprefix成功保証・数値受理coverage・feedback角生成と、全supportに対する追加parameter／lowering誤差。
- Q0の340 shotsを移送する `δφ≤.000748774257 rad` の実装保証。有限float差をこのrad保証に読み替えず、既存diskを二重加算せず、operator diskもradとして足さない。
- m17 native価格、全sampling lawの期待単価μQ,j、校正・feedback費Aj、全18round総費用、Clifford+T費用。

将来の比較条件は `340 μQ,j+Aj < 626 μX,j+626 μY,j`。今回のq1有限価格を代入してm17費用を認定していない。B′にも同じQ機会を与えており、Reference固有の改善ではない。強Full／Truncated PFとの公平な比較も後続判断に残す。

**次担当はGPT／ユーザー。** 実prefix・全support feedback誤差の資格と、長roundの価格接続のどちらを次の限定scopeで扱うか判断する。CodexはQ3を開始せず、結果・SOURCE・検算・remote照合の公開をもって[mandatory STOP](../../../artifacts/pr_reference_feedback_q2/2026-11-10/run02/STOP.json)。Q0再実行、sampling、m17 whole回路、340-shot RPE、全18round再会計、strict合成／guard、総T評価はいずれも0件。

[Q1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f3705321515f35c08c702e377739b2c3e6e9df13/docs/research/pr_reference_feedback_q1_20261110/REPORT.md
[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f3705321515f35c08c702e377739b2c3e6e9df13/docs/research/pr_reference_feedback_q1_20261110/NEXT_VALIDATION_PLAN.md
