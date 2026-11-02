# P3 Stage 3後：独立価格標本とStage 4会計の非実行設計

2026-11-02 JST。基点はStage 3完成結果`90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5`。
今回は次の限定実行案を具体化する文書作業だけで、新科学計算・保存価格の再集計・sampling・回路生成・compileは0。
実行コード・科学SOURCE・新しい科学結果は作成していない。既存の全STOPを維持する。

## 1. 次に得たい情報と推奨範囲

[Stage 3完成報告][S3]では、同じGaussian保持を共有する固定cohortでPauli残差のEとFullがDFより低価格だった。
linear H4/隣接1Å/STO-3G/中性singlet/system8/full256、保存DF tol-only1e-8/rank10、
案A/T=.8/S2/q8/r4/K4/M_B・M_C/P=0に限定した観測である。
各arm4 trajectoryであり、母平均・一般的方式順位・同精度N/Gは未評価。
[Stage 2][S2]の保存BはPauli39.125909965750068、DF34.527999187486905で、finite bias両軸は未解像のまま。
ここでは保存報告値を引用しただけで、平均・比・差・新しい精度余裕を計算していない。

次の推奨は、**同じtupleのFull価格だけを各arm32本の新しい独立cohortで取得する小規模確認**。
平均価格とその標本SEを得て、単回価格差が追加標本にも見られるかを調べる。
32本は事前に固定する予算であり、要求相対精度・検出力・希少tail捕捉の達成を保証する標本数ではない。
pilotの価格差を見てから立てた後続設計であり、pilot以前から事前登録されていたとは扱わない。

固定programのcase studyとして報告するだけなら、追加標本は必須ではない。
期待単回費用を定量化して後続会計へ渡したい場合に、この確認を推奨する。
Stage 4の設計は並行して整理できるが、実際のN/G会計は別scopeとする。
Stage 2 finite biasの追加高精度化は、今回の価格確認の前提にしない。

## 2. Stage 3追加価格案：固定する対象と新規件数

| 項目 | 限定実行案（未実行） |
|---|---|
| 系・分割 | 上記H4、案AのT_eff保持/V_square残差、M_B/M_Cのみ |
| law | Stage 2の保存184/360成分、順序・符号・係数・確率・角度bindingを維持 |
| 時間 | binary64 .8のexact rational、S2/q8/r4/K4、32 event位置/trajectory |
| 保持 | run02保存Gaussian half/controlled half/actual inverseと保存DF basisを再利用。新Givens合成・再対角化0 |
| 制御・wrapper | 同じPauli D2/DF central制御、同じbasis reuse、ancilla8、cosine/+Im sine、measure込み、P=0 |
| compiler | Qiskit1.3.0、rz/sx/x/cx、level1、seed17、backend/coupling/layout/routing指定なし |
| 新標本 | P32本、DF32本、計64 trajectory・2048 event位置。各本1回のPCG64/choice flat32 sampling |
| seed案 | P:320201–320232、DF:320301–320332。各範囲を昇順のindex0–31に対応。元pilotの310201–310204/310301–310304と分離 |
| 新native対象 | Full64 bodyのみ。残差E・R・Emptyの新body/価格/非加法性は取得しない |
| 新価格 | Full64 body×2軸＝128 compile。cos/sinに同じevent列を使用 |
| 独立参照 | 保存Tから同じ半指数を1回評価しcache。保存system halfの行列を1回評価・照合。新H/full reference/state生成0 |
| 追加処理 | 64 Full bodyを独立辞書oracleへ照合。全PASS後だけ128 compile |
| 方式間関係 | PとDFは独立cohort。同じindexの差をpaired標本としない |
| 旧結果 | pilot8本/36価格は保持し、新32本/armの確認平均へpoolしない。旧23件の再compile0 |

これは実行承認を求める対象の案であり、取得済み件数ではない。
seedを価格・order coverage・成功率で選び直さず、固定64本を一度だけ取得する。
order4が出ないことを理由に再抽出・forced event混入・標本追加をしない。
未観測tailの母平均寄与を0と認定せず、population上界はnullを維持する。

## 3. 再利用するSOURCEと最小の技術変更

| 役割 | 再利用候補と差分 |
|---|---|
| 科学入力・辞書 | Stage 2のdictionaries/results/common_operatorsと元snapshot。再SCF/FCI/DF生成・新分割なし |
| 保存native | run02のnative_evidence.zip/native_bases.json/preflight.json。QPY原bytes・fingerprint・gate順・phaseを対応づける |
| event/law | matched_partition_stage3.component_inputs/residual_sequence、rte.sample_rte_events。保存lawの照合は行うが、新B/実効平均は評価しない |
| Full合成 | trajectory_partsのouter-major規約を再利用。現APIはEも生成するため、新adapterでFullだけをcomposeする最小接続を用意し、旧SOURCEは変更しない |
| oracle | DictionaryOracle.trajectory(part='F')とprogram_check。gate列を参照targetの代用にしない |
| 価格 | matched_partition_stage3.priceと既存QPY保存処理。run03のspawn/max_tasks_per_child=1を再利用 |
| 統計 | rte_compiled_cost.CompiledMetricAccumulator、必要時にh4_conditional_cng.cost_summaryの軸間共分散。旧分子runner全体は起動しない |

run02の[科学SOURCE][NATIVE]とrun03の[process修正SOURCE][PROCESS]を参照する。
13件固定の旧回復runnerはそのまま追加標本取得には使えない。変更するのは新namespaceの小さなrunner/adapterだけ。
保存DF basis QPYのgateとqubit位置をregistryへ戻す際、保存lawのbasis ID/hashとの一致を確認する。
共有library・NumPy・venv・OS設定や旧封印SOURCEは変更しない。

保存artifactには独立参照E_Tの行列自体がないため、「参照指数の再評価0」とはしない。
新しい半指数1回と保存halfの照合は、将来の科学実行範囲へ明示的に含める。
同じcomplex128・τ/4/4τ・積誤差伝播政策を維持し、閾値緩和・phase alignment・dense fallbackは行わない。
旧81 native PASSは保存証拠として再利用し、新64 FullのPASSとは区別する。

人工検査は保存QPY再接続、Full-only合成と旧F一致、signed order0/2/4、scalar/actual adjoint、
cosine/+Im、process非共有、人工価格の平均/SE/共分散を対象に絞る。
実装・人工検査後に新SOURCEをcommit固定して公開し、そのSOURCEで一度実行する。
独立taskは空き資源に応じ並列化し、各compiler processでは1 targetだけを処理する。
一時的なCPU競合や無関係worktreeの更新だけで正常workerを中断しない。

## 4. 将来の価格集計と停止規則

新32本/armを解析単位として、各arm/測定軸/metricごとに標本平均、
不偏標本分散、標本SE=sqrt(s²/32)、観測min/maxと全個別価格を保存する。
主metricはFull RZ、CX/depth/sizeとhistogramは副指標。同じtrajectoryのcos/sinは各32標本であり、64独立標本へ数え替えない。
同一armのcos/sin価格共分散をmetricごとに残し、将来の重み付き会計に使えるようにする。
方式間平均差を表示する場合は独立標本SE=sqrt(s_P²/32+s_DF²/32)を使い、paired SEを使わない。
これは経験的な平均費用診断であり、非漸近信頼区間・母平均の厳密順位・rare-tail上界を取得したとしない。
標本SEが小さくても未観測tailを除外できず、別途根拠のないpopulation_mean_upperはnull。

途中の価格やSEに応じた早期成功・top-up・倍増はしない。終了時に不確かさが残れば未解決として報告する。
入力/phase/native oracle非PASS、runtime/compiler不一致、compile例外・出力欠落は途中出力を保全してSTOP。
既に動くworkerを回収し、正常出力を破棄しない。科学失敗の自動retryや別seedへの代替は行わない。
全event、native/input/compiled QPY、metrics、SOURCE/inputとworkerの由来、集計、STOPを新namespaceへ保存する。
公開・remote照合後mandatory STOP。新N/G・新signal・finite-bit law平均・追加精度・別tupleは実行しない。

## 5. Stage 4は同精度N/G会計：実行前に決める内容

既存[P3全体準備§8][ALL]が定めるStage 4は、単一Tの同じsignal精度/failure政策でのN/C/Gである。
全roundのenergy/RPE総費用とは区別する。次の表は接続設計であり、今回は数値処理していない。

| 入力 | Stage 4準備で固定すべき意味 |
|---|---|
| 精度・failure案 | ε_sig=.05/.01/.005/.001/.0005、軸ε=ε_sig/sqrt(2)、各軸α=.025。採用・実行は別の指示で確定 |
| B | 各armの保存lawに対応するBと上界/丸めの扱い。decimal50照合を厳密上界certificateへ格上げしない |
| signed total bias | 各armの補正済みnominal信号と同じ保存raw φ/ν・共通referenceから定義。outer＋finiteの点差を安易に絶対値加算しない |
| 数値不確かさ | nomの数値/binding誤差とreference/transferの寄与を一度ずつ扱う。outerとfiniteの共有split誤差を二重計上しない |
| 実装誤差 | finite-bit sampling/angleとnominal lawの差、compiled作用誤差ρを数値solver誤差から分離。未測定を実測0としない |
| 価格 | 新確認cohortのFull平均・SE・cos/sin共分散。旧pilotとは分離し、E+RをFull価格の代用にしない |
| N | fixed_signal_accounting.fixed_signal_shotsのHoeffding式を候補とし、bias絶対値の保守側と残るallowanceでheadroomを定義 |
| G | G=N_cos*C_cos+N_sin*C_sin。価格MC標本数32と量子shot数Nを区別。SEは同trajectoryの軸共分散を保持して伝播 |

headroomは概念上 `h_a=ε_a−bias_abs_upper_a−u_remaining_a`。
同じ数値誤差をbias_abs_upperとu_remainingの両方へ入れない。
正のheadroomやBの必要な上界を正当化できない行はN/G=nullまたはUNRESOLVEDとし、0で補完しない。
現時点で新しいheadroom・N・G・倍率・方式順位は計算していない。

Stage 4には二つの異なる報告対象がある。
条件付き感度診断なら、ρ=0・finite-bit gap=0などを**仮定**として明示することはできるが、実装保証とは呼ばない。
実装を含む保証を求める場合は、その保証に必要なlaw/angle/compiler誤差とB上界の根拠を別scopeで用意する。
すべての探索的会計へ全law証明を一律に要求せず、仮定と保証の差を表に残す。

finite biasが未解像でも、その不確かさを含む保守側biasで余裕を判定できるなら、成分値を解像する高精度化は必須ではない。
将来の会計で数値不確かさが適格性/順位を左右する場合に限り、追加精度の必要範囲を改めて提案する。
これは将来の判断規則であり、現在の数値でその条件を満たしたと認定したものではない。

## 6. 次の担当・今回の完了条件

今回の非実行設計は、公開・全文照合で完了。科学的主要claimの採択や詳細科学レビューの実施ではない。
科学結果が変わっていないため、旧validation manifestの科学statusは変更しない。
Stage 2 UNRESOLVEDとStage 3のFAIL/partial/completeの全結果・STOPを保全する。

次の担当はユーザー：**新32本/arm・Full128 compile、独立参照半指数1回と保存half照合、新64 Full oracle、平均/SE/共分散まで**を実行するか判断する。
この範囲の明示許可後、Codexが関連実装・人工検査・SOURCE固定・限定実行・公開をまとめて行う。
Stage 4のN/G実行や高精度化をこの価格確認へ暗黙に含めない。
新科学実行の明示許可とSTOP維持は共通AGENTS.mdおよび統一指示§8.1/8.2に基づく。
承認証明書・原文保存・grant発行は不要で、具体的な範囲へのチャットの許可で足りる。

[S3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run03/REPORT.md
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/REPORT.md
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c/src/trotterlib/matched_partition_stage3.py
[PROCESS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0d9d3bbe66750d0b20acd9e71d55525f429b229/scripts/recover_h4_p3_stage3_process.py
[ALL]: h4_df_pauli_matched_partition_execution_readiness_20261102.md
