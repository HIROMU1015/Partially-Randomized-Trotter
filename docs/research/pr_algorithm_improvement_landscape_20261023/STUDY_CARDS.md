# 研究ごとの詳細カード

各カードは調査専用分類。元status/STOPを変更せず、SOURCEの静的読解と過去の保存結果を記録する。資料日・含有branchは研究初出や実行HEADと同一とは限らない。

<a id="rep-01"></a>

## REP-01 Pauli groupingとDF回路基盤

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-01-22（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation_numGate_highorder_DF / codex/execute-provided-command-sequence |
| 固定資料commit | 4d90167f0634a67ca1ed6bf2630af075c35a149f |
| 目的 | 回転数・RZ層を減らす |
| 変更部分 | Pauli grouping／DF基底変換と対角回転 |
| 改善機構 | 可換性と低rank構造で実装単位を変える |
| 比較対象 | ungrouped Pauli／grouped PF／DF PF |
| 科学条件 | 旧水素鎖、basis・距離・rank・PF窓は個別artifactの条件。共通条件へ統合しない |
| 評価指標 | Pauli rotations / RZ layers / T-depth proxy |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 実装基盤を確認。最新PRや全energy taskの優位性とは分離する |
| 調査用成績分類 | 実装・評価基盤 |
| 既存状態の説明 | 既存実装・保存fitあり |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 個別runの終了。今回再実行しない |
| 未解決 | rawと現在の係数版・原稿の対応 |
| 再利用 | DF/Gaussian builder、grouping、費用関数 |
| PRへの応用 | 直接応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0001: README.md](https://github.com/HIROMU1015/Evaluation_numGate_highorder_DF/blob/4d90167f0634a67ca1ed6bf2630af075c35a149f/README.md)

静的実装参照：[EV-0002: src/trotterlib/df_trotter/circuit.py](https://github.com/HIROMU1015/Evaluation_numGate_highorder_DF/blob/4d90167f0634a67ca1ed6bf2630af075c35a149f/src/trotterlib/df_trotter/circuit.py), [EV-0003: src/trotterlib/config.py](https://github.com/HIROMU1015/Evaluation_numGate_highorder_DF/blob/4d90167f0634a67ca1ed6bf2630af075c35a149f/src/trotterlib/config.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rep-02"></a>

## REP-02 旧DF rank screening

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-04（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / rte-basis-cost-attribution-diagnostic-20261013 |
| 固定資料commit | 403204edf22e5712a71ac9ac7eebdbcd18626421 |
| 目的 | rank削減と誤差の交換を調べる |
| 変更部分 | DF rankを変更したscreening |
| 改善機構 | 低rankで決定論費用を節約する |
| 比較対象 | full reference／低rank DF |
| 科学条件 | 旧screening固有入力。現在のtol-only rank方針と同一視しない |
| 評価指標 | DF residual / PF bias / cost proxy |
| 検証段階 | 実装/報告記録はあるが重要raw/inputが不足し、実計算の成否を十分判断できない。 |
| 実際の結果 | 旧screening値は現行科学結果として利用不可。履歴として保存 |
| 調査用成績分類 | 重要資料不足・判断不能 |
| 既存状態の説明 | DO NOT USE / stale |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。raw不足により成否未判定。 |
| 停止理由 | stale evidenceを現行値にしない |
| 未解決 | 生成元・入力identity・旧rawの回収可否 |
| 再利用 | rank-policyとsnapshot固定の教訓 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | C：成否や再利用の判断を左右する原入力／結果が不足。stale/prose-onlyの数値を有効結果へ昇格しない。 |

根拠文書：[EV-0004: VALIDATION_STATUS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/VALIDATION_STATUS.md), [EV-0005: notes/uwc_current_implementation_and_results.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/notes/uwc_current_implementation_and_results.md)

静的実装参照：[EV-0006: src/trotterlib/df_hamiltonian.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/src/trotterlib/df_hamiltonian.py), [EV-0007: src/trotterlib/uwc.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/src/trotterlib/uwc.py), [EV-0008: src/trotterlib/partial_randomized_pf.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/src/trotterlib/partial_randomized_pf.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

資料不足：[GAP-008](EVIDENCE_GAPS.md#gap-008)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rep-03"></a>

## REP-03 BLISS型UWC＋grouped PF

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-07-20（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / agent/finite-rte-df-tail |
| 固定資料commit | 231d17e393f573fd29c659eaf1fd9bae2a68d8f9 |
| 目的 | sectorを保持してtail normとPF費用を削減 |
| 変更部分 | 粒子数sectorで消えるlinear/quadratic shiftを前処理 |
| 改善機構 | Hamiltonian係数を変えて誤差・normalizationを下げる |
| 比較対象 | 同じgrouping規則の元Hamiltonian |
| 科学条件 | 旧H2–H6/STO-3G系列、theta=.01 quadratic、二次PF。H3 sector訂正の前後を区別。L_Dとfit窓の完全条件は欠損raw待ち |
| 評価指標 | alpha / RZ layers / total-cost proxy |
| 検証段階 | 実装/報告記録はあるが重要raw/inputが不足し、実計算の成否を十分判断できない。 |
| 実際の結果 | 本文上の小差・悪化訂正は実行メモ。数値的な改善成否を確定しない |
| 調査用成績分類 | 重要資料不足・判断不能 |
| 既存状態の説明 | prose-only / raw artifacts missing |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。raw不足により成否未判定。 |
| 停止理由 | 独立監査可能なraw不足 |
| 未解決 | JSON/CSV、sector診断、theta sweep、実行環境 |
| 再利用 | uwc.py、前処理位置とsector guard |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | C：成否や再利用の判断を左右する原入力／結果が不足。stale/prose-onlyの数値を有効結果へ昇格しない。 |

履歴・実行単位：誤ったH3 target sectorの旧悪化値と訂正後記述を併記。orbital_optimization系は予約interface／NotImplementedError。

根拠文書：[EV-0005: notes/uwc_current_implementation_and_results.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/notes/uwc_current_implementation_and_results.md)

静的実装参照：[EV-0007: src/trotterlib/uwc.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/src/trotterlib/uwc.py), [EV-0008: src/trotterlib/partial_randomized_pf.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/src/trotterlib/partial_randomized_pf.py), [EV-0009: src/trotterlib/grouped_uwc_comparison.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/src/trotterlib/grouped_uwc_comparison.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

資料不足：[GAP-001](EVIDENCE_GAPS.md#gap-001)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pf-01"></a>

## PF-01 Yoshida・Morales・新4次係数の比較

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-19（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / finite-time-research-consolidation-20260922 |
| 固定資料commit | 20d64c2fea1ed2d2188777df4f3d962ee94df486 |
| 目的 | 誤差次数と一step費用の釣合いを改善 |
| 変更部分 | 既存高次係数・新4次候補・公開係数版を区別 |
| 改善機構 | 誤差係数の低減で必要step数を減らす |
| 比較対象 | S2／標準4次／複数8次・10次 |
| 科学条件 | 高次PF repositoryの分子・H-chain別固定条件。PR分割L_Dは非該当。公開版と旧係数版を区別 |
| 評価指標 | Pauli rotations / RZ depth proxy / energy bias |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | PF family内の限定比較がある。高次数が常に有利とはしない |
| 調査用成績分類 | 限定条件で改善 |
| 既存状態の説明 | 公開係数整理・有限時間検証記録 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 既存比較scopeで完了 |
| 未解決 | PRの負時間tail・normalizationと強い低次対照 |
| 再利用 | 固定係数列・stage数・既存PF builder |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0010: review_response/README.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/README.md)

静的実装参照：[EV-0011: src/trotterlib/product_formula.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py), [EV-0012: review_response/validate_morales_y8m10b_h2.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/validate_morales_y8m10b_h2.py), [EV-0013: src/trotterlib/processed_cost.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pf-02"></a>

## PF-02 Processed PF／YP8m8

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-19（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / finite-time-research-consolidation-20260922 |
| 固定資料commit | 20d64c2fea1ed2d2188777df4f3d962ee94df486 |
| 目的 | kernelの有限時間誤差・状態重なりを改善 |
| 変更部分 | processor–kernel–inverse processorを分離 |
| 改善機構 | processingによる状態・近似演算子の改善と反復kernel短縮 |
| 比較対象 | kernel単体／非processed Morales |
| 科学条件 | YP8m8の公開kernelとprocessor。決定論unitaryの相似変換と固定状態を区別。PRへの移送は未実施 |
| 評価指標 | eigenvalue error / state overlap / processor/kernel rotations |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | kernelとprocessed回路の検証・費用区分を保存。PR全費用への利益は未確定 |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | published coefficient and direct-check records |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 主結果への採用は未確定 |
| 未解決 | processorの前後費用、状態変換、random平均の意味 |
| 再利用 | kernel/processor係数・既存検証記録 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：非processed Y8m10bとの名称・stage数の混同を避ける。

根拠文書：[EV-0010: review_response/README.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/README.md)

静的実装参照：[EV-0011: src/trotterlib/product_formula.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py), [EV-0012: review_response/validate_morales_y8m10b_h2.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/validate_morales_y8m10b_h2.py), [EV-0013: src/trotterlib/processed_cost.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pf-03"></a>

## PF-03 PF固有位相・D6係数とsurvival代理量

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-24（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | PF係数入力の妥当性を確かめる |
| 変更部分 | survival／D6／支配固有位相を比較 |
| 改善機構 | 安い状態依存係数で探索を支える |
| 比較対象 | full-H参照・survival fit・QPE RMSE |
| 科学条件 | linear H4 1Å/STO-3G/DF12、L_D=0…11、δ=.0125/.025/.05/.1/.2/.4、q=1/2/4。D6 fit窓とphase窓を区別 |
| 評価指標 | energy bias / branch weight / C_use |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | survival phaseとQPE RMSEは同じ量でない。C_useは実行窓の経験的envelope |
| 調査用成績分類 | 評価基盤・限定検証 |
| 既存状態の説明 | local validation |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 局所検証で停止 |
| 未解決 | H12係数・長q・別snapshotへの移送 |
| 再利用 | 係数定義・conditioning/branch診断 |
| PRへの応用 | 直接応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0014: docs/pf_delta_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/pf_delta_validation.md)

静的実装参照：[EV-0015: src/trotterlib/pf_delta_validation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/pf_delta_validation.py), [EV-0016: scripts/run_pf_delta_validation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/scripts/run_pf_delta_validation.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pf-04"></a>

## PF-04 有限時間校正・第一研究S0–S4

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-25（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / first-study-completion-analysis-20260926 |
| 固定資料commit | 6a1e54d5e830a20b8817791f9f1f65729815fc12 |
| 目的 | 校正を改善して安全なPF予算を下げる |
| 変更部分 | 状態置換／proxy差／fit誤差の分離、二状態診断 |
| 改善機構 | 誤差の支配成分を改善すれば校正過大見積りを削減できる |
| 比較対象 | 既存校正と強い共通対照 |
| 科学条件 | 第一研究のH4/toy/HF等、分子ごとにbasis・active space・PF・time gridを固定。PR分割なし |
| 評価指標 | calibration error / budget safety / regret |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 機構分離は成立、固定S4診断はcomplete_no_benefit。研究未完了と解釈しない |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | complete_mechanism_study_no_method_benefit |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | S5へ進まない既存STOP |
| 未解決 | PRで同じ取得費・bias台帳を満たすか |
| 再利用 | 誤差分解と同情報比較の設計 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0017: PF_first_study_final_synthesis_20260925.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/6a1e54d5e830a20b8817791f9f1f65729815fc12/PF_first_study_final_synthesis_20260925.md)

静的実装参照：[EV-0018: review_response/run_pf_first_study_s4_state_convergence_v1_1.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/6a1e54d5e830a20b8817791f9f1f65729815fc12/review_response/run_pf_first_study_s4_state_convergence_v1_1.py)

保存result/decision metadata：[EV-0019: PF_first_study_final_decision_20260925.json](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/6a1e54d5e830a20b8817791f9f1f65729815fc12/PF_first_study_final_decision_20260925.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pf-05"></a>

## PF-05 有限時間校正・第二研究Direction C

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-06（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / pf-finite-time-research-completion-audit-20261016 |
| 固定資料commit | b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1 |
| 目的 | cheap predictorの安全性と資源価値を調べる |
| 変更部分 | spectral情報・rank・geometry・prospective固定予測 |
| 改善機構 | 安全余裕と点推定を分けて有用なdecision windowを調べる |
| 比較対象 | same-time cheap／B0／真値scorer |
| 科学条件 | H-chain G/Rと一般分子prospectiveは別契約。各分子basis・PF・rank/time条件は固定参照へ。PR L_Dなし |
| 評価指標 | safe/unsafe / budget advantage / abstention |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | H-chain限定支持と一般分子への不十分な移送、情報追加後のbudget利得なしを保持 |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | Track G/R completion accepted; Direction C |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | H-chain追加science終了 |
| 未解決 | 一部HF/HCl/prospectiveのabsolute identity tuple |
| 再利用 | prediction/truth分離・abstentionとheadroom診断 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0020: docs/second_study_v2/evidence_integration_20261006/approved_closure.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/second_study_v2/evidence_integration_20261006/approved_closure.md), [EV-0021: docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md)

静的実装参照：[EV-0022: review_response/run_second_study_v2_s1a_predictor.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/review_response/run_second_study_v2_s1a_predictor.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

資料不足：[GAP-005](EVIDENCE_GAPS.md#gap-005)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pf-06"></a>

## PF-06 FS-C旧入力回収とFS-R1校正follow-up

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-16（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / pf-finite-time-research-completion-audit-20261016 |
| 固定資料commit | b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1 |
| 目的 | state改善・local評価で校正予算を下げる |
| 変更部分 | 旧source回収から別new-source prediction freezeへ移行 |
| 改善機構 | selected-timeの安い情報で校正誤差を減らす |
| 比較対象 | historical H01/H02／new-source B0と4arm |
| 科学条件 | 旧N2/CO historical Hamiltonianの指定SHA。new sourceのN2/COは独立identity。Phase B truth未実行 |
| 評価指標 | identity / prediction error / future budget scoring |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 旧回収失敗を保存。new-source Phase A freezeは成立、利益採点は未実行 |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | NO_GO_SOURCE_MISSING / NO_GO_RECONSTRUCTION_MISMATCH; Phase B unauthorized |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 旧identity不一致と新Phase B未承認 |
| 未解決 | 旧H01 pickle／Hamiltonian／CISD／ordered groupsの原本 |
| 再利用 | source再baselineとfreeze barrier |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

履歴・実行単位：C0.5 source missing→C0.6 reconstruction mismatch→R0別系列→R1 Phase A／B0。原truthを混合しない。

根拠文書：[EV-0021: docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md), [EV-0023: artifacts/pf_first_study_fs_c05_execution_closure_20261007/source_missing_followup.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/artifacts/pf_first_study_fs_c05_execution_closure_20261007/source_missing_followup.md), [EV-0024: artifacts/pf_first_study_fs_c06_functional_equivalence_20261007/README.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/artifacts/pf_first_study_fs_c06_functional_equivalence_20261007/README.md)

静的実装参照：[EV-0025: artifacts/pf_first_study_fs_r1_b0_phase_b_barrier_20261008/phase_b_controller.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/artifacts/pf_first_study_fs_r1_b0_phase_b_barrier_20261008/phase_b_controller.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

資料不足：[GAP-004](EVIDENCE_GAPS.md#gap-004)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pf-07"></a>

## PF-07 BCH高次項・response・direct-scaling

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-16（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / pf-finite-time-research-completion-audit-20261016 |
| 固定資料commit | b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1 |
| 目的 | 高次PF係数・有限時間モデルを安く予測 |
| 変更部分 | compact BCH、D4混合/D8、response・few direct点 |
| 改善機構 | 相殺と混合補正を保持して過小評価を抑える |
| 比較対象 | HF/CISD/exact response／direct grid |
| 科学条件 | H05やF02/F03、H-chain sizingは別固定設計。basis/active space/PF/time窓は各引用元。PR L_Dなし |
| 評価指標 | model error / branch eligibility / acquisition cost |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 限定identityや予測成功はあるが、モデル増量の単調改善・サイズ普遍則・net資源利益は未確立 |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | 既存機構結果・U1/U3/U4残余 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 既存終了／保留を維持 |
| 未解決 | 取得費・剰余・枝・未使用系への移送 |
| 再利用 | compact BCH、response、失敗条件の記録 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0021: docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md)

静的実装参照：[EV-0026: review_response/audit_h05_higher_order_acquisition.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/review_response/audit_h05_higher_order_acquisition.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="cir-01"></a>

## CIR-01 WP06-a/b full/support-union基底計画

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-24（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | DF event列の基底変換費を削減 |
| 変更部分 | singleton source runだけsupport限定へ置換 |
| 改善機構 | runの共有を残し小supportの単発変換だけ省く |
| 比較対象 | full-basis共有／event毎support限定 |
| 科学条件 | H4 rank12/1Å/STO-3G、二次partial S2、主にL_D3、δ=.02、短列3/6、固定Qiskit条件 |
| 評価指標 | RZ / CX / depth / operator residual |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 局所列で削減を観測。一律support限定は混在列で悪化し得る |
| 調査用成績分類 | 限定条件で改善 |
| 既存状態の説明 | local holdout pass |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 局所検証完了 |
| 未解決 | full-wrapper・別compiler/encodingにおける増分価値 |
| 再利用 | basis-plan API／support_run_le_1 |
| PRへの応用 | 直接応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0027: docs/research_direction_sequence_policy.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md)

静的実装参照：[EV-0028: src/trotterlib/research_direction_sequence_policy.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_sequence_policy.py), [EV-0029: scripts/run_research_direction_sequence_policy.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/scripts/run_research_direction_sequence_policy.py)

保存result/decision metadata：[EV-0030: artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/research_direction_sequence_policy/2026-09-22/wp06b_sequence_policy_proxy_bridge_v1.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="cir-02"></a>

## CIR-02 P-A interval-union DP

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-25（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | run内区間分割で回路をさらに減らす |
| 変更部分 | interval-union DPと明示一区間baseline |
| 改善機構 | support共有範囲の選択で基底変換を節約する |
| 比較対象 | support_run_le_1／one_segment_per_source_run |
| 科学条件 | H4/STO-3G/DF12/L_D3/δ=.02/K2、H5 transfer、後続forced-support order2の30task |
| 評価指標 | compiled RZ / plan changes / operator equivalence |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 初期blind削減は保持するが、後続非退化比較で一区間baselineとの差0。interval固有claimを停止 |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | stop_pa_interval_dp_as_primary_and_return_to_pc |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | interval subdivisionの独立寄与を支持せず |
| 未解決 | より異なるsupport構造に明示的機構があるか |
| 再利用 | run-level full/support-union policy |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：pilot→H5/H4 blind pass→全event order0等の機構監査→forced-support order2で全metric差0。

根拠文書：[EV-0031: docs/research/pa_joint_synthesis_v1_formalization.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/pa_joint_synthesis_v1_formalization.md), [EV-0032: docs/research_direction_joint_synthesis_mechanism_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_joint_synthesis_mechanism_validation.md), [EV-0033: docs/research_direction_joint_synthesis_blind_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_joint_synthesis_blind_validation.md)

静的実装参照：[EV-0034: src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py), [EV-0035: src/trotterlib/research_direction_joint_synthesis_formalization.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_joint_synthesis_formalization.py), [EV-0036: scripts/run_research_direction_joint_synthesis_formalization.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/scripts/run_research_direction_joint_synthesis_formalization.py)

保存result/decision metadata：[EV-0037: artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_mechanism_validation_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_mechanism_validation_v1.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="opt-01"></a>

## OPT-01 round別r/K・誤差/失敗確率配分

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-24（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | shot込み総費用proxyを下げる |
| 変更部分 | round schedule、beta/alpha配分、compiler-context整合 |
| 改善機構 | 誤差headroomと単価を同時に配分する |
| 比較対象 | PR候補と決定論endpointへ同じ再最適化 |
| 科学条件 | H4 linear1Å/STO-3G/DF12、L_D3/12、δ=.01/.02等、直接wrapper q≤32。長RPEは外挿scope |
| 評価指標 | shot-weighted RZ / uncertainty intervals / prep sensitivity |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 共通配分が大きく効き、強い対照とcompiler感度で頑健な区間優位は未確立 |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | undetermined_under_compiler_and_transfer_sensitivity |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 局所compiler精密化を区切った |
| 未解決 | 長q移送・状態準備・外部instance |
| 再利用 | 公平な予算再配分と不確かさの分離 |
| PRへの応用 | 直接応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0038: docs/research_direction_wp11_synthesis.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_wp11_synthesis.md)

静的実装参照：[EV-0039: src/trotterlib/research_direction_full_opt2.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_full_opt2.py), [EV-0040: src/trotterlib/research_direction_wp11_synthesis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_wp11_synthesis.py), [EV-0041: scripts/run_research_direction_wp11_synthesis.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/scripts/run_research_direction_wp11_synthesis.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="opt-02"></a>

## OPT-02 P-B signal-aware selection

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-25（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | 位相誤差だけの選択をsignal情報で改善 |
| 変更部分 | target weight／radiusによるscreening |
| 改善機構 | leakageとvisibilityを考慮して不適切な候補を除く |
| 比較対象 | energy-only選択 |
| 科学条件 | H4 1Å/STO-3G/DF12、L_D0…11、6delta、q1/2/4 |
| 評価指標 | target weight / radius / selection change |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 現gridに実用的なselection差がない。一般signal問題の否定ではない |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | 現H4 gridで停止 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | meaningful inversion／near-tie改善を得ず |
| 未解決 | 別taskでsignalがdecisionを支配するか |
| 再利用 | signal/energyの分離診断 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0042: docs/research_direction_signal_weight_pilot.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_signal_weight_pilot.md)

静的実装参照：[EV-0043: src/trotterlib/research_direction_signal_weight_pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_signal_weight_pilot.py), [EV-0044: scripts/run_research_direction_signal_weight_pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/scripts/run_research_direction_signal_weight_pilot.py)

保存result/decision metadata：[EV-0045: artifacts/research_direction_signal_weight_pilot/2026-09-25/pb_h4_s2_prefix_grid_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/research_direction_signal_weight_pilot/2026-09-25/pb_h4_s2_prefix_grid_v1.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="opt-03"></a>

## OPT-03 P-C geometry tracking／energy差

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-25（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | geometry間の相殺でenergy差を安くする |
| 変更部分 | DF prefix追跡・係数interpolation |
| 改善機構 | 共通構造と符号付きbiasを利用する |
| 比較対象 | independent prefix／独立endpoint推定 |
| 科学条件 | linear H4/STO-3G/DF12/L_D3、局所.80–1.20Å→後続.70–1.60Å、二次partial S2、文書内delta窓 |
| 評価指標 | coefficient prediction / difference bias / breakdown flags |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 局所pilot支持。後続stretch予測破綻、tracked prefixは独立と同じ。主題候補を停止 |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | stop_pc_current_h4_family_as_primary |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | tracking固有差とbreakdown予告が不足 |
| 未解決 | 別familyで安い追跡が新しい選択を作るか |
| 再利用 | paired geometry条件とsigned-biasの記録 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：局所pilotの肯定的結果を後続tracking失敗で消さない。

根拠文書：[EV-0046: docs/research_direction_geometry_energy_difference_pilot.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_geometry_energy_difference_pilot.md), [EV-0047: docs/research_direction_geometry_tracking_breakdown.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_geometry_tracking_breakdown.md)

静的実装参照：[EV-0048: src/trotterlib/research_direction_geometry_energy_difference_pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_geometry_energy_difference_pilot.py), [EV-0049: scripts/run_research_direction_geometry_energy_difference_pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/scripts/run_research_direction_geometry_energy_difference_pilot.py), [EV-0050: src/trotterlib/research_direction_geometry_tracking_breakdown.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_geometry_tracking_breakdown.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="opt-04"></a>

## OPT-04 P-D energy/tail-aware高次PF選択

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-25（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | PF次数をtail負担と協調選択 |
| 変更部分 | 5公式をenergy／tail／finite modelで選ぶ |
| 改善機構 | 高次PFの負stageがrandom tailの絶対時間とnormalizationを増やす交換を利用 |
| 比較対象 | B1a/B1b/B2/B4公平再最適化 |
| 科学条件 | H4/STO-3G/DF12、L_D3/4/5、pilotδ=.4、S1 T=.8・δ候補、負時間/control/内部HDを区別 |
| 評価指標 | PF bias / Gamma_R / stage proxy / finite work |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | pilot選択差は成立、強いB1b/B2/B4は同じ候補を選びfinite補正固有利得なし |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | P-D S2停止; S1 Case B+undetermined_boundary |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 弱いouter-stage対照だけが差を作った |
| 未解決 | 具体的な別機構・同情報baselineとの差 |
| 再利用 | signed-time guardとenergy/tailの分離 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0051: docs/research_direction_energy_tail_pareto.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_energy_tail_pareto.md), [EV-0052: docs/research/研究概要・現状.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/%E7%A0%94%E7%A9%B6%E6%A6%82%E8%A6%81%E3%83%BB%E7%8F%BE%E7%8A%B6.md)

静的実装参照：[EV-0053: src/trotterlib/research_direction_energy_tail_pareto.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_energy_tail_pareto.py), [EV-0054: scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/scripts/tracks/algorithm_codesign/verify_ra_d0_source_review_v3.py), [EV-0055: scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py)

保存result/decision metadata：[EV-0056: artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json), [EV-0057: artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/evidence_manifest_v1.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="opt-05"></a>

## OPT-05 R3 split／formulaのmulti-fidelity選択

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-26（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | 詳細評価数を減らし選択を保つ |
| 変更部分 | cheap DF-prefix診断による候補選別 |
| 改善機構 | 安価な情報で不要な詳細評価を除く |
| 比較対象 | SPRINT等／一般multi-fidelity best-arm |
| 科学条件 | 理論・静的scope、独立pilot未実行。新分子/rank/L_D/delta未固定 |
| 評価指標 | regret / number of detailed evaluations |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | quantum-specific method deltaを固定できず数値pilotなし |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | STOP_R3_NO_METHOD_DELTA |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 一般手法を超える生成/認証差がない |
| 未解決 | 具体的な情報削減則が提示されるか |
| 再利用 | cheap/full情報の公平比較表 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0058: docs/research/r3_prior_art_and_minimal_contract.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/r3_prior_art_and_minimal_contract.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="err-01"></a>

## ERR-01 FR 位相・半径分離

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-27（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | finite-RTEの位相認証を鋭くする |
| 変更部分 | 正scalar分離・Hermitian/anti-Hermitian構造境界 |
| 改善機構 | 位相を変えないscalarを切り分けて保守性を減らす |
| 比較対象 | OLD/STRONG norm-disk・同情報scalar最適化 |
| 科学条件 | 2×2 toy FR-1、4×4非一様FR-R1b。分子/DF/L_Dなし、正負時間/K0/2/4、固定T/q/delta/r |
| 評価指標 | phase bound / one-sided witness / soundness |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 限定strict gainはあるが固定予算の実用認証差なし。直接帰結・反例・未解決を分離 |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | TECHNICAL_NOTE / MECHANISM_ONLY_NO_PRACTICAL_GO |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 実用GOなし、FR-R2等を開始しない |
| 未解決 | 構造情報の安い取得と非oracle利益 |
| 再利用 | mean operatorとchannelの区別、scalar処理 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0059: docs/research/fr_research_claim_and_manuscript.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/fr_research_claim_and_manuscript.md)

静的実装参照：[EV-0060: src/trotterlib/fr_revision_nonuniform.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/fr_revision_nonuniform.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pr-01"></a>

## PR-01 PR-2 圧縮＋残差random補完

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-04（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / codex/h2-outer32-metadata-export-20261023 |
| 固定資料commit | e3531956af6a6d359e2f1fb320e9a5179f41eb98 |
| 目的 | discard biasを補い回路を短縮 |
| 変更部分 | DF prefix保持と残差finite RTEをmatched signal accuracyで比較 |
| 改善機構 | bias・normalization・安いsample回路を同じshot費用へ戻す |
| 比較対象 | discard B0／full deterministic B1／partial B2／full random B3 |
| 科学条件 | H4 linear1/1.30Å/STO-3G/DF12、rank3/6/9等、T=.8、q1/2/4/8、M2固定B2 rank3候補、r/Kは固定5構成表 |
| 評価指標 | full-wrapper RZ/CX/depth / analytic shots / paired SE |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | development点最小と固定held-out構成のtransferを支持。一般rank最適・新算法・全RPE優位ではない |
| 調査用成績分類 | 限定条件で改善 |
| 既存状態の説明 | TRANSFER_SUPPORTED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | M2後mandatory STOP、後続PMは別承認で実施 |
| 未解決 | 希少tail・強いPF/合成対照・全energy task |
| 再利用 | 入力snapshot、residual再構成、matched-accuracy比較 |
| PRへの応用 | 直接応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

履歴・実行単位：旧S0 hash STOP→別series V1–V4/S2→M1-A/M1-B1→M2。GAP-006の旧pilot入力と新系列を混ぜない。

根拠文書：[EV-0061: docs/pr2_matched_accuracy_m2_transfer_result_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md), [EV-0062: docs/pr2_matched_accuracy_m1_b1_result_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m1_b1_result_validation.md), [EV-0063: docs/research/pr2_v1_v3_result_packet_ef86868_20260928.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/pr2_v1_v3_result_packet_ef86868_20260928.md), [EV-0064: docs/research/pr2_v0_input_recovery_audit_bf5d3b2_20260928.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/pr2_v0_input_recovery_audit_bf5d3b2_20260928.md)

静的実装参照：[EV-0065: scripts/run_pr2_matched_accuracy_m1_b1_result_validation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/scripts/run_pr2_matched_accuracy_m1_b1_result_validation.py)

保存result/decision metadata：[EV-0066: artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json), [EV-0067: artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json)

資料不足：[GAP-006](EVIDENCE_GAPS.md#gap-006)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pr-02"></a>

## PR-02 PR-3 tail限定Richardson

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-27（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | tailのbiasを外挿で減らす |
| 変更部分 | Re/Im平均信号の低次数外挿 |
| 改善機構 | 複数levelのlinear combinationでtail biasを相殺 |
| 比較対象 | 通常PR最良／tail-exact oracle |
| 科学条件 | 固定2qubit toy、tail levels4/8/16、target complex RMSE .05、分子/DF/L_Dなし |
| 評価指標 | signal bias / shots / backbone work |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | biasは下がったがshotとbackboneを戻すと不利。一般外挿の否定ではない |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | STOP_PR3_VARIANCE_BACKBONE_DOMINATES |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 統計分散とbackbone費用が支配 |
| 未解決 | 同meanを安く取得できる別構造があるか |
| 再利用 | 外挿係数の分散会計 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0068: docs/pr2_pr3_minimal_pilot_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/pr2_pr3_minimal_pilot_validation.md)

静的実装参照：[EV-0069: src/trotterlib/pr2_pr3_minimal_pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/pr2_pr3_minimal_pilot.py)

保存result/decision metadata：[EV-0070: artifacts/pr2_pr3_minimal_pilot/2026-09-27/pr2_pr3_minimal_pilot_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/pr2_pr3_minimal_pilot/2026-09-27/pr2_pr3_minimal_pilot_v1.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="pr-03"></a>

## PR-03 PR-1/4/5/6初期候補群

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-09-27（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / all-r-coherent-opt2-reoptimization |
| 固定資料commit | f29eafbf4412193e8e8660d9f051889c2cba684b |
| 目的 | PRの表現・sampling・誤差自由度を広げる |
| 変更部分 | 共通比較軸、測定実装監査等の候補整理 |
| 改善機構 | 候補ごとにtailと実装・測定を接続する |
| 比較対象 | 通常PRとstrong baseline |
| 科学条件 | 段階Aの概念計画。PR-2/3実行と他候補の保留を区別 |
| 評価指標 | candidate-specific proposed metrics |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 計画を実施済み成果へ数えない。後続に別名の対応候補があるか照合する |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | PR-4監査まで、PR-5/6保留 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 既存段階Aの制限を維持 |
| 未解決 | 各候補の後続系列への対応と未採択条件 |
| 再利用 | 候補分類・既存APIの対応 |
| PRへの応用 | 不明 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0071: partial_randomization_research_options_20260927.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/partial_randomization_research_options_20260927.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="eval-01"></a>

## EVAL-01 Track A PM-1/PM-2 precision資源地図

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-05（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / codex/h2-outer32-metadata-export-20261023 |
| 固定資料commit | e3531956af6a6d359e2f1fb320e9a5179f41eb98 |
| 目的 | PRが競争力を持つ精度・準備費条件を整理 |
| 変更部分 | nearby discard補完と保存値precision/prep感度 |
| 改善機構 | bias headroomと単価・shotを同じ目的関数に戻す |
| 比較対象 | B0 rank4/5等／B1/B2/B3 |
| 科学条件 | H4 linear1/1.30Å/STO-3G/DF12/T=.8、二次DF-prefix、PM-1 εsig=.05、PM-2固定precision表示点、Qiskit1.3 opt1 |
| 評価指標 | N / one-shot RZ / G / prep breakpoints |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 近接discard補完後も限定B2点利得が残る。小差の統計winnerは未確定 |
| 調査用成績分類 | 評価基盤・限定検証 |
| 既存状態の説明 | PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 資源研究の既存STOP |
| 未解決 | 強い高次PF・native synthesisと実準備 |
| 再利用 | 既存precision/resource table |
| PRへの応用 | 直接応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0072: docs/pr2_pm2_precision_resource_result_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm2_precision_resource_result_validation.md), [EV-0073: docs/pr2_pm1_discard_result_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm1_discard_result_validation.md)

静的実装参照：[EV-0074: src/trottertracks/resource_applicability/pm0_evidence_attribution.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/src/trottertracks/resource_applicability/pm0_evidence_attribution.py)

保存result/decision metadata：[EV-0075: artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result.json), [EV-0076: artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result_verification_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04/result_verification_v1.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="eval-02"></a>

## EVAL-02 Track A H4–H8強対照と希少tail

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-21（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / codex/h8-rare-rte-closure-policy-20261021 |
| 固定資料commit | 862d732fba0a5f353c522a9219dec78ae2f21a4b |
| 目的 | 既存PRの限定資源利益と限界を評価 |
| 変更部分 | geometry／独立確認／B0+I／rare-order層別 |
| 改善機構 | 短回路とnormalizationの交換を有限標本で評価 |
| 比較対象 | strong deterministic PF／通常PR |
| 科学条件 | H8 linear1Å/STO-3G/tol-only1e-8 DF28、二次partial PF、r4/K4、p8q2・p7q4等固定候補、T=.8/P0、δ=T/q。H4/H6は別source条件 |
| 評価指標 | shot-weighted compiled RZ / stratified costs / residual sensitivity |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 層別有限標本と残余費用の条件付き感度で採択。母平均優位保証・未観測費用0補完なし |
| 調査用成績分類 | 限定条件で改善 |
| 既存状態の説明 | H8 rare-RTE current-scope closure adopted |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 現在のclaim範囲で終了、追加sampling不要の既存判断 |
| 未解決 | 全support費用、実準備、長RPE。GAP-002は旧H4来歴 |
| 再利用 | 共通task／strong baseline／rare層の会計 |
| PRへの応用 | 直接応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0077: docs/research/h8_rare_rte_closure_policy_20261021.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md), [EV-0078: docs/research/track_a_research_handoff_20261018.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2c22c8edad6a497240f57f8654da3778c0df1fdd/docs/research/track_a_research_handoff_20261018.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：[EV-0079: artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/77bf6472f901feab8cdc4871369bc8ddfc04be7b/artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21/result.json)

資料不足：[GAP-002](EVIDENCE_GAPS.md#gap-002)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="eval-03"></a>

## EVAL-03 PR高次PF探索と次数逆転の既存証拠

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-23（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / codex/h2-outer32-metadata-export-20261023 |
| 固定資料commit | e3531956af6a6d359e2f1fb320e9a5179f41eb98 |
| 目的 | PF次数の比較可能性と既存順位を整理 |
| 変更部分 | S2／Yoshida4／m5_bestの既存取得・来歴照合 |
| 改善機構 | 次数・stage・normalization・単価の交換を既存結果から記録する |
| 比較対象 | 通常deterministic PF／higher-order PR |
| 科学条件 | H2/H4/H6別snapshot、1Å/STO-3G、tol-only1e-8、H4 rank10/prefix3・H6 rank19/prefix5等、T=.8、q1/2/4/8、K4/outer32。比較ledger差を保持 |
| 評価指標 | signal-error ledger / compiled cost / recorded ordering |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 既存逆転候補と政策/来歴差を記録。今回原因調査を開始しない |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | 科学的比較C・因果機構は未認定 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 別チャット対象。既存結果の読解のみ |
| 未解決 | common allowance／native-runtime binding／因果説明 |
| 再利用 | 固定係数・branch・policy差の索引 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0080: docs/research/track_a_pf_order_reversal_existing_evidence_audit_20261023.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/research/track_a_pf_order_reversal_existing_evidence_audit_20261023.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-01"></a>

## RB-01 B-F finite-RTE係数設計

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-05（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | finite tail modelを係数設計へ戻す |
| 変更部分 | 同5-stage対称4次familyのF/L探索 |
| 改善機構 | finite modelで誤ったleading選択を避ける |
| 比較対象 | 同情報・同32評価budgetのL／共通参照 |
| 科学条件 | 固定development、q1/2/4/8、Rbudget5/10/20/40/80、K2と事前trigger K4。詳細分子/DF/splitは契約参照 |
| 評価指標 | shot/native action proxy |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 保存cellの公平cross-scoreでF固有のdecision valueなし。未完了bridgeは未補完 |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | 元INCONCLUSIVE、R0 BF-A、現仮説限定closure |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | finite model追加価値なし |
| 未解決 | 別familyへ移す具体機構・同budgetとの差 |
| 再利用 | 結果前探索budgetとcross-score |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

履歴・実行単位：元serialization failure→read-only recovery。元runを完了へ上書きしない。

根拠文書：[EV-0081: docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md)

静的実装参照：[EV-0082: scripts/tracks/algorithm_codesign/run_bf1.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/run_bf1.py)

保存result/decision metadata：[EV-0083: artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1/result.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-02"></a>

## RB-02 B-M DF multirate／compact BCH

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-05（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 決定論block頻度の協調設計 |
| 変更部分 | nested列の三次誤差分解とcache |
| 改善機構 | 繰返し構造を再利用して評価を安くする |
| 比較対象 | 同情報compact BCH再帰＋同DF backend |
| 科学条件 | 記号監査、固定nested sweep。DF実入力の新順位・pilotなし |
| 評価指標 | symbolic coefficient identity / evaluation cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 固定adapterの三次係数は既知compact再帰と同じ。BM-1新手法検証なし |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | BM05_EQUIVALENT_NO_ESTABLISHED_METHOD_DELTA_MANDATORY_STOP |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立method deltaなし |
| 未解決 | 異なる具体生成則があるか |
| 再利用 | compact再帰・cacheをbaselineにも与える |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0084: docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md)

静的実装参照：[EV-0085: scripts/tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-03"></a>

## RB-03 BS block合成・SP synthesis placement

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-06（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 高価な角度合成をrandom cheap gateへ置換 |
| 変更部分 | PAI/notch辞書とplacement mask |
| 改善機構 | native T単価と重み二次momentの交換 |
| 比較対象 | NONE／deterministic native合成／ordinary finite mean |
| 科学条件 | SP0.5 fixed8angles、SP1 synthetic2qubit wrapper、12wrapper・48mask、εnative1e-6、whole-wrapperはadditive primitiveモデル |
| 評価指標 | expected T / moment/range / sufficient shots |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | primitiveではstrict trade-off。wrapperでshot増加・不適格もあり一律有利でない |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | PRIMITIVE_TRADEOFF_EXISTS; SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | SP1後STOP、BS0.5はnew-method deltaなし |
| 未解決 | full wrapper/DF接続・bias/range・準備費 |
| 再利用 | synthesis-placementとmoment会計 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0086: docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md), [EV-0087: docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md), [EV-0088: docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md)

静的実装参照：[EV-0089: scripts/tracks/algorithm_codesign/audit_sp1_saved_result.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/audit_sp1_saved_result.py), [EV-0090: scripts/tracks/algorithm_codesign/audit_sp05_saved_result.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/audit_sp05_saved_result.py)

保存result/decision metadata：[EV-0091: artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/saved_field_audit_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_sp1_wrapper_result/2026-10-06/v1/saved_field_audit_v1.json), [EV-0092: artifacts/track_b_sp05_economics_result/2026-10-06/v1/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_sp05_economics_result/2026-10-06/v1/result.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-04"></a>

## RB-04 R0/R1 finite-mean再配分

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-06（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 有限Taylor平均を保持して回転を安くする |
| 変更部分 | Euler再配分・共通角と価格law |
| 改善機構 | normalizationと角度辞書・native費用を交換する |
| 比較対象 | ordinary／PTSC-K0／A／collected CTS |
| 科学条件 | 2qubit finite P3、distinct-basis ordinary-controlled、x1/8・1/4、3native precision。PR molecular splitなし |
| 評価指標 | B² / T/CX/1Q / bias/shots |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 限定native Pareto上の有効点とtrade-offを保存。一般sampler最適・全PR利益は未確立 |
| 調査用成績分類 | 限定条件で改善 |
| 既存状態の説明 | R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 既存R1/R1.5 STOP |
| 未解決 | same-information CTSと一般入力費 |
| 再利用 | finite-mean・位相を保つ再配分 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0093: docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)

静的実装参照：[EV-0094: src/trottertracks/algorithm_codesign/rte_reallocation/model.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/rte_reallocation/model.py), [EV-0095: scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/analyze_r1p5_saved_attribution.py)

保存result/decision metadata：[EV-0096: artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_r1p5_saved_attribution/2026-10-06/attribution_summary_v1.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-05"></a>

## RB-05 RA-D0 LP・G1–G5固定辞書

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-10（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 同辞書内のnormalization/費用を最適化 |
| 変更部分 | LP／exact rational認証／任意IS比較 |
| 改善機構 | 返り語の集約と係数再配分を最適化する |
| 比較対象 | saved CTS law／同一辞書strong baseline |
| 科学条件 | 2qubit distinct-basis controlled、p3/4・1/4、x1/4、固定6vertex/digital class、有限r |
| 評価指標 | second moment / bias / native resources / class lower |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 指定ideal/digital classで同辞書優位主線を閉じる。全乱択法のno-goではない |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | G5_DIGITAL_SIX_VERTEX_CLASS_EXCLUDED_BY_SAVED_CTS_LAW |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 強いCTS対照による限定class除外 |
| 未解決 | 辞書外の生成差・取得費を伴う別構成 |
| 再利用 | exact class境界とstrong comparator |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：RA-D0 technical STOP／active-support修正／exact backend等を別runとして保持。G5の理論閉鎖と技術停止を区別。

根拠文書：[EV-0097: docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md)

静的実装参照：[EV-0098: src/trottertracks/algorithm_codesign/return_aggregation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/return_aggregation.py)

保存result/decision metadata：[EV-0099: artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/result_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/result_v1.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-06"></a>

## RB-06 G6 Return Aggregation／General Full

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-22（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / track-b-g6-v1-2-draft-review-integration-20261022 |
| 固定資料commit | 5f750a23f023d75c8c4fbcabc6937d6953ce61f2 |
| 目的 | finite Taylorを全列挙せず生成 |
| 変更部分 | formal return集約・parent/child・local Green query・finite bits |
| 改善機構 | 集約したrotationを局所生成しglobal normalizerを回避する |
| 比較対象 | ordinary／partial return／closed low-order／CTS |
| 科学条件 | 正有理p、Hermitian involution、odd m、short-step条件。理論とsynthetic providerを分離。H4 C1は別study |
| 評価指標 | mean correctness / bit cost / query cost / conditional native resources |
| 検証段階 | 適用域付き理論、finite-bit構成、人工/既知synthetic検証。H4移植はRB-17へ分離。 |
| 実際の結果 | 方法論と限定理論を保持。RAは広い構成、General Fullは生成方式、Closed P5は内部特殊化 |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | DRAFT_COMPLETE_AWAITING_USER_GPT_DECISION |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | v1.2 draft公開後STOP、投稿GOなし |
| 未解決 | 全DF provider・費用認証・全手順の独立新規性 |
| 再利用 | 局所generatorとfinite-bit/zero-fill仕様 |
| PRへの応用 | 直接応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：v1.0→v1.1→v1.2 draft。degree-zero parentでもchild rotation/集約係数が残る。

根拠文書：[EV-0100: docs/research/track_b_g6_v1_2_draft_review_integration_20261022/handoff.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/handoff.md), [EV-0101: docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md), [EV-0102: docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_2_draft.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_2_draft.md)

静的実装参照：[EV-0103: src/trottertracks/algorithm_codesign/return_aggregation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/src/trottertracks/algorithm_codesign/return_aggregation.py), [EV-0104: src/trottertracks/algorithm_codesign/g7_generator.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/src/trottertracks/algorithm_codesign/g7_generator.py), [EV-0105: src/trottertracks/algorithm_codesign/g9_p5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/src/trottertracks/algorithm_codesign/g9_p5.py)

保存result/decision metadata：[EV-0106: docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/exact_model_core.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/exact_model_core.json), [EV-0107: docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/input_identity.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/input_identity.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-07"></a>

## RB-07 G7 conditional native economics

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-10（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | return集約の費用価値を調べる |
| 変更部分 | helper/directとnormalizationの条件付き会計 |
| 改善機構 | normalization低減とnative回転費用を交換する |
| 比較対象 | ordinary／partial／P3+tail |
| 科学条件 | known synthetic p、P3 controlとP5 development、finite mean、抽象controlled-Q provider。DF/basis/L_Dなし |
| 評価指標 | native T intercept / provider slopes / attempt caps |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | m5登録3対照への条件付きT利得、m3 closed対照には不利。provider/prep/古典費用が残る |
| 調査用成績分類 | 限定条件で改善 |
| 既存状態の説明 | G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 限定取得終了 |
| 未解決 | 強いP5/CTS対照・実provider |
| 再利用 | native affine cost ledgerとsigned joint phase |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0108: docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md)

静的実装参照：[EV-0109: src/trottertracks/algorithm_codesign/g9_native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/g9_native.py)

保存result/decision metadata：[EV-0110: artifacts/track_b_g7_budget_control_result/2026-10-10/v1/result_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g7_budget_control_result/2026-10-10/v1/result_v1.json), [EV-0111: artifacts/track_b_g7_budget_control_result/2026-10-10/v1/saved_output_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g7_budget_control_result/2026-10-10/v1/saved_output_audit.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-08"></a>

## RB-08 G8 on-demand synthesis／provider budget

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-10（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 全angle表の事前取得を避ける |
| 変更部分 | local generator→cache miss時native RZ取得 |
| 改善機構 | 必要になった角度だけ取得しcacheを共有する |
| 比較対象 | G7固定inventory／closed low-order |
| 科学条件 | G7 development P3/P5、仮想provider δ=1e-6、finite bits/短trace。物理CQ未取得 |
| 評価指標 | cold/warm calls / conditional error budget / CPU trace |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | on-demand接続は成立。一般termination・大L scaling・物理provider性能は未保証 |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | G8_ON_DEMAND_PATH_AND_CONDITIONAL_PROVIDER_BUDGET_COMPLETE |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 条件付きbudget検証でSTOP |
| 未解決 | 全support費用上界、provider単価、全N古典費用 |
| 再利用 | on-demand/cache・誤差会計 |
| PRへの応用 | 直接応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0112: docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md)

静的実装参照：[EV-0109: src/trottertracks/algorithm_codesign/g9_native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/g9_native.py)

保存result/decision metadata：[EV-0113: artifacts/track_b_g8_on_demand_result/2026-10-10/v1/result_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g8_on_demand_result/2026-10-10/v1/result_v1.json), [EV-0114: artifacts/track_b_g8_on_demand_result/2026-10-10/v1/saved_output_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g8_on_demand_result/2026-10-10/v1/saved_output_audit.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-09"></a>

## RB-09 G9 Closed P5 matched-native

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-10（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 低次数return集約を安く生成 |
| 変更部分 | closed P5 group normalizerとnative direct実装 |
| 改善機構 | 同ensembleをclosed normalizerで生成してzero/予算保守性を減らす |
| 比較対象 | General Full／ordinary／partial／P3+tail／matched CTS |
| 科学条件 | 3system qubit synthetic Clifford+T provider、p1/5・3/10・1/2、x5/7、m5、Re/Im1/200、固定22axis failure配分 |
| 評価指標 | expected T / CX / 1Q / prep slope / range |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 登録direct比較でT-primaryの有効点、CTSにはCX利点。全座標優越・任意IS最適ではない |
| 調査用成績分類 | 限定条件で改善 |
| 既存状態の説明 | G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | v2完了後STOP |
| 未解決 | DF native移植・finite-bit差・setup/online償却 |
| 再利用 | g9_p5群式・native価格・strong comparator |
| PRへの応用 | 直接応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：v1 API technical failureを保存し別v2完了。FullとP5の理想受理ensemble一致と全trial law差を区別。

根拠文書：[EV-0115: docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md), [EV-0116: docs/research/track_b_g7_g9_strong_comparator_audit_20261019/handoff.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6576e606d4edcc6c53f095ee5bdf750defa99a42/docs/research/track_b_g7_g9_strong_comparator_audit_20261019/handoff.md)

静的実装参照：[EV-0117: src/trottertracks/algorithm_codesign/g9_p5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/g9_p5.py), [EV-0118: scripts/tracks/algorithm_codesign/audit_g9_v2_saved_outputs.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/audit_g9_v2_saved_outputs.py)

保存result/decision metadata：[EV-0119: artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_affine_T_comparison_v2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_affine_T_comparison_v2.json), [EV-0120: artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_output_audit_v2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_output_audit_v2.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-10"></a>

## RB-10 G10 degree-matched native比較

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-10（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 一般odd degreeでreturn方式の価値を確認 |
| 変更部分 | degreeごとの同target・native費用比較 |
| 改善機構 | 高degreeで集約率と角度価格が変わる |
| 比較対象 | 同degree ordinary／partial／closed/CTS等 |
| 科学条件 | 3system qubit known provider、p1/5・3/10・1/2、x5/7、固定degree集合・budget。m間は同一finite targetでない |
| 評価指標 | T intercept / prep slope / CX/1Q / resource cap |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | within-degreeの条件付きtrade-offを取得。次degree/分子/全PRへ一般化しない |
| 調査用成績分類 | 限定条件で改善 |
| 既存状態の説明 | G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | v3完了、G11未許可 |
| 未解決 | 強いsampling・古典取得費・実DF provider |
| 再利用 | degree別generator/streaming記録 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：v1/v2 resource/IO等のtechnical失敗→別v3完了。旧prefix利用制限、訂正auditorを保持。

根拠文書：[EV-0121: docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md)

静的実装参照：[EV-0122: src/trottertracks/algorithm_codesign/g10_generator.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/g10_generator.py), [EV-0123: scripts/tracks/algorithm_codesign/audit_g10_v3_saved_outputs.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/audit_g10_v3_saved_outputs.py)

保存result/decision metadata：[EV-0124: artifacts/track_b_g10_degree_result/2026-10-10/v3/evidence_manifest_v3.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_g10_degree_result/2026-10-10/v3/evidence_manifest_v3.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-11"></a>

## RB-11 DF希薄三次補正・generic whole

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-12（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 三次誤差を局所補正して四次化 |
| 変更部分 | whole指数rotation対局所Pauli線形補正 |
| 改善機構 | 少数軌道支持の補正を安い確率rotationで挿入する |
| 比較対象 | 同じ局所情報を持つPauli対照 |
| 科学条件 | synthetic3/6active modes、rank2×3 fixture。CDとE/Tは別条件。分子geometry/basis/L_D非該当 |
| 評価指標 | conditional order / native T / completion lower/upper |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 局所支持・平均構成は保持。固定加法planの3条件でwhole下界が対照上界より高く、1条件未確定 |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | GENERIC_WHOLE_SERIES_CLOSED_TO_AUTOMATIC_EXTENSION |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | generic whole主線への追加投資を区切る |
| 未解決 | 一般compiler・native phase・同task総費用 |
| 再利用 | 局所support判定・補正意味論 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：多数のnumeric/storage/phase停止→saved-only E/T recovery。原v7 TECHNICAL_INCONCLUSIVEは保持。

根拠文書：[EV-0125: docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md)

静的実装参照：[EV-0126: src/trottertracks/algorithm_codesign/df_correction_comparison/comparison.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/df_correction_comparison/comparison.py)

保存result/decision metadata：[EV-0127: artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/comparison_rows.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/comparison_rows.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-12"></a>

## RB-12 Gaussian積生成

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-13（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | DF event積の基底変換費を下げる |
| 変更部分 | Gaussian積/相対basisの直接入力 |
| 改善機構 | 隣接基底変換の積をまとめる |
| 比較対象 | 同情報既知Gaussian compiler |
| 科学条件 | 静的DF-RTE構成監査。新分子・rank/split/delta比較なし |
| 評価指標 | conditional wrapper cost / method delta |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 候補固有の追加手順が未定義で閉鎖。性能で敗れた実験結果ではない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | 独立新手法としてCLOSED、性能UNDETERMINED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 既知手順との差がない |
| 未解決 | 安い具体accessと費用差 |
| 再利用 | Gaussian積とphase/actual-adjoint条件 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0128: docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-13"></a>

## RB-13 D1–D3 finite-target・境界相殺・保証価格

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-14（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | representation/native/保証の交換を同taskで評価 |
| 変更部分 | 同finite P5の6方式、boundary cancellation、保証×価格ledger |
| 改善機構 | 平均を保ちlaw・range・単価・r選択を協調する |
| 比較対象 | ordinary／partial／P3+tail／Full／P5／matched CTS |
| 科学条件 | synthetic3qubit involution、p1/5・3/10・1/2、τ5/7、r1/2、εaxis1/200、strict native誤差固定。molecule/DF/L_Dなし |
| 評価指標 | Bernstein sufficient N / T/CX/1Q / prep sensitivity |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 同targetの資源交換とboundary効果を保存。全方式に共通最適化、十分forecastであり新生成利益の証明でない |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | D1/D2/D3 v2 completed; current numerical exploration closed |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | D1–D3探索は一区切り |
| 未解決 | DF scaling・online古典・full wrapper |
| 再利用 | same-mean accounting、literal inverse cancellation、四ledger |
| PRへの応用 | 直接応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：D3 v1型変換失敗→別v2完了。価格精度変更とnative error creditを区別。

根拠文書：[EV-0129: docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)

静的実装参照：[EV-0130: src/trottertracks/algorithm_codesign/rte_development/accounting.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/rte_development/accounting.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-14"></a>

## RB-14 Parent allocation／S4 KKT

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-15（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | 角度をcheap辞書へ寄せる |
| 変更部分 | even parent係数をchildへcost-aware配分 |
| 改善機構 | L1増加を許し回転価格を下げる |
| 比較対象 | common allocation／ordinary／closed P5、同じIS機会 |
| 科学条件 | 固定development finite P5、r1/all-accept、正係数・fixed price proxy、native取得後の再選択なし |
| 評価指標 | T/CX/1Q / second moment / sufficient N |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 限定配分成果・訂正KKTを保持。commonはL1最小でありchild配分はL1改善自由度でない |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | D＋E正式採択・現数値探索を区切る |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 同fixture追加探索を区切る |
| 未解決 | 新しい価格構造・未知normalizer/online費 |
| 再利用 | parent_allocationと強い同情報比較 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

履歴・実行単位：v1 failure→v2/v3→誤ったKKT訂正→S4補完。旧proxy結果・markers不変。

根拠文書：[EV-0131: docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md)

静的実装参照：[EV-0132: src/trottertracks/algorithm_codesign/parent_allocation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/parent_allocation.py)

保存result/decision metadata：[EV-0133: artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/saved_output_audit_s4.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v4/saved_output_audit/saved_output_audit_s4.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-15"></a>

## RB-15 M1-S shared ± labels

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-17（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / track-b-m1s-integrated-handoff-20261017 |
| 固定資料commit | 5f079dbe67bb35eff35613857b169fdf812af4b8 |
| 目的 | 干渉信号の減衰と位相を改善 |
| 変更部分 | Hadamard正負branchの同位置Pauli label共有 |
| 改善機構 | 共通乱数で中心化変動の一部を抑える |
| 比較対象 | ordinary／independent Pauli exponential、同総label数等を区別 |
| 科学条件 | H4 1Å/STO-3G/DF12、generation-prefix G3/G6/G9、保存sector-ground、δ1/8・1/16、R2/4、T1/4。C-sec理論は別適用域 |
| 評価指標 | visibility / phase bias / conditional sufficient shots |
| 検証段階 | 適用域付き理論＋保存H4数値診断＋同保存信号の事後解析。 |
| 実際の結果 | 適用域付き理論と局所signal/整数witnessを保持。strong新primitive・whole-cost・多roundRPE未確立 |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | 限定理論受理・統合後mandatory STOP、正式方向未採択 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 資源路線へ自動進行しない |
| 未解決 | full-H/HD旧行列、先行定理対応、実wrapper/状態/decoder |
| 再利用 | 共有law・visibility/phase trade-off台帳 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

履歴・実行単位：Gate0 v4 technical STOP→v5→部分独立照合→finite signal v2→gamma→π/3→Stage S。共通保存信号を独立実験に増やさない。

根拠文書：[EV-0134: docs/research/m1s_integration_20261017/integrated_research_summary.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md), [EV-0135: docs/research/m1s_integration_20261017/evidence_and_results_index.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/evidence_and_results_index.md)

静的実装参照：[EV-0136: src/trottertracks/algorithm_codesign/m1s_finite_signal_v2/pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/src/trottertracks/algorithm_codesign/m1s_finite_signal_v2/pilot.py)

保存result/decision metadata：[EV-0137: artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/result.json), [EV-0138: artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/saved_identity_status_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cdcb7f6f151df217960611f502eb1bac8be63f58/artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/saved_identity_status_audit.json)

資料不足：[GAP-003](EVIDENCE_GAPS.md#gap-003)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-16"></a>

## RB-16 G6 finite-bit／B3 live API・古典費用

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-20（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / track-b-c1-0-technical-preflight-20261020 |
| 固定資料commit | a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed |
| 目的 | 非列挙生成を監査可能なlive経路へ接続 |
| 変更部分 | dyadic/angle guard、closed ports、offline verifier、限定pilot |
| 改善機構 | 重複query/保存/検査を最適化し生成費を抑える |
| 比較対象 | 旧G7 law／同条件baselineの必要性 |
| 科学条件 | 固定G7/P5 workloadとrelease別API、correctness512と限定16896 pilotを区別。分子/DFなし |
| 評価指標 | correctness / CPU/bit/cache/storage scope |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | live接続と限定workload観測。全N・Closed P5/CTS同条件古典費用・新量子優位は未確立 |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | 限定pilot技術記録・mandatory STOP |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 別releaseごとのSTOPを維持 |
| 未解決 | same-budget stronger comparatorと全N償却 |
| 再利用 | finite-bit契約・API separation・provenance |
| PRへの応用 | 直接応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0139: docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md)

静的実装参照：[EV-0140: src/trottertracks/algorithm_codesign/g6_w2c_b3_limited_pilot_live_v1/execution.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed/src/trottertracks/algorithm_codesign/g6_w2c_b3_limited_pilot_live_v1/execution.py)

保存result/decision metadata：[EV-0141: docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/result_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/result_summary.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-17"></a>

## RB-17 C1 G6–DF移植・H4 logical pilot

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-21（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / track-b-c1-freeze-and-manuscript-synthesis-20261021 |
| 固定資料commit | 185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e |
| 目的 | return構成を分子DFへ移す |
| 変更部分 | exact input packet、controlled adapter、whole-path接続 |
| 改善機構 | 同じtailを別lawで生成して回転/zeroを減らす |
| 比較対象 | ordinary／General Full／Closed P5、人工5方式 |
| 科学条件 | H4 linear1Å/STO-3G/DF12、R1 generation-prefix L_D3、13bases/324labels、T=.8/q1/r4/m5、microstep.2、1024paths/方式、TAIL_ONLY_PLUS_SCALAR |
| 評価指標 | logical counts / weight records / CPU/setup / selected correctness |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 3方式探索を完走した記録、logical差は小さくzero削減未観測。strict native/N/全wrapperは未取得 |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | C1追加研究を一時凍結、G6数学的方法論維持 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 現在条件で追加投資を凍結 |
| 未解決 | rigorous model binding、deterministic halves、十分N、strong公平CPU比較 |
| 再利用 | exact packet・adapter・失敗原因とphase修正 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

履歴・実行単位：旧partition hash STOP→別R1→state hash STOP→reference zero-sign修正→旧P5 worker停止→別P5R完走。

根拠文書：[EV-0101: docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md), [EV-0142: docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md), [EV-0143: docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/SOURCE_CONNECTION_AUDIT.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/SOURCE_CONNECTION_AUDIT.md)

静的実装参照：[EV-0144: src/trottertracks/algorithm_codesign/g7_generator.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/src/trottertracks/algorithm_codesign/g7_generator.py), [EV-0145: src/trottertracks/algorithm_codesign/g9_p5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/src/trottertracks/algorithm_codesign/g9_p5.py), [EV-0146: src/trottertracks/algorithm_codesign/g6_f3_generator.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/src/trottertracks/algorithm_codesign/g6_f3_generator.py)

保存result/decision metadata：[EV-0147: docs/research/track_b_c1_5a_p5_worker_recovery_20261021/result/summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/result/summary.json), [EV-0106: docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/exact_model_core.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/exact_model_core.json)

資料不足：[GAP-007](EVIDENCE_GAPS.md#gap-007)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rep-04"></a>

## REP-04 N1 representation selector

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-11（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / n1-decode-recovery-execution-20261011 |
| 固定資料commit | 4c719a72eb8caae2f77fc746bd4672d610d97610 |
| 目的 | 表現近似の回路節約を資源利益へ変える |
| 変更部分 | Gaussian completion/cluster候補とbias込みselector |
| 改善機構 | basis費を減らしつつ元Hの精度を保つ |
| 比較対象 | exact fallback／同情報候補 |
| 科学条件 | synthetic JW4、ν2、signed DF2、T=.6/q1/2/4/8、δ=.6/.3/.15/.075、εcomplex.04 |
| 評価指標 | certified shot×RZ/CX |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | decode修正後も固定全contextでexactが選ばれ、certified目的改善なし |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | N1 STOP |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 現selector scan不成立 |
| 未解決 | 別表現の具体生成差とheadroom |
| 再利用 | selectorのbias/費用統合・exact fallback |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0148: docs/research/n1_decode_recovery_v1_results_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_results_20261011.md)

静的実装参照：[EV-0149: src/trottertracks/representation_exploration/algorithm_design.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/src/trottertracks/representation_exploration/algorithm_design.py)

保存result/decision metadata：[EV-0150: artifacts/n1_decode_recovery_v1/2026-10-11/run1/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/artifacts/n1_decode_recovery_v1/2026-10-11/run1/result.json), [EV-0151: artifacts/n1_decode_recovery_v1/2026-10-11/run1/comparison_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/artifacts/n1_decode_recovery_v1/2026-10-11/run1/comparison_summary.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rep-05"></a>

## REP-05 N2 signed charge・L1 fit

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-11（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / n1-decode-recovery-execution-20261011 |
| 固定資料commit | 4c719a72eb8caae2f77fc746bd4672d610d97610 |
| 目的 | charge表現で相互作用を圧縮 |
| 変更部分 | signed low-rank fitとcompute/phase/uncompute |
| 改善機構 | 係数fitを少数算術操作へ変える |
| 比較対象 | direct／known HWP |
| 科学条件 | synthetic JW4、後続はfixed4×2 SとεH=.0021の一件。化学geometry/basis/PF/L_Dなし |
| 評価指標 | fit residual / LP primal/dual / circuit cost |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | weighted entry-L1訂正は成立。旧charge回路の費用増とfit改善を混同しない |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | N2 HOLD |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 安い算術回路とfitの両立未支持 |
| 未解決 | bit幅・adder・ancilla・phase費用 |
| 再利用 | signed-fit規約とprimal/dual検査 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0152: docs/research/n13_redesign_results_20261011.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n13_redesign_results_20261011.md)

静的実装参照：[EV-0149: src/trottertracks/representation_exploration/algorithm_design.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/src/trottertracks/representation_exploration/algorithm_design.py)

保存result/decision metadata：[EV-0153: artifacts/n13_redesign/2026-10-11/remote_retrieval_result_commit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/artifacts/n13_redesign/2026-10-11/remote_retrieval_result_commit.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rep-06"></a>

## REP-06 N3 occupation／active-energy certificate

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-13（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / n3-achievements-closure-20261013 |
| 固定資料commit | ccc575ee339be091fd4031272c38bee13866f097 |
| 目的 | active空間縮約誤差を構造認証 |
| 変更部分 | occupation/Schur/Fantope/SDP/SOS比較 |
| 改善機構 | 結合方向とoccupationを残し保証幅を狭める |
| 比較対象 | 強い既存certificate／同品質費用 |
| 科学条件 | 限定one-external mode、real one-body＋非負density、固定粒子。stress family5/6→4/5、εproj=.01等、PF/DF/L_Dなし |
| 評価指標 | active-energy width / PSD witness / prototype cost |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 限定理論・保存幅改善は保持。元精度で追加縮約なし、強い既存族との差不足 |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | 現系列CLOSED・主研究不採択 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 同quality task利益未確立 |
| 未解決 | 一般二体・全energy/solver費、未保存因果情報 |
| 再利用 | occupation certificate・rational witness |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0154: docs/research/n3_saved_attribution_v1_claim_and_observability_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/docs/research/n3_saved_attribution_v1_claim_and_observability_20261013.md)

静的実装参照：[EV-0155: src/trottertracks/representation_exploration/algorithm_design.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/src/trottertracks/representation_exploration/algorithm_design.py)

保存result/decision metadata：[EV-0156: artifacts/n3_achievements_closure/2026-10-13/research_decision.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/afecaf591e62b8ab7ba5fb5c20f53f17a6d7be37/artifacts/n3_achievements_closure/2026-10-13/research_decision.json), [EV-0157: artifacts/n3_achievements_closure/2026-10-13/research_decision.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/artifacts/n3_achievements_closure/2026-10-13/research_decision.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rep-07"></a>

## REP-07 THRIFT／mixed relative evolution

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-13（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | mixed可解blockでstepを削減 |
| 変更部分 | interaction/mixed relative template |
| 改善機構 | 可解部をexactに扱いstep誤差を下げる |
| 比較対象 | 固定7方式・費用model |
| 科学条件 | H4 linear1Å/STO-3G、8modes/N4、DF16全保持、T1/4、r1…1024、εU=.001、PR L_Dなし |
| 評価指標 | operator-error frontier / arithmetic budget |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | ideal step削減はあるが追加算術を回収する登録予算不足 |
| 調査用成績分類 | 肯定と否定が混在 |
| 既存状態の説明 | 現template CLOSED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 詳細backend投資見送り |
| 未解決 | 別の安いmixed oracleとcontrol費 |
| 再利用 | mixed access契約と費用headroom |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0158: docs/research/qc_mixed_access_feasibility_v1_closure_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_mixed_access_feasibility_v1_closure_20261013.md)

静的実装参照：[EV-0159: src/trottertracks/representation_exploration/qc_mixed_access_math_v1.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/src/trottertracks/representation_exploration/qc_mixed_access_math_v1.py)

保存result/decision metadata：[EV-0160: artifacts/qc_mixed_access_closure/2026-10-13/research_decision.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/artifacts/qc_mixed_access_closure/2026-10-13/research_decision.json), [EV-0161: artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/descriptive_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/execution_audit/descriptive_summary.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-01"></a>

## QC-01 C1 macrocycle／符号相殺

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-14（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | ground-connected PF位相の誤差を相殺 |
| 変更部分 | 固定macrocycle composition |
| 改善機構 | 異符号主要誤差を合成で消す |
| 比較対象 | single/subdivision／S4／C2強対照 |
| 科学条件 | 非分子2qubit Pauli4H、3fragment、24macro、6h、target1e-5/1e-6、価格3/6。DF/L_Dなし |
| 評価指標 | eigenphase / overlap / conditional cost proxy |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 通常contextで事後oracleを含め登録優位なし。bad contextは適格不足で未判定 |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | 固定catalog内negative・新catalogなし |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | fixed catalogの利益を得ず |
| 未解決 | catalog外の具体機構 |
| 再利用 | branch/overlapと強いcomposition baseline |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0162: docs/research/qc_c1_saved_oracle_diagnostic_v1_results_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_c1_saved_oracle_diagnostic_v1_results_20261014.md)

静的実装参照：[EV-0163: src/trottertracks/representation_exploration/c1_macrocycle_pilot_v1.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/src/trottertracks/representation_exploration/c1_macrocycle_pilot_v1.py)

保存result/decision metadata：[EV-0164: artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/result_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/artifacts/qc_c1_saved_oracle_execution_v1/2026-10-14/result_summary.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-02"></a>

## QC-02 C2 branch付きPF外挿

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-14（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | PF biasを外挿で除く |
| 変更部分 | 同branch二点Richardsonと精度配分 |
| 改善機構 | 先導bias相殺で許容stepを広げる |
| 比較対象 | 既知外挿・同情報強対照 |
| 科学条件 | 静的監査、実分子/DF/L_D/precision pilot未固定 |
| 評価指標 | proposed energy bias / multiple-width controlled cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 式は条件付き成立、独立した安いbranch/remainder取得規則未特定 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | 現案は実装停止 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 既知手順との差・情報取得費が未解決 |
| 未解決 | ground同定・残余certificate・複数幅全費用 |
| 再利用 | branch/remainderと外挿分散の条件 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0165: docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-03"></a>

## QC-03 C3 self-energy／active space

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-14（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | virtual couplingを保持してmode削減 |
| 変更部分 | Schur/Feshbach root・resolvent access |
| 改善機構 | 小active空間にvirtual補正を戻す |
| 比較対象 | 既知downfolding／Neumann／QSVT/LCU |
| 科学条件 | 静的監査、E<c等の条件。新分子/active space/PF窓未固定 |
| 評価指標 | proposed energy/root error / normalization/body rank |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 条件付きroot構造は整理、安い具体access未特定 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | 現案は実装停止 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 既知構成と高body・取得費の壁 |
| 未解決 | normalizer/control/ground solverと全費用 |
| 再利用 | active-space残差を消さない台帳 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0165: docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-04"></a>

## QC-04 C4 cost-aware proposal／whole path

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-14（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | 同meanのsampling費を削減 |
| 変更部分 | 価格law・rejection/zero-fill・path DP |
| 改善機構 | 高価なeventを低頻度化しweightを補正する |
| 比較対象 | 同情報importance sampling／weighted automata |
| 科学条件 | finite-RTEの静的law監査、分子/DF/L_D/delta pilotなし |
| 評価指標 | moment / range / sufficient shots / online cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | q∝\|a\|/sqrt(g)等の既知法対応。moment×平均価格だけで全費用優位を示さない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | Phase B停止・独立生成差未特定 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 圧縮された安いpath取得則が未成立 |
| 未解決 | inverse stack、finite bits、tail/range、zero-cost |
| 再利用 | 全trial denominator・proposal費用の条件 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0166: docs/research/qc_c4_finite_rte_gpt_handoff_v1_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_c4_finite_rte_gpt_handoff_v1_20261014.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：[EV-0167: artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/method_status.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/method_status.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-05"></a>

## QC-05 一体再配分・exact-ground PF応答

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-15（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | ground-energy誤差を同情報で低減 |
| 変更部分 | 一体項再配分・leading応答・finite-step成分比較 |
| 改善機構 | ground状態の誤差構造を係数配置へ戻す |
| 比較対象 | 同情報compact BCH／強い費用対照 |
| 科学条件 | 現固定toy/DF構成と4登録h条件、10%費用基準。分子/DF/L_D/deltaの完全値は固定契約参照 |
| 評価指標 | leading error / finite-step cost |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 生成法はREDUCES_TO_KNOWN、別scopeのfinite-step基準も4条件未達 |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | 現固定構成CLOSED; NO_UNIFORM_REGISTERED_COMPONENT_GAIN |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 同情報生成差と登録費用gainなし |
| 未解決 | 別の明示的生成/情報レバー |
| 再利用 | equal-access first-pair比較 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0168: docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md)

静的実装参照：[EV-0169: src/trottertracks/ground_pf_repartition/qc_onebody_repartition_h_boundary_reaudit_v2.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/src/trottertracks/ground_pf_repartition/qc_onebody_repartition_h_boundary_reaudit_v2.py)

保存result/decision metadata：[EV-0170: artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/result_identity.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/result_identity.json), [EV-0171: artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/saved_results_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/artifacts/qc_onebody_repartition_h_boundary_reaudit_v2_execution/2026-10-15/saved_results_summary.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-06"></a>

## QC-06 H-E1・M-a/M-b PF小改変

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-16（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | 固有値特化correctorでPFを改善 |
| 変更部分 | outer commutator消去・processor/counterterm候補 |
| 改善機構 | 対象固有値で消える誤差方向を除く |
| 比較対象 | known processed PF／Hejazi／同情報BCH |
| 科学条件 | 静的K0と固定小構成。新分子/rank/L_D/delta追加なし |
| 評価指標 | method delta / leading structure / proposed cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 既知処理への還元や生成規則不足。未実装を未検討扱いしない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | H-E1現定義CLOSED、M追加探索STOP、K1未認可 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 現定義の独立方法差未成立 |
| 未解決 | 多fragment生成・有限step・control全費用 |
| 再利用 | known correctorとfinite-step限界 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0172: docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-07"></a>

## QC-07 SOS-PF生成・K0/K0.5

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-16（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | 低energy構造をPF誤差へ利用 |
| 変更部分 | PSD化・DF/BLISS/Gram・SOS gap |
| 改善機構 | 正値構造で低energy誤差を抑える |
| 比較対象 | 既知SOS/low-energy構成 |
| 科学条件 | 静的監査、現分割生成の固定scope、K1-Fなし。分子/rank/L_D/window新固定なし |
| 評価指標 | gap/remainder / generation cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 生成探索の終了と保証理論の非採択を分離する |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | 現分割生成CLOSED、A2 NOT_ADOPTED、K0 UNDETERMINED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 具体的有利な生成と有限保証・総費用が未成立 |
| 未解決 | 有用なgap/remainderと同情報強対照 |
| 再利用 | PSD/SOSとBLISSの重複整理 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0172: docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-08"></a>

## QC-08 A2 RF0/RF1/EW2有限step相対保証

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-18（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-current-route-handoff-20261018 |
| 固定資料commit | f7e103bf3a0ffcee6162bd4b8d244f9a2f4502c5 |
| 目的 | 低energy相対形式保証でstepを広げる |
| 変更部分 | 正stage S2の相対係数・有限剰余・energy-window tail |
| 改善機構 | H−LIの正値構造から低energy誤差を抑える |
| 比較対象 | centered Cbar2強化対照 |
| 科学条件 | H4 1Å/STO-3G/保存DF12/N4/8modes全block、one_body→block0…11、d=.001Ha/αspec=.1。PR splitなし |
| 評価指標 | sufficient-step enclosure / checker/preprocess cost |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 理論・checkerは保持。固定DF pilotでは強いcentered対照の十分stepが大きい |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | A2 NOT_ADOPTED、現実用化経路追加投資なし |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 現経路への追加投資見送り |
| 未解決 | 鋭いΛ・構造保持剰余、PRへの保証接続 |
| 再利用 | RF/EW保証・rational checkerと負の結果 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0173: docs/research/qc_a2_current_route_handoff_20261018/achievements_and_limits.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7e103bf3a0ffcee6162bd4b8d244f9a2f4502c5/docs/research/qc_a2_current_route_handoff_20261018/achievements_and_limits.md)

静的実装参照：[EV-0174: src/trottercert/a2_pilot/adapter.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7e103bf3a0ffcee6162bd4b8d244f9a2f4502c5/src/trottercert/a2_pilot/adapter.py)

保存result/decision metadata：[EV-0175: artifacts/qc_a2_phase_b_execution/2026-10-18/saved_results_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7e103bf3a0ffcee6162bd4b8d244f9a2f4502c5/artifacts/qc_a2_phase_b_execution/2026-10-18/saved_results_summary.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-09"></a>

## QC-09 DF covariance縮約

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-18（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-iv-covariance-method-audit-20261018 |
| 固定資料commit | 88e4665590bb93c16fdd070e14980d9d3cbff699 |
| 目的 | DF相関を安く縮約してenergy情報を取得 |
| 変更部分 | factor contraction／streaming／cache／signed aggregation |
| 改善機構 | 既知表現変換・相関再利用で取得/回路費を節約する |
| 比較対象 | 同DF/state/precisionを持つ既知対照K* |
| 科学条件 | 静的source/数式/API監査。新分子・basis・rank/L_D/time比較なし |
| 評価指標 | method delta / access cost / proposed variance |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 同情報の既知対照が同じ演算列を選べ、独立した情報削減・計算量差は確認されない。 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | REDUCES_TO_KNOWN_METHOD |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立した安い生成/情報規則が未提示 |
| 未解決 | 実装上の便利さ・runtime・元Hの全費用は未検証。未知費用は方法差ではない。 |
| 再利用 | encoding/control/scalar台帳と既知還元 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0176: docs/research/qc_iv_covariance_method_audit_20261018/handoff.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/88e4665590bb93c16fdd070e14980d9d3cbff699/docs/research/qc_iv_covariance_method_audit_20261018/handoff.md), [EV-0177: docs/research/qc_iv_selection_hold_20261018/handoff.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7eff8a33bef4ee1f29698ab10c6da46988dc8db5/docs/research/qc_iv_selection_hold_20261018/handoff.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：[EV-0178: docs/research/qc_iv_selection_hold_20261018/decision.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7eff8a33bef4ee1f29698ab10c6da46988dc8db5/docs/research/qc_iv_selection_hold_20261018/decision.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rep-08"></a>

## REP-08 非DF Pauli対DF fixed-signal比較

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-23（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / codex/non-df-h2-process-recovery-20261023 |
| 固定資料commit | c8990de7721545faff88b636108034dedb80a463 |
| 目的 | DF前提を外し表現とPRを比較 |
| 変更部分 | Pauli/DF各PF/PRの4固定方式 |
| 改善機構 | 表現で単価・tail norm・shot負担を変える |
| 比較対象 | 同MO積分・同reference signalの4方式 |
| 科学条件 | H2 linear1Å/STO-3G、DFtol-only1e-8 rank3、T=.8/q4/δ=.2/S2、PauliPR L_D7・DFPR L_D1、r2/K4、64paired trajectories、P0 |
| 評価指標 | C / analytic N / compiled RZ G / paired SE |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | H2固定4方式の条件付き比較完了。母平均rare-order費用は未同定、H4は人工準備のみ |
| 調査用成績分類 | 技術成果あり・科学的利益未確定 |
| 既存状態の説明 | COMPLETE_MANDATORY_STOP; H4 readiness only |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 元run01失敗保全、別run02完了後STOP |
| 未解決 | 未観測order費用・H4等への移送 |
| 再利用 | 表現identity・ordinary control・process分離基盤 |
| PRへの応用 | 直接応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

履歴・実行単位：PauliPR/DFPF thread競合で失敗→別process実行。旧成功2case再利用、新2case取得。H4未実施。

根拠文書：[EV-0179: docs/non_df_h2_process_recovery_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/docs/non_df_h2_process_recovery_validation.md), [EV-0180: docs/non_df_h4_readiness_validation.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7da77da0581c7a79c31c981cd9c6c1fcb05d98f/docs/non_df_h4_readiness_validation.md)

静的実装参照：[EV-0181: src/trotterlib/h2_non_df_process_recovery.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/src/trotterlib/h2_non_df_process_recovery.py), [EV-0182: src/trotterlib/isolated_qiskit_execution.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/src/trotterlib/isolated_qiskit_execution.py), [EV-0183: src/trotterlib/h2_non_df_process_integration.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/src/trotterlib/h2_non_df_process_integration.py)

保存result/decision metadata：[EV-0184: artifacts/non_df_h2_process_recovery/2026-10-23/comparison.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/artifacts/non_df_h2_process_recovery/2026-10-23/comparison.json), [EV-0185: artifacts/non_df_h4_readiness/2026-10-23/validation_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7da77da0581c7a79c31c981cd9c6c1fcb05d98f/artifacts/non_df_h4_readiness/2026-10-23/validation_summary.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="ext-01"></a>

## EXT-01 P1* bus/CAR固定word

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-17（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / qc-algorithm-difference-construction-20261018 |
| 固定資料commit | beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b |
| 目的 | fermionic構造でinteracting演算を生成 |
| 変更部分 | bus条件付き変位の固定12segment word |
| 改善機構 | CAR由来の相殺で低次残余を消す |
| 比較対象 | 既知rectangle＋Strang構成 |
| 科学条件 | 紙上/固定exact solver ansatz、分子/basis/DF/L_Dなし |
| 評価指標 | formal order / fixed-ansatz satisfiability |
| 検証段階 | 固定分子/保存Hamiltonianまたは実装の既存局所検証。対象条件・scopeはscientific_conditionsと固定資料参照。 |
| 実際の結果 | 固定wordは低次で不成立、後続紙上障害が固定solver結果を包含。一般bus不可能性ではない |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | NO_SOLUTION_IN_FIXED_ANSATZ |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 現固定ansatzの障害 |
| 未解決 | 辞書外の具体的生成機構 |
| 再利用 | 必要条件とpositive control |
| PRへの応用 | 低い |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0186: docs/research_exploration/research_strategy_redesign_20261017/research_exploration_retrospective.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/research_strategy_redesign_20261017/research_exploration_retrospective.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="ext-02"></a>

## EXT-02 J1 joint-loop・echo構成

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-17（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / qc-algorithm-difference-construction-20261018 |
| 固定資料commit | beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b |
| 目的 | 複数factorを安い共同閉路で生成 |
| 変更部分 | 20pulse／16/24pulse紙上構成 |
| 改善機構 | 未完了legの共有で操作数削減を狙う |
| 比較対象 | completed gate echo／rectangle Strang |
| 科学条件 | 静的固定pulse word、分子/DF/L_D/数値delta窓なし |
| 評価指標 | formal residual / method delta |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 成立したものは既知生成、独自候補は三次で不成立。追加探索保留 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | PAUSED_PENDING_NEW_MECHANISM |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 新機構がない |
| 未解決 | 新しい明示的閉路生成則 |
| 再利用 | bus依存残余の失敗機構 |
| PRへの応用 | 低い |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0187: docs/research_exploration/p1star_j1_pause_decision_20261017/research_decision_record.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/p1star_j1_pause_decision_20261017/research_decision_record.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：[EV-0188: docs/research_exploration/p1star_j1_pause_decision_20261017/provenance.json](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/p1star_j1_pause_decision_20261017/provenance.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="ext-03"></a>

## EXT-03 Q1 integrable level insertion

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-17（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / qc-algorithm-difference-construction-20261018 |
| 固定資料commit | beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b |
| 目的 | 可解構造を非一様電子系へ拡張 |
| 変更部分 | Richardson–Gaudin型level insertion/Riccati候補 |
| 改善機構 | 可解channelを増やし安い生成を得る |
| 比較対象 | 既知可解族と同じaccess |
| 科学条件 | 固定L4/m2等の紙上条件、分子/DF/L_Dなし |
| 評価指標 | closure conditions / method delta |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 現候補の非一様性がchannel閉包を壊す。一般可解法の否定ではない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | PAUSED_PENDING_NEW_LEVEL_INSERTION_RULE |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 新しい挿入則未提示 |
| 未解決 | 別の具体的insertion rule |
| 再利用 | commutator障害とaccess条件 |
| PRへの応用 | 低い |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0189: docs/research_exploration/q1_pause_decision_20261017/research_decision_record.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/q1_pause_decision_20261017/research_decision_record.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：[EV-0190: docs/research_exploration/q1_pause_decision_20261017/provenance.json](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/q1_pause_decision_20261017/provenance.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="ext-04"></a>

## EXT-04 集団遷移・pair collision・R1-P

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-17（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / qc-algorithm-difference-construction-20261018 |
| 固定資料commit | beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b |
| 目的 | 多電子集団操作を安く生成 |
| 変更部分 | pair collision／CG／multi-jump cycle等 |
| 改善機構 | 集団量子数へ移して相互作用を簡略化する |
| 比較対象 | 既知CG/Paldus・phase・標準gate |
| 科学条件 | 紙上/静的構成、固定fermionic family、分子/DF/L_Dなし |
| 評価指標 | generation/access conditions / method delta |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 既知生成への還元や具体規則不足。量子優位の数値比較はない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | 集団遷移PAUSED、R1-P NOT RETAINED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立した生成・情報削減が未成立 |
| 未解決 | 同情報強対照を越える具体規則 |
| 再利用 | 既知還元・fermionic抽出の障害 |
| PRへの応用 | 低い |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0186: docs/research_exploration/research_strategy_redesign_20261017/research_exploration_retrospective.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/research_strategy_redesign_20261017/research_exploration_retrospective.md), [EV-0191: docs/research_exploration/collective_transition_pause_decision_20261018/research_decision_record.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/collective_transition_pause_decision_20261018/research_decision_record.md), [EV-0192: docs/research_exploration/r1_nonretention_decision_20261017/research_decision_record.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/r1_nonretention_decision_20261017/research_decision_record.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="ext-05"></a>

## EXT-05 G coherent adaptive subspace

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-18（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / qc-algorithm-difference-construction-20261018 |
| 固定資料commit | beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b |
| 目的 | 測定再利用で変分部分空間を安く更新 |
| 変更部分 | coherent cell partitionと共通測定record |
| 改善機構 | 同じ測定情報を候補へ再labelして追加取得を減らす |
| 比較対象 | 同dictionary/seed/accessのVQNHE/QSE等 |
| 科学条件 | 固定有限tree class、紙上構成。実分子/basis/DF/L_D/PF窓なし |
| 評価指標 | energy interval / additional classical cost / update law |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 強い既知wrapperが候補評価・情報law・更新を再現でき、独立差未確認 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | CLOSED / NOT RETAINED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 既知法の特殊化・組合せ |
| 未解決 | 全絶対E0誤差とnative/情報費 |
| 再利用 | 強対照へ同じ測定情報を与える設計 |
| PRへの応用 | 低い |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0193: docs/research_exploration/coherent_subspace_g_closure_20261018/technical_outcomes_and_reduction.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/coherent_subspace_g_closure_20261018/technical_outcomes_and_reduction.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="ext-06"></a>

## EXT-06 固定条件付き測定

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-18（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / qc-b01-limited-feasibility-20261018 |
| 固定資料commit | bbd12642e2540e0d5c6341339820f0b153091d10 |
| 目的 | 条件付き回転でenergy測定分散を削減 |
| 変更部分 | spectator読出し→active Givens→readout |
| 改善機構 | 条件付き可換構造を測定に使う |
| 比較対象 | separate pool／公表conditional法の強い対照 |
| 科学条件 | 4mode/N2、H=cI+F+R、hopping+density。残余Rは保持。PF/DF/L_Dなし |
| 評価指標 | fixed-state variance / measurement/access cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 弱い分割対照との差はあるが既知公表生成へ具体還元。全測定下界ではない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | CLOSED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立方法差なし |
| 未解決 | 残余Rのfresh copies・状態/回路費 |
| 再利用 | conditional分散と残余会計 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0194: docs/research_exploration/conditional_measurement_closure_20261018/technical_outcomes_and_reduction.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/bbd12642e2540e0d5c6341339820f0b153091d10/docs/research_exploration/conditional_measurement_closure_20261018/technical_outcomes_and_reduction.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="ext-07"></a>

## EXT-07 B-01 unknown-gap／energy情報

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-18（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / qc-b01-pause-decision-20261018 |
| 固定資料commit | d26aa812226819c406d6ce645dbb3c1056b0e8ee |
| 目的 | 小gapに依存しにくいenergy情報を得る |
| 変更部分 | real-time Hadamard recordからfilter情報を取得 |
| 改善機構 | filterの情報をcontrolled実時間測定へ接続する |
| 比較対象 | 既知Gaussian convolution・coarse energy取得 |
| 科学条件 | D1 access下の静的検討、具体Hubbard/molecular利益未検証。PF/DF/L_D/window新固定なし |
| 評価指標 | sample/depth targets / soundness/completeness / eta acquisition |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 目標依存と取得費を同時に閉じる独立規則未構成 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | PAUSED_PENDING_NEW_EVIDENCE |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 構成研究着手根拠なし |
| 未解決 | ground population下界・正規化・全confidence台帳 |
| 再利用 | access条件と不足情報の整理 |
| PRへの応用 | 低い |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0195: docs/research_exploration/b01_pause_decision_20261018/technical_evidence_and_limits.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/d26aa812226819c406d6ce645dbb3c1056b0e8ee/docs/research_exploration/b01_pause_decision_20261018/technical_evidence_and_limits.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：[EV-0196: docs/research_exploration/b01_pause_decision_20261018/source_registry.json](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/d26aa812226819c406d6ce645dbb3c1056b0e8ee/docs/research_exploration/b01_pause_decision_20261018/source_registry.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="ext-08"></a>

## EXT-08 T1 logical accreditation・独立原理P1–P8探索

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-17（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae / qc-algorithm-difference-construction-20261018 |
| 固定資料commit | beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b |
| 目的 | PF以外の資源レバーを探す |
| 変更部分 | accreditation接続、障害起点、独立構成原理の比較 |
| 改善機構 | 保証取得と生成操作を変えて費用削減を狙う |
| 比較対象 | 公表accreditation／measurement／既知生成法 |
| 科学条件 | 静的候補・紙上構成、実分子/DF/L_D/time比較なし |
| 評価指標 | method delta / access/total-cost requirements |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 候補名を新規性と数えず、既知還元・構成欠如・不成立を分類 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | T1-D撤回、他は限定探索結果／個別STOP |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立差・資源利益が研究継続を正当化するほど未確立 |
| 未解決 | 新しい具体規則と最小識別task |
| 再利用 | 既知baseline・障害・候補判別条件 |
| PRへの応用 | 不明 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0186: docs/research_exploration/research_strategy_redesign_20261017/research_exploration_retrospective.md](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/beecb0c9fbaa8c8aab3a76b88b5e8f2e0faba63b/docs/research_exploration/research_strategy_redesign_20261017/research_exploration_retrospective.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rep-09"></a>

## REP-09 core吸収・physical pair表現

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-10（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / hamiltonian-construction-feasibility-20261011 |
| 固定資料commit | ae7434452450e80ec007d65417335e18a063fa1e |
| 目的 | tail norm低減を回路節約へ接続 |
| 変更部分 | 占有対角core吸収／physical pair involution |
| 改善機構 | diagonal吸収とpair表現でtail normalizerを下げる |
| 比較対象 | whole-Pauli／direct Pauli／反射法 |
| 科学条件 | synthetic3mode JW正平方DF2とphysical3+aux1 pair、q1 actual cost、q1/2/4 mean、εcomplex.05/.02、固定compiler |
| 評価指標 | lambda / shots / RZ/CX/depth |
| 検証段階 | 既存synthetic/toy・限定prototypeまたは保存値解析。実分子の一般優位性へは移送しない。 |
| 実際の結果 | 強いwhole-Pauliでcore中心の利得が消え、pairのshot節約はbasis/central費用に負けた |
| 調査用成績分類 | 改善なし／現固定構成不支持 |
| 既存状態の説明 | ATTRIBUTION_PAIR_COMPLETE_AWAITING_GPT_REVIEW |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 固定toyで中心改善として不支持 |
| 未解決 | 別入力で費用増を避ける具体生成 |
| 再利用 | core/direct辞書・次数別標本の比較 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0197: docs/research/representation_attribution_pair_results_20261010.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/representation_attribution_pair_results_20261010.md)

静的実装参照：[EV-0198: src/trottertracks/representation_exploration/attribution_pair.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/src/trottertracks/representation_exploration/attribution_pair.py)

保存result/decision metadata：[EV-0199: artifacts/representation_attribution_pair/2026-10-10/run1/result.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/representation_attribution_pair/2026-10-10/run1/result.json), [EV-0200: artifacts/representation_attribution_pair/2026-10-10/attribution_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/representation_attribution_pair/2026-10-10/attribution_summary.json)

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="rb-18"></a>

## RB-18 interstep correlated randomization

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-13（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / m1s-prior-art-survey-20261017 |
| 固定資料commit | 4301c0f87a16be45b620304c3fa935d5aa2d721a |
| 目的 | basis switch等のpath費用を下げる |
| 変更部分 | 相関law・history-adaptive block・whole-path IS |
| 改善機構 | path構造を利用し重み補正を保って高価な遷移を減らす |
| 比較対象 | PRHS／qSHIFT／全経路importance sampling |
| 科学条件 | 静的law/source/文献対応、既存paired finite RTEを参照。新molecule/DF/L_D/delta pilotなし |
| 評価指標 | coherent mean / moment/range / whole-path/native/classical cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | marginal保存はcoherent mean保存でない。単純相関やISは既知対応、独立candidate未成立 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | KNOWN_PROCEDURE_MAPPING; ADDITIONAL_CONCRETE_METHOD_DELTA_NOT_ESTABLISHED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 具体method deltaが揃わずpilotなし |
| 未解決 | finite-bit・signed phase・D interleaving・安いpath生成 |
| 再利用 | channel/first-moment区別と全trial台帳 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | A：固定資料から目的・主要結果・停止理由を概略判断可能。全rawの独立再認証を意味しない。 |

根拠文書：[EV-0201: docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md)

静的実装参照：[EV-0202: src/trotterlib/rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trotterlib/rte.py), [EV-0203: src/trotterlib/rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/rte.py)

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-10"></a>

## QC-10 H1 先導固有値補正

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-16（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | ground状態情報でPF固有値誤差を低減 |
| 変更部分 | 低body表示・多fragment残余の静的監査 |
| 改善機構 | 対象stateで消える誤差構造を利用 |
| 比較対象 | 既存BCH／一体再配分／SOS監査 |
| 科学条件 | 静的検討。新分子・basis・DF rank・L_D・delta窓の数値pilotなし |
| 評価指標 | method delta / information and implementation cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 既存式との重複とfinite-stepへの未接続。高body表示から一般RDM下界を主張しない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | H1 PAUSED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立した具体的生成/情報取得規則未成立 |
| 未解決 | 情報取得費と有限step剰余 |
| 再利用 | 既存scopeとknown-baseline対応 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0172: docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-11"></a>

## QC-11 H2 mixed correlation／filter

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-16（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | 安い相関信号から元Hのenergyを読む |
| 変更部分 | filter ratioとleakage/分母/shot台帳 |
| 改善機構 | 有用な低energy相関へ測定を集中 |
| 比較対象 | 既知filter・直接energy測定 |
| 科学条件 | 静的検討。新分子・basis・DF rank・L_D・delta窓の数値pilotなし |
| 評価指標 | method delta / information and implementation cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | ratio恒等式だけでは独立情報取得法にならず、現時点で有利な生成は未確定 |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | H2 PAUSED; UNDETERMINED_DATA_REQUIRED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立した具体的生成/情報取得規則未成立 |
| 未解決 | 正の分母・leakage・全測定費 |
| 再利用 | 既存scopeとknown-baseline対応 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0172: docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-12"></a>

## QC-12 H3 frame／interaction picture

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-16（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | 可解frameでtime evolutionを安くする |
| 変更部分 | Gaussian共役・DF-square・frame access |
| 改善機構 | 可解部分をframeへ移して残余を小さくする |
| 比較対象 | THRIFT／CFQM／SPRINT／既知共役 |
| 科学条件 | 静的検討。新分子・basis・DF rank・L_D・delta窓の数値pilotなし |
| 評価指標 | method delta / information and implementation cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | sampleされた和の指数・controlを無料にせず、独立した生成差を特定できていない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | H3 PAUSED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立した具体的生成/情報取得規則未成立 |
| 未解決 | 同accessで安いframe/相対演算子生成 |
| 再利用 | 既存scopeとknown-baseline対応 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0172: docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-13"></a>

## QC-13 H4 copies／phase reference

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-16（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | 利用可能stateから絶対energyを安く読む |
| 変更部分 | copies・coherent preparation/inverse・位相参照を分離 |
| 改善機構 | 準備済みstate情報を再利用 |
| 比較対象 | 直接energy/RDM測定／既知reference |
| 科学条件 | 静的検討。新分子・basis・DF rank・L_D・delta窓の数値pilotなし |
| 評価指標 | method delta / information and implementation cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | uncontrolled evolutionの定数位相不変性を確認。全測定方式の不能定理ではない |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | H4 PAUSED |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 独立した具体的生成/情報取得規則未成立 |
| 未解決 | 許可されたcopy/逆/参照accessと全費用 |
| 再利用 | 既存scopeとknown-baseline対応 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0172: docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。

<a id="qc-14"></a>

## QC-14 exact fermion encoding／controlled構成

| 項目 | 記録 |
|---|---|
| 時期 | 資料日 2026-10-17（研究初出と同一とは限らない） |
| repository／含有branch | HIROMU1015/Partially-Randomized-Trotter / qc-a2-certificate-connection-design-20261018 |
| 固定資料commit | 7af83266f96b714bb3fa2f1ef5e4bc89faf37db3 |
| 目的 | sector/encodingを使い要求位相を保つ |
| 変更部分 | JW/DFと既知符号化のSOURCE/API照合 |
| 改善機構 | qubit/Pauli構造を変えて論理blockを安くする |
| 比較対象 | 同情報の既知encoding＋通常compiler |
| 科学条件 | 静的preflight、対象H/basis/rank/L_D/delta/precision未採択 |
| 評価指標 | method delta / control/scalar/preparation cost |
| 検証段階 | 既存計画・理論/静的SOURCE監査。新しい科学的数値検証を実施したとは記録しない。 |
| 実際の結果 | 既知手順を超える具体的block生成則は未提示。adapter不足と算法不足を分ける |
| 調査用成績分類 | 計画・理論／静的監査まで |
| 既存状態の説明 | A1 CLOSED（後続方針）、preflight限定完了 |
| 証拠の強さ | 既存の適用域付き理論／局所数値／静的監査の帰属。外部再現・immutable CI・一般科学結論には格上げしない。 |
| 停止理由 | 具体生成差未成立、④選定保留 |
| 未解決 | どのblockをどのaccessから生成するか |
| 再利用 | encodingとordinary controlの接続条件 |
| PRへの応用 | 条件付き応用 |
| 資料十分性 | B：概略判断は可能。参照先の全条件・全rawの確認範囲に限界がある。未取得資料と未実施検証の詳細はopen_questionsおよびGAPを参照。 |

根拠文書：[EV-0204: docs/research/qc_exact_encoding_controlled_preflight_20261017/technical_preflight_report.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_exact_encoding_controlled_preflight_20261017/technical_preflight_report.md), [EV-0176: docs/research/qc_iv_covariance_method_audit_20261018/handoff.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/88e4665590bb93c16fdd070e14980d9d3cbff699/docs/research/qc_iv_covariance_method_audit_20261018/handoff.md)

静的実装参照：独立した実装pathは今回の索引で照合していない。理論/計画のみのものと、文書からの追加追跡が必要なものを区別し、未実装の証明としない。

保存result/decision metadata：主要文書の固定リンクを入口とする。本欄の空欄はraw不存在の判定ではない。

実行SOURCEのhistorical identityはカタログの`source_snapshot_references`と原文書を参照する。上の実装参照は結果含有commitの静的参照を含み、実行HEAD・実行認可と同一とは表示しない。
