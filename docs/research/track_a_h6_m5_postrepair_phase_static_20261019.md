# H6 m5_best：compile後の残差を生む箇所の限定調査

2026-10-19。**修正候補の最優先箇所を、1qubit再合成で各置換の位相をDAGへ逐次加算する処理へ絞った。根本原因の確定・実H6の修正完了ではない。** 保存callback 626件のうち、位相bitが変化した65件はすべて`Optimize1qGatesDecomposition`だった。新compile・回路再生・QPY/NPZ復元・実vector再解析は0。旧STOPと58軸を保全し、local commit後mandatory STOP。pushは行わない。

調査器SOURCE：`c3b43e9e1355adb14e5e6a6ac130bed608b9e3dd`。対象の科学SOURCE：`02cb8b124302af334d0b02fe53213d3e5a932b78`。対象結果commit：`32bd4e571c250c77f1134abd8395aeba218f8703`。branch：`track-a-h6-m5-postrepair-phase-static-20261019`。

[直前の一回結果](track_a_h6_m5_phase_repair_run01_results_20261019.md)、[以前の人工BasisTranslator調査](track_a_h6_m5_phase_repair_design_20261019.md)、[今回のsummary](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/summary.json)。対象はH6直鎖1 Å/STO-3G/tol-only1e-8 DF19、full m5_best/p19/q2、T=.8/P=0、symmetric-directional/full Hadamard cosine、保存low状態一つ。新候補・sineは対象にしない。

## 1. 保存記録から確認できたこと

[読取専用調査器](../../scripts/inspect_h6_m5_postrepair_phase.py)で、SHA結合したcallback・lowering/compiled receipt・completion・worker STOPの5 metadata roleを照合した。全626行の連続index、finiteなphase、basis名、最終phase/countsとcompiled receiptの一致を確認した。浮動小数値の新しい差やdeterminantは計算せず、保存bit表現と整数件数を比較した。[集計](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/callback_summary.json)、[位相bit変更の65境界](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/phase_transitions.csv)。

| 観測項目 | 保存記録の結果 |
| --- | --- |
| 1qubit再合成の呼出し | 88回 |
| 位相bitが変わったcallback | 65件、全て同じ1qubit再合成pass |
| 最初の位相bit変更 | callback count 9（0始まり） |
| BasisTranslator | count 4。位相とgate countsは入力から不変 |
| 最初の数値誤差を生んだpass | 未確定、null |

正しい再合成でもgate表現に応じてglobal phaseは変わる。したがって、count9を「最初の誤り」とは呼ばない。位相が不変のcallbackにも角度や演算子の誤差がないとは言えない。個別置換のphase増分、途中の回路・作用結果、局所再構成誤差は保存されておらず、既存ログだけで65境界のどれが問題かを確定することはできない。

## 2. 固定compilerの対応箇所

[SOURCE結合](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/source_binding.json)では、既取得の公式Qiskit1.3.0 commit `896d1c8b72888a52a46081a6d1f4de8c56db8aea`の11fileをSHA照合した。installedのPython optimizerとbuiltin pass managerの2fileもbyte一致。Rust binaryをこのsourceからbuildした証明はないため、その点は未確認として保持する。共有Qiskitは変更していない。[固定SOURCE trace](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/source_trace.json)。

実行されたPythonの`Optimize1qGatesDecomposition.run()`はaccelerated Rust関数を呼ぶ。Python側の補助関数だけを直しても、このrun経路には適用されない。

最優先候補はRustの`optimize_1q_gates_decomposition`の置換受理箇所（1238行）から、DAGの`add_global_phase`、数値phase setterへ進む経路。各置換で`sequence.global_phase`をbinary64として加算し、そのたびにbinary64の2πで折り返している。多数の加算の途中で小さい寄与が失われる可能性がある。

別の要因も残る。各1qubit列の行列積・Euler位相抽出、ZSXXの内部phase計算と角度簡約もbinary64である。SOURCEの局所角度簡約の既定値1e-12は、今回の全回路raw差1e-9未満を保証する上界ではない。これらを今回の主原因と断定したり、無断で閾値を変えたりしていない。

以前のBasisTranslatorの定数phase欠落は別の人工証拠として有効だが、それを回避した今回も残差があった。今回のphase観測とalready-in-basisの早期returnに対応するSOURCEから、同じ等価則の修正を繰り返すより、残る1qubit再合成を優先して調べる。

## 3. 人工scalarでの分離確認

[人工証拠](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/artificial_scalar_evidence.json)。量子gate・分子入力を持たない空のDAGで、初期phase1.7、微小量2^-54を65,536回加算し、最後にその正確な合計を引いた。保存binary64値を有理数として合計した正しい結果は元の1.7である。

固定installed DAGの逐次加算は`1.699999999996362`となり、正確な和に対して約−3.638e-12のscalar差が出た。binary64加算/wrapの単純modelとbit一致し、一度だけ和を代入する人工対照は元の1.7に一致した。この例ではmodulo境界を越えておらず、逐次加算自体の情報落ちを分離している。

**これは修正候補の仕組みを示す人工反例で、H6の約1.472e-9を再現・説明・上界付けした結果ではない。** H6の保存phaseを使った補正値の計算、入力をH6残差に合わせた調整は0。

## 4. 具体的な修正案と次の最小検証

[未実装DRAFT](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/repair_plan_DRAFT.json)では、最初の変更を受理済み1qubit置換のphase蓄積に限定する。既存の行列→gate列の生成、gate順序、置換判断はまず維持し、開始phaseと各`sequence.global_phase`を補償和または正確なbinary64和で蓄積して、pass境界で一度だけ折り返し・丸め・DAG代入する。修正は変換時に正しい位相寄与を運ぶためのもので、保存H6誤差を引く事後補正ではない。

実装はproject専用passまたは専用buildに隔離し、共有Qiskitのfileは置換しない。新しいphase集計receiptとgate-metadata hashを記録し、gate列・置換判断が同一であるか検証する。この段階ではproduction修正を実装・接続していない。

次に必要なのは、この一変更の隔離実装と人工の全演算・制御付き比較。その後に実H6検証を行う場合も、同じcosine一軸・保存low/reference、最大1compile/2作用、3raw差を各1e-9未満、先行FAILでSTOPの最小範囲とする。新SOURCE/入力/実装policy/runtime/契約・新grant・fresh gate・別実行承認が必要。sine・sampling・費用会計・全58軸の再取得へ自動拡張しない。

これで改善しなければ、位相蓄積以外のmatrix/Euler計算・局所簡約を別の変更として調べる。閾値・opt level・PF公式を同時に変えて、どの変更が効いたか分からなくしない。

## 5. 検証・保全・担当

[人工12件](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/artificial_validation.json)PASS（0.413秒）。logの欠測・重複・順序・nonfinite・未知basis・改変拒否、変更bitと誤差原因を区別すること、空DAGのscalar反例と対照、既定CLIがscope表示だけであることを確認した。作者自身の確認で、独立AIレビューではない。

[保全](../../artifacts/resource_applicability/track_a_h6_m5_postrepair_phase_static/2026-10-19/preservation.json)は既存2,500file不変、科学SOURCE/input/runtimeの1,964sealも一致。直前の5stage・grant消費・marker・STOP・58受理軸を変更していない。旧1e-9未満基準・AD03主比較採用保留・新費用非互換を維持する。回路の復元/再生/新compile、実vectorのphase差/determinant再解析、費用再会計/fit/ranking、新有効grantは全て0。

**ユーザー**：位相蓄積修正案の隔離実装・人工検証へ進むか判断する。

**Codex（次の指示がある場合）**：採択された変更を隔離実装し、人工検証と新契約を準備する。実H6の再compile・回路再生は、別の明示承認まで行わない。

local commit後**MANDATORY STOP**。公開push・既存実行への自動適用は行わない。
