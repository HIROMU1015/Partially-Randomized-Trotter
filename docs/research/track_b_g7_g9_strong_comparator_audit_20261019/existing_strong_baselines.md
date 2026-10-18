# 強い既存対照の成果・限界（保存結果の引用）

[入口](README.md) / [比較条件](comparison_conditions.md)。以下は[G9 v2報告][G9R]の既存記述・表示表と、[保存affine比較][G9AFF]の判定fieldを引用する。差・比・新rankingは計算していない。G9保存監査25件PASSは**過去の記録**であり、今回再実行した検査ではない。[G9AUD][G9AUD]

## 1. 共通scope

known development p=(1/5,3/10,1/2)、x=5/7、m=5のfull finite P5 first operator moment。3-system-qubit synthetic provider、I1 Pauli情報、exact Clifford+T model（CQ delta=0）、strict RZ ε=10^-6、H160/K256・η=ρ=10^-12、共通direct/compiler、G9の22-axis/resource failure policyでの会計。分子/geometry/basis/DF rank/split/δ窓は非該当。QPE総資源・hardware shot実測・独立再現ではない。

期待費用は**二axis合計**。T欄はnative-event切片、primaryは`T_intercept + K*T_prep/readout`。T_prep/readoutは共通非負parameterのまま。CXもnative-event座標であり、他の未知費用を0とした総費用ではない。

## 2. 元報告のprimary direct表示表

| 方式 | N/axis | native T intercept | native CX | K（prep/readout） | workspace |
|---|---:|---:|---:|---:|---:|
| ordinary | 1,246,046 | 348,759,216.423 | 20,586,846.721 | 2,492,092.000 | 1 |
| partial-return + tail | 949,142 | 258,865,913.425 | 15,975,424.661 | 1,898,284.000 | 1 |
| closed P3 + tail | 933,915 | 267,617,674.368 | 16,067,266.079 | 1,867,830.000 | 1 |
| general local full | 1,076,085 | 257,506,695.137 | 15,748,232.374 | 1,846,058.549 | 1 |
| closed P5 full | 917,129 | 255,860,636.626 | 15,647,565.043 | 1,834,258.000 | 1 |
| matched CTS | 1,596,220 | 364,311,583.986 | 9,766,516.587 | 3,192,440.000 | 1 |

これは**丸め済みの元表示表を同じ順序で転記**したもの。primaryの符号根拠は元のexact affine保存値であり、この表示表から新しい閾値・削減率・勝者表を作っていない。1Q、accepted tail cap、hard attempts、worst-event T等は[元CSV][G9CSV]に保存済み。depthはCSVにも`cost` SOURCEにもない。[G9N][G9N]

## 3. 既存の判定をどこまで言えるか

| 既存結論 | 固定根拠 | 制限 |
|---|---|---|
| General Fullはordinary/partial/P3よりT切片が小さい。共通非負prep/readout Tを含む既存pair判定でもFull側が非劣 | G9報告の明記、G9AFF `/pairs/2`、`/pairs/6`、`/pairs/9` | 登録law・G9 policy・このprovider/directに限定 |
| General Full対matched CTSのT-primaryではFull側の既存非劣判定 | G9AFF `/pairs/13`の`left_no_higher_for_common_nonnegative_prep_T=true` | CTSはCXで有利。総vectorのdominanceではない |
| Closed P5は登録direct6方式のT切片とKが最小。共通非負prep/readout Tを加えてもT-primaryの順序は維持 | G9報告に明記。対FullはG9AFF `/pairs/12`でright側非劣、対CTSは`/pairs/14`でleft側非劣 | 同familyのm=5 fast path。独立新アルゴリズムや一般Full不可能性ではない |
| matched CTSはCXで小さい。一方Closed P5よりT-primaryは大きい | G9報告の既存結論、元direct表・CSV | T/CXトレードオフ。採択していない価格換算で単一winnerを作らない |
| direct workspace1、helper診断workspace2 | G9報告・契約・native SOURCE | system3を除くモデル内ancilla。state/backendのpeakを含まない |

T-primaryの既存affine判定を引用しており、UNKNOWNなstate/prep/readout価格を0と選んだ判定ではない。1Qは元CSVのnative-event/outer部分を参照可能だが、今回新たな1Q順位を分類しない。depth・hardware誤差・実state/native prep/backend・all-N古典費用の優位は未確立。

## 4. Closed P5とGeneral Fullの役割

G9の独立formal auditは同じ理想full-return ensembleの係数一致を保存している。General Fullのraw/coin zeroに対しClosed P5はgroup canonical lawで受理1。normalizer・finite-bit proposal・保守的budgetが異なるため、NやKは同一とはしない。元報告はこの差を小m fast pathと一般器の役割の違いとして扱い、独立競合法の発見や一般法の失敗と採点していない。[G9R][G9R] [G9MATH][G9MATH] [P5][P5]

Closed P5のO(L²)はgroup構築/rootの**算術operation**の範囲。bit費用やlinear selectorは無料ではなく、全sampleをO(L)と保証しない。実装sample APIがあることと、全N production性能が実測済みであることは別。matched CTSもI1の有限dictionary/event lawは実装されているが、G9 runnerはreference会計であり全Nの専用production sampler/監査性能を測っていない。

## 5. G7およびB3/C1への適用

[G7結果][G7R]の「登録P5でFullはordinary/partial/P3より条件付き期待Tが小さい」は保持できる。ただしG7は4方式・helper/abstract CQのaffine会計で、Closed P5/CTSは未登録。これを強い対照全体への資源改善に拡大できない。

G9に進めば、**同じ入力について強い対照を含むnative会計が既にある**。C1が同じprovider/direct/compiler/ε/key/情報accessとG9 policyに合う場合、旧native-event rowsやcacheを再利用候補にでき、同じ取得の再実行は不要。C1が別provider/helper/最適化/物理誤差・confidenceを使う場合は、identity差を埋める前に費用を混ぜない。G9helper5 rowsも保存済みだが、primary direct6 rowsの代用ではなく、CTSだけdirectを禁じて対照を弱くしない。

B3が新しい4方式CPU結果を得ても、Closed P5/CTSの同じ入口/監査/保存/cold取得・全N費用は別途未確定である。G9のguard CPU約3.094秒はsmall reference/native会計を含み、production μやFull対P5/CTSのCPU順位ではない。C2の未知も残る。

## 6. 証拠の強さ・失敗履歴

G9 v2は11 rows/22 axes完了、25 saved checks PASS、19 key（18再利用/1取得）、1,866 native-event bindingsという既存local証拠。float matrix tolerance10^-10はphase/意味論診断であり、confidenceは固定analytic/coefficient/composition予算による。全event参照はproductionへ注入していない。immutable CI・外部再現・全compiler最適性ではない。[G9MAN][G9MAN] [G9AUD][G9AUD]

旧G9 v1はTypeErrorで`G9_TECHNICAL_INCONCLUSIVE`、比較0 row、marker消費済み。v2は別SOURCE/別承認/別markerの一回実行で、v1の結果を消した再開ではない。今回いずれも変更・再実行していない。[G9V1][G9V1] [G9BOUND][G9BOUND]

CTSのT=0 real correction10 eventは保存監査で保持されている。これを除く強い対照の再定義、任意IS/CTS辞書/精度配分に対する最適性、一般mへの外挿を行わない。G9の既存数式・先行研究確認を繰り返していない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[G7R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G9AFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_affine_T_comparison_v2.json
[G9AUD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/saved_output_audit_v2.json
[G9BOUND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_v2_api_boundary_source_review_20261010.md
[G9CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/resource_rows_display_v2.csv
[G9MAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/evidence_manifest_v2.json
[G9MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_p5_native_preparation/2026-10-10/mathematical_preparation_v1.json
[G9N]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_native.py
[G9R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[G9V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_results_and_gpt_handoff_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
