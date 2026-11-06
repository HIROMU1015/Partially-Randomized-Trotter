# O1 M1：固定4候補のCX費用下界の静的検証

2026-11-06、SERVER 2（GPUA100-gleap02）。**4候補とも UNDETERMINED。** 今回、X/Yそれぞれのfull no-prep期待CX価格について確立できた下界は非負性の `L_X=L_Y=0` だけである。正の下界を証明できなかったので、固定候補の厳密な総費用改善を排除できない。改善可能性を確認した結果でもない。静的調査・本REPORTの公開後に **mandatory STOP** する。

## 固定条件と候補別判定

H4-A、直鎖H4・1 Å・STO-3G、保存DF rank10、共通 `T_eff/V_square`、faithful tail identity、元Hの保存 `state` を厳密基底状態と仮定、状態準備費用除外。q=64、delta=.05、K=2、r=4/8だけを対象とする。C0の0.1 Ha／95%目標、7round、誤差・失敗確率配分を変更しない。

以下のshot数・必要単価上限・baseline **B=9,827,008,128 CX** は、[最新固定結果][RESULT]の引用であり、再計算していない。Bは[C0][C0]の強DF-S2・4 substeps・既存relative fusion／spin policyの全7round保存費用である。

| 辞書 | r | 保存 N_X=N_Y | 必要な期待CX単価（未満） | 確立した L_X,L_Y | 判定 |
|---|---:|---:|---:|---:|---|
| Pauli | 4 | 594,869,749,589,604 | 8.259798161513152e-6 | 0, 0 | UNDETERMINED |
| DF | 4 | 292,028,168,233,855 | 1.682544561956532e-5 | 0, 0 | UNDETERMINED |
| Pauli | 8 | 1,133,345,886 | 4.335396744008651 | 0, 0 | UNDETERMINED |
| DF | 8 | 780,863,721 | 6.292396396272071 | 0, 0 | UNDETERMINED |

単価は `(N_X E[C_X]+N_Y E[C_Y])/(N_X+N_Y)`。排除に必要なのは、**同じ確率法・同じcompiler・full no-prep scope** に対する正当な下界で `N_X L_X+N_Y L_Y >= B` を示すことである。現在の0下界はこれを満たさない。必要単価は表示丸め値で、厳密な分数は[既存result][RAW]に保存済み。今回は分数の再評価も行っていない。

比較compilerはQiskit 1.3.0、`rz/sx/x/cx`、optimization 1、seed 7、backend／coupling mapなしのall-to-all。C0の価格CIなし・`not_certified`という留保も維持する。

## 全trajectoryを定めるSOURCEと取消可能性

調査対象SOURCEはcommit `c7e9587fb7398ebdab156d83abc869d82e5f42c9` で固定した。[準備SOURCE][PREP]、[信号検証][SIGNAL]、C0の保存報告を再利用し、回路builder等は実行していない。

**確率法。** [rte.py][LAW]の `finite_rte_distribution`、`_make_event`、`iter_sample_rte_events` による。tau=lambda_R*delta/r、K=2では次数 n=0,2、重みは
`w_n=|tau|^n/n! * sqrt(1+(tau/(n+1))^2)`、確率は `w_n/(w_0+w_2)`。各eventはrotation用成分1個とproduct用成分n個を、正規化された成分確率p_iから独立に復元抽出する。productを順に作用させ、その後に角度 `atan(tau/(n+1))` のsigned-Pauli rotationを作用させ、phase `(-1)^(n/2)` を保持する。DF成分もGaussian基底で共役したZ/ZZまたはidentityである。

正の確率を持つ成分数を辞書ごとにMと書けば、eventのラベル付きsupportは `M+M^3`、trajectoryはそのq*r個のCartesian積で記述できる。同じ成分の再選択、同じ基底の連続、identity（保存係数が非零の場合）は禁止されていない。異なるラベル列が同じoperatorへ写る可能性があり、この数は異なるunitary数ではない。supportの列挙や、その確率の新規数値評価は行っていない。Pauli/DFは同じ物理分割でも成分確率とlambda_Rが異なる。

**full trajectory。** [resource bridge][BRIDGE]の `_canonical_events` はq*r eventと、そのorder・成分・phase・probabilityを法に照合する。`build_controlled_trajectory` は、各PF stepで共通のcontrolled retained-half、r event、同じretained-half、核scalarのancilla位相を置き、これをq回反復する。event境界を保護するbarrierは挿入されない。共通のretained-halfがあることだけでは、任意tailとの合成後・compiler後にそのCXが残る証明にならない。

| 部分 | 静的に確認できたこと | 下界への制約 |
|---|---|---|
| Hadamard X/Y wrapper | [wrap_controlled_no_prep][WRAP]はancillaのH、Y側のSdg、H、測定を追加する | wrapper自身はCXを要求しない。両軸のcompiled価格が同一であることも仮定しない |
| scalar／identity | [Pauli実装][PAULI]はall-Iをglobal phaseとして保持し、後のcontrolで相対位相にする。[DF実装][DF]はancilla PhaseGateへ集約する | scalar位相をCXとして課金できない。identity eventがあってもretained-halfを含む全trajectoryがzero-CXとは言えない |
| product／rotation | n=2の同一Pauliを2回選んだproduct部分は代数的にP_i^2=I。rotationとevent phaseは残る | 「K=2なら必ずproduct分のCXが追加される」という論拠は使えない。実compilerがこの取消を全て発見するとも主張しない |
| Gaussian変換／DF基底 | [one_body_circuit][ONEBODY]は基底変換を制御せず、対角部分だけ制御する。[DF builder][DF]は同一基底の往復を再利用・取消可能 | 各Gaussian変換を別々にcontrolした費用、またはevent数×基底変換費用を下界にできない |
| controlled diagonal | [既存primitive][OPS]は非零角のcontrolled RZ/RZZ、DF productはCZを生成する | 分解前のgate数は最終CX下界ではない。角度の特殊値、逆対、隣接部分との取消を別途排除する必要がある |

matched bridgeのDF経路は `build_event` をeventごとに呼ぶので、builder内の同一基底reuseと、eventをまたぐcompiler取消は区別する。任意の基底fusion・大域最適化が既に行われると仮定していない。

## 論理下界と固定compiler下界の区別

**条件付きの論理的事実。** 測定直前までのunitary部分を正確に実装し、利用するentangling gateがCXだけなら、CXが0個の回路は各qubitの一量子bit unitaryの積である。ordinary-controlled `diag(I,U)` のancilla対systemのoperator Schmidt rankは、Uがscalar identityでないとき2になる。従って、そのtrajectoryに少なくとも1個のCXが必要である。wrapperの局所H/Sdgはこのrankを変えない。これはfull operatorの議論であり、特定の保存stateに対する信号だけを再現する別回路の下界ではない。

しかし今回の静的確認では、固定H4の**全supportの合成Uが非scalarであること**、あるいは十分な確率質量についてそれを保証することは証明できていない。角度が各eventで非零であること、元Hが非自明なこと、平均信号が非零であることだけでは、任意の積全体の非scalar性を保証しない。保存信号はその代替証明にしていない。zero-CXの実H4 trajectoryを発見・証明したわけでもない。

もし全trajectoryについて1 CX以上を確立できれば、両軸の期待価格も1以上となり、既存の必要単価が1未満のr4両候補の排除に足りる。r8両候補には1 CXでは足りず、より強い下界が必要である。これは**条件付きの含意**で、候補別の確立済み下界ではない。

**固定compilerで確認したこと。** [価格測定][COMPILE]の `transpile_and_measure_cost` はwhole wrapperを一括transpileし、最終 `count_ops` のcx数を測る。primitive価格の和ではない。Qiskit 1.3.0の[opt1 SOURCE][QOPT]には初期段階と最適化段階の `InverseCancellation`（CX/CZ等を含む）、`Optimize1qGatesDecomposition`、翻訳・depth/size固定点までの反復がある。[取消pass][QCANCEL]は逆gate対を取り消し、[1q最適化][Q1Q]はidentityとなる1q列を除去できる。このため、CX-CX逆対や基底往復に由来する取消を無視した加算は正当化できない。

ここでopt2/3用の `CommutativeCancellation` や大域unitary synthesis等がopt1で一般に実行されるとは仮定していない。反対に「opt1は軽いので各blockのCXが必ず残る」とも言えない。all-to-allにはrouting由来の必須SWAP費用もない。正確なoperatorの非局所性からcompiled下界へ移すには、固定角・分解・取消と数値精度の下で必要な非局所性が保持されることまで示す必要がある。今回その全support証明はない。

既存専用venvの `VERSION.txt` は1.3.0。上記3つのQiskit SOURCEは公式tag `1.3.0` と全文一致を読取り照合した。Qiskitのimport、回路生成、transpileは行っていない。これは該当SOURCEの照合であり、compiler出力の全support検証ではない。

確率部分集合Sだけに正当な下界ell_b(omega)を与え、
`L_b = sum_{omega in S} p(omega) ell_b(omega)`
とする経路も数学的には可能である。他の回路価格の非負性を使う。ただし、Sの確率質量と固定compiler後の下界が必要で、代表trajectoryのCX数や旧r2平均価格では代用できない。今回この和は評価していない。単純なevent数・未最適化CX数・既存代表回路のCX数は採用しない。

## 終了判断と、必要な場合の最小pilot案

**数学的な候補終了：4点とも不可（UNDETERMINED）。** C0の負結果は維持する。今回得た情報は、固定候補の改善を保証せず、厳密な不可能性も示さない。研究上これ以上の費用をかけず終了する判断と、下界による排除は区別する。大規模価格MC・全RPE計算へ進む根拠は今回も得られていない。

続ける場合の**未実施・未採択の小規模診断pilot案**は次の通り。

- 対象はまず同じq64/delta=.05/K2のPauli r8、DF r8の2点だけ。各点8 trajectoryを事前固定。PCG64 seedはPauli `2026110612`、DF `2026110613`。r4価格評価やsample追加はこの案に含めない。これはr4の不可能認定ではなく、追加測定の優先順位の提案である。
- 同じ保存入力・retained block・faithful identityから、既存 `prepare_matched_resources`／`configure_law` と価格providerを再利用する。保存信号configの照合ラベルをcanonical pricing lawの代わりに使わない。新fit・再DF・新signal・新shot算術は不要。
- 各trajectoryにfresh q*r eventを既存lawから抽出し、同じ列のX/Yを両方full no-prepでcompileする。重複drawを削除せず、旧proxyで補わない。Qiskit1.3.0、rz/sx/x/cx、opt1、transpiler seed7、all-to-allを固定する。既存並列providerも同じbuilder／wrapper／価格測定を使う。
- rawの両軸CX、標本平均、sample SEを記録し、既存の必要単価との診断的比較までに限定する。8標本の平均・SEは価格CIでも全law下界でもない。平均が閾値以上でも数学的不可能性に読み替えず、閾値未満でも全RPE改善とは言わない。
- SOURCE／入力／law／full scopeの不整合やcompile異常なら未確認価格を採用せずSTOP。正常なら固定8標本の完了でmandatory STOP。外れ値による選別・追加draw・seed変更・自動retry・全round拡張はしない。統計的な排除を求めるなら、有効な価格範囲と事前固定した片側CI等を別途設計する必要がある。

**必要な承認範囲：** ユーザーがこの2点・標本数・seed・SOURCE／入力・compiler・停止規則を採用した場合に、その範囲の実H4資源準備、event sampling、回路build/compile、CX価格集計を明示承認すること。現在の静的検証指示にはこの実行許可を含めない。許可の証明書やgrant発行は追加しない。

## 実施範囲・保全

primary agentによる静的SOURCE確認、固定結果の引用、上記の数学的検討、REPORT作成だけを実施した。実装変更はなく、unit testや過去の科学テストの再実行も行っていない。新H4 simulation、保存数値再解析、回路build/compile、価格MC、shot計算、全RPE再最適化は未実施。M2は保留のまま追加作業なし。既存SOURCE／raw／result／grant／marker／STOPに変更はない。

専用branchは `codex/track-a-o1-m1-cx-lower-bound-20261106`、親は最新結果commit。差分は本REPORTのみとし、公開後にremote ref・commit parent・tree・REPORT全文を照合する。**mandatory STOP。追加科学計算・価格測定・候補拡張・詳細科学レビューへ自動進行しない。**

次の担当は**ユーザー**：有限候補を研究上終了するか、上記の限定pilotを採用・実行承認するかを判断する。**Codex**は現在の公開・照合完了後に停止する。

[RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/docs/research/track_a_o1_m1_cost_feasibility_science_20261106/REPORT.md
[RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/artifacts/track_a_o1_m1_cost_feasibility_science_20261106/run01/result.json
[PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c64007e7a3310bddcc5c65b04463f5f98986f1e8/docs/research/track_a_o1_m1_cost_feasibility_preparation_20261106/REPORT.md
[SIGNAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185104f64a4594355a3813d04d9e50bd4abdb6b5/docs/research/track_a_o1_m1_attenuation_science_20261106/REPORT.md
[C0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md
[LAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/src/trotterlib/rte.py#L1459
[BRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/src/trotterlib/o1_m1_resource_bridge.py#L103
[WRAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/src/trotterlib/rpe_hadamard_interrogation.py#L120
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/src/trotterlib/pauli_signal_adapter.py#L161
[DF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/src/trotterlib/df_rte_qiskit.py#L214
[ONEBODY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/src/trotterlib/o1_m1_circuit_signal_validation.py#L40
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/src/trotterlib/df_trotter/ops.py#L295
[COMPILE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/src/trotterlib/rte_compiled_cost.py#L843
[QOPT]: https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L565
[QCANCEL]: https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/inverse_cancellation.py#L27
[Q1Q]: https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py#L60
