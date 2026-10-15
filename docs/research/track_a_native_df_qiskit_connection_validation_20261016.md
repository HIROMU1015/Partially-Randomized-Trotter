# Native DF理論ゲートモデル：Qiskit接続検証 — 2026-10-16

**人工入力に限り、Qiskit 1.3.0で接続検証を完了した。**
既存回帰27＋新規接続15＝42 tests PASS、failure/error/skip 0。
元builderを変更せず、理論カウンタの明示zero RZ取扱いのバグ1件を修正した。
モデルA/Bの科学的採用、実H6/H8への適用、compiled母平均の認定ではない。

- [理論式・固定分解・モデル定義](track_a_native_df_theoretical_gate_model_20261016.md)
- [カウンタ](../../src/trotterlib/native_df_gate_model.py)
- [接続tests](../../tests/test_native_df_qiskit_connection.py)／[参照loader・独立Qiskit分解](../../scripts/native_df_qiskit_validation_support.py)
- [実行runner](../../scripts/check_native_df_qiskit_connection.py)
- [最終receipt・環境・SOURCE・各case](../../artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/connection_validation_v1.json)
- [修正前の不一致](../../artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/baseline_discrepancy_v1.json)
- [技術監査・保全](../../artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/technical_audit_v1.json)

## 1. 固定SOURCEと読込み方法

Repository `HIROMU1015/Partially-Randomized-Trotter`。
起点 `track-a-native-df-theoretical-gate-model-20261016`、固定commit
`1ea42fbf1bd1c8341ffd410b8067467b4ba4e170`。
独立branch/worktree `track-a-native-df-theoretical-qiskit-validation-20261016`。

| コード | 固定版と今回の扱い |
| --- | --- |
| `native_df_gate_model.py`、既存27tests | 1ea42fbf…を起点に、zero RZバグだけ修正。旧commitと旧receiptは不変。 |
| `df_trotter/ops.py`, `df_partial_s2.py`, `df_rte_circuit.py`, `df_rte_qiskit.py`, `rte.py`等 | 起点版を実際にimportして人工入力で呼ぶ。変更0。 |
| [native v5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trottertracks/resource_applicability/ax2b_native_df_v5.py) | 別SOURCE `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`からGit bytesを一時領域へ取得して実際にimport。srcへの混入0。 |
| nativeのrelative imports | 同じ87b6b9f…の`ax2a_control_plan.py`, `ax2b_stream_fingerprint_v5.py`, `ax2b_diagnostics_v4.py`。 |
| nativeから参照する共通library | importした全共通.pyについて、起点版と87b6b9f…版とlocal bytesが一致することをassertした。新counterは別扱い。 |

各path/bytes/SHA-256は最終receiptの`loaded_source_closure`に記録。
別SOURCEを同一と推定したのではなく、共通部分は実際にbytesを比較した。
packageの大量exportを避けるnamespace shellを使い、`trotterlib/__init__.py`等の無関係な初期化は実行しない。
**数学的builder本体のstubや再実装への置換ではない**。
分子I/O・sampling・DF分解・transpileの入口には、人工tests内だけの拒否guardを置いた。
guardはOS sandboxではなく、通常package import全体の互換性を保証しない。

最新版共通指示は[server原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b78e5f39917d92d6c60694d3e0b0d06394d7eb5/docs/research/provenance/research_project_instructions_server_original_20261012.md)を適用。
主担当による技術検証であり、別担当の独立科学レビューではない。

## 2. ローカル環境

既存`.venv311`をそのまま使用。install/update/環境設定ファイル変更0。

| 項目 | version |
| --- | --- |
| Python | 3.11.0rc1 |
| Qiskit | **1.3.0** |
| Qiskit Nature | 0.7.2 |
| NumPy | 1.26.4 |
| SciPy | 1.14.1 |
| OpenFermion | 1.6.1 |

指定された実験Qiskitとversionは一致。ただしPython・全依存・platform・compiler profile全体の同一性は認定しない。
実際に使ったRZ/RZZ/CRZ/XX+YY/H/P/CX/BogoliubovTransformのdefinition source SHAをreceiptに保存。
新しいvenvは作らず、bytecode書込みを`-B`で避け、Matplotlib cacheは一時領域へ隔離した。
system Python 3.10.12にはQiskit等がなく、[fallback検証](../../artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/no_qiskit_fallback_v1.json)は
27 stdlib testsと`PARTIAL_STDLIB_ONLY`を報告する。これは接続検証PASSへ数え足さない。

## 3. 人工入力・検証方法

分子geometry、化学basis、分子DF rank、研究用split L_D、実分子delta windowは**該当なし**。
2 system qubit＋ancilla、解析的な2×2 unitaryをGaussian入力とする。
one-body eigenvalues=(.7,-.2)、人工squared block eta=(.4,-.9), lambda=-.6。
分子integrals、Hamiltonian、state、保存H6/H8の配列は使わない。

primitiveはtime=-.3/0/.3、PFはT=.8、q=1/2、S2/global Yoshida S4、
ordinary/symmetric-directional、scalar=.17、cosine/sineを検査。
RTEはtau=.4、次数0/2/4、正Z・負ZZ・負I、same/different basis、複数eventを人工列挙する。
乱数・trajectory samplingは使わない。

照合を三つに分けた：

1. actual builderとIRのgate名・順序・ordered qubits・数値parameterを照合。
   角度は絶対/相対許容差2e-15、命令名・順序・qubitは完全一致を要求する。
2. Qiskit側で**独立に実装した同じ固定分解**を実行し、`count_ops()`の整数RZ/CX/SX/X数とカウンタを照合。
   `model.lower`の出力をそのまま数えて一致とする検査ではない。
3. 選んだ人工回路で`Operator`行列を直接比較。unitary判定の許容差3e-12。
   `Operator.equiv`等によるglobal phaseの無視はしない。

transpileは一度も呼ばない。
**元の高水準命令数、明示分解費用、transpile後費用は別の量**として保つ。
全量の最適化・routing・実compiled費用との一致は今回のPASS条件ではない。

## 4. A/Bと回路構成の対応

| 構成 | 確認できた範囲 | 残る区別 |
| --- | --- | --- |
| Gaussian forward/inverse | 実`U_to_qiskit_ops_jw_givens`出力、append順、inverse、固定分解、unitary一致 | 任意のopaque/custom Gaussianや全nは未検証 |
| one-body/DF squared diagonal | 実producer→appendとカウンタが一致。正/負/0 time、普通制御・directionalを含む | 実分子metadataは未使用 |
| CRZ/CRZZ/CZ等 | Qiskitのgate定義と宣言分解のunitary・整数費用が一致 | 内部transpile最適化費用とは別 |
| native PF S2/S4 | Aのraw列・固定分解費用・位相・wrapperが一致 | native PF builderはBのGaussian相殺をemitしない |
| PFのB | forward/reverse、step、S4 piece境界でunitary保持と費用減を確認 | 数学的に許される別構成であって元PF出力命令数の再現ではない |
| native scalar | trajectory末尾のP(-scalar*T)を1回だけ計上 | stepごとのPを追加しない |
| wrapper | H/U/(Sdg)/H一致、measurement有無で命令差1 | state準備・measurementをRZへ換算しない |
| RTE A | builderの`cancel_adjacent_equal_bases=false`で一致 | reuse=trueのbuilderとAのraw数が異なるのは定義差 |
| RTE B | registered full basisでreuse=trueのbuilderと一致 | history/order依存のsupport-restricted planは対象外 |
| identity barrier | phase0でもactive basisを閉じるsource挙動を保持 | identityを自由に移動させる最適化はしない |
| ID/操作/配置 | registryが同IDで異なる操作列・qubit配置を拒否。カウンタも偽った同ID/hashを拒否 | 同一supportだけではreuseしない |
| occurrence境界 | nativeの別tail composeと閉境界モデルが一致 | 同じeventを一つのoccurrenceへflattenすると別のreuse scopeになる |

例えば同basisのk0 eventを二つのoccurrenceとして閉じるとraw14、
一つにflattenするとraw8となった。unitaryは一致するが、**BOUNDARY_SCOPE_DIFFERENCE**として記録する。
H6/H8へ都合のよい方を事後選択する認可ではない。

Gaussianの物理的orderingは独立なFock minor oracleでも確認した。
occupation bit p=mode p、2-mode swapの二重占有stateが符号-1になることを含む。
単独Gaussian Gには実Qiskit NatureのRZ構成によるvacuum phaseがあるため、
物理oracleとの比較だけは `G = G[0,0] * FockLift(U)` と明示して照合した。
この位相を勝手に消したGをカウンタへ渡したわけではない。
実回路とIRの比較では位相をそのまま保持し、Ginv/D/Gでの相殺、ancilla relative phaseを別に検証した。
将来Gaussian自体を制御化するなら、この位相に関する追加検証が必要。

## 5. 発見したバグと限定修正

**COUNTER_IMPLEMENTATION_BUG：明示zero RZの過剰除去。**
元`append_diagonal_primitives`およびnative directional appenderは、渡されたRZリストをそのままemitする。
旧カウンタだけが`angle == 0`を追加で省いていた。

手入力 `(phase=0, rz=[(0,0)], rzz=[])` について、旧カウンタのraw/RZ/CXは全modeで0だった。
actualの `(raw,RZ,CX)` はuncontrolled=(1,1,0)、ordinary=(1,2,2)、directional=(3,1,2)。
修正前のactual word・費用をbaseline artifactに保存した。

修正は新カウンタのRZ用zero-skipを除去するだけ。RZZ exact-zero規則・元producer・native builderは不変。
通常のone-body/DF squared producerはzero RZを既に省くため、その入力域の構成は変わらない。
既存人工testの期待値を訂正し、actual appenderとの回帰を追加した。
数式のaは「渡されたRZリスト項数」と明記した。旧commitと旧27件receiptは改変しない。
モデルA/Bの科学的意味の変更や、経験的補正係数の導入はない。

## 6. finite-RTEと期待費用

実`finite_rte_distribution`、`iter_rte_events`、Qiskit event builderを使用した。
`iter_rte_events`はこの人工有限集合の全列挙であり、samplerは呼ばない。
次数kごとのproduct k個→rotation1個、係数符号、component確率の正規化、identity位相を確認した。
理論helperと実distributionのbinary64確率は3e-16以内で一致する人工例を得た。
これをすべてのtau/Kのbit一致や超越関数の丸め誤差certificateとは扱わない。

| 人工法則 | 全列挙数／model | actual Qiskit費用との比較 |
| --- | ---: | --- |
| 2 components、K=4、r=1、異なるbasis、identityなし | 42 | A線形期待値・B DPと有理数として一致 |
| 2 components、K=2、r=2、同basis、identityなし | 100 | event境界のreuseを含めてA/B一致 |
| 3 components、K=0、r=2、identityあり | 9 | actual期待値はA/Bのraw/RZ幅に入り、CX/SX/Xは一致 |

比較の確率は、実distributionの保存binary64 order probabilitiesをFraction化して再正規化し、
componentの明示した有理数確率と組み合わせた**同じ有限法則**で統一した。
実eventのbinary64 probability計算そのものと無限精度の確率を同一視しない。
各caseの有理数期待費用はreceiptに保存。

identity phaseでは正負I回転のfloat和0でPをemitしないこと、4個の負productでphase=4piの場合に
数学的identityでもliteral Pを1個emitすることを確認した。
**一般identity位相の期待費用は引き続き0〜1 P/occurrenceのbound**。
人工列挙のPASSだけで一般問題をexactへ変更しない。
全次数を数えられることも、未観測rare-orderのcompiled母平均の保証とは異なる。

## 7. 検証結果と性能範囲

最終receiptは42 tests PASS、134 observation記録。
unitary直接比較の最大差は約2.00e-15、固定分解の最大差は約1.72e-14、
Fock minor oracleの最大差は約3.34e-16だった（正確な値はreceipt）。
これは有限の人工例の浮動小数検証であり、一般nの形式証明ではない。

実行は約4秒、peak RSS約369.5 MiB。既存小型suiteの時間とsourceを見て、
小回路は直列、BLAS/OpenMP/Numba/Rayonの要求thread数1、metadataとgate-cost cache再利用を選択した。
processを増やして大きな科学importを繰り返す利益は検証していない。speedup倍率は主張しない。
phase/total wall上限なし。実分子・GPU・重いcompileに関する性能保証もない。

再現例（同等の既存依存環境で）:

```bash
python -B scripts/check_native_df_qiskit_connection.py
```

固定参照commitがlocal Git objectに必要。欠ける場合は参照取得を報告してから別途準備する。
runnerはinstallやserver接続を行わない。receipt指定時は既存pathを上書きせず拒否する。
[attempt01](../../artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/connection_validation_attempt01.json)も保全した。
attempt01はwhole-partial unitary照合を追加する前の42件PASSであり、最終source hashは最終receiptを正とする。

## 8. 後段へ残す事項

- 実H6/H8のsource/input/basis metadata・sampling law・tau/K・reuse policyへのbinding。
  今回は保存科学データを読み込んでいない。
- opaque/custom Gaussian、controlled Gaussian、support-restricted/history依存basis-plan、
  大きなn/q/r、他Qiskit version、通常package import全体の確認。
- 一般identity phase期待値の厳密計算、閉occurrenceをまたぐ追加最適化、depth/routing。
- 同じsignal target・bias・経験的数値allowance・normalization・N_axisを固定した
  theory/compiled比較。今回は保存費用の再解析も実施していない。
- Aと実reuse設定、BのPF仮想相殺と閉境界scopeをどの比較契約に採用するか。
  実分子へ適用する前にGPTが科学的主張の範囲を判断する事項であり、ここでは変更しない。

今回のコードの単純バグは解消し、技術接続検証を完了。
実H6/H8計算・保存値再解析・sampling・transpile・server接続・新grant・ledger変更・原稿改訂は0。
既存SOURCE/results/STOP・dirty/untrackedを保全し、公開後**MANDATORY_STOP**。
**DRAFT_NOT_AUTHORIZATION / H6_NOT_AUTHORIZED / H8_NOT_AUTHORIZED / next_stage_authorized=false**。
この表示は今回から派生する新実行についてのもので、別環境の既存jobや認可を変更しない。


## 2026-10-16 後続metadata監査

[H6/H8保存metadata接続監査](track_a_native_df_h6_h8_metadata_binding_20261016.md)。代表event記述は接続可能、PF primitive/registry・全lawは未取得。旧42人工結果と本書の限界を変更せず、実費用計算は行っていない。
