# H4 C候補証拠run01：reference実装欠陥により不受理・停止

2026-10-23 JST。利用者の「次の作業に進んで大丈夫」、続く「続きを行って」を、前段階で提示した固定契約の実H4数値証拠候補生成への承認として受領した。専用branch `codex/non-df-h4-c-evidence-run01-20261023`、基点 `c4497d8b49a3d796513df439dd025c72fb964a21`、実行SOURCE `49c18ccc35f14041df45bfeff455ba5e67543182`。

**新grantで一回起動したが、referenceの状態縮約に実装欠陥を確認し、科学的な受理0軸で停止した。** Pauliの20軸はrawを保存したが、bias・精度判定は不受理。DFは不具合によるintegrity STOPで、最終20軸は未取得。新SCF/FCI/DF生成/状態最適化/sampling/compile/shot/G再会計・科学retry・push・正式certificate受理は0。

## 固定条件・grant・実行

対象はlinear H4/隣接1.00 Å/STO-3G、8 system qubit、共通保存状態Nα=Nβ=2。Pauli-JW184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body。S2、T=.8、q8、r4/K4、ε_sig=.05/.01/.005/.001/.0005・cosine/sineの40軸を維持した。Bの必要u上限の1/2を設計比較目標とし、旧allowance/guard・候補・比較条件を変更していない。

旧Cの無効契約・SOURCE seal・技術検証結果は保全し、別versionの[active契約](../artifacts/non_df_h4_c_evidence/2026-10-23/launch01/active_contract.json)と[承認記録](../artifacts/non_df_h4_c_evidence/2026-10-23/launch01/approval.json)を作成した。承認記録は利用者の原文と、直前の提案に基づく担当者のscope解釈を区別している。

新grantは `8af94df114e6485c7a5f10d12e77352347aa21b2e9525647a8ad8ebc988f4554`。`run01`は未使用namespaceとして発行し、CONSUMEDを排他的に保存して一回だけdispatchした。消費・予約は返却しない。両表現は独立process、旧runtimeのpreparation復元bridgeも独立witnessで監視した。

## 確認した実装欠陥

SOURCEの `h4_ball_evidence.py:322` は次を呼んでいた。

```python
reference = signal(state, self.expm(full * acb(0, -T), ('reference', bits)))
```

一方、`signal`の定義（同216行）は `(dagger(state) * out)[0,0]` である。この呼出しのoutは行列なので、計算しているのは `(ψ† exp(-iTH))[0,0]` となる。必要なreferenceは `ψ† exp(-iTH) ψ` であり、**右側のstate作用が欠けている**。helperに列vectorだけを許すshape検査もなかった。

保存されたPauli結果ではreferenceのRe/Im上下端がすべてexactに0となった。rawの`TARGET_NOT_MET`20軸は、この誤ったreferenceを使ったbiasに基づく。したがって、物理モデル・方式が不適格であることや、Bの必要予算が達成不能であることを示さない。

[新しい人工1 qubit回帰](../artifacts/non_df_h4_c_evidence/2026-10-23/diagnosis01/test_reference_contraction.py)で、状態|1⟩・恒等行列に対するreferenceが本来1であるのに、封印コードの呼出しでは0となることを再現した。**1 test/1 FAIL/ERROR0/skip0**。H4入力・指数計算・乱数を使わない反例である。

前回のproduction end-to-end人工fixtureは最初のbasis vectorを使っており、この入力では誤ったfirst-column縮約と期待値が一致する。そのため既存110 PASS・backend13 PASSでは欠陥を見逃していた。過去のログを変更せず、今回の新しいFAILと保証範囲の不足を追記した。[source診断](../artifacts/non_df_h4_c_evidence/2026-10-23/diagnosis01/source_diagnosis.json)。

## 4方式×5精度×2軸の保存状態

| 方式 | 精度条件 | 保存raw軸 | 今回の扱い |
|---|---|---:|---|
| pauli_jw_PF | 固定5条件×両軸 | 10 | raw TARGET_NOT_METだが、reference欠陥により科学判定不受理 |
| pauli_jw_PR | 固定5条件×両軸 | 10 | 同上。normalization欄もrawとして保全し、別の科学受理は行わない |
| df_native_PF | 固定5条件×両軸 | 0 | 同じreference呼出しを使うためintegrity STOP、最終tier未取得 |
| df_native_PR | 固定5条件×両軸 | 0 | 同上。途中のpreparation・call予約・監視証拠を保持 |

[全40軸CSV](../artifacts/non_df_h4_c_evidence/2026-10-23/axis_outcomes.csv) / [JSON](../artifacts/non_df_h4_c_evidence/2026-10-23/axis_outcomes.json)は、既存Bの固定registryと保存済みstatusの対応表である。新しいbias・shot/Gを計算していない。旧20行N/G=null・40軸UNDETERMINEDを維持する。保存rawや将来の修正結果で旧runの正常native終了を補完しない。

## 停止・native終了証拠

17:02:39 JSTに新grantを一回消費し起動。両preparation bridgeはnative exit0で保存を完了した。Pauli workerはexit0で20軸rawを保存した。欠陥のsource確認・人工再現後、自分のDF witnessをpid/start_ticks/commandで同定し、pidfd経由SIGTERMを送った。witnessが自分のDF workerを停止・回収し、native exit−15を永続化した。17:06:43 JSTにsupervisorは`C_WORKER_FAILURE`でexit1、全体`FAILED_NO_RETRY`となった。

| 今回のprocess | native exit | wait/reap証拠 |
|---|---:|---|
| Pauli preparation bridge | 0 | 保存済み |
| DF preparation bridge | 0 | 保存済み |
| Pauli worker | 0 | 保存済み。数値結果の科学受理とは別 |
| DF worker | −15 | 欠陥によるintegrity停止。正常完了ではない |
| supervisor | 1 | 失敗終了を保存・回収 |

各worker終了直後のnative記録、raw wait receipt、witness STOPを照合し、同一pid/start_ticksの残存0を確認した。[観測](../artifacts/non_df_h4_c_evidence/2026-10-23/launch01/observation03.json)、[停止理由](../artifacts/non_df_h4_c_evidence/2026-10-23/launch01/integrity_stop_request.json)、[実行STOP](../artifacts/non_df_h4_c_evidence/2026-10-23/run01/STOP.json)。

旧H4監視消失とは別の新runである。旧native exit/reap/STOPは未証明、元監視消失原因UNKNOWNのまま。今回のwitness動作によって旧欠落が回収できたとはしない。

## call予算・部分証拠・保全

| 契約のcall | 予約済み | 最大予算 |
|---|---:|---:|
| Operator/preparation復元 | 2 | 2 |
| Probability mapping | 2 | 2 |
| Reference包含 | 2 | 6 |
| Nominal law mean包含 | 4 | 12 |
| Implemented PR law mean包含 | 1 | 6 |
| PR normalization包含 | 2 | 6 |
| Dense matrix exponential | 13 | 39 |

これはcall前に記録する**予約数**であり、保存された完成証拠の件数へ読み替えない。DFは途中で停止しており、最終tierやcertificateが完成したとはしない。予算超過0、科学retry0、禁止された科学call0。

両表現の復元model、Pauliの64bit tier/result、grant/CONSUMED、call ledger、各native記録、停止理由、人工反例を保持した。動作中の封印SOURCEやruntimeを変更していない。旧SOURCE738・結果/claim1147・調査17・B/C inventory99ファイルの保全を確認した。[保存結果監査](../artifacts/non_df_h4_c_evidence/2026-10-23/completion_audit.json)。報告書完成時のindex・現状文書・manifestへの追記は、新runの来歴として旧記載を保ったまま行う。

## 次の最小作業（未実行）

[最小修正案](../artifacts/non_df_h4_c_evidence/2026-10-23/diagnosis01/proposed_reference_fix.patch)を保存した。reference行列をstateへ作用させてから縮約し、signal helperが行列outを拒否するshape検査を加える。**修正は未適用**で、実行SOURCE・runtimeはそのまま保全する。

次は**ユーザー**が、修正・非真空/複素重ね合わせ状態の人工回帰・新SOURCE/runtime seal、および別grantでの固定40軸再実行の範囲を明示承認する。**Codex**は承認後に新しい来歴・未使用namespaceで行い、消費済みrun01のresume/retryや結果上書きをしない。DF event範囲・literal compiled wrapper転送・正式certificate受理の従来制約は別途残る。N/G再会計・公開pushは別承認。

今回のrunは不具合を確認して停止・証拠保存・local commitまで行う。**mandatory STOP。自動修正適用・科学再実行・certificate受理・N/G会計へ進まない。**
