# H4 Pauli-JW central-control A_AXES：両軸費用の完成

2026-10-24 JST。添付61fce570の明示承認でPR sine255本（index0,2..255）とPF sine1本を新規計測する。既存cosine257件とPR index1 sine1129 RZはSOURCE/入力SHA/compiler/native対応を確認して再利用する。基点7c5e25dce757b7daa165095839ca879ed0bb0c10はGitHub取得済み。現時点はSOURCE固定前の記録で、科学結果・終了監査を追記する。

Linear H4/隣接1.00 Å/STO-3G、8system＋ancilla wire8、Pauli184非identity項・PR prefix27、S2/T.8/q8/r4/K4、元保存law・D2 central-control。Qiskit1.3.0、rz/sx/x/cx、opt1/seed17、backend/coupling/layout/routing指定なし、状態準備費用除外。

[Adapter](../src/trotterlib/h4_a_axes.py)、[runner](../scripts/run_h4_a_axes.py)、[人工検査](../tests/test_h4_a_axes.py)、[証拠入口](../artifacts/non_df_h4_a_axes/2026-10-24/README.md)。D2のPF/PR index0はcontrolled_body.qpyをそのまま利用。保存cohort index2..255はcosine入力からH–body–H–measureの間のbodyを抽出し、演算順序・全global phase・wireを保持する。元hadamard_wrapperのalready_controlled=TrueでSdgを最後のHの前へ追加し、二重control・barrier・Pauli replay・central生成を行わない。

元D1 QPY compiler、D2並列batch、A_COHORT_COSのgrant/authority/native launcherを再利用。再利用control flowの変更は新namespace・grant/status名と254→256の終了対象数だけで、科学builder/compilerは変更しない。既存統計_statistics/H.summarizeも使用する。新人工14 PASS、旧44件の意味対応証拠をSHA保全して再利用し、今回再実行したとは扱わない。主agent自身の検査で、独立別担当reviewではない。

新SOURCE commit後に入力QPY/価格/runtime/schemaをsealし、専用active contractと未使用one-shot grantを発行する。call最大body再利用256・sine wrapper256・費用compile256・paired統計1batch、cosine compile/body replay/central生成/sampling/state-signal/N_G0。各compileはtask起動前に非返還予約する。独立対象を実際の空き資源に合わせ並列処理し、各processの小行列BLAS/OpenMPは1threadとして過剰な多重thread生成を避ける。共有設定や他processは変更しない。

技術的中断はユーザーが明示承認済み。原因を特定し、科学条件を維持し、成功済み対象を再利用して失敗/未完了だけを新SOURCE/seal/契約/namespace/未使用grantで回復する。消費済みcallや旧run/STOPは変更しない。recovery_planは範囲を読取専用で整理するだけで、自動retry/実行を行わない。科学的phase/演算不一致が未解決ならSTOP。人工fixtureのruntimeハッシュ項目不足はSOURCE前に修正して失敗ログを保存し、科学runの中断・回復とは区別する。

全価格が揃った場合にPR256本の両軸平均/SD/SE/nominal t95/paired差/sample covariance/分位点/一致数、PF両軸のRZ/CX/depth、旧genericと同IDの比較、DFの異law参考値を1batch整理する。保存cohortのexact平均とimplemented lawの母平均・未観測order4 tailを区別する。価格が揃ってもrho/精度/N/Gの保証は成立せず、今回N/G再解析は行わない。

旧historical hash FAIL/原因UNKNOWN・元H4 native UNKNOWN・旧N/G null・C/L COMBINATION_BLOCKED・旧compiler MISSING_EVIDENCE・正式certificate未受理を維持する。最新版運用§8.4/§10.1/10.2適用。同ファイルに§10.3は存在せず未確認規則を仮定しない。専用branch codex/non-df-h4-a-axes-execution-20261024へcommit/push確認後mandatory STOP。次のユーザー判断は条件付きC/N/Gの限定解析を別承認するかであり、Codexは自動進行しない。
