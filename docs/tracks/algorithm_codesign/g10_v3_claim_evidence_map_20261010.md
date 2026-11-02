# G10 v3：採用レビューのclaim・証拠・適用範囲

GPTレビュー日2026-10-10、資料化2026-10-11 JST。
利用者の「こんな感じで進める」により[レビュー原文](../../research/track_b_G10_v3_scientific_review_20261010.md)
の文書化方針を採用する。原結果分類はCOMPLETEのまま。新規科学authorizationは発行しない。

| Claim / 方針 | 証拠の種類と正本 | 適用範囲・今回の扱い |
| --- | --- | --- |
| 同一有限P_mの比較完了 | [G10原結果handoff](g10_v3_results_and_gpt_handoff_20261010.md)、corrected/final監査、outer receipt | 登録17行・34軸、source-bound local evidence。独立外部再現ではない |
| reduced-word return集約恒等式 | [G6数学監査](g6_independent_mathematical_audit_20261010.md) §2–7 | Q_i²=I、正p、奇数m、0<x≤1。既存証明の引用で、今回の再証明ではない |
| finite-bit平均・bias・access | [G6 finite-bit監査](g6_finite_bit_and_access_audit_20261010.md) とG10固定contract | ideal exact mean、digital補正、native誤差を区別。恒等式だけで実装費用を無料にしない |
| 一般fullの追加registered T利益は不支持 | 採用GPTレビュー§4、[保存有理算術](../../../artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10/saved_arithmetic.json) | 各m内のclosed対照に対しT/Kとも正差。全入力への不可能性とはしない |
| m3 partial/P3のprep依存性 | exact 17-row tableからaffine式を照合、図1 | h境界約287.5266。単価の採択や新しいwinner探索ではない |
| m7 fullのsampling-only救済の条件付き限界 | 保存fixed_dictionary_policy_lowerと登録P5＋tail、GPTレビュー§6 | 固定辞書・native列・precision・confidence policy、0≤h≤970。lower対達成予算の比較 |
| lower endpointの意味 | 保存lowerとの差を有理算術で確認、図2 | 約978.0709は下界未分離への境界。実際のwinner crossover/最適lawではない |
| root回転価格の具体的変化 | exact event partの先頭3 eventsずつ | 各4 T差、受理回路share約87.8%。全差の一意原因・他seed/backendの定理にはしない |
| P5系のliteral CTSへのT利益とCX trade-off | exact native T/CX/1Q/K、保存CTS lower、図3 | 同じfinite taskのliteral CTS。CTS family全体への優位・全資源支配ではない |
| support減少と古典scalingを分ける | GPTレビュー§8、非列挙source、保存interface trace | 255対2,250 bindingsをproduction実行時間・メモリ・一意回路数と同一視しない |
| 既知要素と独立method deltaを分ける | 採用GPTレビュー§10、G6 prior-art監査 | 新しいpriority監査は実施しない。全体新規性・投稿十分性は未確定 |
| G9からG10への方針更新 | G9レビュー§14の事前分岐、G10レビュー§1/12–15 | 固定入力の一般full性能探索を区切り、構成・native限界ノートへ整理 |

共通native条件：p=(1/5,3/10,1/2)、x=5/7、synthetic 3-qubit provider、m=3,5,7、同じP_m内の
full first operator moment、literal direct controlled lowering、strict Rz 10^-6、epsilon_axis=1/200。
Pauli取得可能なI1文脈。分子geometry/basis、DF rank、L_Dは適用外であり、degree間を等exponential精度で比較しない。

## この作業の独立性と限界

ローカルscriptは保存されたexact tableと二つのevent partから差・比・affine境界を再計算するだけである。
既存lower、予算、合成列、行列誤差、samplerは再生成しない。corrected/final監査を正本として保持する。
GPTレビューが言及する別添`analyze_display.py`、`check_exact_m7.py`等のZIPは今回の入力に含まれず、
それらの再実行やGPTの独立64項対数certificateの再現とは表示しない。

## 次にGPTへ戻す節目

- 別の科学的利益を持つ新構成を提案する。
- 比較対象、主評価、科学条件、一般化検証のscopeを変える。
- 独立論文の主要claimと新規性を確定する。

本文構成・表記・保存値の照合はこの採用scope内でCodexがまとめて進める。
m9、新入力、合成・sampling最適化、DF/分子、G11、再実行は未認可。**mandatory STOP維持**。
