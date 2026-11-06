# O1 M1：固定C0負結果後の改善余地・静的技術整理

2026-11-06、SERVER 2（GPUA100-gleap02）。baseは`50ee55bc30f30838e1ac0704bc94f720914640de`。**静的SOURCE確認と公開済み結果の引用だけを行った。新計算・保存値再集計/再ランキング・最適化実行・実装・テスト・詳細科学レビュー・正式採択は行っていない。** 成果は本書だけで、公開照合後mandatory STOP。

round別RTE設定と配分の最適化資産は既にある。M1の全RPE・Pauli/DF・実半径・full no-prep価格を一体最適化する接続は未完成であり、既知手法を使った減衰軽減が強DF-PFに対する総費用改善を残すかは未判定である。C0負結果を変更せず、改善可能とも不可能とも結論しない。

## 固定して引用する結果

根拠は[固定C0報告][c0]。linear H4、1 Å、STO-3G、保存DF rank10、H4-Aの共通`T_eff/V_square`（LD0）、faithful tail identity、元Hの保存`state`を厳密基底状態と仮定し、状態準備費用を除く。`epsilon_E=.1 Ha`、成功目標95%、`delta=.05`、m=0..6、r=2、K=2、`beta_RPE=.30`。PF/RTE/stat/model/実装/reserve配分は`.06/.03/.15/.03/.01/.02`、各軸alphaは`.05/14`。Qiskit1.3.0、`rz/sx/x/cx`、optimization1、seed7、all-to-allである。

| arm | 公開済み条件付き総CX平均 |
|---|---:|
| Pauli PR | 5.69387927e28 |
| DF PR | 2.86880473e28 |
| 強DF-S2・4 substeps・spin policy/relative fusion | 9.82700813e9 |

21/21価格pointがpilot条件を通過したが、C0では総費用改善を示さなかった。q64のattenuationはP `1.740598031e-11`、DF `3.183271063e-11`、必要shot/axisはそれぞれ`3746527790271109800132608`、`1120156469303095630757888`。強baselineは各round1134 shot/axis。価格MCは各PR arm/round32 trajectory、最大CX relative sample SEは約4.7018%。これらは報告からの転記で、新しい算出ではない。`not_certified`、`price_uncertainty_qualified=false`、価格CI未算出と科学STOPを維持する。UWCに対する判断はない。

入力の固定先はcommit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。今回NPZをdecodeせず、raw JSONを再集計していない。

## 既存資産と再利用の境界

以下のSOURCE・文書はすべて固定commit `50ee55bc30f30838e1ac0704bc94f720914640de`内のpathを参照する。sparse checkout未展開の文書は`git show`で読んだ。[概要][overview]・[手順][method]の2026-09-20記述は履歴であり、M1の現在の数値根拠は上のC0報告を優先する。

| 既存path / 処理 | 再利用できるもの | M1へ必要な差分・未検証部分 |
|---|---|---|
| [rpe_short_round_optimization.py][short]：`optimize_rpe_short_round(s)` | 宣言したr/K格子の完全探索、cost比較条件・失敗・保証statusの区別 | `DFPartialS2Preparation`型、q<=4限定。`MatchedResourcePreparation`と全7roundへ直接適用できない。delta/split/配分は最適化しない |
| [rpe_delta_round_schedule_validation.py][schedule]：`validate_rpe_delta_round_schedule` | PF条件によるdelta screening、round別r/K、独立行列照合という手順 | 成分作用数×暫定shotのproxyで、full価格ではない。DF ground-state診断とidentity抽出を元H state・faithful policyへ無条件転用しない |
| [rpe_allocation_sensitivity_validation.py][allocation_src] | 固定された価格・法の下でbeta/alpha配分を比べる構造 | 旧q1/2/4直接価格＋q8 proxy用。matched実半径とmodel/実装枠、全roundの成功条件を維持する接続が必要 |
| [o1_m1_resource_bridge.py][bridge]、[o1_m1_parallel_price.py][parallel] | 同一物理分割のP/DF law、実半径shot算術、fresh-IID MC/full no-prep価格、強DF baseline | round別に異なる設定を渡す薄い接続と、その設定ごとの独立信号/回路確認が必要。既存r2/K2の価格・signalを別lawへ流用しない |
| [df_partial_randomized_pf.py][split]：`split_df_hamiltonian_by_ld` | 保存済みDF列をprefixで分け、one-body/scalarを維持する既存処理 | 現M1 matched-AはLD0固定。別splitは新しい同一物理分割・P/DF対応・operator検証が必要で、既存selectorを呼ぶだけでは完了しない |
| [o1_m1_resource_bridge.py][bridge]：`build_strong_df_baseline` | 保存factor、明示substeps、spin policy、relative fusionを用いる強DF-S2 | C0の4 substepsは一つの固定比較点。次の比較ではbaseline側も同じ精度/成功/誤差/compiler条件で適格なstep設定を検討する |

過去結果も方法の存在根拠に留める。[旧schedule検証][schedule_doc]はH4・1 Å・STO-3G・rank12・LD3、暫定CA/10、beta=.4の3 delta候補・全56行列点のPASSを報告しているが、全RPEの実compiled総費用比較ではない。[旧配分感度][allocation_doc]は同じrank12/LD3、delta=.1/r4/K2、q1/2/4/8、Qiskit1.3.0/opt1/seed17、RZ指標でbeta/alpha配分の効果を報告した限定診断である。[中央RTE proxyのstatus][status]は決定論half sweep・外側境界・制御化・Hadamard wrapper等を含めず、local dirty-worktree evidenceと明記する。**rank10/LD0、seed7、CXのC0への数値・proxy係数の移送根拠にはならない。**

## normalization・shot・一shot価格のtradeoff

既存[finite-law/attenuationコード][rte]と[shot式][shot]は、概念的に次を別々に扱う。

\[
\tau_m=\lambda_R\delta/r_m,\qquad
A_m=B_{K_m}(\tau_m)^{-q_mr_m},\qquad
\rho_{m,\mathrm{lb}}=A_m(\rho_{\mathrm{PF},m,\mathrm{lb}}-\epsilon_{Z,m}),
\]
\[
N_{m,b}=\left\lceil\frac{4\log(2/\alpha_{m,b})}
 {\rho_{m,\mathrm{lb}}^2\sin^2\beta_{\mathrm{stat},m}}\right\rceil,
\qquad G=\sum_{m,b}N_{m,b}\,\mathbb E[C_{m,b}^{\mathrm{no\text{-}prep}}].
\]

これはSOURCEの記号整理で、新しい数値評価・保証の導出ではない。

- **r_m**：中央RTEのtauを小さくする一方、trajectoryのevent数はq_m r_mになる。減衰軽減を狙えるが、一shot価格増と合わせた最適点・改善倍率は未評価。現M1 builderの外側one-body half stepはdelta/2のままで、r増だけで外側PF biasが改善したとは扱えない。
- **K_m**：大きくすると打切り残差を抑える一方、有限normalizerへ非負の次数重みが追加され、eventの次数/価格も変わる。固定tauでKを上げればattenuationも必ず改善する、とはいえない。低いKもRTE誤差配分を通る必要がある。
- **beta/alpha配分**：law固定ならnormalization自体は変えず、許される統計誤差と失敗配分を通じてshotを変える。旧配分改善率をC0へ掛けたり、model/実装/reserve枠を無断でstatへ移したりしない。
- **PF刻み・split**：deltaや決定論側を変えるとPF誤差・tail norm・半径・round範囲・価格が一緒に変わる。r/K調整と区別して再検証する。M1の共通物理分割も変更条件として明示する。
- **回路価格・実行速度**：basis reuse/fusionは一shot価格の論点。古典worker並列化やfingerprint高速化は計測のwall timeの論点で、量子総CX削減を意味しない。C0の強baselineは既にspin policy/relative fusionを用いている。

従ってC0の大きな負結果を「既知配分法の小さな改善だけで回復する」とも、「固定r/Kの結果だけで全設定が不可能」とも読めない。改善残存には、r等によるshot減少がfull no-prep単価の増加を上回り、決定論sweep・基底変換・全必要roundを含めた総費用が同条件の強DF-PFより小さいことが必要である。lambda・PF誤差・normalization・単価の単独最小では判定できない。

## 未採択の最小検証案と具体的な承認範囲

以下は仮説を切り分ける案で、候補の優先順位・有効性を数値評価したものではない。現在は新SOURCE未作成で、科学実行GOではない。

1. **ユーザーが別途指示する最小実装準備**：既存matched bridgeを使うround別設定の薄い接続と人工テストだけ。元state、faithful identity、同一P/DF分割、符号/scalar、各lawのnormalization・scope照合を維持する。新しい管理基盤は不要。
2. **限定信号検証案（別途実行承認）**：同じ保存rank10入力・delta=.05・K=2で、q=64のr=4と8の2条件をP/DF共通に調べ、C0 r2/K2は固定記録の対照として引用する。r4/8は既存格子から取った未採択例で、最良候補ではない。新SOURCEを固定後、operator/controlled相対位相、元stateのPF/RTE複素信号・normalization・半径・誤差配分と、これらの照合に必要な実回路buildを承認範囲に含める。shot/価格/compile、sampling、全round総費用、追加格子はこの案の範囲外。q64単独で全RPE改善を結論しない。
3. **全RPE比較案（条件確定と別途実行承認）**：選択規則と全mのr_m/K_m、必要ならPF/split候補、beta/alphaとmodel/実装/reserve、energy区間・分枝、baselineの許可step集合を先に固定する。目標は元H厳密基底状態仮定/no-prep、同じ0.1 Ha/95%、同じcompiler・主CX。PRだけ調整してbaselineを不必要に固定した比較を最終優位性にしない。各roundの独立信号確認、shot導出、fresh-lawのfull no-prep build/compile/価格MC、全round集計を明示的な対象にする。価格不確かさ・候補選択後の比較規則も実行前に決め、SEをCIと呼ばない。delta等を変える場合は必要horizonも固定し直し、7roundを機械的に維持しない。

**GO（次計画を検討できる条件）**：既存SOURCEの必要差分が明確で、元H state・scalar/identity/分割一致と独立参照、事前配分・正半径・分枝/成功条件を保てること。全RPEの同条件費用と不確かさが揃うまでは、改善は仮説のまま。

**STOP**：現時点は科学実行未承認・未実装のためSTOP。将来もoperator/phase/誤差/半径の不適格、未取得価格、未検証proxy、baseline条件不一致、候補選択/不確かさ未規定なら費用優位性の判定を止める。欠損を0にせず、条件・閾値の事後変更で救済しない。事前固定した有限候補で、適格な全RPE費用と不確かさを確認しても同条件の強DF-PFを下回る根拠が得られなければ、その候補集合の調査を終了する。格子拡張・条件変更を自動実行せず、全PRの不可能性とは区別する。

round適応、配分、PF刻み、prefix分割は[既存研究手順][method]と実装にある既知の調整項目で、接続だけを新アルゴリズムの新規性としない。外部文献の網羅的比較・UWC優位性・正式な継続/採択判断は今回実施していない。新fit・rank探索・別Hamiltonian・DF_stateへの置換も本案に含めない。

次の担当：**ユーザー**が後続準備/限定検証の要否・候補と承認範囲を判断する。**Codex**は本書の公開・照合後mandatory STOPし、追加実装・科学計算・詳細レビューへ進まない。旧C0結果・SOURCE・raw・許容値・grant・marker・STOPは変更しない。

[c0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md
[overview]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/研究概要・現状.md
[method]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/研究方法・解析手順.md
[short]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/src/trotterlib/rpe_short_round_optimization.py
[schedule]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/src/trotterlib/rpe_delta_round_schedule_validation.py
[schedule_doc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/rpe_delta_round_schedule_validation.md
[allocation_src]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/src/trotterlib/rpe_allocation_sensitivity_validation.py
[allocation_doc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/rpe_allocation_sensitivity_validation.md
[bridge]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/src/trotterlib/o1_m1_resource_bridge.py
[parallel]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/src/trottertracks/o1_m1_parallel_price.py
[split]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/src/trotterlib/df_partial_randomized_pf.py
[status]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/VALIDATION_STATUS.md
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/src/trotterlib/rte.py
[shot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/src/trotterlib/finite_rte_signal_validation.py
