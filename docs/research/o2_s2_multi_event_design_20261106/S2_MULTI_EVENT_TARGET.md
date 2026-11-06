# O2-S2：複数event対象の選定案

2026-11-06 JST。静的設計。基点は[直前preflightの固定commit][BASE]。今回の方向性承認は、科学実行・実装・重要科学レビューの許可を含まない。

**最小候補は、D1保存row 10／D2 Closed P5の `r=2, τ=5/7, x=5/14, degree=5`。** 同じ保存stepを時間順に二つ接続する。個別pair価格と価格依存proposalの新評価は未実施であり、本書は対象の提案である。

## 1. 比較可能性を決める条件

共通のsynthetic operatorを

\[
R=\tfrac15Q_0+\tfrac3{10}Q_1+\tfrac12Q_2
\]

とする。3 system qubits、`Q0=Z0`、`Qi=Vi† Zi Vi`、`V1=R_XX01(π/4)`、`V2=R_XX12(π/4) R_ZZ01(π/4)`、`R_P(θ)=exp(-iθP/2)`。符号σ=+1、state=`000`、workspaceは既存のancilla 1個。[D2契約][D2C]・[結果記述][D2R]に明記されている。

これは既知development入力である。分子geometry、basis、DF rank、split `L_D`、δ窓は非該当。`000`を元の物理Hamiltonianの厳密groundとみなさない。最終目的の元H厳密ground仮定・状態準備除外は維持し、真のenergy/gap/path信号を無料情報にしない。

\(P_5(z)=\sum_{k=0}^{5}z^k/k!\) とすると、D1の有限targetは

\[
F_r=[P_5(-i(\tau/r)R)]^r
\]

である。[D1 `collect_pending/evaluate`][D1P]は `x=tau/steps` をgeneratorへ渡し、stepのfinite targetをr乗して比較する。有限targetと厳密な \(e^{-i\tau R}\) は異なり、その差は共通biasで扱う。

## 2. 候補と条件対応

| 候補 | 有限target・整合条件 | 再利用と追加処理 | 最小案での扱い |
|---|---|---|---|
| G9/G10の1 P5 event | `x=5/7, m=5`、`P5(-i(5/7)R)` | event binding/価格は保存済み。ただしevent間境界を含まない | 今回の主対象には不足 |
| 同G9/G10 eventを2回反復 | `[P5(-i(5/7)R)]²` | 既存D2 `τ=5/7,r=2` と異なる。総時間と有限targetが変わる | D2との同target比較として採用しない。別条件の承認なしに変更しない |
| **D1 row 10→D2 Closed P5** | **`τ=5/7,r=2,x=5/14,m=5`、`[P5(-i(5/14)R)]²`** | 正しいxの保存binding・cache・境界kernelがある。個別pair価格表、joint proposal/weight/統計会計を追加取得する必要 | **第一提案**。新しい時刻・r・次数やIR/angleを生成せずに対象を定義できる |
| D2の他RA arm、同r=2 | 同R/τ/x/degree、各armの有限meanはD1条件に従う | ordinary、partial return、Closed P3 tail、Full return等の保存入力がある。law/zero-fillは各方式の実定義が必要 | 拡大対照の候補。全arm評価は今回の最小案に自動追加しない |
| D3 G1_P0のr=2 | 同物理入力・旧価格、より鋭いTaylor保証 | 同targetでもbias/remaining/Nの保証条件が変わる | 採択するなら全baselineへ同じ保証を適用。最小案はD2旧保証の再利用を提案 |
| D3 P1を含むr=2 | 同物理入力でも別の保存native価格/sequence | D3の別cache・bindingとの対応が必要。旧P0と混在不可 | 追加precision/price探索にしない。今回選ばない |

G9/G10とD1 r=2はgeneratorの名前が同じでも、x・root/rotation・finite coefficient・digital proposal・native価格が同一とは限らない。G9 m5の価格表をD1 row 10へ流用しない。[直前preflight][PREFLIGHT]のx=5/7入力と区別する。D3の各保証・価格caseは[既存報告][D3R]に従い、保存された最良case/rの再選定はしない。

## 3. 推奨入力と固定pointer

| 役割 | 固定commit／path |
|---|---|
| D1科学SOURCE | `874ca12d347f8f35e5d8d64f65f24abc4a2df523`、[pipeline][D1P]・[accounting][D1A] |
| D1保存結果 | `fc96571881f301435a655f057709df4d6c1678e0`、[resource_map][D1M] の `rows[10]` |
| step入力 | 同commit、[events/row_10.json][D1E]。各recordの `index,event,native_ir,native_cost`。eventには `coefficient,proposal,weight,word,phase_i_power,ratio` 等がある |
| native sequence | 同commit、[synthesis_cache.json][D1CACHE]。ratioに対応するactual sequence、phase、T/1Q count、strict errorを保持 |
| D2科学SOURCE | `28fc3ca17ec51bb7656a498e3eea9b72966eb3fa`、[契約][D2C]・[boundary.py][BOUNDARY] |
| D2保存結果 | `d1c59b28f4370cc7be65cff60d1ec877fd1491f5`、[resource_map][D2M]、[報告][D2R] |
| D3科学SOURCE | `dceccaa774229527a46a1386131583a4f618f8ad`。case対応は[固定報告][D3R] |

D2報告の登録metadataではClosed P5は63 step events、3,969 ordered pairs。今回これを新たに集計した数とは表現しない。row 10の保存eventsを公開GitHubからも取得し、固定Git blob `a84047e2bcc9575fd9baf061678e1552d4cf264f` と対応確認した。科学的な再計算・全入力再監査ではない。

D2保存resource mapは元lawでの期待価格・saving・worst・相殺depth分布等の**集約値**を持つ。個別pair価格表は保存せず、`pair_circuits_retained=0`。集約値から各pair価格を復元した扱いにはしない。

## 4. 接続順序と保持するmean

pathを \(\omega=(i,j)\) とし、時間順はleft step i→right step j、行列順は \(V_jV_i\)。native IRのlistを `IR_i + IR_j` の順で扱う。非可換因子を入れ替えない。

保存eventの正の有限係数を \(a_i\)、実digital proposalを \(P_i\)、weightを \(w_i=a_i/P_i\) とする。符号・複素phaseは保存されたcontrolled event \(V_i\) に保持する。元lawはD2契約の独立step drawなので

\[
P_0(i,j)=P_iP_j,\quad A_{ij}=a_ia_j,\quad W_0(i,j)=w_iw_j.
\]

joint proposalをqへ変えても全正係数pathを正確に被覆し、同じ回路・係数・phaseを保ち、実qで \(W_q=A_{ij}/q_{ij}\) と補正すれば、実装mean \(\sum_{ij}A_{ij}V_jV_i\) は同じになる。理想P5積との係数丸め/native近似差は既存共通biasに残す。marginalの一致だけでは十分でない。

D2と同じliteral inverse cancellationはこのmeanを保持する。signed rotationの実sequenceとそのadjoint、制御に関係するglobal W phase、wire、angle keyを保持する。新angle、合成string内部の追加簡約、非可換交換、任意fusionは対象外。Gaussian/basis変換一般の共有を検証したことにはならず、このsynthetic入力では保存IRに現れる隣接inverseの共有だけを扱う。

## 5. 実行可能性と未採択事項

**限定した明示pair-table検証は設計可能。** 固定入力の取得、演算順序、同mean補正、合法な境界passまで接続できる。一方、price table・normalizer・新qのM2/L/Nと費用は未評価である。非列挙sampler、分子全RPE、性能改善、新規性、科学GOは未確立。

次の[最小実行仕様案](S2_MINIMAL_EXECUTION_PLAN.md)で、既存保証/precisionの再利用、主資源、wrapper境界、zero-priceの扱い、古典online費の評価範囲を選択事項として示す。これらをユーザー／GPTが採択し、必要最小限の実装と新しい科学評価を明示許可するまで実行しない。旧D2/D3のone-shot許可やSTOPは再利用しない。

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/79253e0aa761c06ca9439b795d5619c7865f860c
[PREFLIGHT]: ../o2_s2_p5_access_preflight_20261106/S2_P5_ACCESS_PREFLIGHT.md
[D1P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/874ca12d347f8f35e5d8d64f65f24abc4a2df523/src/trottertracks/algorithm_codesign/rte_development/pipeline.py
[D1A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/874ca12d347f8f35e5d8d64f65f24abc4a2df523/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[D1M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/resource_map.json
[D1E]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/events/row_10.json
[D1CACHE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/synthesis_cache.json
[D2C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28fc3ca17ec51bb7656a498e3eea9b72966eb3fa/artifacts/track_b_d2_boundary_source/2026-10-13/v1/contract_v1.json
[BOUNDARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28fc3ca17ec51bb7656a498e3eea9b72966eb3fa/src/trottertracks/algorithm_codesign/rte_development/boundary.py
[D2M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1/resource_map.json
[D2R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1c59b28f4370cc7be65cff60d1ec877fd1491f5/docs/tracks/algorithm_codesign/d2_boundary_results_and_gpt_handoff_20261013.md
[D3R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/534d2347cb8af59a85ff58c3eb4e8aaf6e5a1c1f/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md
