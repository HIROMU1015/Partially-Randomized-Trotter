# O2候補比較：全RPE費用へ接続できる条件

2026-11-06 JST。既存資料・SOURCEの静的照合による技術整理。候補の正式採択、科学的GO、新規性・論文化可能性の最終判定ではない。検証は計画のみで、実行していない。

入口は [REPORT](REPORT.md)、統計の根拠は [O2-A](O2_A_SAMPLING_STATISTICS.md)、return/nativeの根拠は [O2-B](O2_B_RETURN_AGGREGATION.md)。両trackを一つの新アルゴリズムへ統合しない。

## 1. 共通taskと比較の境界

元の物理Hamiltonianについて `H|E₀⟩=E₀|E₀⟩` を満たす同じ厳密基底状態を候補・対照に与える。状態準備を除く。未知のE₀、gap、完全な状態vector、pathごとの真の信号、最適価格・normalizerを無料の情報に加えない。新しい分子、精度、成功確率、compiler、主資源指標は固定しない。

\[
G_{\rm total}^{(a)}=\sum_{m,b}N_{m,b}\bar C_{m,b}^{(a)},\qquad
\bar C_{m,b}^{(a)}=\mathbb E[C_{m,b}^{\rm full,no\text{-}prep,(a)}].
\]

`a` ごとにRZ/CX/T/depthを別に扱う。古典setup、normalizer/query、価格取得、online生成、cache memoryは別欄に残す。交換率なしに古典秒数とgate数を足さない。ただし古典生成が実行可能か、双方に同じ情報・cache機会があるかは候補成立の条件である。

価格にはdeterministic D halves、中央tail、controlled位相、basis共有、合成、ancilla準備・X/Y readoutを含める。物理入力状態の準備除外は、候補固有の追加演算を無料にする意味ではない。event価格の和をfull-wrapper価格に代用しない。根拠は [既存の費用依存地図 §1–4][COSTMAP] と [主RPE SOURCE][RPE] の `evaluate_rpe_round_candidate/build_rpe_resource_summary`。

### zero-fillとshotの単位

G6等のzero-fillでは、confidence式の標本数が**全planned classical trial**で、実際の量子測定回数と異なる場合がある。その場合は全trial数 `K` と各trialの無条件量子価格を使って会計するか、量子測定へ進む確率と条件付き価格を明記する。zeroを統計の分母から消したり、accepted-only単価に全trial数を掛けたりしない。zeroの判定前に実行した回路があれば、その実費も残る。classical trialにも実測shotにも同じ `N` を無説明で使わない。[O2-B §3–5](O2_B_RETURN_AGGREGATION.md)

### 成功条件

同じ補正平均を保っても、raw半径、weight range/moment、量子測定分布まで同一とは限らない。例外的に、同じraw複素平均を持つ非加重±1のX/Y測定なら各軸のBernoulli周辺分布も同じである。weighted/zero-fillへはこの説明を拡張しない。

各軸の有限confidenceを、正の信号半径・統計位相予算、全roundの失敗確率配分、branch/alias条件、共通E₀への系統誤差へ接続する。主RPE SOURCEの単位基準半径は条件付きモデルであり、元Hの厳密基底状態だけからは保証されない。[研究目的 §2.2–2.3][PURPOSE]。条件付き18round成功・energy接続は既に保存されている一方、最終資源優位は未確定である。[固定後続整理 §5][ENERGY]

## 2. 分類

| 分類 | 候補 | 残る具体的条件 | 既知法との関係 |
|---|---|---|---|
| 1：独立方向 | S1 既知variance-aware会計のRPE接続 | 許された情報から得る有限confidenceが、同じ手法を与えたbaselineと比較して有用か | 既知保証の適用。samplingの新方法ではない |
| 1：独立方向 | S2 非加法whole-path価格を使う既知ISの適用 | 価格・likelihood・normalizerを実行可能な費用で取得し、range項まで含むN×価格に利益が残るか | whole-path IS、resource-optimal IS、Parent/S4が直接対照 |
| 1：独立方向 | R1 構造化return集約のfull-wrapper接続 | 非自明returnの重複を安く利用でき、生成・合成・統計・D部分を戻して利益が残るか | G6–G10の既存構成・native技術の適用/改良 |
| 2：組合せの問い | J1 return構造と非加法path価格を同時利用 | 集約後に変わるsupport/weight/価格を安く取得でき、同情報の最適化済み対照後にも増分が残るか | 組合せという名称だけでは差にならない。追加の取得可能性・成立条件が必要 |
| 3：残す根拠が弱い方向 | W群、§4 | 限定range-only再重み付け、marginal相関、Full一律優越、無料oracle等 | 既存障害・既知還元・負結果を維持 |

この順序は優先順位でも実行順でもない。S1は保証の共通整備で終わる可能性があり、S2/R1/J1に独立した研究寄与があるとは判断していない。

## 3. 候補ごとの9項目

### S1：既知variance-aware有限confidenceの適用

| 項目 | 技術照合 |
|---|---|
| 改善機構 | 同じ平均・回路familyに対し、最大weightだけでなく二次momentまたは正当なvariance上界を使う。単価低下を必須とせず、十分shotの保守性が減る条件を問う |
| Signal | estimator変更が平均を保存することと、native/finite-bit biasを差し引いた統計許容幅が正であることを確認する |
| 統計 | Track Bには既にBernstein会計がある。主RPEはIID±1 Hoeffding。Bernsteinのrange項は残り、実測varianceのplug-inや事後stopをそのまま保証にしない |
| コスト | 回路を変えなければ同じfull単価。pilotやvariance取得が必要なら測定・古典費用を計上する。両方式に同じ保証を適用する |
| RPE接続 | round/axis別許容幅、半径、失敗配分、全roundのenergy/alias条件まで追う。単位半径の仮定を追加しない |
| 先行研究 | 既知Bernstein/empirical Bernsteinの適用。新統計定理でも新sampling lawでもない |
| 既存証拠 | [既存accounting][BERN] `confidence_budget`、[相関法照合 §4.4][CORRELATED]、O2-AのSOURCE/文献対応 |
| 未解決点 | Track Bの補正平均・native biasを、主RPEのraw半径・energy成功条件へ移せるか。両対照に適用後の差は未評価 |
| 判断条件 | 同情報で有効な上界が得られずrange項/取得費が支配する、または双方の改善だけで方式間の差が残らないなら、独立研究候補としての継続根拠は弱まる |

### S2：whole-path ISと生成アクセス

| 項目 | 技術照合 |
|---|---|
| 改善機構 | basis相殺・D境界・control・合成を含むpath価格の非加法性を、既知ISで利用できる条件を問う |
| Signal | 同じ実装path familyとtarget、support包含、正確なlikelihood補正が条件。再重み付けだけで元のPF/native biasを減らさない |
| 統計 | `p/√c`は既知のcost×second-moment最適化。有限confidenceのNではrange・ceil・axis許容幅も必要。weight積は長roundで悪化し得る |
| コスト | path生成後に価格を知るだけでは、その価格でpathを引くsamplerを得たことにならない。normalizer、support列挙、rejectionの全試行、価格oracle/cache取得を課金する |
| RPE接続 | 各roundの全interrogationに対するlaw/weightを用い、fresh trialの条件を守る。局所event価格のlawを長roundの最適lawとしない |
| 先行研究 | [resource-optimal IS §II, Thm.1][IS]、cost-aware qDRIFT、whole-path IS、Parent/S4。非加法価格を使うこと自体もIS一般から外れない |
| 既存証拠 | [sampling freedom §3–6][FREEDOM]、Parent/S4、O2-Aのアクセス表。IS §II.1の長積議論は独立・加法条件付き |
| 未解決点 | 全support列挙/新compileを必要とせず、必要な価格情報とlawのnormalizerを利用可能にする構造があるか |
| 判断条件 | 同情報の既知ISより差がない、生成がsupport規模へ膨れる、有限confidenceの負担が単価低下を打ち消す場合は、その具体候補を終了検討 |

### R1：構造化returnの集約だけを変更する方向

| 項目 | 技術照合 |
|---|---|
| 改善機構 | 既知returnの係数/wordをまとめ、繰返すchild処理・回路実行やbasis移動を減らせる条件を問う。G6 kernelを再実装しない |
| Signal | 同じ有限polynomial numeratorとcontrolled相対位相を保つ。normalizationが変われば、同じ補正平均でもraw平均は変わる |
| 統計 | 全trial分母・zero-fill・finite-bitの重みを保つ。B/range/M2の変化をNへ戻す。非加重同raw平均の場合だけ統計を共通化できる |
| コスト | strict native全support、child回転、shared basis、D halves、prep/readout/full wrapperを含む。return/provider取得費を量子節約と別に記録 |
| RPE接続 | 同じ元H、同じfinite-time精度を経て全roundへ。異なるdegreeを同じexponential精度とせず、低次数の強い対照も最適化する |
| 先行研究 | G6 Full/Closed、G7/G8 provider、G9/G10 native、D1–D3、Parent/S4、既存native融合。構成存在と分子有用性を分ける |
| 既存証拠 | O2-Bの段階表と固定原文。G9 Closed P5/Fullの理想ensemble対応、G10の次数別負材料、C1のlogical限定結果を維持 |
| 未解決点 | 非自明returnが多く、取得が安く、強いno-aggregation/CTS/PF対照後にもfull価格差が残る入力クラス。現C1の全不足解消は目標にしない |
| 判断条件 | 子回転が残って差が小さい、zero回避がない、取得費が重い、native全wrapperに差が残らない場合は、その入力で終了検討。G6一般の数学的否定にはしない |

### J1：return構造とwhole-path価格の組合せ

| 項目 | 技術照合 |
|---|---|
| 改善機構 | 集約で変わるsupport/重みと、ordered pathの共有価格を一緒に扱える構造を問う。新しいjoint samplerを設計していない |
| Signal | return集約のoperator対応と全path likelihoodの両方が必要。片方の正しさで他方を代用しない |
| 統計 | 集約後のrange/M2/有限bit biasと全trial lawから一組のNを評価する。別実験のNや単独改善率を借用しない |
| コスト | 集約後のfull-wrapper価格と、return-query/normalizer/価格取得/online生成を共同で定義する。単独の節約率を加算・乗算しない |
| RPE接続 | roundごとに同じ有限mean/energy条件へ戻す。遅いroundだけ改善しても早いbranchの成功条件を維持する |
| 先行研究 | whole-path IS＋既存return kernel/Parent/native最適化が直接対照。追加性質が特定できなければ既知法の組合せ実装で終わる |
| 既存証拠 | RA-KERNEL/RA-NATIVE/ALLOCATIONという再利用入口はある。組合せの分子全RPE費用を確認した証拠はない。[再利用地図 §4][REUSE] |
| 未解決点 | 両方を調整すると初めて使える安価なquery/normalizerや価格構造があるか。単独では得られない効果は現時点で未確認 |
| 判断条件 | R1/S2の独立適用・同情報既知対照へ還元される、joint情報取得が重い、同target/統計/価格の境界が閉じない場合は組合せ独立候補を終了検討 |

## 4. 研究候補として残す根拠が弱い方向

| 方向 | 既存の障害・負材料 | 限定範囲 |
|---|---|---|
| 固定path集合・同targetを単に再重み付けし、最大weightだけの十分shotを使う | ユーザーが提示した前回レビューのrange-only障害と整合。価格低下だけで総費用改善を主張できない | O2-Aに照合条件と元review未特定を明記。必要shot下界、ceil込み全最適化、variance-aware一般の不可能性へ広げない |
| `p/√c`やParentの再命名 | 既知law・既存KKT/実装がある | 新しい有限confidence最適性/取得性を実証した意味ではない |
| marginal確率を保つ相関 | joint ordered operator meanは変わり得る。符号/重み/生成費が残る | 正確なwhole-path likelihoodを使う既知ISまで否定しない |
| Full returnを常に最良とする | G9同ensembleの重複、G10低次数対照のT/CX trade-off、C1の小さいlogical差・zero回避なし | 元synthetic/限定H4の保存条件の負材料。RA一般の劣位ではない |
| common-parentの再配分だけでBも必ず小さくする | SAMPLING-FREEDOMではcommon K0が当該parent familyのL1最小 | 全unitary辞書の最適性ではない |
| exact groundなので統計費・PF混入を省く | 準備状態が厳密でも近似PF/finite meanの半径・bias・測定揺らぎは残る | 取得不能な真のsignalやE₀をlaw選択のoracleにしない |
| 人工512件やC1保存pathから分子資源優位へ進む | API/限定logicalと、physical全support/native/N/full-wrapperは別 | 既存PASSを保持し、未取得部分を0/FAILにしない |

## 5. 次の最小検証候補と終了条件

以下は将来提案であり、実行契約・再開承認ではない。全UNKNOWN解消や全round新compileを一括要求しない。

| 問い | 必要になった場合の最小検証 | 公平な比較・終了条件 |
|---|---|---|
| S1/S2の有限confidence適合 | ユーザーが選ぶ一つの固定有限ensemble/taskで、同じ補正mean・bias・情報アクセスの下、既存Hoeffding/Bernstein、既知lawのNとfull単価を対応させる | 両armへ同じ既知保証と価格情報。range/取得費で利益が消えたらその具体案を終了検討。保存値の再会計も別途科学実行許可が必要 |
| R1に分子接続の価値があるか | 非自明return/child回転/共有basisの構造を説明できる入力一件について、physical DF対応を確定し、その有限supportに必要なstrict nativeと共通confidence/full-wrapper境界を確認する | 低次数Closed/CTS/通常PR/no-aggregation/強いPFへ同じ最適化。差がchild・D部分で消えたら拡大しない。C1を使う場合は凍結変更の明示判断が必要 |
| J1に独立した効果があるか | まず安価なreturn/価格/normalizer取得という追加性質をユーザー／GPTが特定し、採用する場合だけR1のみ・S2のみ・組合せ・同情報既知法を同taskで比較 | jointのNと価格を直接対応。単独利益の積を予測値にしない。単独適用へ還元されたらjoint独立claimを終了検討 |
| 全RPE改善へ接続できるか | 上の限定taskが必要な根拠を与えた場合だけ、共通E₀・alias・全round失敗配分と、支配的価格/統計負担の接続を確認する | 既存V1/V2/energy接続を再利用し、変更された保証部分だけ扱う。短roundやprimitiveの利益だけでは次の主張へ進まない |

担当は、ユーザー／GPTが扱う問い・対象・比較条件と研究上の採否を選択し、ユーザーが必要な科学実行または重要レビューをそれぞれ明示許可すること。Codexは許可された最小範囲のみ実施する。今回の完了に追加計算は不要であり、公開・取得照合後にmandatory STOPする。

[COSTMAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/END_TO_END_COST_ANALYSIS_MAP.md
[RPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rpe_resource_accounting.py
[PURPOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/研究目的・研究課題.md
[ENERGY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md
[BERN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trottertracks/algorithm_codesign/rte_reallocation/accounting.py
[CORRELATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[IS]: https://arxiv.org/html/2603.13495v1#S2
[FREEDOM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_sampling_law_freedom_survey_20261030.md
[REUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
