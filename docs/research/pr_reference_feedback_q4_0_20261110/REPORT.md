# 新A Q4-0：強い推定器baseline・比較設計・貢献候補

2026-11-10 JST／SERVER 2。**可変reference位相による適応測定と未知visibilityの推定は既知であり、Qの約50%削減だけから最良推定器への優越や新しい測定原理を認定できない。** 次の最優先比較は、同じprefix・未知ρ・bounded bias・有限coverageを使う強いX/Y集合推定器。さらに二項CIを導入するならQ側も同じCIへ強化する。

旧Q0は弱い1252対照だけを評価したものではなく、同じHoeffding CI-hull familyの強X/Yを総数340以下で排除する限定証拠を持つ。ただし、その最小shot数は未確定で、他のCI／推定器への下界ではない。今回、その下界や50.381058%を再計算・新しい最適値へ置換していない。

## 成果物・今回の範囲

- [BASELINE_COMPARISON](BASELINE_COMPARISON.md)：推定器、同じ情報・成功条件、native共有信号の静的補題、測定能力、費用モデル。
- [PRIOR_ART_AND_NOVELTY](PRIOR_ART_AND_NOVELTY.md)：8本の主要一次文献、訂正・確認版・本文位置、追加文献の確認限界、既知の原理と貢献候補。
- [NEXT_NUMERICAL_VALIDATION_PLAN](NEXT_NUMERICAL_VALIDATION_PLAN.md)：未実行のH（強X/Y Hoeffding）／E（matched二項CI）／C（条件成立後の保存価格接続）の固定入力、最適化変数、幾何、科学call、gap／STOP規則。

基準commitは[`08f9cbe77456ec29d50a992d7165c4d726df7bf7`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/08f9cbe77456ec29d50a992d7165c4d726df7bf7)。同じbranch `codex/algo-improve-new-a-total-cost-20261109`を継続し、新Q4-0 directoryにこの4文書だけを追加する。研究階層はアルゴリズム改善→新A→Q系列→Q4-0。

今回行ったのは固定GitHub成果に対応するlocal Git object／SOURCE／保存結果fieldの静的読取、一次文献の限定照合、外部codeの静的確認、数学的条件整理、文書作成・公開。**新しい科学計算・数値再解析・shot最適化・信号/count走査・sampling・build/compile・simulation・RPE再実行・再価格／T評価は全て0件。** 研究moduleのimport、外部notebookの実行、科学testの再実行もしていない。実装変更がないため新test suiteを作らず、文書リンク・参照・変更範囲だけを技術確認した。

正式共通指示2026-11-09、共通／worktree AGENTS、PROJECT_MAPを適用。primary agentだけで作業し、AI subagentなし。既存SOURCE／raw／結果／grant／ledger／marker／STOP／KEEP、他Track、共有設定・他者process/fileは不変更。独自のCPU/RAM/worker上限・新しい管理基盤も作っていない。

## 1. 維持する科学結果と、下界の範囲

固定条件はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2／LD3／h=.02／K2、canonical independent finite bits、Compiler A/Qiskit1.3、ideal native/reset/measurement。m17はq131072/r32/bits88、既存18round、精度1.5936e−4 Ha、failure .05、cap .4。元科学SOURCE f2638a78を変更しない。

| 既存成果 | 正確な位置付け |
|---|---|
| Q0 L2 X/Y | 1252は座標Hoeffding–L2の十分証明内の最小総数。α配分・整数最適化を含むが、一般の統計推定器の下界ではない |
| Q0単一Pauli | 前段情報を使いX又はYを一つ選ぶ対照に、保存許容集合での等投影反例がある。任意角readoutの不可能性ではない |
| Q0強X/Y | 連続中心π/4・未知ρ/B・同じHoeffding CI-hullでは総数≤340を排除。最小総数は341〜1252で未確定。二項CI、joint LR、全prefix likelihood、逐次adaptive方式、実H4の到達信号への下界ではない |
| Q0〜Q3-DのQ | prefixからφを一度決め、m17全340 trialで同じ角を使う。branch／有限精度／native／丸めを含む固定モデル内十分保証。m17内部のadaptive再選択を追加していない |
| Q3-D native費用 | Reference-B1は従来X/Y比の全18round RZ削減率下限50.381058%。B′にもQを与えた比較ではReference RZ下限0.287865%の削減。強い別推定器との比較値ではなく、総Tや実機時間ではない |

SOURCE・RESULTS・CHECKSはQ0の[REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34327b904540556bce7509f2a3168cbff36daab7/docs/research/pr_reference_q0_20261109/REPORT.md)と、Q3-Dの[REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08f9cbe77456ec29d50a992d7165c4d726df7bf7/docs/research/pr_reference_feedback_q3cd_20261110/REPORT.md)を正本として照合した。Q0の圧縮RESULTSは展開して保存済み結論fieldを読み、scanの再集計／ランキングはしなかった。

ユーザー指示に記載された直前GPTレビューの支持は受領した判断として扱う。今回、そのレビューを代行・再実施したものではなく、本文を別途取得・読了したとも主張しない。

## 2. 先行研究から分かった対応

Higginsらのfeedback、Wiebe–GranadeのBayesian PE、de Neeveらのtime-adaptive PEは、可変reference位相を使う既知の測定familyである。Qのprefix中心に対するquadrature readoutもこのfamilyに入る。アルゴリズム全体のpolicy・interrogation長・損失関数が同一という意味ではない。[P1/P5/P7の一次資料対応](PRIOR_ART_AND_NOVELTY.md#1-確認した一次資料と読んだ範囲)。

Belliardoらは未知visibilityをjoint Bayesian推定／nuisanceとして扱っており、「visibility未知に対応した」という事実だけは新規性根拠にならない。Qはρを点推定せず、[ρL,1]を射影して有限幅を保証する。その**特定の一様CIと有限実装・資源接続**に違いが残る可能性があるが、同条件の強い方式への優越は未確定。[P8](PRIOR_ART_AND_NOVELTY.md)。

Kimmel–Low–YoderのRPEには2021訂正があり、Russoらはbranch／consistencyの条件を補強している。今回、旧定数の無断流用もQ3の再検証も行わない。Bayesian median/MSE／credible mass、CR下界、有限sample worst-case coverageを同じshot指標にしない。

文献の確認版・該当式・節・未取得範囲は一次文献表に記録した。Roccia2018の詳細本文とTAPEのBitbucket実装は今回未取得。P8著者notebookは固定blobを静的に読み、likelihood/update/utility等の再利用候補を特定したが、coverage資格のあるdrop-in実装ではない。限定調査なので「同等手法が存在しない」という結論は出していない。

## 3. 今回具体化した比較上の問題

**第一に、共有nuisance。** 強X/Yは同じ複素信号を両軸CIへ同時に入れる必要がある。各軸を別々に逆写像して交差するだけでは弱い外側集合になる。Q3-Cの角ごとのvirtual-state補題だけを、その共有性の証明にしない。

保存H/P/Sdg templateを使う[静的代数補題](BASELINE_COMPARISON.md#3-joint信号を共有するためのsource条件静的補題)を導いた。同じnative bodyのcoherenceを既知のδ=π/2−aだけ回したz'により、X/Y/Qを軸0／a／φbinとして同時に表せる。旧raw diskへ収めるnorm条件も明示した。**native Yの軸aを数学的π/2へ丸めない**ため、後続のjoint geometryはrectangleの単純な円交差から既知parallelogramの円交差へ拡張する。これは今回の静的導出であり、数値optimizer／新成功保証の正式採択をした結果ではない。

**第二に、center domain。** Q0のπ/4 witnessは連続中心の外側問題に対する下界。実N347 countからその中心が厳密に到達するとの証明ではない。全ψの一様上界は実履歴に安全側に移るが、その最悪点の下界を実H4の必要shotへ移さない。最初はQ0と同じrobust taskで比較し、domainを都合よく狭めない。

**第三に、最適化機会と最適性gap。** Aの配分はprefix中心に依存してよい。L2用の最良αをjoint hullの最良αと仮定せず、整数nと連続αを最適化する。可行解、下界、未解決gapを分ける。強いCIを導入するならQ側も同じ強化を受ける。旧340以下の排除を別CIへ流用しない。

## 4. 最も価値のある次の限定検証

提案は[H/E/C bundle](NEXT_NUMERICAL_VALIDATION_PLAN.md)。Hは同じHoeffdingでの強X/Y最適化、Eは等尾の外向き二項CIをQ／X/Y双方に認める閾値診断、Cは保証が成立した解だけを保存native価格へ接続する。別の科学条件や量子回路を作る前に、既存の340と強対照の距離を直接判別できる。

技術的には、旧192-bit scalar helper、count型、branch/feedback、保存価格のFraction会計を再利用する。必要な新規部分はjoint polygon-annulus geometry、continuous α／centerのcertifier、外向きbinomial inverseと独立checker。これらはまだ資格済みrunnerとして存在しないことを明記し、旧Q0を再実行する計画にはしていない。

原論文のBayesian policyを同じworst-case基準へ移すには、nuisance尤度とcoverage証明、場合によってはpredictable adaptive設定に対応したconfidence sequenceが必要。posterior credible massや固定nのCIをadaptive停止へそのまま使わない。既知方式との対応は既に整理したが、その新しい数値campaignまで今回自動開始しない。

## 5. 貢献候補・未立証・次の担当・STOP

残る貢献候補は、既知のadaptive quadratureを、PRのcoherent mean、未知半径、有界bias、prefix成功、数値受理、native circuit costへ保証付きで接続した具体的caseと、そこで明確になる有利条件である。confidence集合の反転やresource-aware設計という考え方そのものを新規発明としない。

未立証なのは、matched強baselineへの優越、任意推定器に対する340最適性、一般の未知visibilityでの新定理、PR固有の測定原理、別Hamiltonianへの一般化、Full/Truncatedへの競争力、総T削減である。CTでは任意feedback角のword policy／strict相対位相／誤差配分・domain／総費用が残る。旧CT結果・STOPを維持する。

**次担当：GPT／ユーザー。** ①論文化する主claimを保証付きPR実装・native資源評価に置くか、②全ψ robust taskとH/E/Cの範囲を採用するか、③新しい科学数値実行を許可するかを判断する。**Codex**は承認後に必要実装と計画内科学callを行う。今回の4文書をcommit/pushし、remote SHA・全文・hash照合後、**mandatory STOP**。数値結果がない部分を推測で補わず、自動で追加実装・科学計算・正式レビューへ進まない。
