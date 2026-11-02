# Track B G2 科学的研究レビュー：degree-local RTE再配分の資源価値と次の判断

- **レビュー日**：2026-10-09（JST）
- **対象研究**：Partially Randomized Trotter / Track B、RA-RTE（Resource-Aware Reallocated Randomized Time Evolution）
- **G2 evidence**：`HIROMU1015/Partially-Randomized-Trotter`、branch `track-b-g2-saved-diagnostic-20261009`、commit `b260189b020ab7dfabb16bf424f49a6efff40d75`
- **G2原報告**：[`g2_saved_diagnostic_handoff_20261009.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md)
- **数理監査**：[`g2_independent_math_audit_20261009.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_independent_math_audit_20261009.md)
- **数値概要**：[`result_v1.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/artifacts/track_b_g2_saved_diagnostic/2026-10-09/result_v1.json)
- **前回の科学レビュー**：[`track_b_G1_scientific_review_20261009.md`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/research/track_b_G1_scientific_review_20261009.md)
- **レビュー分類**：科学的判断必須。**LIMITED_CONTINUE / NO_FULL_V4 / DECISIVE_FEASIBILITY_AND_BASELINE_CHECK**（本書での推奨ラベル。GitHubに既に記録された実行statusではない）
- **証拠境界**：現G2はdevelopment/post-hocの固定保存表診断。登録B2/B3の科学比較、独立再現、実際の量子計測、DF/分子への移送、一般的な新規性の証明ではない。

## 1. 最終結論

Track B自体は継続する余地があるが、**degree-local B3再配分を実用的な主アルゴリズムとして昇格させるには証拠が不足している**。既存3頂点にも同じimportance sampling（IS）とprecision選択を与えた限定理想目的において、J1はTおよび1Qで約0.36–1.52%の差を示した。CXの追加差はない。T側の差はzero-T eventを含む**未達成infimum**で、有限確率法による実装改善ではない。1Q側は理想実数proposalに限る。新たな数学的構造の存在と、資源上の有用性・独立新規性を分ける必要がある。

次の担当は**Codex**。ただしフルv4・大規模登録LP・新分子へ進まず、**有限実装可能性を検査し、成立しそうなら既知return対照を同じtaskで評価する限定作業**を一つの科学的判断パケットとして任せる。結果後はGPTへ戻し、Track Bの研究継続／理論ノート化／方向転換を再評価する。

## 2. 研究目的と履歴

研究Aは固定方式のmatched-accuracy resource/applicability study。研究BはPRのランダム化・時間発展アルゴリズム自体の改善。BF（finite-aware PF係数）ではF/L選択一致、BM（multirate error adapter）はcompact BCHと同値、SP（確率的合成配置）は選択的優位を示さず、BS（block synthesis）はgeneric LCUとの差が不明確だった。その後、R0において有限Taylor平均を保存する非負adjacent-degree representationと、当該class内のnormalization最小構成Aを導いた。R1/R1.5の固定2-qubit toyではAのnative費用・合成bias・shotの相互作用が確認され、RA-RTEでは表現・合成精度・samplingの共同設計に進んだ。

G1では固定P3・7 prototypesの理想係数classが3変数の6頂点多面体であり、B2（ordinary/PTSC-K0/Aの混合）がその部分集合であることを確認。SoPlexの人工B2/B3型LPも8件PASS。ただし、登録LPの科学的結果ではない。G2はこの構造を用いた**小さな完全診断（明示した目的のみ）**であり、本研究の初めての核心に近い判別材料である。

## 3. G2で確認された内容

### 3.1 固定条件

- 2-qubit distinct-basis **controlled** finite $P_3$、$\widehat R=\tfrac34Q_0+\tfrac14Q_1$。
- $x\in\{1/8,1/4\}$、$\sigma=+1$。負符号側は同費用・誤差identity照合用であり独立試行ではない。
- 論理prototype：O0/O2/P2/P3/A0/A1/A2。保存合成精度 $10^{-3},10^{-4},10^{-6}$。
- ordinary、PTSC-K0、Aの3頂点で63 profile/$x$、追加J1/J2/J3を含む6頂点で252 profile/$x$。計504 profile。
- 新しい角度・native合成、circuit build、Hamiltonian/DF、LP最適化、trajectory/量子測定はいずれも0。
- 当初の入力検査失敗はrotation-first labelの誤読。**0 profile実施**の段階で修正し、scope・source修正・再実行の履歴を保存。診断process呼出しは2回、完了したprofile passは1回、focused testsは最終19件PASS。これを「一切retryなし」とはしない。消費済みscience one-shotのretryは0。

### 3.2 目的関数と正確な範囲

event係数 $c_i\ge0$、cost $C_i\ge0$、保存合成誤差上界 $e_i$ に対し、

$$
K=\sum_i c_i\sqrt{C_i},\quad s=\epsilon_{\rm axis}-\sum_i c_i e_i,\quad \Phi=(K/s)^2.
$$

固定dictionary、ideal degree equality、affine bias bound、他資源のcapなし、理想実数probability、という条件の**二次モーメント×費用の診断目的**。$\Phi$は実際のfinite-confidence総T/CX/1Q costでも、元RA-D0の登録B2/B3 boundsでもない。$e_i$は保存値の $2\delta_i$ を使用する。

| x | 指標 | 既存3頂点の最良 | 6頂点側の最良 | 6/3の値 | 解釈 |
|---|---|---|---|---:|---|
| 1/8 | T | A | J1 | 0.9961846242 | 約0.3815%差、未達成infimum |
| 1/8 | CX | PTSC-K0 | PTSC-K0 | 1.0000000000 | 追加差なし |
| 1/8 | 1Q | A | J1 | 0.9964092198 | 約0.3591%差、理想real proposal |
| 1/4 | T | A | J1 | 0.9847924627 | 約1.5208%差、未達成infimum |
| 1/4 | CX | PTSC-K0 | PTSC-K0 | 1.0000000000 | 追加差なし |
| 1/4 | 1Q | A | J1 | 0.9858219778 | 約1.4178%差、理想real proposal |

これらのstrict interval差は**限定モデル内での数学・算術上の差**として採用する。外部誤差・synthesizer依存性・新サンプリングの取得費用に対するrobust性を主張してはいけない。T、CX、1Qの最適proposalは一般に異なるため、同一の運用回路が全資源で勝ったのではない。

### 3.3 Tのzero-cost問題

全eventのcostが正なら、既知ISの二次モーメント費用最小のproposalは $\pi_i\propto c_i/\sqrt{C_i}$。しかし正係数のzero-T eventとpositive-T eventが混在すると、理想最小値は**infimum**としてしか達成できない。zero-costへ確率を集めて極限に近づけるほどpositive-cost側の確率が消え、重みの二次モーメント $m_2$ と最大重み $L$ が増大する。有限shot cap、Bernstein range、他資源の正のコストやstate preparationを戻さず、$\Phi_T$の改善をT資源削減と呼べない。人工的に小さな正費用を足して結論を作るべきでもない。

### 3.4 1Qのfinite-confidence解析的予測

1Qでは全eventが正費用でideal IS lawが存在する。native 1Q用proposalに旧R1のshots会計と一回のRe/Im readout合計5 gatesを戻した報告値は：

| x | 構成 | shots/axis | two-axis 1Q予測 |
|---|---|---:|---:|
| 1/8 | A | 811,506 | 465,382,325.19 |
| 1/8 | J1 | 831,338 | 463,413,930.45 |
| 1/4 | A | 1,022,850 | 468,173,605.23 |
| 1/4 | J1 | 1,115,481 | 461,948,987.57 |

J1はshotsが多いにもかかわらず、この**解析的**予測では総1Qが約0.4%と約1.3%低い。ただしnative-1Q最適proposalをreadout込みで再最適化しておらず、dyadic law、exact mean、joint resource capsの独立certificateはない。単一toyで結果後に1Qだけを主要実証指標へ昇格させることはpost-hocの選択となるため、しない。

### 3.5 線形価格による機構の説明

G1の係数多面体では、prototypeへの非負linear price $\ell_g$ を掛けて

$$
F=F_0+\alpha s+\beta r+\zeta b.
$$

既存三頂点と追加三頂点を比べると、追加構成のstrict利益が存在する必要十分条件は

$$
\min\{0,\mu\zeta,\alpha+\beta\}<\min\{\alpha+\zeta,\alpha,\beta\},\quad \mu=(x^2+2)/(x^2+6).
$$

G2では同precisionのIS価格で18条件の符号を確認し、T/1QにおけるJ1選択の原因を分析した。これは局所的なmechanism explanationとして価値がある。ただし最終bias分母、range、other capsは非線形・追加制約なので、この符号条件だけでfinite-confidenceの勝者を証明しない。

### 3.6 数学的完全性の実際の射程

G2の独立監査は、理想係数classの6頂点、affine bias、非負linear $K$、他資源capなしという条件下で、$K/s$の最小がpure precision profileに存在することを一般的に証明している。したがって同モデルで252/$x$の列挙は完全。これは特殊な問題を有限に閉じる点で有用だが、以下には延長できない：

- 保存済みnumerical K3のdyadic membershipやouter/inner差。
- Bernstein rangeまで含む全proposalの最小、整数shots、複数resource caps。
- zero-costでinfimumが達成されない状況の有限law。
- 追加dictionary、Pauli word collection、cancellation-aware表現。
- 実際のPR wrapperやlarge-DF、独立条件へのtransfer。

## 4. 科学的解釈

**確立した主張**：同じ既存event tableを使用し、既存3構成と新6構成に同等のIS freedomとprecision freedomを認めても、ideal linear-fractional目的では新J1に小さな厳密差が存在する。

**確立していない主張**：(a) 有限T countの改善、(b) multi-resource Paretoで実現可能な新法、(c) return/CTSを含む既知法より優れる、(d) state/DFに対する低費用性、(e) operator-realizationの一般新規性、(f) oracle-freeなcost取得法、(g) 漸近的計算量の優位性。

この差は数学的には非零だが、1.5%以下の理想値だけから**独立論文の主要method claim**を支えるのは難しい。過去の恣意的5%/10%閾値を引き継ぐ必要はない一方、精密なgate synthesizerと選択されたtoyに依存する0.4–1.5%の差を、実務的重要性や一般性の証拠と取り違えない。

## 5. 新規性と強い先行研究

- Cugini–Atif–Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms* (2026), [arXiv:2603.13495](https://arxiv.org/abs/2603.13495)。固定protocolの回路cost×二次モーメント最適化は既知。ISを使うこと自体に独立新規性はない。
- Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum 5, 040352 (2024), [DOI](https://doi.org/10.1103/PRXQuantum.5.040352)。辞書と確率／quasiprobabilityを凸最適化で設計する一般論は既知。ただし同論文の主要対象はchannel/processであり、現在のcoherent first operator momentとcontrol phaseの比較条件は明示する必要がある。
- Peetz–Smart–Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026), [DOI](https://doi.org/10.1038/s41534-025-01168-w)。CTS/Pauli coefficient collection・channel-oriented simulationは強い対照。distinct-basis 2-qubitでPauli展開可能なのでI1情報を無条件に排除しない。取得費用と同targetのcontrolled realizationをそろえる。
- Zhao–Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021), [DOI](https://doi.org/10.22331/q-2021-08-31-534)。Taylor／LCUの構造的な簡約・相殺は既知の重要な出発点。R0に保存済みのreturn恒等式を、そのまま新手法とはしない。
- 元のpartially randomized Hamiltonian simulation / RTE、PTSC、通常PFに関する既知構成はR0.5 prior-art auditで確認済み。一般involutionとI0 accessの制約は特徴だが、同じtargetのgeneric LCU全体の最適性は否定できない。

**独立論文として狙い得る差分**は、degree-structured first-moment表現の新しい構成と、用いる情報・費用を制御した再利用可能な選択法、もしくは一般次数に拡張した非自明な定理と確かな実装利益。今回のJ1最適点だけで、ここまでのmethod contributionが確立したとは扱わない。

## 6. 既知return・CTSの位置付け

R0からG2へ引き継いだ同targetのidentity-returnで、$\chi=\sum p_i^2=5/8$では

$$
B_{\rm return}^2/B_A^2\approx0.9809287044\ (x=1/8),\quad
0.9287211847\ (x=1/4).
$$

これはそれぞれ約1.91%、7.13%の**normalization二次モーメント低下**であって、資源費用の減少率ではない。return zero-degreeの新しいrotation ratioは $3067/24456$ と $763/3012$。同2 labels×2 x×3 precision、計12件のstrict phase-preserving event cost/error/IRが未取得。off-diagonal $O2$ の既存列は再利用可能。returnの条件付きindex sampling／取得費用も数える必要がある。これらを省いてJ1の実用上の優位性は主張できない。

CTSについて、既存の別contextでのPauli結果を同一toyのcostと偽らない。一方で今回の2-qubit toyはPauli展開可能なため、「I0一般involution classなのでCTS対照は無関係」とするのも不公平。同一対象でのPauli coefficient acquisition/native costが未取得のまま、新methodの最終主張へ進めない。

## 7. 進行方向の選択

| 方針 | 採否 | 理由 |
|---|---|---|
| 全v4・大規模登録B2/B3 LPへ直行 | **NO-GO** | 理想差が小さく、zero-T feasibility・既知return/CTS不明。実装だけ先行する情報価値が低い |
| J1/finite IS の実現可能性を保存値から狭く判定 | **GO** | 中心仮説に直接答え、未使用データの取得なし。最も安価に反証できる |
| return12実装費用を最初から無条件取得 | **保留** | J1有限lawが不成立なら先に取得する意味が弱い。次工程内の条件付き実施なら可 |
| higher-order・新angle grid・DF/分子へ拡大 | **NO-GO** | 研究帰属・最小機構が確定していない |
| R0/G1理論成果を文書として保持 | **GO** | 狭いが証明された成果。論文の独立新規性は未確定 |
| B3でなくreturn-aware／cancellation-awareへ方向転換 | **保留** | 正規化の大きな改善機構はあるが、単純return自体は既知。新method差が必要 |

## 8. 次のCodex作業：一括・段階条件付き

**担当：Codex。目的：G2の差が実装可能な方法を支持するかを最小費用で判別し、既知対照の不足を閉じる。** 全てdevelopmentであり、新held-out確認ではない。科学的設計条件を結果に応じて勝手に変えない。

**Phase A（先行・保存値だけ、必須）**

1. x={1/8,1/4}、旧固定P3/controlled context、旧21 columns/x・3 precisionのsource identityを守る。J1と同じ情報・cost/errorを使える旧B2/A/PTSC側を、同等のsampling/precision自由度の下で評価する。旧B2全体の最適性を証明できない場合、比較が限定されることを明記する。
2. 正確なfirst-moment、controlled relative phase、finite-support probability、$m_2$、最大重み$L$、Bernstein confidence、整数shots、shot cap、readout/state preparationの適用範囲、T/CX/1Q費用とworkspaceを整合させる。得られないnumerical certificateはmissingとし、実装可能と偽らない。
3. T=0イベントについては正のproposalとfinite nの実現可能点を構成するか、探索範囲内のinconclusiveを示す。極限値への任意の微小cost置換を禁止する。
4. 1Qの報告済みanalytic advantageを、roundingおよび他資源費用を含む有限lawで検査する。T/CX/1Qを恣意的に合算した新たなpost-hoc scalarで研究GOを作らない。今回見えた1Qだけを事前登録済みのprimary結果だったと呼ばない。
5. 同一登録x上のpost-hoc結果として、条件付きの具体的なfeasible J1と公平な既存対照の費用差、または再現可能なblockerを返す。旧science/v3/G1/G2のresult・source・consumed marker/STOPは不変。

**Phase B（Phase Aが実行可能な非自明差を支え、return比較が科学判断を変えると見込める場合のみ）**

- 既知returnの同target controlled実装を最小の12 event条件から取得する。新しい辞書・角度探索、再最適化範囲の拡大はしない。費用・error・phase・workspace・sampling/acquisitionコストを正式に揃える。exact returned lawと既存event再利用の同一性を確認する。
- 戻したreturn comparatorにJ1/A/PTSCと同等のprecision/IS調整の機会を与え、少なくとも同一taskでのfeasible budget comparisonを行う。returnが強い場合、B3の利点を探すため別xやprecisionを追加しない。
- Phase Aで差が消えるかfinite feasibilityを解決できない場合は、Phase Bを自動実行せず、結果と欠落理由をGPTへ返す。新しい科学対象を選ぶ判断はGPTへ戻す。

**実行上限・技術裁量**：条件、target、比較classは固定。具体的なoptimizer/rounding/独立検証実装はCodexの裁量。Phase Bの対象は最大12 returned event条件と保存済みO2の再利用に限定。元の55k/111k LP grid、G1の再実行、別geometry、new Hamiltonian、large DF/GPU、通常のPR scienceへ無許可進行しない。結果、変更したsource、実行回数、取得・失敗証明をcommit/pushしmandatory STOP。

## 9. 次回GPTレビュー（G3）の必須判断

1. **有限law**：J1のideal IS差が、整数sampling/strict mean/confidence/multi-resource capsで実装可能か。Tのinfimumを達成済みと偽っていないか。
2. **fairness**：B2/A/PTSC側の同等のsampling/precision freedomと、特定のresource primary選択のpost-hoc性を考慮したか。
3. **strong baseline**：return（および最終的にはCTS）に対し、同target・同情報cost・同誤差規則で何が残るか。
4. **情報と有用性**：改善量は証明/数値不確かさ、synthesizerの離散性、追加古典計算・実装複雑さに比べて重要か。ゼロ以上のtiny differenceを論文の独立主結果へ自動昇格させない。
5. **研究採否**：明確な資源/選択則の差が残るなら対象を小さくしたprospective実回路検証を検討。差が消えるならR0/G1を理論noteとして整理し、現B3主線を縮小・停止。未決なら追加計算の情報価値を評価してから一度だけ設計。negative resultを一般no-goにしない。

## 10. GO/STOPと論文化の着地点

**本研究へ継続**：同条件の実装可能J1が、優遇されない既存対照に対して意味のあるPareto資源点を作り、その因果機構（degree coupling解放とcost/bias構造）を説明できる。返される費用表や最終選択法がoracle-freeならより強い。結果を見た後の材料で仮説を変えた場合は探索的結果と明記し、新しい独立条件で検証する。

**限定理論・mechanism note**：R0 restricted theoremとG1 polytope等の数学が残るが、J1のfinite実装改善が小さい／脆い、known returnに支配される、新たな再利用可能な選択法がない。この場合も一般のRTEやrepresentation再配分を否定しない。

**現行B3主線STOP**：強い同情報比較で、差が既知IS/returnの調整に吸収される、またはmethod deltaが固定toyのsynthesizer-notchにしか現れない。設定追加でpositive例を探さず、この仮説を閉じる。Track B全体の別研究候補はGPTで再評価する。

論文は「LPで最適化」「平均を保存」「tiny toyで1Qが1%安い」だけでは主張が弱い。目指すのは、**degree-awareなcoherent first-moment representationの構成・情報費用・選択条件・実装資源・失敗域**を閉じた成果である。短時間finite-P3の結果を基底状態エネルギー推定/QPE全体の資源改善と同一視しない。

## 11. 判断履歴と変更理由

- G1後の暫定的な「v4 sourceへ進む」から、ユーザーの指摘を受けてGPTによる価値判断を先行させた。
- G1科学レビューでは、強いreturn・IS対照と、252/$x$ pure-profile診断を先に実行する方針に変更。
- G2ではその診断が完了し、degree-local J1に非零のstrict ideal差が残る一方、Tは未達成infimum、1Qは理想実数law、CXの追加差は0、return/CTS native比較は未取得と確定した。
- **したがって**、巨大なLPを開く情報価値はさらに低くなり、「限定finite-realizability確認→条件付きreturn比較→G3」で科学判断を行う方が合理的になった。単にテストPASSが増えたから進行を変更したのではない。

## 12. 出典

**Repository 固定証拠**：
- [G2 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_saved_diagnostic_handoff_20261009.md)
- [G2 independent math audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_independent_math_audit_20261009.md)
- [G2 result JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/artifacts/track_b_g2_saved_diagnostic/2026-10-09/result_v1.json)
- [G2 return missing inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/artifacts/track_b_g2_saved_diagnostic/2026-10-09/known_return_rows_v1.json)
- [G1 scientific review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/research/track_b_G1_scientific_review_20261009.md)
- [R0 mathematical proof](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/672d6bc667eaa7b9ca4979b012f1530499d701b8/docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md)

**先行研究の一次資料**：
- Davide Cugini, Touheed Anwar Atif, Yigit Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495 (2026): https://arxiv.org/abs/2603.13495
- Bálint Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum 5, 040352 (2024): https://doi.org/10.1103/PRXQuantum.5.040352
- Joseph Peetz, Scott E. Smart, Prineha Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026): https://doi.org/10.1038/s41534-025-01168-w
- Qi Zhao, Xiao Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021): https://doi.org/10.22331/q-2021-08-31-534

**再現・証拠区分**：このレビューは公開済み固定repoの文書・算術結果と上記文献による科学的判断である。新しいsource実装・元science one-shotの再実行・独立の登録最適化・controlled synthesis・DF/分子検証は行っていない。新規の論文優先性を全関連文献について網羅的に保証するものでもない。
