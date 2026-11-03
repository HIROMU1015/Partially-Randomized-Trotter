# M1-S：RPE適用条件・総回路費用の科学的設計レビュー

- **レビュー日**：2026-10-17（日本時間）
- **対象**：Track B／M1-S（gate共有±によるcoherent平均信号）
- **区分**：限定科学レビュー（Level 2）
- **開始承認**：ユーザー「レビューを開始して」。直前にM1-Sの限定継続方針を承認済み。
- **状態**：`SCIENTIFIC_DESIGN_REVIEW_COMPLETED / NARROW_V1_DESIGN_RECOMMENDED_NOT_ADOPTED / NO_NEW_SCIENTIFIC_COMPUTATION / MANDATORY_STOP`
- **位置付け**：採択済みの「M1-Sを限定継続する」方向の下で、Decision PacketのD2–D10を科学的に検討する。新規実験の正式採択、具体的な数値閾値の採択、科学計算の実行承認は含まない。

## 1. 結論と推奨

**結論：限定継続を支持する。直ちにmulti-round RPEの数値実験・native compileへ進まず、保存済み有限step複素信号に基づく位相予算・shot費用の限定screening（V1）を、第一の追加検証候補とする。**

最終的に目指す科学的課題は、対象エネルギーに対するRPEでの総資源改善である。しかし、今回の保存oracleには厳密固有状態・十分な対象固有重み・alias-free初期energy branchの認証がない。したがって、**直近の数値検証task A（保存raw-H信号の単一時刻位相）**と、**最終目標task B（対象エネルギーのmulti-round RPE）**を明確に分離する。

- **直近の選択**：task Aに対する、同じ標的・同じ位相誤差予算・同じconfidence・同じ統計policyの下での無補正phase/shot feasibility screening。保存済みcell全体を対象とする。これは既存結果の事後解析であり、独立検証ではない。
- **将来の選択**：task Bを目標として保持する。ただし、対象状態・energy prior・round列・全wrapper・native費用の証拠が不足しているため、今回Bの実行契約を採択しない。
- **追加検証の順序**：V0（V1の科学条件と実行範囲の固定）→別途実行承認後のV1→科学的判定→必要な場合だけV2（限定native wrapper費用）またはV3（不可欠な未取得信号）。すべてを自動的に順番実行する計画ではない。
- **主張の限界**：G3/G6でのvisibility改善は、phase biasの増加と併存する。現段階のpositiveは「条件付き有限step信号改善」であって、shot数の減少、ground-energy推定精度、native総資源改善、新規性を示すものではない。

今回のレビューでは、D2–D10の**科学的な推奨・先行して固定すべき事項・未決のまま残す数値**を区別する。推奨を採択済み契約へ自動反映しない。

## 2. 取得・確認した一次資料と証拠水準

### 2.1 固定GitHub原資料

|資料|固定commit|今回の確認範囲|
|---|---|---|
|[M1-S RPE Decision Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/870af3f65e3e31e19d2b002c556216c7340c06f0/docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md)|`870af3f65e3e31e19d2b002c556216c7340c06f0`|D1–D11、P1–P3／R1／S1／C1／G1–G2、SOURCE適合性、V0–V3|
|[次期検証契約候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/870af3f65e3e31e19d2b002c556216c7340c06f0/docs/tracks/algorithm_codesign/m1s_rpe_next_verification_contract_candidate_20261017.md)|同上|未採択条件・STOP・実行境界|
|[機械可読な契約候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/870af3f65e3e31e19d2b002c556216c7340c06f0/artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1/next_verification_contract_candidate.json)|同上|`open_scientific_choices`のnull、承認フラグfalse|
|[有限step pilot結果に対する前回科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md)|`e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b`|物理的解釈・保存結果・trade-off・次期条件の根拠|
|[finite-step完了報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md)|`025c5a90888b02b73d2815fbb10d6bd1d2823602`|108行完了・技術監査・未検証事項|
|[T1b理論SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/docs/tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md)|同上|coherent first moment、平均化条件、scalar rateの適用域|

既存SOURCEのうち、同じcommitの`src/trotterlib/rpe_resource_accounting.py`、`rpe_hadamard_interrogation.py`、`df_rpe_hadamard_compiled_cost.py`および`docs/rpe_resource_accounting.md`を静的に参照した。具体的に、`_hoeffding_shots`はHoeffdingの十分予算を計算するが、既存resource evaluatorはeffective eigenstate、単位unattenuated radius、既存RTEの保証など**M1-Sへ直接移せない仮定**を明記している。M1-S用の物理wrapper・cost sourceが完成しているとは扱わない。

指定branch `track-b-m1s-rpe-feasibility-conditions-20261017` の公開HEADは `870af3f65e3e31e19d2b002c556216c7340c06f0` と取得時に一致した。ただし、今回source／結果の全byte独立照合、保存済み108行の新規再集計、コードの実行は行っていない。Codexの156件static PASSと保全結果は**保存報告としての確認**であり、数値coverageや科学的GOを独立認証したものではない。

### 2.2 先行研究の確認

- Güntherほか、*Phase Estimation with Partially Randomized Time Evolution*、PRX Quantum **7**, 020332 (2026)、[arXiv:2503.05647](https://arxiv.org/abs/2503.05647)。既知の部分乱択Hamiltonian simulationとsingle-ancilla位相推定を扱う。今回のM1-S専用native費用保証を与える文献ではない。
- Kimmel–Low–Yoder、*Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation*、Phys. Rev. A **92**, 062315 (2015)、[原論文](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.062315)、[2021年Erratum](https://doi.org/10.1103/PhysRevA.104.069901)。Erratumは旧π/2スケールの成功判定の問題を訂正し、π/3スケールの判定を説明する。ここから今回の具体的費用定数・目的精度・confidenceを直接転用しない。
- Cugini–Atif–Subaşı、*Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*、[arXiv:2603.13495](https://arxiv.org/abs/2603.13495)（2026）。cost-aware sampling一般論は別の改善余地を示すが、今回のM1-Sに特定の最適proposalや費用効果が存在する証拠とはしない。

文献調査は本レビューの設計判断に必要な範囲に限る。網羅的priority・独立新規性の確定は行っていない。

## 3. 現在の結果から何が言えるか

保存結果のscopeはlinear H4 1.00 Å／STO-3G／8 spin orbitals／DF rank12、G3/G6/G9、`delta={1/8,1/16}` Ha^-1、`R={2,4}`、`T={1,4}` Ha^-1、shared ±／ordinary／independent ±である。zero controlを含む108行は、108個の独立物理instanceでも108回の量子shotでもない。

[前回レビュー]に照らし、G3/G6では両対照に対する絶対visibility改善が数値的に分解された。G9は絶対差が不分解である。同時に、とくにG3ではsharedのraw-H参照位相biasがordinary/independentより大きい。したがって、**visibilityが大きい方式ほど、実際のRPEで少ないshotで済む**とは推論できない。

- `phase_error_radius_rad`は**数値誤差の半径**であり、**物理的／アルゴリズム的なphase biasの許容量**ではない。
- raw-H基準の`phase_bias_rad`にはsplit-targetとのモデル差が既に含まれる。同じモデル差上界を再加算してはならない。
- 保存状態は厳密ground stateとして認証されていない。単一Eに対する位相保証へ直結できない。
- T1bの小step・有限減衰予算のsecular／scalar rate保証は、今回の有限Tの複素信号と同一の命題ではない。有限時間にphaseへの非secular寄与が残っても矛盾しない。
- sharedの概念的label数が同じでも、branch演算の実装費用は同じとは限らない。

これらの限定は本レビューでも変更しない。

## 4. 中心的な科学判断：Task AとTask Bの二層設計（D2）

### A：直近の判別可能なtask

**Task A：保存raw-H信号`z_raw(T)`の主値位相を、現在の有限step方式がどの程度のbiasと信号振幅で表現できるか。** 採択済みH4／state／law／既存Tだけを対象とし、有限shotに必要な位相余裕を条件付きで診断する。

Aを直近の第一候補とする理由は、保存資料からすでに必要な複素信号・位相bias・数値半径・対照が対応づくためである。Aはオラクルを実機の補正情報として無料利用することを認めない。開発データの**事後診断**であって、新しい未知instanceへの転移実証ではない。

### B：研究としての最終目的

**Task B：指定した対象固有エネルギー`E_*`を、multi-round RPEで要求精度・confidenceにより推定すること。** Task Bには、対象stateのスペクトル保証、alias-free energy prior、`T_k=2^k t_0`の全round、decoder、wrapper実装、費用が必要である。

保存された`T=1,4`をRPEの全round列と見なしてはならない。`T=2`を自動補完したり、将来の`K`や`epsilon_E`を勝手に選んだりもしない。AのpositiveをBのGOへ格上げしない。

**判断**：Aをfirst screeningとし、Bを継続研究のtargetに保持する。Bへ進める条件をV1結果を見て後付け変更するのではなく、今回の文書で識別条件を定め、詳細な数値基準はユーザー採択の別契約に残す。

## 5. 精度・confidence・位相誤差の科学的契約（D3–D5）

### D3：誤差の意味

Bの主要な精度意味論としては、まず**高確率の絶対energy誤差**

\[
\Pr(|\widehat E-E_*|\le\varepsilon_E)\ge 1-\alpha_{\rm tot}
\]

を推奨する。これはRPEのbranch成功条件と対応し、failure時損失を別途定義しなければならないRMSEよりも、現時点では保証内容を明確にしやすい。**現時点で`epsilon_E`・`alpha_tot`の数値を採択する根拠はないため、未決のまま残す。**

Aではエネルギー誤差を名乗らず、各保存時刻の円周位相誤差予算`gamma`と測定統計誤差用の残余`q`を使用する。既存pilotの信号半径`2^-24`、phase半径`2^-18` radを新RPEの許容biasや`epsilon_E`へ流用しない。

### D4：時間／branch

将来Bで採用するなら、`T_k=2^k t_0`、初期energy interval、`+iTH`の符号と測定軸規約、branch decoderを固定する。単なる先験区間の幅だけでは初期branchの実装可能性を保証しない。現packetのnearest-consistent-branch候補に従い、**選んだdecoderに対する保守的な十分条件**として

\[
 b^+_{mjk}+q_{mjk}\le\gamma_k<\pi/3,\qquad
 \gamma_K/T_K\le\varepsilon_E,
 \qquad\sum_{k,a}\alpha_{k,a}\le\alpha_{\rm tot}
\]

を検討できる。ただし、π/3は全decoderに共通する必要限界ではない。今回、`t_0`、round数、先験区間、`gamma_k`の具体的数値は採択しない。

Aの既存Tはあくまで既存測定点であり、multi-round Bの新しいhorizonへの拡張承認ではない。

### D5：bias routeと情報利用

**直近Aのprimary routeは、無補正のraw直接routeとすることを推奨する。** 保存された`TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H`を円周距離の条件付きbias上界に変換し、すでに含まれているモデル差`eta_j(T)=|T|B_j`を二重計上しない。

\[
 b^+_{\rm method\to raw}
 \le \min\{\pi,|\widehat b_{\rm saved}|+r_b\}
\]

という変換は、既存の誤差coverageとbranch条件の成立時に限る。実機で既知でない正確なraw-H位相を使って配分・補正を最適化する方式とは区別する。補正ルートは将来の研究候補として残せるが、補正の取得情報・bias保証・前処理費用が未確定なのでprimaryには採用しない。

Bではさらに`arg z_raw(T)`と`E_*T`の相違を会計する。対象固有stateへの厳密重み`g>1/2`を**独立に保証できる場合に限り**、disk幾何から

\[
 |z_{\rm raw}|\ge 2g-1,\qquad
 d_{\mathbb S^1}(\arg z_{\rm raw},E_*T)
 \le \arcsin\bigl((1-g)/g\bigr)
\]

という十分条件を利用できる。現在の保存oracleにそのgを認定できないため、これは**条件式の提示**であって結果ではない。state fidelity/overlap、Hamiltonianモデル差、実装誤差はそれぞれ別会計とする。

## 6. Shot設計と位相余裕：今回の中心的な改善条件（D9）

各shotが新しいIID乱択labelを引き、boundedなHadamard X/Yの±1結果を出すとき、Decision PacketのHoeffding型十分予算を採用可能である。各軸の失敗確率を`alpha_{k,a}`、真のmean信号振幅下界を`A^-`、統計に残す位相余裕を`q`とすると、

\[
 0<q<\pi/2,\quad
 t=A^-\sin q/\sqrt2,\quad
 N_{k,a}\ge\left\lceil\frac{2\log(2/\alpha_{k,a})}{t^2}\right\rceil.
\tag{S1}
\]

これは最小必要shot数ではなく、**その仮定と統計policyの下での十分shot数**である。同一乱択回路を複数shotで再使用する場合には無条件に適用できない。古典的なtrajectory sample数をquantum shot数と重ねて掛けてもならない。

**理想平均信号と実際の測定信号の区別は特に重要である。** 旧pilotの保存値はexact-HDを含む理想coherent mean `z^0`に対するもので、物理的なstate-preparation、finite synthesis、SPAMを含む`z^p`の保証を与えない。V1で保存振幅・biasを使ってS1に代入する場合は、必ず**理想信号モデル上の仮説的十分shot forecast**と表示する。物理shot保証を名乗るには、別途 `|z^p-z^0|<=d_impl` の独立した根拠から振幅下界と位相上界を転送し、その誤差を払う必要がある。

### 独立した紙上の判別式：visibility利得とbias負担の交換

同一task、同一`gamma`、同一軸別confidence policyの下で、補正をしない方式mについて最大限使える位相余裕は原則

\[
 q_m=\gamma-b_m^+,\qquad 0<q_m<\pi/2,
\]

である（これを超えないよう数値・coverageの誤差を払う）。整数丸めと軸差を無視した同一policyのcontinuous shot proxyなら、

\[
 \frac{N_{\rm sh}}{N_{\rm cmp}}
 \approx
 \left[\frac{A^-_{\rm cmp}\sin(\gamma-b^+_{\rm cmp})}
 {A^-_{\rm sh}\sin(\gamma-b^+_{\rm sh})}\right]^2.
\tag{S2}
\]

したがって**shot削減の可能性を考える際の中心量は`A^-`単独ではなく**、

\[
 \boxed{\quad A^-_m\sin(\gamma-b_m^+)\quad}
\tag{S3}
\]

である。sharedのvisibilityが高くても、bias増加が位相余裕を使い切ればshot利得は消える。この関係は既存PacketのR1/S1からの**紙上代数的帰結**であり、今回保存数値を新たに代入・ランキングしていない。

`b_m^+\ge\gamma`なら**選んだ無補正・十分条件routeには正の統計余裕が残らない**。これは任意の補正推定器の不可能性、M1-S一般のno-goとは異なる。現在の`b^+`が数値coverageに条件付きであることも維持する。

primary statistical policyは**全方式で同じfresh-IID-per-shot、同じper-axis confidence配分規則、同じ丸め規則**とすることを推奨する。方式ごとの情報上の非対称な最適化は許さない。より精密な分散適応・shot配分最適化は次の段階に残せるが、それを暗黙のbaseline無料最適化にも、新方式専用の恩恵にも扱わない。

## 7. 候補集合・baselineの公平性（D6）

**直近Aの主比較**（理想coherent mean、非物理shot forecast）：保存済みH4で、`G3/G6/G9 × R{2,4} × delta{1/8,1/16} × T{1,4}`の全非零cellにおいて、shared ±対ordinary、およびshared ±対independent ±を、それぞれ同一law・時間・数値誤差会計で対応させる。ゼロ時間のcontrolはcorrectness確認として保持する。G9が現在未解決であることを理由に落とさず、精度内不分解を`UNRESOLVED`として扱う。

以前のG3/G6の改善を見てからの事後比較であるため、後で得られる条件付きpositiveを独立held-outの証拠にしてはならない。将来Bの実用claimに進むなら、既存部分乱択RTEやdeterministic endpointを含む強い対照に、同じ精度、state情報、compilation最適化の機会を与える。

V1では新分子・新geometry・新DF prefix・新精度・新roundを追加しない。また、結果に合わせてcellの最良点を選び直さない。全cellに同じ判定規則を先行適用する。

## 8. 回路費用・状態準備・classical費用（D7–D8）

### D7：物理実装と費用指標

直近のV1では、**native回路の利益を認定しない**。shot構造の成立可能性を先に調べる。次のV2が必要になった場合に、fault-tolerant logical T gate-workを優先的なscalar resource候補とし、CX、1Q、depth、workspaceを独立した副指標／Pareto軸として報告することを推奨する。これは過去Track Bの資源研究との接続に適しているが、**今回Tだけを採択して他資源の不利を消す趣旨ではない**。

shared ±の2分岐のbranch指数演算数は、native価格の2倍を意味しない。例えば各label位置の対称control演算は、数学的には`exp(-i theta Z_a\otimes P)`のように表現できるが、合成価格、Clifford network、routing、HD block、control位相、state準備を含む**全wrapper価格**は未取得である。

source上の`+iTH`と旧RPE wrapperの`-iTH`の符号、Hadamard X/Y測定、negative branchの同順sign reversal、identity相対位相は、将来のadapter実装の必須照合点とする。HDのexact指数を、費用ゼロの物理primitiveとして扱わず、DF sweep/PFなどの候補を別途誤差付きで会計する。

### D8：状態準備・固定費のscope

**end-to-end claimをする場合には、各shotで発生する状態準備・readoutの資源を含むことを原則とする。** 状態準備の回路が方式間で同じでも、必要shot数が違えば総費用に差が残る。

\[
 G_m^{(g)}=F_m^{(g)}+\sum_{k,a}N_{mjk,a}
 (C_{mjk,a}^{(g)}+P_{mjk,a}^{(g)})
\tag{G1}
\]

を同一単位のmetricごとに用いる。ただし状態準備を除外した`no-preparation`感度指標も、end-to-endと明確に区別して併記できる。古典setup秒とlogical T数は変換規則なしに加算しない。前処理価格を無測定のままゼロと報告しない。

G1はshot数Nが設計上の決定値、C/Pが対応する1 shot当たりの期待費用として意味づけられる場合の会計式である。shotごとのrandom native費用の分布や、decoder依存の停止則を導入する場合は追加の定義を必要とする。

### Break-evenの必要な区別

単一時刻・固定費なし・同一統計policy・整数化無視・軸単価対称という**限定proxy**なら

\[
 \frac{C_{\rm sh}+P_{\rm sh}}
 {C_{\rm cmp}+P_{\rm cmp}}
 <\left(\frac{A^-_{\rm sh}\sin q_{\rm sh}}
 {A^-_{\rm cmp}\sin q_{\rm cmp}}\right)^2
\tag{G2}
\]

がそのproxy内のbreak-even条件である。この条件式の右辺を初めて評価する数値解析も、左辺のnative値の取得も今回行っていない。**二つの十分予算の比較は真の最適shot数や実機wall-timeの優位性の証明ではない。**

## 9. 改善判定、unresolved、STOP（D10）

判定を次の階層に分けることを推奨する。

1. **`TASK_OR_EVIDENCE_UNRESOLVED`**：target位相、bias/visibility下界、confidence、stateの保証、初期branch等が欠落しており、そのrouteで適格性を判断できない。数値を推測しない。
2. **`PHASE_BUDGET_NOT_CERTIFIED_UNDER_SELECTED_ROUTE`**：選んだ無補正方式と上界で`b^+ + q <= gamma`を示せない。十分条件が不通過という意味であり、実際に失敗したこと・一般的no-goではない。
3. **`IDEAL_SIGNAL_PHASE_AND_SHOT_DESIGN_FEASIBLE_CONDITIONALLY`**：対象taskと保存数値coverageの条件の下で、理想平均信号に対して正の`q`、正の`A^-`、共通Hoeffding十分予算を構成可能。物理実装・SPAM・native回路費用の検証ではなく、資源改善はまだ判定しない。
4. **`POTENTIAL_RESOURCE_REGION`**：上記に加え、G2のような明示的な費用比の許容領域が見つかる。ただし価格未取得ならresource GOではない。
5. **`CONDITIONAL_RESOURCE_ADVANTAGE_IN_SELECTED_MODEL`**：後の承認済み評価で、同じタスク・同じ費用単位・同じcoverageのもと、実行可能な方式間の費用が所定の不確かさを超えて分離する。全既知法への優位・実機高速化とは呼ばない。

相対的な利得を科学的に頑健と認定するなら、費用の上下界を伴う`G_sh^+ < (1-mu)G_cmp^-`等を事前定義する必要がある。しかし、**`mu=0`、5%その他の具体的materiality数値は、今回の原資料と研究目的だけでは決定できないため採択しない**。欠損を0、信頼半径を誤差なし、upper-vs-upperを頑健優越と読み替えない。

検証結果がnegativeでも、限定されたtask・誤差保証・baseline・候補集合へのnegativeと明記する。positiveでも、新規実験や別分子への拡大を自動許可しない。

## 10. D2–D10の科学判断一覧

|ID|本レビューの推奨|未決／次の契約で固定すべきこと|
|---|---|---|
|D2 推定task|**直近A／最終B**の二層。AでBを代替しない|Bの対象E、state/gap・prior、実行可能性|
|D3 誤差・confidence|Bは高確率絶対energy誤差を主要意味論候補、Aは位相誤差。統計policyは共通化|`epsilon_E`、`alpha_tot`の値、Aの`gamma`|
|D4 時間・branch|Aは保存Tのみ。Bはdyadic列・decoder・alias-free保証を別設計|`t0`、`K`、energy interval、branch初期化|
|D5 bias情報|Aは無補正raw直接routeを優先、二重計上禁止|state overlapの保証、補正の情報・価格、実装誤差|
|D6 cell・baseline|Aは保存全非零cellと2内部対照、G9も保持。Bは強い対照が必要|Bのmatched RTE/deterministicの具体的集合、held-outの要否|
|D7 費用|V1はshot feasibilityのみ。将来V2ではlogical T-workと多資源併記を候補|native gate set、HD近似、合成精度、全wrapper測定scope|
|D8 準備・古典費用|end-to-endではshot×状態準備費用を含む。classicalは別単位|状態準備の品質・実価格、固定費、並列化・wall-time定義|
|D9 乱択・shot|fresh-IID per shot／同一Hoeffding・axis confidence ruleを候補|具体的配分、再利用policyを使うなら別証明|
|D10 判定|位相適格性→shot feasibility→cost region→conditional resourceに段階分け|materiality、費用半径・coverage、失敗分類の固定|

**判断の優先順位**：D2/D5を先に確定せずにD3/D4の数値を埋めたり、native価格の見通しなしにD10の資源閾値を選んだりしない。今回の推奨は研究設計であり、ユーザーの正式採択を代行しない。

## 11. 次の最小作業と必要な承認

### 11.1 Codexに委ねる次の技術作業（V0）

ユーザーが本レビューの**「直近Aを用いる事後位相・shot screening設計」**を採択した場合、Codexは以下を一件の技術作業として完結できる。

1. 固定レビュー・Decision Packet・保存result/CSV・既存SOURCEのidentityを確認し、**全保存cellを対象とするV1契約候補を作る**。G3だけを優先してG9を抜くなどの結果依存の除外をしない。
2. 位相bias上界、visibility下界、`gamma`/`alpha`の科学的入力field、S1・S3・G2の適用条件、`UNDETERMINED`分類をmachine-readableに定める。今回推奨したno-correction・raw直接routeの意味を保持する。
3. フィールド未決の段階では数値閾値を勝手に代入しない。角予算・confidence・費用モデル・source identity・resource capsについて、ユーザー／科学判断が必要な項目を明記する。
4. 既存結果を読み取る以外の**新しいshot見積り、順位表、位相再解析、数値閾値探索、native compile、RPE simulationを実行しない**。SOURCE準備と静的・mock・no-run preflight、独立branchでのcommit/pushのみ実施する。
5. 旧v4/v5、finite pilot各試行、SOURCE、raw結果、authorization、grant、消費済marker、親／他worktree、mandatory STOPを保全する。

V0後、採択可能な数値条件が具体化すれば、新規科学再解析であるV1について**SOURCE/対象/予算/STOPを固定し、別途の明示的な実行承認**を得る。実行後はmandatory STOPし、結果の科学的解釈が重要な場合は開始承認を得てGPTへ戻す。V1がpositiveでもV2/V3へ自動移行しない。

### 11.2 V1より先に解決すべき「研究判断」

次の技術文書で実際に数値を使用するには、最小限、**単一時刻phase task Aを採択するか、位相許容幅／confidenceを具体的に固定する根拠は何か**が必要となる。対象energyのabsolute precisionを今回の保存Tから推測しない。値が未決のままなら、V1を`NOT_EXECUTABLE`としてSTOPさせる。

Bを第一の直接検証にしたい場合は、本レビューのA先行推奨を採らず、energy priorとstate保証の回収が先になる。その場合も旧H4結果から未取得値を推測しない。

### 11.3 レビューを新たに必要とする条件

研究主線をB直接検証へ切り替える、未検証の位相補正を導入する、state保証の新定理を採択する、比較の科学的公平性や主要評価指標を実質変更する場合は、レビューの要否を改めて判断する。V0の単なる機械契約・guard・文書修正では同じレビューを繰り返さない。

## 12. 主要な反対仮説・限界・判断の確信度

**代替仮説1：** G3/G6のvisibility利得は本質的にphase marginの減少で相殺され、同じtaskでは統計利得にならないかもしれない。S3を見ずに結論できない。

**代替仮説2：** 位相が成立しても、共通の状態準備費用とdirectional wrapperのnative合成費用により、総T-workはかえって増えるかもしれない。費用情報が未取得の現状では否定も肯定もできない。

**代替仮説3：** 使用できるstateのenergy overlap保証・初期energy priorが無く、単一時刻raw信号での前進がBへ移送できない可能性がある。この場合、Aを進めたこと自体は診断価値を持つが、Bについての主張は限定される。

**代替仮説4：** より適切なbias補正・statistics・baselineで評価が変わる可能性がある。今回は最初から広い新アルゴリズム探索を立ち上げず、無補正の具体的な比較で一旦識別する。

|論点|現時点の確信度|
|---|---|
|原資料と旧pilot結果のscope、設計式の意味|高（取得した資料・採択された条件付きモデル内）|
|A先行がB直行より小さい承認単位になること|中〜高（既存資料依存・研究上の推奨）|
|既存式からのS2/S3の紙上導出|高（明記した共通統計policy・連続近似の条件下）|
|実際にshotが減るか|未判定（新規数値再解析を未実施）|
|native T／CX／depthやRPE総資源が改善するか|未判定（物理wrapper／費用／state保証が不足）|
|新規性・一般化・投稿可能な主張|未確定（非網羅の文献確認）|

## 13. 承認、STOP、版管理

- **本レビューで実施したこと**：固定原資料と先行研究の限定確認、数式の紙上検討、D2–D10の科学的推奨、V0/V1/V2の進行境界の定義、詳細Markdown記録作成。
- **実施していないこと**：保存済み科学数値の新規再集計・shot forecast・bias再採点・候補ランキング・native compile・Hamiltonian／Pauli／signal／matrix計算、SOURCE・GitHub・正式契約の変更、計算authorization・grant・markerの発行。
- **正式採択済み**：ユーザーがM1-Sの「限定継続」方向を承認したこと。
- **今回の推奨であり未採択**：Aを最初のscreening taskとし、Bを最終的なRPE taskとして保持する二層設計、no-correction raw直接route、共通shot policy、費用評価の優先順位、および数値を伴う新契約。
- **停止状態**：既存のmandatory STOPを維持する。V1の数値再解析やV2のnative compileは別途の実行承認が必要。旧結果・source・ledger・authorization・grant・markerを保全する。

**最終判定：`NARROW_V1_FEASIBILITY_DESIGN_RECOMMENDED_NOT_ADOPTED / ENERGY_RPE_END_TO_END_GO_UNDETERMINED / SCIENTIFIC_EXECUTION_NOT_AUTHORIZED / MANDATORY_STOP`。**
