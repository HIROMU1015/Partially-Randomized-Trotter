# Digital event lawと量子費用の比較

[入口](README.md) / [break-even](break_even_conditions.md) / [不足証拠](missing_cost_evidence_and_execution_plan.md)。SOURCE読解と有限和の通常の整理のみであり、event列挙・期待値の新数値評価・回路生成は実施していない。

## 1. 全trialを分母にする

方式aの非zero eventをe、全trial空間での実digital確率をπ_a,eとする。raw word/groupなどの来歴が異なるeventは、同じunitaryでも別indexとして数えてよい。構造価格・費用は来歴を含めて評価する。

\[
z_a=\sum_e\pi_{a,e},\quad
w_{a,d}=\sum_e\pi_{a,e}c_{a,d}(e),\quad
C_{q,a,d}=N_a w_{a,d}.
\]

zはdigital受理率であり、理想Z=B_RA/B_o、valid parent条件付きA_uとは別。canonicalは正常モデルでz_a=1。FullはR/C zeroがあり、πの和はsubprobabilityとなる。accepted-onlyへ再正規化せず、全N weighted estimatorの分母を保つ。未知B_RA/Zを生成器に渡さず、全event辞書・reference oracleも無料取得としない。

z_a>0ならν_a,d=w_a,d/z_aと書けるためC_q=N_a z_a ν_a,d。これは恒等式で、未知の受理率や価格が測定済みという意味ではない。z_a=0ならw_a,d=0（有限・非負event価格）として扱い、0除算しない。**均一単価**c_dが認証されている座標だけN z c_dに簡約できる。word/child/angle依存のnative価格を一定と仮定しない。

## 2. SOURCE上のliteral IR

[provider][CP] `controlled_event_ir` と[G][G] `_event` の対応である。r_e=還元後word長、i=child、φ_e=元のgroup/parent degree mod4とする。

| 座標 | accepted eventあたりの構造 | 限界 |
|---|---|---|
| circuit実行数 | 1 | prefix zeroが量子前に判明し、実行層がskipする条件 |
| total logical CQ | r_e+2 | word forward r_e＋child forward1＋child actual adjoint1 |
| label別CQ | word中のlabel個数＋child labelに2 | control役割・forward/adjointの物理価格が等しいとは限らない |
| helper | H4、抽象RZ2、明示skeleton CX2 | CQ内部のT/CX/1Q/depthを含まない |
| phase | φ_e=2ならouter Z1、φ_e=0なら0 | raw word還元後も**元degreeのphase**を保持 |
| workspace | system以外にouter1＋helper1 | provider追加ancilla、state準備workspaceは未知 |
| σ± | RZの符号を反転 | 同じ取得sequenceのactual adjointを使い、scalar/global phaseを落とさない |

word CQは`reversed(event['word'])`順、child compute/uncomputeのcontrolはhelper、wordのcontrolはouterである。CX2は**skeleton-only**であり、総native CX2とは主張しない。RZ2も最終native T/1Q数ではない。literal IRをwhole-circuit最適化済み回路としない。

一般native座標dの価格を、例えば

\[
c_{a,d}(e)=c^{word}_{d}(e)+c^{child,fwd}_{d}(i)
+c^{child,adj}_{d}(i)+c^{RZ,+}_{d}(\theta_e)
+c^{RZ,-}_{d}(\theta_e)+c^{skeleton}_{d}(e)
+c^{prep/readout}_{a,d}(e)
\]

と分解できるのは、実装がliteral additive会計を採用する場合である。gate countと異なりdepthは重なり・依存関係・scheduleを要する。「直列実行したevent depthの和」「最大一回路depth」「parallel makespan」を区別する。peak workspaceもN z倍にしない。forward/adjointの価格・厳密位相・物理誤差をprovider証拠に結ぶ必要がある。

## 3. Ordinaryとの比較で静的に閉じる範囲

これは新normalizer/複雑性定理ではなく、同じproposalのSOURCE対応と非負有限和からの直接の整理である。条件は、同じp順序/x/m/K/H/η、同じroot/丸め・tie処理、登録query-model、fresh-IIDの法則である。Ordinary `_ordinary(0)`→group roots/normalized intervalsと、Full `order_intervals`は同じrootリスト・規格化・dyadic手順である。label lawも同じ。従ってraw prefix (l,w) の確率

\[
q_{l,w}=\widetilde q_l\prod_{j=1}^{l}\widetilde p_{w_j}
\]

は同じである。Fullはnonreduced wをzeroとし、reduced wではpacketのdigital threshold a_w（0≤a_w≤1）で受理し、child law h_i|wを引く。Ordinaryは同じraw prefixから全てeventを返し、childをp̃_iで引く。

\[
\pi_{F,w,i}=q_{l,w}a_w h_{i|w}\quad(w\ \mathrm{reduced}),\qquad
\pi_{O,w,i}=q_{l,w}\widetilde p_i.
\]

F3 traceが追加乱数を引かないこともSOURCEで確認したが、実測fresh-IIDやbackend保証ではない。以下は両constructor/preflight/guardが正常で固定lawを維持する条件下での期待値である。

g(l,w)がchild/angleに依存せず、reduced wで両方式に同じ非負価格を与え、Ordinaryのその他raw prefixでも非負なら、

\[
\sum_{w\,reduced}q_{l,w}a_w g(l,w)
\le\sum_{w\,all}q_{l,w}g(l,\operatorname{red}(w)).
\]

右辺のgは元degreeのphaseを保持する。child lawの和1とa_w≤1を使い、Ordinary側の非負項を落としただけである。[W2b][W2B]のN_F≤N_Oと合わせ、次の**期待総量**は`CERTIFIED_FULL_NONWORSE`となる。

- accepted logical実行数：N_F z_F≤N_F≤N_O。実現したaccepted数同士のpathwise比較ではない。
- total logical CQ：g=|red(w)|+2。**label別の合計価格**への変換はしない。
- helper抽象RZ/H、skeleton CX、元degreeに従うouter Zの各呼出数。
- word部分だけのlabel別forward CQ数。同labelへのchild呼出分はこの結論に含まれない。

明示outer/helper workspace2は全方式で`CERTIFIED_TIE`（provider追加領域を除く）。strict改善には対応する総量不等式がstrictである別の根拠を要し、ここで一律strictを宣言しない。N_F≤N_O**だけ**から量子費用全体が下がるのではなく、共通proposalとchild非依存価格という追加構造を使っている。

child/angleに依存する座標では、各reduced wについて

\[
a_w\sum_i h_{i|w}c_{F,d}(w,i)
\le\sum_i\widetilde p_i c_{O,d}(w,i)
\]

が認証されれば、同じ非負和の比較とN_F≤N_Oで非劣を示せる。これは**十分条件**であり、満たさないprefixがあるだけで総量劣位とはならない。ここではa_w/h/価格を評価していない。一般native T/CX/depth、total label別CQ、state依存価格、古典費用には一律結論を出さない。

## 4. Partial/Closed P3、hard cap、state/readout

Partial/Closed P3はgroup/conditional lawが異なり、上の共通raw-prefixによる比較を適用できない。N_F対N_P/N_3もW2bのmoment・range・ceil条件に依存する。全event価格の積比較をそのまま使う。Fullの受理率が1以下であることだけで優位にはならない。

量子event数の契約hard capは各axis N_a。期待数N_a z_aが小さくてもhard capが小さくなるとは限らず、P/3より大きい場合がある。新しいaccepted数のhigh-probability capを導出しない。失敗を含むbackendの課金はnominal期待event数に吸収せず、実行prefixと失敗保存へ別課金する。

[E2E費用契約][E2E]の不足は維持する。zero判定前にstate準備をするarchitectureなら、その準備費はacceptedだけに掛けてはいけない。zero後に遅延できる固定architectureか、全trial前処理かを証拠化する。readout、fresh-IID、basis/sector/phase、weighted all-N accumulatorは別途接続を要する。旧G7は実measurement0である。

## 5. 既存費用証拠がある範囲

[G7既存報告][G7R]には、登録development P5ケースのexact digital受理率、label別IR係数、取得RZに基づく期待T affine差が保存されている。その既存結論は「当該literal/exact-query/RZ会計で、Fullが登録3対照より期待Tで小さい」である。**この過去結論は引用可能**で、受理率や全価格が一切未取得という状態ではない。今回raw数値を読み出して再集計・再ランキングしていない。新しい入力、F3監査費用、完成物理providerの優位ではない。

古いP3 control（m=3）は今回m≥5の主比較外であり、Closed P3が有利だった履歴を消さない。[G8報告][G8R]のon-demand cold/warm、miss、有限RZ keyの取得履歴もあるが、同backend/seedによる一致は独立再現ではなく、未知全angle/providerを覆わない。G8の別failure/confidence policyをG7 budgetへ混ぜない。

native CQ、実state/preparation/readout、全回路depth、全N classical取得費用は未確定。実行済みDF/PR/RTEのnative値をidentity一致なしに移植しない。cache keyはratio/precision/tool/version/phase/actual adjointを保持し、cold/warm・hit/miss・lookup・eviction・取得費用を両方式に公平に課す。数学的angle enclosureを物理certificateに格上げしない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[E2E]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/measurement_and_cost_contract.md
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G8R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/README.md
