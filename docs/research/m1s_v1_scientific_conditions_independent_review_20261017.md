# M1-S V1：保存信号shot feasibilityの科学条件・実行契約に対する限定独立レビュー

- **レビュー日**：2026-10-17（JST）
- **研究**：Track B／M1-S Gate共有±による部分乱択時間発展
- **レビュー対象**：V0終了時点のV1候補契約、科学条件、SOURCEの数学的意味、実行・STOP境界
- **開始承認**：ユーザーの「レビューを開始して」
- **重要度**：Level 2（GPT-6 xhighで十分な、既存方針の範囲内の科学条件判断）
- **判定**：`SCIENTIFIC_REVIEW_COMPLETE / V1_DESIGN_CONDITIONALLY_SUPPORTED / NUMERIC_ACCURACY_TARGET_NOT_JUSTIFIED / CONTRACT_NOT_YET_ADOPTABLE / EXECUTION_NOT_AUTHORIZED / MANDATORY_STOP`
- **本レビューでの新規科学数値計算**：なし。保存signal・bias・shot・Hamiltonian・Pauliの再計算、native compile、SOURCE改変、GitHub更新、実行authorization発行は行っていない。以下の簡単な関係式の導出は紙上での解析。

## 1. 結論

**Task Aの保存済みcoherent平均信号を使って、位相biasとvisibilityの交換から「同一の統計設計で必要な十分shot予算を削減できる条件」を調べるV1は、限定研究として合理的である。** ただし、科学的な意味は「保存oracle情報を用いた事後的、条件付きのshot設計上限」であり、「実用RPEに必要な最小shot数」でも「基底状態エネルギーの資源優位」でもない。

現行のV0 SOURCEにおける確率・幾何・shot式は、明記された仮定のもとで概ね整合している。科学的な障害は主として**単一の許容位相誤差 `gamma_rad` に物理的・研究目的上の基準がまだ存在しないこと**である。`alpha_total`、算術解像度、resource上限にも正式な選択が残る。

**推奨**：

1. 既に採択された「Task Aを先行し、Task Bを最終目標として維持する」「無補正raw直接route」「G3/G6/G9全cellとordinary/independentを比較」の方針を維持する。
2. fresh-IID/Hoeffding/等軸配分、条件付きcoverage、同じ十分shot設計だけの比較、scientific GOを自動判定しない分類を採択候補として支持する。
3. `gamma_rad` を観測済みG3/G6のbiasや好ましいshot比から逆算して定めない。**特定の目標精度がない現Task Aでは、単一γの科学的な一意決定は不可能**である。利用場面から外生的に要求角精度を固定できる場合にのみ既存SOURCEの単一γ契約を採択する。そうでなければ、同じ保存結果に基づく**γ依存の成立域・境界の事前固定された記述的診断**へ、科学契約の小改訂を推奨する（§4、§11）。
4. 正確な数値correctness・platform coverage・実行時間が未検証のため、sourceを動かす前にCodexによる限定的技術preflightを行う。ただし科学データへの数値代入、shot forecast、科学的再集計はユーザーの**別途明示的実行承認後**とする。

これは本レビューの**科学的推奨**であり、`adopted=true`を書き込むことやV1を実行することの承認ではない。

## 2. 一次資料・取得状態・固定identity

|資料|固定commit／所在|今回の確認|
|---|---|---|
|[V1実験契約候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/docs/tracks/algorithm_codesign/m1s_v1_saved_signal_contract_candidate_20261017.md)|`56ca76401e6cb332a4dbece764cadf3116a17285`|GitHub本文を取得・読解|
|[V1 machine契約候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/artifacts/track_b_m1s_v1_v0_preparation/2026-10-17/v1/execution_contract_candidate.json)|同上|取得・主要field照合|
|[未決scientific items](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/artifacts/track_b_m1s_v1_v0_preparation/2026-10-17/v1/scientific_open_items.json)|同上|取得・読解|
|[算術・coverage・resource設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/docs/tracks/algorithm_codesign/m1s_v1_arithmetic_coverage_and_resource_design_20261017.md)|同上|取得・読解|
|[V1 SOURCE `spec.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/src/trottertracks/algorithm_codesign/m1s_saved_signal_v1/spec.py)、[`analysis.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/src/trottertracks/algorithm_codesign/m1s_saved_signal_v1/analysis.py)、[`bounds.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/src/trottertracks/algorithm_codesign/m1s_saved_signal_v1/bounds.py)、[`control.py`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/src/trottertracks/algorithm_codesign/m1s_saved_signal_v1/control.py)|SOURCE `de622bd77ebd61a6d2cf5b69e29c4e620c8563b3`|GitHubで静的に読解。実行・import・数値testなし|
|[前回の科学設計レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56ca76401e6cb332a4dbece764cadf3116a17285/docs/research/m1s_rpe_applicability_scientific_design_review_20261017.md)|新V0契約が受領したレビュー原本を同日のGitHub資料として参照|ローカル原本（29,915 bytes）および固定資料で主要方針を確認|
|[前回の有限step結果科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md)|`e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b`|GitHub取得、結果の適用域を確認|
|[RPE Decision Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/870af3f65e3e31e19d2b002c556216c7340c06f0/docs/tracks/algorithm_codesign/m1s_rpe_applicability_and_total_cost_decision_packet_20261017.md)|`870af3f65e3e31e19d2b002c556216c7340c06f0`|GitHub取得・式と未決事項の確認|
|[有限step完了報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md)|`025c5a90888b02b73d2815fbb10d6bd1d2823602`|完了scopeと条件付き証拠を確認。raw scienceを再実行せず|
|[既存RPE会計SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/870af3f65e3e31e19d2b002c556216c7340c06f0/src/trotterlib/rpe_resource_accounting.py)|同上|`_hoeffding_shots`・RPERound仕様・effective-eigenstateの仮定を静的確認|

V0公開branch `track-b-m1s-v1-v0-preparation-20261017` のremote HEADは `be2da2c3c4ed1d826a76e8a08a98ff6c92ce90c5` と確認した。v1 payloadとpublication receiptは別commitであり、正本のSOURCE・候補payloadは上記固定commitに従う。SOURCEとrawの全ファイルについて独立SHA再監査を実施したとは主張しない。保存監査128/12/9 PASSはCodexの技術報告であり科学的な正しさの証明ではない。

### 文献の確認

- [Hoeffding, *Probability Inequalities for Sums of Bounded Random Variables*, JASA 58 (1963)](https://doi.org/10.1080/01621459.1963.10500830)：有界確率変数の集中不等式。原論文の書誌・abstractを確認。以下の±1特殊化は紙上で再導出。
- [Kimmel–Low–Yoder, *Robust Calibration ... via Robust Phase Estimation*, PRA 92, 062315 (2015)](https://doi.org/10.1103/PhysRevA.92.062315)、[2021 Erratum](https://doi.org/10.1103/PhysRevA.104.069901)：原論文と訂正の存在を確認。ただしTask Aを多round RPEのbranch保証と混同しない。
- [Güntherほか, *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026)](https://doi.org/10.1103/ynxb-p2xq)：RPEと部分乱択シミュレーションの既知の比較枠組み。今回のV1が同論文より優位だとの主張はしない。

## 3. 科学的に何を保証する実験か

Task Aは保存raw-H理想信号 `z_raw(T)` の**各単一時刻・円周位相**である。`+iTH`、保存state、全既存G3/G6/G9、δ/R/Tの組、shared±／ordinary／independent±を変更しない。全108行（非零予測72＋zero control36）と48対応比較は、既存入力についての完全な記述範囲である。

重要な三つの階層を分離する。

1. **保存信号の条件付き数値証拠**：pilotの値、位相bias、数値半径。IEEE/BLAS/trig等の仮定に基づくengineering結果であってplatform certificateではない。
2. **仮想Hadamard測定によるshot設計**：各shotで±1の独立なX/Y観測が得られ、その平均が保存`z^0`へ一致すると仮定した十分統計予算。実測量子shotsでも最小必要shotsでもない。
3. **実装可能なRPEアルゴリズムの保証**：物理wrapperの実際の平均`z^p`、stateの品質、実装近似、全roundのalias/decoder、事前に利用できるbias/振幅下界、総回路費用が必要。V1で提供しない。

**特に、保存真値から構成する`b^+`・`A^-`は、開発用のoracle診断値である。** 実際の量子計算のshot scheduleをこれらの値から選ぶには、真値に依存せず取得可能な上界・下界と、その取得費用が別途必要である。事後的な保存oracleの使用を「無料の運用上の知識」に読み替えてはならない。V1で計算したshot数はこれを明示した`IDEAL_SIGNAL_HYPOTHETICAL_SUFFICIENT_SHOT_FORECAST`だけとする。

## 4. 数式と意味論の独立検討

保存された中心値・条件付き半径を使う場合、候補SOURCEの定義は

\[
 b_m^+=\min\{\pi,|\widehat b_m|+r_{b,m}\},\quad
 A_m^-=\max\{0,\widehat A_m-r_{A,m}\},\quad
 q_m=\gamma-b_m^+.
\tag{1}
\]

同じ保存raw-H位相が比較の標的であり、`phase_bias_rad`は`TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H`である。既に含まれる`eta_j(T)=|T|B_j`をさらに(1)へ足さない。今回の`gamma`は許容**位相角**であり、元pilotの数値計算半径`2^-18 rad`やsignal半径`2^-24`ではない。

さらに、`0<q_m<π/2`、`A_m^->0`を仮定する。各軸の±1確率変数の平均と真の実部・虚部が一致し、shotごとにIIDなら、Hoeffdingの二側式

\[
 \Pr(|\widehat x-\mathbb E x|>t)\le 2e^{-Nt^2/2}
\]

が成立する。X/Yのそれぞれに`alpha_axis`を配分し、`t=A^-\sin q/\sqrt2`とすれば、両軸が成功するとき複素平均誤差は`A^- sin q`以内である。原点の外側にある複素信号を半径`A^- sin q`以内で近似したとき、円周位相差は`q`以内。円周三角不等式から真のraw位相との誤差は`b^+ + q ≤ γ`となる。したがって各軸に必要な十分予算は

\[
\boxed{N_X=N_Y\ge\left\lceil\frac{4\log(2/\alpha_{axis})}{(A^-)^2\sin^2(\gamma-b^+)}\right\rceil .}
\tag{2}
\]

**式(2)の係数4、axis分割、位相disk幾何は整合している。** ただし、π近傍の主値実数差のunwrapを保証するものではなく、円周距離である。concentrationの仮定にはfresh-IIDなshot間の乱択・測定、安定した平均、±1有界値が必要。旧pilotは実測shot分布の再現性を直接検証していない。

`analysis.py`の式、等軸整数化、family/rowのα配分を静的に確認した。`bounds.py`のMachin型π、sin交代級数、正値log区間は**設計上**合理的だが、全入力に対する終了性や正しさを今回は実行で確認していない。初回数値correctnessは将来の独立した実行承認のもとで行う必要がある。

## 5. 最重要の未決：γをどう選ぶか

### 5.1 なぜ現在の証拠から一意に選べないか

V1はTask Aであり、まだ実用的な対象エネルギー精度、回路誤差、RPE time horizonを固定していない。式(2)のshot数を左右する`gamma`には物理的な外生仕様がない。元pilotの解析分解能をgammaに流用することもできない。特定のgammaをG3/G6の保存結果から「勝てるように」選べば、結果依存の設計となる。

`gamma`の数値を今勝手に`1/100`、`1/16`、`π/3`などへ置く科学的根拠はなく、今回いずれの数値も正式採択候補としない。`π/3`は別のmulti-round decoderの十分条件であって、Task A単独の必要な精度ではない。

### 5.2 γ依存の判別は紙上で可能

同じ`alpha_axis`で整数丸めをいったん除くと、方式mに対するshot設計の主要因は

\[
M_m(\gamma):=A_m^-\sin(\gamma-b_m^+),\qquad
 b_m^+<\gamma<\pi/2.
\tag{3}
\]

同じ位相精度・confidenceなら、**大きい`M_m`は式(2)の連続十分shot予算を小さくする**。一方、整数shot数では天井関数による同順位があり得る。`M_m`の比較は実装費用や真の最適shot数の比較ではない。

より具体的に、同じcellで`a=A^-_sh>c=A^-_cmp>0`、`0≤v=b^+_cmp<u=b^+_sh<π/2`というtrade-offがある場合、次を置く。

\[
X=a\cos u-c\cos v,\qquad Y=a\sin u-c\sin v>0.
\]

単純な三角関数の差の変形により

\[
M_{sh}(\gamma)-M_{cmp}(\gamma)=X\sin\gamma-Y\cos\gamma.
\tag{4}
\]

この仮定では、`X≤0`なら`u<gamma<π/2`の全域でshared側の連続十分shot因子は不利である。`X>0`なら

\[
 \gamma>\gamma_*:=\arctan(Y/X)
\tag{5}
\]

の範囲に限りshared側の因子が優位になる。`γ*`はこの仮定で`u<γ*<π/2`に一意に存在する。以上は紙上の条件式であり、今回`a,c,u,v`を実際のG3/G6保存値へ代入していない。両者のbias順序や振幅順序が異なるcellへ(5)を無条件に適用してはいけない。

この関係は科学設計上重要である。**trade-offがある以上、γの選択なしに全精度でshot改善を主張することはできない。しかし、1つの恣意的γを選ばなくても、改善し得る精度領域を記述する問題は定義できる。**

### 5.3 推奨する二つの契約経路

**第一候補（今回推奨）：γ依存の成立域を事前指定して報告する記述的V1。** どのcellでも同じ領域 `0<γ<π/2` と数式を使い、正のbias余裕、`M_sh-M_cmp`の符号が変わる条件、成立不能・不分解を分類する。数値的な境界値や順位の生成は科学的再解析に当たるので、SOURCE／契約を固定して**別途実行承認後**にのみ実施する。既存SOURCEは単一γ用なので、Codexによる契約・SOURCEの限定改訂が必要。今回この改訂を承認・実装したわけではない。

**既存SOURCEを維持する代替：外生的な単一γを明示してから採択する。** 例えば研究上の実際の単一時刻位相許容値が別途決まっていれば、その値を全72行共通に固定する。V1結果を見た後でgammaを緩めたり、良いcellだけを採択したりしない。数値を決めるにはユーザーの明示選択が必要であり、現在の資料から値を捏造しない。

両経路の結果を混在させ、「gammaを後から最適化した一般的優位」と宣言しない。

## 6. α・confidence scope・統計policy

既存の`alpha_total`には**Task Aの一方式一時刻に対するconfidence**か、全72予測方式行の**同時confidence**かという2つの契約候補がある。目的が別なので、どちらが普遍的に正しいとはしない。

- **第一候補：`PER_METHOD_ROW`**。1つの単一時刻taskを事前に選んで実行する際、X/Yに`α_axis=α_total/2`を配分する。全72方式行の同時成功は名乗らない。理想設計の比較用であり、同じ保存signalから多数選んだ最良rowを独立held-outとは扱わない。
- **代替：`FAMILY_ALL_72_METHOD_ROWS`**。72行の2軸計144件に同時上界を置くため`α_axis=α_total/144`。全行を並列に実測することを要請する特殊な保証であり、単一方式の選択後のphase taskには通常不要。方式間の独立性はunion boundに必要ない。

confidenceの数値`α_total`も、研究taskが要求する保証水準から決まる。`1/20`（95%）等は利用可能な慣例的候補ではあるが、既存データから導かれた最適値ではなく、今回自動採択しない。同じαとaxis-policyを全方式へ適用することを科学的条件として推奨する。

**統計policy**：`FRESH_IID_HOEFFDING_EQUAL_AXES`を採択候補として支持。各量子shotでfresh-IIDな乱択label、軸内独立、±1 outcome、同一計算法を要求する。これは将来の物理的実装で別途照合が必要な仮定であり、既存pilotがその測定プロトコルを実測したという意味ではない。

## 7. 算術解像度・coverage・数値correctness

1. **`coverage_acceptance=SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY`を候補として支持**。旧IEEE/BLAS等の未検証仮定を条件付きで受け入れ、physical/SPAM保証へ拡大しない。保存半径はfailure確率αの対象ではないので、統計的confidenceと足し込まない。
2. **`accuracy_semantics=HIGH_PROBABILITY_CIRCULAR_PHASE_ERROR`を支持**。raw-H信号に対する円周位相誤差であり、基底エネルギー誤差、RMSE、principal実数差のunwrapを意味しない。
3. **`arithmetic_bits`を現時点で一意に決めない**。8–256 bitsはSOURCEの対応可能範囲であって、必要十分な精度の証明ではない。科学条件γと算術許容区間が決まってから、解析的な区間打切り基準とresource capの範囲で1つの精度を事前固定する。曖昧な場合は`ARITHMETIC_RESOLUTION_UNDETERMINED`、上限超過は`TECHNICAL_STOP`とし、結果に応じたp上げ再試行をしない。
4. **数値correctnessは未検証**：Machin π、sin/logの区間算術、ceil上界、有理数オーバーフロー、入力JSONのbinary64厳密変換、未評価行の処理を、実行権限の範囲内で独立したtoyまたは限定preflightへ含める必要がある。単なる静的128PASS、12テストでは数値保証に昇格しない。

科学的条件を事後的に探索する代わりに、sourceが失敗・不解像・不適用を明示して停止する設計は適切である。

## 8. 比較・公平性・科学的解釈

**`comparison_policy=COMMON_SUFFICIENT_DESIGNS_NO_OPTIMALITY`を支持。** 全72方式行に同じγ・α policy・保存誤差会計・ceiling規則を使用し、48対照を報告する。零時刻の36 controlには科学shot数を割り当てない。

- **同じ十分設計の整数Nが小さい**：採択条件に対する理想測定の予算差を示す。量子実機での最適shot削減の保証ではない。
- **G3/G6の旧visibility差**：条件付き分解済みであっても、新しいshot優位の証明ではない。位相biasを払った残余を必ず比較する。
- **G9の旧絶対差はunresolved**：shot上限に小差が出ても、これを物理的な改善・一般的有意差へ読み替えない。
- **事後選択**：同じH4既知development入力を使って設計するため、陽性cellを選んで別入力に一般化できない。
- **confidence範囲**：`PER_METHOD_ROW`は単独rowの保証のみ。複数方式のquantum outcomeで最良を選択する場合や全表同時保証を宣言する場合には、契約を別途設計する。

**`materiality_policy=DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO`を支持。** primary outputは全行の適格性・数式の区間・条件付きshot設計。新しいGO/NO-GOや最小資源アルゴリズムの選定はこの実行で認定しない。

## 9. 費用・resource制限

V1ではnative gate synthesis、状態準備費用、classical online費用、全RPE総資源を数値化しない。`N_sh/N_cmp`だけから最良量子アルゴリズムの主張はできない。固定費を無視し、1 shotの同じ費用定義を採択し、整数shotの共通設計が評価できた場合のみ、記述的なbreak-even条件

\[
\frac{C_{sh}+P_{sh}}{C_{cmp}+P_{cmp}}<\frac{N_{cmp}}{N_{sh}}
\tag{6}
\]

を用いてよい。**ただしこれは軸別価格が同一で固定費が無い限定モデルの条件**であり、HDやcontrolled回路のnative実価格は未取得。次のV2では実際のwrapper/control/synthesis/HD近似とstate準備の上限を同じprecisionで会計する。

V0契約のCPU1/GPUなし、メモリ2GiB、入力1MiB、出力・log各16MiB、rational 65536 bits、shot整数4096 bits、bound 100万step、series各4096項は**技術候補**として理解可能。一方、`wall_time_seconds=null`／`cpu_time_seconds=null`のままでは、resource上限を科学契約として十分に固定したとは言い難い。重いFraction演算・π/log/sin区間評価に対して、実際のworst case計算時間は未検証である。

**推奨**：Codexが科学入力を用いない技術preflight等から妥当な有限の時間・CPU予算候補を提示し、ユーザーが科学実行前に採択する。数字を推測して固定しない。既存resultやmarkerを再利用しない。sourceの実行防止はこれまでどおり閉じたまま維持する。

## 10. 科学的条件の個別判断

|scientific field|本レビューの科学的推奨|採択状態|
|---|---|---|
|研究方向|Task A先行・Task Bを最終目標。raw直接無補正、全保存cell|**既にユーザー採択済み**|
|`accuracy_semantics`|`HIGH_PROBABILITY_CIRCULAR_PHASE_ERROR`|**支持／ユーザー採択待ち**|
|`statistical_policy`|`FRESH_IID_HOEFFDING_EQUAL_AXES`|**支持／ユーザー採択待ち**|
|`confidence_scope`|第一候補`PER_METHOD_ROW`、family同時保証が必要なら`FAMILY_ALL_72_METHOD_ROWS`|**第一候補を推奨／未採択**|
|`coverage_acceptance`|`SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY`|**支持／ユーザー採択待ち**|
|`comparison_policy`|`COMMON_SUFFICIENT_DESIGNS_NO_OPTIMALITY`|**支持／ユーザー採択待ち**|
|`materiality_policy`|`DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO`|**支持／ユーザー採択待ち**|
|`gamma_rad`|外生的な位相要求があればその値で単一γ。なければγ依存の成立域報告への契約小改訂|**固定値は根拠不足・未採択**|
|`alpha_total`|taskに応じた明示的失敗確率。慣例候補の数値を正式値へ格上げしない|**未採択**|
|`arithmetic_bits`|γと停止条件が決まった後、固定precisionと失敗分類を一括選定|**未採択**|
|resource caps|既存memory/calls等は技術候補、実行時刻・CPU上限は技術preflightを踏まえ採択|**未採択**|
|科学計算実行|V1は保存科学結果の数値再解析としてSOURCE-bound別承認が必要|**未承認・STOP**|

単一γ以外の数値未決を「理論が誤り」や「研究のGOがない」と混同しない。一方、**必要な数値が欠けたまま科学実行を承認可能とするのも不適切**である。

## 11. 推奨する次の最小作業

科学的に優先するのは、**恣意的な一点γで勝敗を固定しない、精度条件に対するbreak-evenの所在を見分ける設計**である。現SOURCEを可能な限り再利用する。

1. **ユーザー**：今回のレビュー推奨を採択するか判断。外部から要求された単一時刻のphase accuracyが存在するなら、それを唯一のγとして提示する。ないならγ依存の記述的V1への限定小改訂を採択するか判断。
2. **Codex**：採択された経路に応じ、契約・machine項目・SOURCE/guard・非数値静的検査を最小限調整する。γ依存routeの場合は(3)–(5)の条件・仮定・未解像分類を固定し、数値境界の実評価はしない。旧V0候補と固定SOURCEを保全し、別branch/worktreeに新候補を公開、mandatory STOP。
3. **ユーザー**：精度の意味・confidence scope・α・算術解像度・resource・科学実行範囲を最終的に採択。必要な独立技術preflightを挟む。
4. **ユーザー**：新規数値再解析の**別途明示実行承認**。採択だけでは実行不可。
5. **Codex**：固定SOURCE・input・新namespace・one-shot制限に従い限定V1を一回実施し、結果・失敗・未評価行・保全監査を保存、mandatory STOP。
6. **GPT**：結果の研究上の意味を改めてレビューする必要がある場合のみ、新しいユーザーの開始承認後に判断。V1のpositiveからV2／Task Bへ自動昇格しない。

### 科学的STOP／判断の基準

- **入力・契約・SOURCE不一致、数値半径のschema異常、算術制限超過**：technical STOP。旧raw/result/grant/marker保全。
- **`b^+≥γ`、`A^-≤0`、必要なoracle reference/phase branch条件不足**：選択された十分条件routeでは保証を構成できない、またはunresolved。アルゴリズム一般の性能negativeではない。
- **共通の十分shot設計が比較可能**：その予算差とbreak-even費用条件を記述。片方だけ適格ならその事実と適用域を記述。native GOは出さない。
- **未取得の運用上の振幅・bias保証、state品質、物理wrapper価格**：V2／Task Bのend-to-end性能は必ず未認証とする。

## 12. 限界・反論・結果に対する確信度

- **高い確信（数学・静的意味論）**：Hoeffding係数、disk位相幾何、rawモデル差の二重計上防止、同一alpha/axis政策の必要性、Task A/Bの区別。
- **中程度（設計上の推奨）**：単一γでなく精度依存の境界を調べる方が探索の恣意性を減らせること、row confidenceを先行すること、技術preflight後の実行契約採択という順序。
- **未確立**：現H4でのγ境界・整数shot優位・platform条件、実機回路の費用・state条件、RPEにおける精度・confidence保証、異なる分子や入力への一般化、独立新規性、end-to-end resource GO。

反論候補：(i) 厳格なTask Bに直結しないAを省略すべき、という意見には、Aを**狭い機構screening**としてのみ使用し、大きな実験の実施可否を無条件に決めないことで対応する。(ii) γがないなら慣例的に数値を決める方が早い、という意見には、観測済みのbias trade-offに応じて簡単に結論が逆転するため、精度要件のないまま一点を主成果にすべきでないと判断する。(iii) 全体shot数の定理を作るべき、という意見には、Task Bのstate保証・alias・wrapper情報がなく、現在の証拠では不適切と判断する。

## 13. 終了・版管理

- **本レビューで実施**：上記固定GitHub資料の取得・静的SOURCE読解、既存数学的条件の独立紙上照合、γの紙上判別式の導出、外部文献書誌・概説の限定確認、科学条件と次の承認境界の整理。
- **本レビューで未実施**：V1 SOURCE実行、保存結果の数値再計算・shot数予測・新規ランキング、数値correctness・platform認証、native合成・RPE simulation、GitHub write・ユーザーの採択代行。
- **保持する保全対象**：旧v4 STOP/null、Gate0 v5と独立reference、finite pilotの両試行・marker、固定SOURCE・科学契約・raw結果、V0 SOURCE `de622bd77ebd61a6d2cf5b69e29c4e620c8563b3`、V0候補payload `56ca76401e6cb332a4dbece764cadf3116a17285`、公開receipt `be2da2c3c4ed1d826a76e8a08a98ff6c92ce90c5`、parent/other worktree。
- **レビューと計算の承認は別**。本書で定めた推奨は`science.adopted`でも`coverage_model_adopted`でもなく、実行authorization／grant／markerの発行理由にならない。新規計算のmandatory STOP維持。

**総括**：`M1S_V1_FAMILY_STATISTICAL_MODEL_COHERENT_WITH_ASSUMPTIONS / FIXED_GAMMA_TARGET_UNJUSTIFIED / GAMMA_CONDITIONAL_FEASIBILITY_RECOMMENDED / FORMAL_SCIENCE_CONTRACT_NOT_ADOPTED / NO_NEW_SCIENTIFIC_EXECUTION / MANDATORY_STOP`。
