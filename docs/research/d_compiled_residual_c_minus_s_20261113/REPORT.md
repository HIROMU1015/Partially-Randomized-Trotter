# compiled価格残差 C−S の評価方法と全wrapper価格への接続（Codex D、2026-11-13）

対象は、同じtrajectory・同じ測定軸のcompiled RZ数Cと構造RZ数Sの差 \(D_{\mathrm{res}}=C-S\)。目標は、同じlaw・full wrapper・compiler profileで \(\mathbb E[C]=\mathbb E[S]+\mathbb E[D_{\mathrm{res}}]\) を評価できる状態にすることである。

今回行ったのは技術調査・最小adapter・人工テスト・科学実行計画だけである。保存科学値の新しい再解析、残差の平均・分散・回帰・相関の再計算、fit、sampling、build/compile/transpile、N/RMSE/Gはいずれも0件。

**結論（最も直接につながる方法と判断材料）**

- **最も直接な方法**：C（Codex C）が閉状態kernel（K1）で出す法則期待値E[S]と、**同じroundで同一pathのSとCを取得したpaired D̄**を足す方式（制御変量型）。
  - full wrapperのcompileが可能なroundで使える。m0〜m6は可能な見込みで、m6の実負荷は未測定。
  - 推定の分散はVar(C)/nではなくVar(D)/nになる。E[S]が厳密なら、S側の標本誤差は入らない。
  - E[S]とDが同じlawに基づくことが必要条件である。
- **compile困難なm8〜m12**：残差モデルが必要。次の順で検証すれば採否を判断できる。
  1. SOURCE上で見つかった**制御ZZ回転の展開差**（下記§3）を、小さな単位compileで確認する。
  2. m6のpaired pilotで、compile負荷とDの符号・大きさを把握する。
  3. 識別可能なon-path ladder（校正m1/m3/m5、holdout m2/m4/m6、遠方holdout m7）で検証する。
- 既存C1係数・旧connected-cluster係数は、現在のm0/R2・m6/R128へ移植できない。現時点で採用できる残差モデルはない。

成果：[実装](../../../src/trottertracks/compiled_residual.py)／[CLI](../../../scripts/report_compiled_residual.py)／[test](../../../tests/check_compiled_residual.py)／[保存paired S/C inventory](PAIRED_SC_INVENTORY.json)／[設計の識別可能性](IDENTIFIABILITY.json)／[B record人工例](B_RECORD_EXAMPLE_ARTIFICIAL.json)／[科学実行計画（未承認）](SCIENCE_PLAN.json)／[HANDOFF_C](HANDOFF_C.md)／[HANDOFF_B](HANDOFF_B.md)／[TESTS](TESTS.txt)

## 0. 固定SOURCE・branch

| 区分 | commit | 用途 |
|---|---|---|
| C最新 | `7aca17fe84c801a78113fdf8cabb910480ac3b96` | K0入力STOP・K1未実行（E[S]4値はnull） |
| C閉状態診断 | `a637ed1b813dc3952339cb05f36beb79168ace23` | support 324成分・9 basis、R2/R128のlaw診断（τ・P(order2)） |
| C価格検証計画 | `d458b5e4f596d628d923263f3b2f1678960f2ffa` | 13round metadata・8 wrapper直接pilot案 |
| C平均価格三方式 | `1db4d6e7ebce872c069e337b76632046f60d8b56` | 方式比較・SOURCE_MAP |
| S0a | `a11efd130e9dfe7086b88386570594ebe54d46af` | 保存4path・XY8件のS parity |
| C1結果 | `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b` | 科学SOURCE `1a87e533`、凍結`7eeabf0a` |
| B最新 | `3216d27a69ac8532d2a22e78f5f94491d202d238` | 構造S envelope・assess。canonical ID `587d20f7…e911` |

- **branch**：Full PF調査とは別課題（リソース評価系列）として、新branch `codex/d-compiled-residual-c-minus-s-20261113`をB最新`3216d27a`から作成した（sparse worktree）。
- **C由来のSOURCE**：B treeにないもの（C1結果、S0a、K0/K1、計画）は、固定Git objectとして読んだ。
- **非介入**：Cのbranch・worktree・cache・process・結果、Full PF成果、旧C1/C2/S0a/raw/grant/ledger/marker/KEEP/STOPには触れていない。

## 1. 保存paired S/CのSOURCE inventory（D-1）

[PAIRED_SC_INVENTORY.json](PAIRED_SC_INVENTORY.json)は保存metadataのcopyである。
- 読んだfileのSHA-256を記録した。
- 新しい残差統計は0件。

| 項目 | 保存されている内容 |
|---|---|
| 1 H/state/snapshot | 各cellのidentity binding（H4 `c1a6c428…`/`9bcc0b0e…`/`12219b7d…`）。C1の`Backend.context`が保存入力bytes・preparation hashを照合済み |
| 2 DF rank・fragment順 | tol-only DF10/DF19（`H4_TOL_ONLY_DF10`）、`saved_snapshot_order` |
| 3 PF公式・prefix・ΔPF | H4 S2/prefix1/δ0.1、H6 m5_best/prefix7/δ1.6 |
| 4 K・R・qPF | K2。H4はq8/16×R4/8（T0.8/1.6）、H6はq1/2×R16/32 |
| 5 trajectory・seed | path単位のindex・cohort・event digest。新cell（R8/R32）はPLANにseedを固定。旧cell（EXPLORATION、独立q、H6 v2）は元研究のsource参照。**H6 q2/R16の8pathはevent digestがnull** |
| 6 sampler law | law_bindingの定義source（元PCG64/float64 CDF sampler）。ideal lawとの距離は未資格 |
| 7 X/Y | cosine/sineは同一pathのpaired応答。独立標本ではない |
| 8 full wrapper | `Backend.price_requests`：同じrequestsからSを数え、`compose`→`build_native_hadamard_wrapper(include_measurement=True)`→compile |
| 9 compiler | `1239304c…`（Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17）。全pathで同一 |
| 10 S SOURCE | `r_price_phase1_science.structural`（`1a87e533`）＋`h4_df10_price_v1.Accounting`（`8457c443`）／H6`ae44283d` |
| 11 C SOURCE | `h46_long_time_native.compose`（carrier `155529e2`）、provider `c31ef424`の`transpile_and_measure_cost` |
| 12 paired統計・役割 | H4：校正3cell（n9/8/33）＋独立検証1cell（n8）。H6も同構成。forced rare 4対は別unit。X/Yは同値応答 |
| 13 保存残差・分散・共分散 | 各cellのpaired price covariance、**paired residual covariance**、nominal mean covariance、凍結3係数と6×6 nominal係数共分散。**D=C−Sのcell平均は保存summaryにない**（pathのC/Sから導けるが、今回は計算していない） |

補足：
- 保存の`paired_residual_covariance`は、凍結fitに対する残差e=C−Ĉの共分散である。cell内ではĈ−Sが軸ごとに定数なので、これは**cell内のCov(D)と代数的に一致する**。統計を再計算せずに再利用できる。
- S0aの保存4pathは、streaming SとC1保存Sが8軸とも一致した（S parityの証拠のみ）。

**再利用可能性の区別**
- 保存paired S/Cはすべて**C1 domain**（H4 q8/16×R4/8）に属する。現在の重点m0（q1/R2）・m6（q64/R128）とは一致しない。
- 13roundのどれにも保存のSまたはCはない（[target_rounds](PAIRED_SC_INVENTORY.json)、Cの`d458b5e4`と同じ結論）。
- R4/R8のτ（λ_R·0.1/R≈0.0716/0.0358）は、targetのR2（0.1431）とR128（0.00224）の間にあり、lawが異なる。保存値があることと、targetへ再利用できることは別である。

## 2. 残差の定義と回路範囲

\[
D_{\mathrm{res}}(\omega,a)=C(\omega,a)-S(\omega,a)
\]

- 同一trajectory ω（event digest・seed・cohort・index）、同一軸a∈{X,Y}で定義する。
- C：controlled時間発展＋full Hadamard wrapper＋測定のcompiled RZ数。
- S：C1と同じ全wrapperの構造word（各outerでforward head→tail→reverse head、scalar −cT一回、Yのsdg、H/measure、finish一回）を、`Accounting`で数えた値。

D adapterで分けている量：

| 量 | 意味 | Bの価格になるか |
|---|---|---|
| `FULL_WRAPPER_COMPILED_RZ` | path C | – |
| `FULL_WRAPPER_STRUCTURAL_RZ_C1_ACCOUNTING` | path S | – |
| `FULL_WRAPPER_C_MINUS_S` | path D | – |
| `MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C` | E[S]（Bの構造envelope） | ならない |
| `FULL_WRAPPER_MEAN_C_MINUS_S` | E[D] | ならない |
| `FULL_WRAPPER_MEAN_COMPILED_PRICE` | E[C] | 資格があれば |

次のものはDとして扱わない（拒否する）：
- internal/local・time-evolution-onlyのS/C
- 別path、別軸
- forced rare stratumをnormal cohortへ混ぜたもの
- raw κ(C)（Sの二重計上になる）
- C1 affineとlocal Dの重ね足し

## 3. compiler残差の依存構造（D-2）

SOURCEの本文読解による整理であり、compileでは確認していない。**残差が局所的に決まるとは仮定しない。**

**(a) 展開の差：制御ZZ回転（SOURCE上で確認、決定論的、符号は正の候補）**
- compiled経路：tailのrotationは`df_rte_qiskit._append_central_application`で`RZZGate(2θ).control(1)`になる（provider `c31ef424`）。
  - Qiskit 1.3.0の`RZZGate.control`は汎用`add_control`へ委ねる。
  - 汎用`add_control`はRZZ定義（cx–rz–cx）の各gateを制御化する。cx→**mcx（CCX）**、rz→mcrz。
  - したがってcompiled側は**CCX·CRZ·CCX**になる。
- S側：`Accounting._gate('crzz')`は**CX·CRZ·CX**（中央だけ制御）として数える。数学的には等価だが、compiled側はこれを使わない。
- Qiskit標準のCCX定義はH 2個、T/Tdg 7個、CX 6個を含む。rz/sx/x/cx展開では、1個のCCXが十数個のRZ候補を持つ（隣接gateとのmergeで変わる）。
- この差はZZ支持のrotation 1回ごとに生じる。Z単独支持は`RZGate.control`→CRZで、Sと同じ定義である。
- 保存C1の凍結係数θL（H4 18.29、H6 23.26／event位置）は同じ桁の正値である。ただし**機序の検証ではない**。
- 意味：C−Sの主成分の一部は「compiler最適化」ではなく、**Sの数え方と実際のnative loweringの不一致**である可能性がある。小さな単位compileで確認できる（計画P0）。

**(b) compiler最適化（opt1）**
- Qiskit 1.3.0 level 1の構成：初期に`InverseCancellation`、続いてBasisTranslator、その後`Optimize1qGatesDecomposition`＋`InverseCancellation`をdepth/size固定点まで反復する。`CommutativeCancellation`と`RemoveDiagonalGatesBeforeMeasure`はlevel 2以上のみ。
- target無しの1q置換基準は**run長**（docstring：length as a weak proxy）であり、RZ数ではない。そのため1q runの再合成でRZは**増えることも減ることもある**。
- `InverseCancellation`で隣接CX対が消えると、両wireの1q runが連結して再合成が進む。固定点反復なので、**wire沿いの長距離の連鎖**が起こりうる。

**(c) 境界・状態**

| 寄与 | S（C1 Accounting） | Cとの差が生じうる点 |
|---|---|---|
| RTE内部 | 同basis共有、隣接inverse basisの完全一致相殺、wire別pending Zのliteral fold（`angle!=0.0`でflush） | compilerは任意の1q runを再合成し、零角判定も異なる。(a)の制御ZZ |
| PF–RTE境界 | head basis/diagonalとtail basisがpending状態を共有 | 1q runが境界を跨いで連結し、merge量が文脈に依存 |
| signed-piece境界 | H4 S2は1 piece。H6は11 piece・負角 | piece間のrun連結。H6はpiece別law |
| outer境界 | 同じcounterでq反復、途中finishなし | outer端のrun連結・CX相殺がq方向に依存 |
| head/tail相殺 | 完全一致のinverse basis対だけ相殺 | compilerは隣接inverse gateを一般に相殺し、連鎖する |
| pending Z | wire別binary64和。非RZ gateでflush | compilerは非RZを挟んでも1q run内でまとめうる |
| controlled phase | tailの相対phaseをp(phase)として1件 | translation後のrunとのmerge |
| scalar一回 | p(−cT)を1件 | 同上（境界1回分だけ） |
| X/Y wrapper・terminal | H、Yのsdg、H、measure、finish一回 | level 1は測定前のdiagonal除去なし。runのmerge |

**局所残差モデルの成立に必要な条件**（提案する場合）：
- 局所unitを、制御・境界を含めて実回路と同じ文脈で定義する。
- run連結とCX相殺の連鎖長に上限があること、またはその打切り誤差を独立holdoutで評価すること。
- outer/piece/wrapperの境界項を別に推定すること。
- order2のproduct（CZ）事象を別stratumとして扱うこと。
- compilerはRZを増やすことも減らすこともあるので、符号制約を置かないこと。

## 4. 既存実装の再利用範囲

| SOURCE | 再利用する内容 | 再利用しない内容 |
|---|---|---|
| C1 `Backend.price_requests`（`1a87e533`） | 同一requestsからS（Accounting）とC（compose→full wrapper→compile）を取る**paired取得経路**、cache binding、seed/digest | 係数θ・C1の校正・検証役割の付替え |
| C1保存summary | cell内Cov(D)（paired residual covariance）、価格共分散、forced rare対 | targetへの数値移植、D平均の再計算 |
| C `mean_compiled_price` | `combine_mean`（残差1modelだけ、full wrapper、clampなし）をmodel modeで再利用。`MeanPriceProvider`（受動copy） | `local_residual_mean`はRTE内部成分だけ（full_wrapper_mean=null）。単独では価格にしない |
| C K0/K1（`h4_structural_kernel`） | E[S]（law期待値）の供給元 | K1の4値をcompiled Cとして扱わない |
| B `rpe_diagnostic_intake.accept_structural_packet` | E[S]をBが受け入れた行をDの入力にする | 構造Sを価格へ渡す経路 |
| B `assess` | 未資格E[C] recordの拒否を確認 | – |
| 旧connected-cluster | κ1〜κ3のwindow形・校正→凍結→holdoutの手順 | 旧H4 DF12/H5 DF9・uncontrolled windowの係数・閾値・独立SE合成 |

**新規の最小adapter `compiled_residual.py`**：
- `paired_residuals`：同一path・同一軸の全wrapper S/CからD recordを作る。
- `cohort_summary`：有限cohortのD平均。X/Yは別記録で、nはpath数。役割・forced stratumは混ぜない。
- `legacy_domain`：旧係数のdomain不一致を拒否する。
- `identifiability`：q/R設計の階数と、holdoutの移送種別を判定する。
- `combine`：E[S]＋E[D]の一致条件を検査する。
- `b_price_record`：未資格のB recordを作る。
- `saved_paired_inventory`：保存paired S/Cのmetadataを集める。

汎用frameworkは追加していない。

## 5. 評価方法の比較（D-3）

| | A：保存paired残差 | B：局所・connected-cluster | C：full-wrapper直接参照 |
|---|---|---|---|
| 得る量 | 保存cohortの有限D̄（C1 domainのみ） | 局所κ(D)と境界項のmodel E[D] | 同一pathのS/CのD̄（target roundで） |
| 計算量 | 0（既存summaryのcopy） | 局所compile群＋law期待の重み（qRに依存しない形）。校正費は別 | path当たり2 compile（X/Y）。m6は8192 event位置で、m12（3.3×10⁷）は直接不能 |
| 保存証拠 | C1 8cell・S0a parity | 現domainの局所係数なし | target roundは0 |
| 科学的仮定 | 有限cohort、同domain | 局所性・打切り・境界の分離・law一致 | 同law・同compiler。E[S]との結合ではlaw一致が必要 |
| 未解決のsystematic | targetへ移送できない | 非局所連鎖、(a)の文脈依存、order2、law距離 | order2の希少性、有限n、sampler対ideal law |
| 13roundへの適用 | 不可 | 将来、検証済みならm8〜m12に適用可 | m0〜m6（m7は負荷次第）。それ以上はmodelが必要 |

- **推奨**：target roundで方法Cを取り、C K1のE[S]と合成する。m8以上は、§3(a)の機序を取り込んだ方法Bの限定形を、方法Cのladderで校正・holdoutする。
- 方法Aは方法開発・機序の照合にだけ使う（新しい保存値の再解析には別承認が必要）。

## 6. 識別可能性（D-4）

[IDENTIFIABILITY.json](IDENTIFIABILITY.json)はexact rationalによる設計の階数だけで、データは使っていない。

| 設計 | 特徴量 | 階数/係数数 | 判定 |
|---|---|---:|---|
| m0（1,2）＋m6（64,128） | 1, q, qR | 2/3 | **識別不能** |
| m0〜m6（R=2q） | 1, q, R, qR | 3/4 | 識別不能。R/qが常に2なので、qとRを分離できない |
| m0〜m12 | 1, q, R, qR | 4/4 | 形式上はfull rank。ただしR/q∈[255/128, 2]でほぼ共線であり、分離には不適 |
| C1 H4（8,4）（8,8）（16,4） | 1, q, qR | 3/3 | 識別可能、lack-of-fit DOF 0。holdout（16,8）は既知水準の組合せ |
| 提案ladder m1/m3/m5 | 1, q, qR | 3/3 | 経路上でのみ識別可能。holdout m2/m4/m6/m7は新しい水準 |

- X/Yは同じ設計点の応答なので、行を増やさない。
- S0経路上だけで使うなら、R≈2qによりqとRを分離する必要はない（経路上の関数として校正・検証すればよい）。
- 別のκ scheduleへ移送するなら、q-only・R-onlyの統制点が必要になる。

## 7. Cとの独立性・結合条件（D-5）

[HANDOFF_C](HANDOFF_C.md)。Dの方法の成立条件と、CのE[S]との結合条件を分けた。

- **Dの成立条件**：同一path・同一軸・全wrapperのS（C1 Accountingと同じword）とC（同compiler）。
- **結合の条件**：
  - 同じcanonical identity・law_binding・physical candidate・profile・軸。
  - E[S]のlawとDの標本lawが一致すること（ideal descriptorとPCG64 CDF samplerの距離が未資格なら、不一致として記録する）。
  - K1はK0 PASSが前提。
  - 残差は1 modelだけ。
  - 不足があればnullのまま。
- CのK1四値は`MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C`であり、compiled Cとして扱わない。E[C−S]が未知のときに0で補わない。

## 8. 次の有限科学実行計画（D-6、未承認）

詳細は[SCIENCE_PLAN.json](SCIENCE_PLAN.json)。いずれも別途、明示の実行承認が必要である。

- **P0 機序単位compile**：制御RZZの汎用control形とCX·CRZ·CX形を、数角度で比較する（小回路、最大8 compile、sampling 0）。§3(a)を確認・反証する。確認された場合は、(i) Sに実loweringを反映した別quantity S′をCへ提案する、(ii) 残差modelに機序項を入れる、のいずれかをユーザーが選ぶ。
- **P1 paired development pilot**：Cの既存8 wrapper案（m0/R2・m6/R128 × normal 2path × 同path X/Y）を**再利用候補として確認**した。Sを同じpathで同時に取ることを明示した（C1 `price_requests`の経路）。
  - 目的は、m6のcompile負荷（wall・RSS）、Dの符号・大きさ、記述的な分散の取得である。
  - 役割はDEVELOPMENTで、校正や全law価格ではない。n=2は精度保証にならない。
- **P2 経路上ladder（P1とK1の後）**：校正m1/m3/m5、holdout m2/m4/m6、負荷が許せば遠方holdout m7。
  - n0=8で開始し、目標精度ε_Cから事前に決めた規則でnを決める（ε_CはB/ユーザーが決める）。
  - order2は法則確率で重み付けた別stratum（forced conditional path）で扱う。
  - 設計用の算術（科学結果ではない）：保存P(order2)から、pathに少なくとも1つorder2が出る確率は、R2で約2.0%、R128で約2.0%。n=2のnormal pathではほぼ観測されない。
- **P3 off-path統制**：S0以外のκ scheduleへ移送する場合だけ行う。

試行数を勝手に増やさない。既存8 wrapper案を正式採択したともしない。

## 9. 技術テスト・保全

- [TESTS](TESTS.txt)：**140 PASS**。既存1件のFAILは今回の変更によるものではない。
  - 内訳：新13件、C再利用23件（mean price 15、R adapter 8）、B 3216d27a一覧の回帰104件。
  - 既存FAIL：C `check_r_variable_compiled_price`の`pf_pr_rpe_pipeline.py`のbyte固定比較（C `4cb61ef4`）。B `f0c4bb55`の標準4次拡張でこのfileが変わっており、本branchでは変更していない。Bの通常test一覧にも入っていない。
  - 新13件の内容：同path対応、X/Y独立record、full wrapper単位、符号と加算、二重計上防止、null保持、DIRECTとMODEL、calibration/holdout、旧係数のdomain拒否、compiler/profile不一致、Bの未資格E[C]拒否、識別可能性、inventoryのcopy・非起動。
  - 主要guardは、無効化すると検出されることを確認した（mutation 6件）。
  - これらは作者自身の技術確認で、実H4残差モデルの科学的成立を示すものではない。
- **値の扱い**：数値はすべて人工fixture、またはsaved metadataのcopyである。
- **保全**：sampling、compile、fit、再解析はしていない。既存の結果・STOP・C/B branchは変更していない。
- **STOP**：commit/pushとremote照合の後、mandatory STOP。
