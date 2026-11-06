# ランダム化推定器：分散削減文献の限定調査

2026-11-06 JST。[REPORT](REPORT.md)、[PR/RPE適合条件](PR_RPE_ESTIMATOR_COMPATIBILITY.md)、[候補と最小検証案](NEW_RESEARCH_DIRECTIONS.md)を併読する。以下は一次本文の関連節の確認と、既存調査の再利用である。新しい定理の証明、重要科学レビュー、性能計算・再解析は行っていない。

**今回新たに具体化した応用候補は、CV4Quantum型の既知期待値を持つ古典path controlを、実Hadamard outcomeへ適用する方向である。** 分散削減もPR＋RPE費用改善も未確認。MLMCには別の量子測定構造が必要で、共通乱数だけで測定ノイズが消えるとはしない。

## 1. 文献の版・取得深度・保証

「新たに確認」は今回指定された固定調査の参照範囲へ追加したという意味であり、全repositoryや先行研究で未調査だったとの主張ではない。版日はarXiv submission historyに基づく。著者の数値実験を本projectの証拠に移さない。

| ID／一次資料 | 取得版・今回読んだ箇所 | 対象・測定モデル／保証 | 追加費用とPRへの位置付け |
|---|---|---|---|
| L-CV：Shyamsundar, Yeong, **CV4Quantum: Reducing the Sampling Overhead in Probabilistic Error Cancellation Using Control Variates** | [2502.08735v1][CV]、2025-02-12。§§1.1–1.2、2.1–2.5、3の関連式、Appendix AのLemma A.2 | QPD/PECの実測結果×weight。factorizable controlの平均・共分散を既知lawから取得し、path間の変動を減らす。未知の最良controlは測定内分散を消さない。係数を同じ標本にfitするbiasにはleave-one-outを提示 | 通常回路を変えず古典controlを同じshotに付加できる。lawのmoment取得、係数学習、記録・線形代数が有料。PECの数値的効果でありPR全RPEの保証ではない。今回の主な新規確認 |
| L-ML：Mohammadipour, Li, **MLMC-qDRIFT: Multilevel Variance Reduction for Randomized Quantum Hamiltonian Simulation** | [2604.26865v2][ML]、2026-05-19。v1は2026-04-29。§§2.1–2.3、Theorem 1/Lemma 2、§3/4の関連説明を確認。v1も参照したが以下はv2に固定 | qDRIFTのobservable期待値。index-sharingのtelescopingと、測定ノイズ床を避けるscaled augmented-state構成を区別。Theorem 1はRMSE/指定gate model、Lemma 2はcoupled state差の2次・4次moment条件付きshot bound | augmented blockは非unitary。block qubit、dilation/postselection、norm補正、observable測定の費用・成功率が必要。PRのfinite operator meanへの移送は未確認。今回の新規確認 |
| L-QML：An, Linden, Liu, Montanaro, Shao, Wang, **Quantum-accelerated multilevel Monte Carlo methods for stochastic differential equations in mathematical finance** | [2012.06283v2][QML]、2021-06-22。Quantum 5,481（2021）。§2.2/2.3、§3.1 Lemma 3、§3.2 Theorem 2 | SDE payoff等の乱数出力平均。差分variance/cost/biasの階層条件とcoherent oracleを使う量子平均推定 | classical samplerを可逆/coherentに呼ぶアクセスを要する。既存PRの「古典pathを選び一shot測定」接口とは異なる。古典couplingを共有するだけの手法と区別。今回の新規確認 |
| L-EB：Maurer, Pontil, **Empirical Bernstein Bounds and Sample Variance Penalization** | [0907.3740v1][EB]、2009-07-21。Theorem 4（PDF p.2）、Theorem 10（p.4） | bounded独立標本の固定n confidenceと標準偏差のconfidence bound | 独立な校正標本・既知rangeが必要。固定nの式を任意停止へ流用しない。O2-A既存参照を今回は一次本文で補足確認 |
| L-CS：Howard, Ramdas, McAuliffe, Sekhon, **Time-uniform, nonparametric, nonasymptotic confidence sequences** | [1810.08240v9][CS]、2022-08-06。Ann. Statist.49(2),1055–1080（2021）の後続arXiv版。§4.1 Theorem 4/Eq.(23) | bounded過程のpredictable predictionとtime-uniform boundary。推定対象は平均条件付き期待値 | 適応停止を許す既知の対照。固定target条件、既知range、失敗配分と実停止費用が必要。O2-Aの既存参照を再利用・本文補足。独立した新候補にしない |

L-CVの主要locatorは§2.1 Eqs.(28)–(32)、§2.2 Eqs.(33)–(35)、§2.3 Eqs.(38)–(39)/Lemma A.2、§2.5.1 Eq.(47)。L-MLでは§2.3.2 Eqs.(29)–(32)、§2.3.3–6 Eqs.(36)–(59)、特にLemma 2/Eqs.(42)–(48)を確認した。原文の保証と以下のPR適用上の推論を区別する。

## 2. Control variate：実測値への適用を優先する理由

既知control平均を引くこと、係数がデータ依存ならbias/依存を扱うこと、path条件付きの測定分散が残ることはL-CVに明示されている。ここからPRに移せるのは**既に正しく取得したaxis outcomeに古典的な平均ゼロ項を加減する統計処理**である。PECのsystem channel恒等式そのものをPRのcoherent signalへ移す必要はない。

本調査の適用上の推論は、canonical RTEのorder/label lawと、実行前に既知のpath metadataがcontrolの入力になり得るという点である。全pathの真の期待値、E₀、state vectorをcontrolとして与えない。係数も無料の最適値を使わず、別の測定pilotで学習して凍結する。未知期待値の量子referenceを追加する案とは違い、control値自体には追加の量子測定を要しない。

weightだけのcontrolはcanonical PRでweightが一定なら無効である。orderだけのindicator bankは既知層別推定と比較する必要があり、それ単独を新候補にはしない。今回残すのは、**orderを条件付けた後のlocal labelに対する既知factorized controlの応用可能性**である。有限feature集合を結果前に決め、その既知平均・共分散を実lawに対して取得できることが前提になる。実際のoutcomeとの相関は未測定であり、存在を仮定しない。

同じ情報を使った層別・cost-aware shot配分が直接対照になる。indicator controlが層別推定へ還元される部分は既知の同じ利益として扱う。大量のpath別controlを列挙して「安い最良controlを得た」とする方法は採らない。安価なfactorizationによって、強い対照の後にも費用上の余地があるかが検証対象である。

## 3. Multilevel：path差分と測定差分を混同しない

L-MLの§2.3.2は、同じindex列を使ってもfine/coarseを別々に測ればshot noiseが残ること、単純なcoherent二分岐とfixed-norm observableでも十分でないことを示す。これは既存の[時間相関監査§4.4][CORR]とも整合する。

同論文の代案は、差分stateをscaled blockへ持ち上げ、levelとともに小さくなるobservableを測る構造である。これを「共通乱数だけで量子shot varianceを削減」と要約しない。正規化されたstateの測定から元補正量へ戻すには、原文Eq.(48)のpath依存norm係数Sも必要になる。

PRへ適用する際の未取得事項は次のとおりであり、これは本調査の適合性判断である。

- 元H stateを含むcontrolled全interrogationに対する同じ複素targetのtelescopingと、finite-RTE weight/biasの対応。
- 非unitary blockの実現、成功flagと全試行費、Sの取得・補正精度。未知state依存normを無料で既知としない。
- total path上のmoment/rangeと有限confidence、追加ancilla・測定・dilationのnative full価格。Theorem 1のfine＋coarse term countをそのままこの実費にしない。

したがって、**測定構造を変える機構としては具体的だが、現PRの次の数値pilot候補にはまだしない**。先にアクセス/測定/費用の有限仕様を示す必要がある。論文一般の否定や新しいno-go証明は行っていない。L-QMLもcoherentアクセスを追加する別路線であり、現在のRTE generatorの存在だけで利用済みとはしない。

## 4. 相関・補助量・その他の境界

| 方式 | 変更されるもの | 今回の選別 |
|---|---|---|
| 古典pathのantithetic/common-random-number coupling | path条件付き平均間の共分散 | 独立測定の条件付きnoiseを消さない。level差の真平均を無料評価した古典simulationの結果を採用しない |
| inter-shotでpathだけを相関させる | outcome列の依存 | mean保存とvariance和・全共分散、適用可能なconcentrationが必要。IID Nを流用できず、具体的追加機構は未特定 |
| PRHS/qSHIFT型inter-step correlation | ordered operator/channel mean、bias、signed weight | 既存監査を継承。marginal一致はcoherent mean保存でない。C/O3側の近接課題として扱う |
| 別の量子referenceを測るcontrol | joint measurementとreference平均の取得 | reference期待値・共分散・追加回路が未知。別shotで測ったreferenceを「同shotの相関」としない。無料E₀/state仮定を足さず、今回支持せず |
| 古典量子Monte Carloのtrial wavefunction control | state振幅やlocal energy評価 | 古典VMC等のアクセスを、元H groundを量子状態として利用できるという仮定から導かない |
| empirical Bernstein/anytime停止だけ | 統計保証と停止規則 | 既存O2-S1の既知処理。candidate/baseline双方へ適用し、分散削減の新原理にはしない |

## 5. B/Cの再調査を広げない

[既存文献索引][OLDLIT]のUWC（L03）、PR（L01）、SPRINT/GRADE（L02）、TE-PAI/TETRIS（L08–10）、PRHS/qSHIFT（L22–23）と固定[相関監査][CORR]の該当対応を再利用した。今回これらの全論文・後続訂正を新たに全面比較したとはしない。

UWCのround別表現と全RPE費用、SPRINTの位相shift/減衰/合成、TE-PAIのchannelとTETRISのHadamard用途の区別は既存調査が扱っている。BはAのaxis confidenceをbranch/energyへ戻す層、Cはmean/biasを変えるO3の層として整理する。新optimizer・randomized PF・合成方式をここで設計しない。

[CV]: https://arxiv.org/html/2502.08735v1
[ML]: https://arxiv.org/html/2604.26865v2
[QML]: https://arxiv.org/html/2012.06283v2
[EB]: https://arxiv.org/pdf/0907.3740v1#page=2
[CS]: https://arxiv.org/html/1810.08240v9
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[OLDLIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/SOURCES.md
