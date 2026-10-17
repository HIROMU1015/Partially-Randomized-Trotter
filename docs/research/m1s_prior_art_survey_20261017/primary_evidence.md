# 一次資料の定理・数式・アルゴリズム構成

2026-10-17 JST。数式は原論文または既存repository資料からの引用・記法の対応整理であり、新規証明ではない。
ページは指定PDFの1始まり。著者の保証を記録し、全定理の独立検証や数値再現をしたとは表示しない。
版・DOIは[文献一覧](bibliography.md)。内部資料の固定リンクは[レビュー索引](review_index.md)。

## E01：共有labelをcoherent branchesへ載せる先行例

[MLMC-qDRIFT, arXiv:2604.26865v2](https://arxiv.org/pdf/2604.26865v2)、
pp5–12、§§2.1–2.3。Eq.(14)はfine経路で全labelを使い、Eq.(15)は奇数labelだけを倍角で使う。
Eq.(30), p10は同じdraw列によるfine/coarse状態をcontrol二枝のsuperpositionに置く。
したがって**coherent controlと共有indexの併用**は明示的な先行構成である。

同p10 Eqs.(31),(32)はPauli observableの場合
$$
\mathbb E[2Z_c\otimes O]=Y_\ell,\qquad
\mathrm{Var}(2Z_c\otimes O)=4-Y_\ell^2
$$
を示す。共有乱数でpath期待値差が小さくなっても、固定norm observableの測定分散は小さくならない。
続くEqs.(33)–(46)はscaled difference-state／block observableという別の構成で対処する。
Eq.(37)のblock mapは一般に非unitary、§2.3.6はdilation実装を別に議論する。
Thm.1, p8のMLMC資源scalingをM1-Sへ転記しない。

**差**：同じ時間方向のfine/coarse、異なる角度・列長、observable差。
M1-Sの同深さopposite-sign wordやsecular attenuationの同一定理ではない。

## E02：同じ乱数から対称積を作るrandomized PF

[Childs–Ostrander–Su, 1805.08385v2](https://arxiv.org/pdf/1805.08385v2)、p3 Eq.(6)。
同じpermutation $\sigma$を使う
$$
S_2^\sigma(t)=\prod_{j=1}^L e^{tH_{\sigma(j)}/2}
              \prod_{j=L}^1 e^{tH_{\sigma(j)}/2}
$$
という対称wordを定義する（論文の複素時間記法）。
p5 Lemma 1は $\|U_j-V\|\le a$、$\|\sum p_jU_j-V\|\le b$から
平均channelのdiamond差を $a^2+2b$で抑える。
Eqs.(15)–(17)のforward/reverse一次公式の混合とEq.(6)の対称公式は別の構成。

**共通**：回文、順序相関、mean operatorを使う誤差解析。
**差**：全項permutationとIID replacement、係数ごとのhalf angleと等角Pauli sampling。
wordの見た目だけでM1-Sとのensemble同値を主張しない。

## E03：平均演算子と複素信号の非漸近bound

[Morisaki–Fujii, 2609.36694v1](https://arxiv.org/pdf/2609.36694v1)、pp2–4,6–7。
Eq.(3)は確率$p_j$で$e^{-i\delta H_j/p_j}$を選ぶchannel、
Eqs.(22)–(25)はHadamard平均信号とmean evolution operatorを明示する。
Eq.(16)の
$$
D_p=\sum_jH_j^2/p_j-H^2=\mathbb E(X_j-H)^2\succeq0
$$
に対し、Appendix Eq.(35), p6は
$$
\|\overline V_{p,t}^{(N)}-U_t\|_1\le\frac{t^2}{2N}\operatorname{Tr}D_p
$$
を全実時間・正整数Nで与える。Eq.(26)のHaar平均複素信号biasはさらに(1/d)を持つ。
Eqs.(20),(27)ではnormalized HS normに比例したsamplingが当該boundを最小化する。
単一Pauli項ならop/HS samplingの重みは一致する。

**比較上の注意**：Haar平均だけの論文、channelだけの論文、長時間評価のない論文と説明しない。
一方、理想unitaryへのbias boundとM1-Sの有限減衰secular参照への誤差は同じstatementではない。
HS／ancilla liftでどの結果まで直接導けるかは未証明として残す。

## E04：partial evolutionと± control、RPE、費用

[Güntherほか, 2503.05647v2](https://arxiv.org/pdf/2503.05647v2)。
§IV Eqs.(22)–(26)、Appendix A.1 Lemma A.1/Eq.(A14), p20は
involution-tail qDRIFTのHadamard平均を
$$
g(\tau,r)=(1+\tau^2)^{-r/2}\langle\psi|(I-i\tau\widehat H)^r|\psi\rangle
$$
とする。ここでの$\tau$・正規化Hamiltonian・angleの定義をM1-Sの線形$\theta$と黙って混ぜない。
phase distortionの補正と減衰の状態依存性も同節で扱う。

Appendix A.4, p24 Eq.(A46)のdirectional controlは
$$
C_\pm U=(|0\rangle\langle0|\otimes U^\dagger+|1\rangle\langle1|\otimes I)
(|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes U).
$$
直後の本文は$U,U^\dagger$へ**独立なRTE randomness**を適用すると明記する。
決定論部分のgate数半減を述べ、random部分まで半減とはしていない。
Appendix B, Theorem B.2（pp27–29）はguiding overlapとRPE RMS保証、
Appendix E（特にEqs.(E20)–(E23), pp42–43）はdet/random費用の会計。
これはM1-Sのshared joint lawの保証とは区別するが、応用・費用面で直接の先行基盤である。

## E05：確率的cross termとnormalizationの価格

[Peetz–Smart–Narang SCU, 2407.21095v2](https://arxiv.org/pdf/2407.21095v2)、p2 Fig.1/Eqs.(1)–(3)。
ancilla X/Yで $\mathrm{Tr}[Oe^{i\phi}V_1\rho V_2^\dagger]$を取得し、
Eq.(2)はnormalization $\lambda$によるshot overhead $\lambda^2$を戻す。
p4 Fig.3/Eqs.(6)–(8)のCTSは左右propagatorを**独立sample**する。
cross termを実装できることと、M1-Sの特定joint law・平均写像・長時間保証は別の層である。

[Wan–Berta–Campbell, 2110.12071v2](https://arxiv.org/pdf/2110.12071v2)、p3 Fig.1/Lemma 2/Eqs.(7)–(10)、
p4 Algorithm 1はrandom unitaryのHadamard測定でCDFを推定する。
Lemma 2の $\sum_k b_k\le e^{t^2/r}$ とcontrolled rotation数$r$により、
sample complexityと一回費用を交換する。M1-Sのfixed-time無補正Arg taskと同一の推定器ではない。

## E06：coherent controlはsystem channelだけでは定まらない

[Abbottほか, 1810.09826v3](https://arxiv.org/pdf/1810.09826v3)、§§3–4、pp3–4 Eqs.(2)–(5)、
Appendix A。Eq.(5)のcontrol-target出力は、各channel $\mathcal C_i$に加え、
実装依存のtransformation matrix $T_i$を使うoff-diagonal項を持つ。
同じsystem channelでも相対位相・環境との関係が違えば干渉出力が違う。

M1-S比較では各branch marginalのchannel一致だけでjoint coherent信号一致としない。
これは一般的な意味論上の注意で、同論文からM1-Sの$W/v/\rho$やRPE成功が導かれたとはしていない。
非CPなcross blockと、物理的な全体CPTP回路を区別する。

## E07：qDRIFT・composite・concentration

| 一次資料 | 確認した構成・保証 | 比較で使える範囲 |
|---|---|---|
| [Campbell 1811.08017v2](https://arxiv.org/pdf/1811.08017v2)、pp2–3/Fig.1、p8 App.D Eq.(D1) | 項重みに比例するIID exponentials。controlled H′=$\vert1\rangle\langle1\vert\otimes H$ | qDRIFTとcontrolled PEへの使用は既知。systemのchannel精度を枝間joint lawの保証に置換しない |
| [Hagan–Wiebe 2206.06409v3](https://arxiv.org/pdf/2206.06409v3)、pp8–9 Definition3/Thms.4–5 | AをTrotter、BをqDRIFT、outer composition、diamond誤差とexponential query費用 | partial splitの一般原理は既知。coherent/RPE固有の条件は別 |
| [Pocrnicほか 2306.16572v1](https://arxiv.org/pdf/2306.16572v1)、§II D–E、pp6–7 Eqs.(11)–(16) | composite channel、inner/outer orderと分割の自由度 | real/imaginary timeの対象差、channel誤差、query costを保持 |
| [Chenほか 2008.11751v4](https://arxiv.org/pdf/2008.11751v4)、pp2–5 Algorithm0.1/Thms.1–2 | mean channel、mean operator、単一random trajectory、固定／全状態concentrationを区別 | M1-S wordの内側依存を無視して独立2N gateの定理を適用しない |

## E08：phase・減衰・modeの先行解析

[Guほか, 2208.04100v2](https://arxiv.org/pdf/2208.04100v2)、p2 Theorem 1は
Hermitian Kraus noiseによるunitary-channel固有位相の先導次数での不変性を扱う。
SI p8 Eqs.(S7),(S8)はisolation distanceと摂動条件、p9 Eqs.(S11)–(S16)は左右eigenoperatorsと信号展開。
M1-Sで使うpair-sum／HS normとは対象を合わせる必要がある。
一般的な一次phase安定性やnon-normal modeの利用自体を独自技法としない。

[Yi–Crosson v1](https://arxiv.org/pdf/2102.12655v1)、p3 Corollary1/Lemma3、
Appendix C/E、および[出版本文](https://www.nature.com/articles/s41534-022-00548-w)の
Framework／Lemma3／Theorem2は、deterministic PFのeigenvector leakageとenergy shiftを分ける。
[出版SI](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41534-022-00548-w/MediaObjects/41534_2022_548_MOESM1_ESM.pdf)
Supplementary Note 4, Eqs.(50)–(57), p5とNote5 Eqs.(61)–(68), p6を今回確認。
off-diagonal leading perturbationの場合の一次energy shift消失と二次boundは既知の比較根拠。
preprintと出版版は定理構成・番号が異なるため、式番号を跨いで引用しない。

**残る問題**：これらの結果がM1-Sの具体的secular bound、三次係数、共鳴floorを
どこまで直接含むか。ここで新しい還元証明を作成していない。

## E09：random evolutionを位相推定に接続する他の先行例

[Kivlichan–Granade–Wiebe, 1907.10070v1](https://arxiv.org/pdf/1907.10070v1)、
pp2–4 Fig.1/Theorem1/Eqs.(4),(5)。Hamiltonian列を乱択し、列中の最小ground gapと
隣接Hamiltonian差からphase誤差・失敗確率を評価する。
これらのgapをM1-Sの固定Hのpair-sum gapと同一視しない。

[Li, 2111.10430v3](https://arxiv.org/pdf/2111.10430v3)、pp8,11,16 Theorems2.2/3.1/4.1、
§4 Eqs.(40)–(56)はimperfect state/unitaryとrandom-unitary register-QPEを扱う。
single-ancilla RPEのdecoder・round/axis配分へ自動的に同じ保証が付くわけではない。

## E10：RPEの訂正とbounded-shot推定

[Kimmel–Low–Yoder, 1502.02677v3](https://arxiv.org/pdf/1502.02677v3)、
**p1 Errata**、§Vと付録。v3冒頭は旧解析のπ/2条件の問題とπ/3条件、
関連文献による定数overhead付きの修復を説明する。
[出版Erratum DOI](https://doi.org/10.1103/PhysRevA.104.069901)も確認したが、
その別PDFは今回取得失敗。v3冒頭を実際に読んだ根拠とする。
π/3を任意のphase estimatorに普遍的な必要条件としない。

[Hoeffding原論文](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)、
原誌p16／取得PDF p5、Theorem2 Eq.(2.6)を紙面で確認。
独立有界変数に対する一側集中boundを、既存V1レビューは±1 X/Y測定・二側化・union bound・円板幾何へ特殊化している。
[既存γレビュー §4 Eq.(2)](../m1s_gamma_v1_completed_results_scientific_review_20261017.md)の転記は
$$
M_m(\gamma)=A_m^-\sin(\gamma-b_m^+),\quad
N_{m,a}^{\rm suff}=\left\lceil\frac{4\log(2/\alpha_{\rm axis})}{M_m(\gamma)^2}\right\rceil,
\quad A_m^->0, b_m^+<\gamma<\pi/2.
$$
十分予算であり最小必要shot数でない。γ境界の式は同レビュー§5、整数丸めの注意は§5.2。
今回の新しい定理・数値forecastにはしない。

## E11：費用を戻す最適化

[Kiss–Grossi–Roggero, 2212.05952v2](https://arxiv.org/pdf/2212.05952v2)、
§3/§4、p5 Theorems1–2、p6 compositeへの接続。
importance-rescaled qDRIFTのbias／concentrationと項別費用を扱う。
異なるsampling lawが作るbiasまで評価する点と、次の固定ensemble再重み付けを区別する。

[Cugini–Atif–Subaşı, 2603.13495v1](https://arxiv.org/pdf/2603.13495v1)、
§II Theorem1, pp2–3 Eqs.(4)–(13)。固定ensembleに対し
$$
\mathrm{NC}_q=\mathbb E_q[c]\mathbb E_q[(p/q)^2],\qquad
q^*\propto p/\sqrt c
$$
で同論文のworst-case second-moment proxyを最小化する。§III/Theorem2はbiasの不変性。
これはM1-Sへsamplingを変更すればphase biasが改善するという結果ではない。
CLT的proxyを有限confidenceの最小必要shotsと呼ばない。

[Kanasugiほか, 2603.22778v2](https://arxiv.org/pdf/2603.22778v2)、
pp4–6 §II B.2–4 Eqs.(10)–(18)、p21 App.B Eqs.(B3)–(B5)は
qDRIFT／partial RPEの減衰とSTAR費用を接続。
同論文のcost coefficient γ=1（qDRIFT）／γ=2（RTE等）と、本調査の許容位相γは別の記号用途。
STAR logical rotation数をClifford+T／compiled CXの価格に読み替えない。

## E12：高次化・経路相関・echoとの境界

| 資料・固定箇所 | 原論文の内容 | M1-Sとの差 |
|---|---|---|
| [Kim–García-Pintos 2608.07720v1](https://arxiv.org/pdf/2608.07720v1)、pp2–3 Eqs.(2)–(13), Thm.1 | $H=A+\alpha B$、random insertion u、u/1−uのsymmetrization、mixed-unitary diamond誤差 | continuous insertionとlabel共有を混同しない。強いaccessのTHRIFTは別モデル |
| [qFLO 2411.04240v2](https://arxiv.org/pdf/2411.04240v2)、pp2–5 §§2–3/Thm.1、§7 | qDRIFT step-size flow、Richardson、observable推定の深さと反復費用 | M1-S無補正signalのphase/radius比較と別のbias処理 |
| [qSWIFT 2302.14811v2](https://arxiv.org/pdf/2302.14811v2)、pp2–4導入/§II、§III以降 | 高次channel補正、one-ancilla observable測定 | 今回はoverview関連箇所。全補正定理・全shot定数は再監査していない |
| [PRHS 2608.29756v1](https://arxiv.org/pdf/2608.29756v1)、p2 Eqs.(3)–(8) | time-slice間のsigned correlated pathで高次matching | shared ± lawとは異なる。負係数の正samplingとsign weight、normalizer費用を要する |
| [qSHIFT 2604.26263v3](https://arxiv.org/pdf/2604.26263v3)、§III.1/Alg.1、p6 Eq.(17) | 履歴依存samplingと累積weight | 誤植・版差は旧監査の注意を維持。controlled first momentへの移送未認証 |
| [SYK/TETRIS 2507.07530v1](https://arxiv.org/pdf/2507.07530v1)、pp8–9 Eqs.(31),(32) | 独立U/U′の二枝、normalization付きobservable、O=Iのmirror-on-average | 同順sign reversalと異なる。mirror名の一致を同一lawとしない |
| [Proctor 2008.11294v2](https://arxiv.org/pdf/2008.11294v2)、App.IV pp12–16 | circuit inverse/quasi-inverse、Pauli/fiducial、known output | 回路能力benchmarkであり相対energy位相の保証ではない |
| [Wallman–Emerson 1512.01098v3](https://arxiv.org/pdf/1512.01098v3)、§I A–B、p4 Thm.1 | twirlingと補償、easy-gate noise仮定の下のstochastic Pauli noise | 理想回路を保つnoise tailoring。qDRIFT sampling biasとの目的差 |

PRHS/qSHIFTの表示不整合は[既存2026-10-13監査](../../tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md)
の指摘を引き継ぐ。今回は主構成を再確認した範囲で、全補遺の再監査はしていない。

## E13：未取得の基礎理論を根拠にしない

[Davies, *Markovian master equations* (1974)](https://doi.org/10.1007/BF01608389)は
弱結合・長時間・平均化に関する背景文献として収集した。publisher abstract／書誌は確認、
原論文PDFの取得はDNS障害で失敗。T1a/T1bと同じ定理、直接系、あるいは還元不能とは認定しない。
一般平均化・pinching・正のspectral measureとJensen下界がどこまで既知定理の特殊化か、
後続GPTが原文を追加確認すべき未解決事項として扱う。
