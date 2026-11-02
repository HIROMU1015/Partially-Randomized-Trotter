# BM-0 DF情報・技術目標・claim比較 v1

2026-10-05 JST。**構成候補のreview資料**。DF-specific new methodの成立、性能改善、BM-1実行を認定しない。
対象は[数学仕様](bm0_native_sequence_and_error_spec_v1.md)の固定A/B/R列と、mによる頻度比変更。

## 1. Claim単位の一次本文照合

次の一次本文の関連箇所を2026-10-05に確認した。文献の不存在証明や全引用網の監査ではない。
返却review内のChatGPT citation番号は可搬な出典ではないため、ここでは一次URLと本文locatorを使う。

| 一次文献・locator | 既知内容 | B-Mに残し得る内容 | 今回主張しない内容／未解決比較 |
|---|---|---|---|
| [Poulin et al., arXiv:1406.4920v1](https://arxiv.org/html/1406.4920v1), §VI、Eq.(26)、§VI.1、§VII | 異なる刻みのcoalescing、係数以外の重要度。sum-of-squaresとの併用も検討し、その検討では改善なし | 特定DF-native変更を採る具体的評価・列生成が既知対照より有用か | multi-rateやDF併用自体の新規性。過去の不改善を今回のno-goとはしない |
| [Casares et al., SPRINT, arXiv:2606.30741v1](https://arxiv.org/html/2606.30741v1), Fig.1、§III.1–3、§IV | 群ごとの異なるPF／randomized tail、既知BCH・near-integrable構成、実装を含む設計 | 更新頻度変更について、追加誤差とnative costをつなぐ限定手順の差 | 三層architectureの新規性。次数と頻度の変更は同一でない。SPRINT全面比較・Toffoli/RZ換算は未実施 |
| [Hagan–Wiebe, Composite Quantum Simulations](https://quantum-journal.org/papers/q-2023-11-14-1181/), [arXiv本文](https://arxiv.org/html/2206.06409), §4–4.1、Theorem 5 | Trotter/qDRIFT partition、outer/inner誤差の会計 | first-moment有限RTEを保持したnative列と共通task比較 | diamond-distance channel保証をcoherent signal保証へ転用しない。partition一般論は既知 |
| [Maxwell et al., arXiv:2606.30738v1](https://arxiv.org/html/2606.30738v1), §III.2–3、§III.2.3、§IV.1 | compact BCH、grouped commutator、CDF chemistryでの評価。三次のO(L) formal commutators | 同情報のcompact法を上回る評価の情報保持・取得cost・安定した選択のいずれか | 「DF小行列」「grouping」だけの新規性。展開word法が高速とは未証明。spectral評価をfinite signal certificateとしない |

特にMaxwellのCDF例も重なる。現時点で明確なnew-method deltaは未確定。
BM-1を提案する目的は、具体的な評価候補が有用かを少数controlで判別すること。
同情報の既知法と同じ結果ならapplication/technical scopeへの縮小判断をGPTへ返す。

## 2. 狭い技術目標と情報access

候補目標：**登録したA/B groupの頻度比変更について、mに共通なleading寄与と変わる内部寄与を
DF情報から評価し、native列・fusion後costとともに採否を出す。** generic optimizerへの入力追加だけを寄与にしない。
K_floorとK_Aの共通化は既知BCH代数の利用。これ自体を新定理としない。

| 量 | I1設計での取得案 | 既存資料の状態／追加作業 |
|---|---|---|
| DF G_i、lambda_i、one-body、c、順序、residual分布 | input conventionを固定 | sourceの規約は読める。現分子の数値は今回取得しない |
| N-sector証明、完全なlogical blockの保存性 | 演算子の数保存／adapter semantic test | 保存stateのNだけでは不十分。新BM adapter検証は未実施 |
| [G_i,G_j]、nested小行列、DF word係数 | 下記の小行列展開 | 分子別値はMISSING。Aのaggregate costやBF scalar cacheから補完不可 |
| K_floor／K_A leading評価 | compact BCHと同じ係数で評価 | 数値未取得。finite-T remainderは別義務 |
| native回数・既存exact fusion | text仕様から列count | 文法案あり。actual BM列は未生成 |
| basis transition／deterministic／random compiled cost分解 | 新しい限定wrapper比較が必要 | A固定evidence attributionでも分解はMISSING。saved totalから逆算しない |
| exact bias、state/signal、exhaustive winner | I2診断／評価専用 | BMモデルの値は未取得。I1選択へ戻さない |

Aの参照はreview指定commitの
[evidence attribution](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c23453c541700c6a41ba71fc5ec9323b53858d6/docs/research/pr2_post_m2_evidence_attribution.md) §C/D/F。
他Trackのruntime/cacheを新しいB science dataとしない。

## 3. 具体的な評価adapter案：ordered DF word

以下は実装前の代数案で、これが既知評価を上回るかは未確認。
`Z_i=F(G_i)`、`D_i=lambda_i Z_i^2`とする。既知bilinear恒等式
`[F(G),F(H)]=F([G,H])`と、N-sectorで`||F(M)||<=N||M||`を使う。
`C_ij=F([G_i,G_j])`なら

\[
[Z_i^2,Z_j^2]=Z_i C_{ij} Z_j+Z_i Z_j C_{ij}
 +C_{ij} Z_j Z_i+Z_j C_{ij} Z_i.
\]

ordered word `W=F(M_1)...F(M_d)`に対して

\[
[F(G),W]=\sum_t F(M_1)\cdots F([G,M_t])\cdots F(M_d),
\qquad [F(G)^2,W]=F(G)[F(G),W]+[F(G),W]F(G).
\]

従って三つのsquared DF generatorsのnested commutatorは、最大24個のdegree-4 ordered wordsへ
直接展開できる案となる。one-bodyはdegree-1、identityのcommutatorは0として扱う。
係数lambda、-i、BCHの有理係数を保持し、**同じ順序のwordだけを係数結合してから**三角不等式を使う。
非可換wordのfactorをソートして同一視しない。

wordの粗い評価は`N^d product_t ||M_t||`。
例えばpairwiseには`4 |lambda_i lambda_j| N^3 ||G_i|| ||G_j|| ||[G_i,G_j]||`が得られる。
この粗いbound自体は新規claimではない。得られるchiは**leading係数のnorm評価**であり、
有限時間biasの保証にはremainderとfinite-RTE誤差、数値guardが追加で要る。

pair診断はdense積でO(L^2 n^3)。全tripleを素朴展開すればworst caseはO(L^3 n^3)、
word inventoryもO(L^3)。全nested評価をO(L^2)と書かない。
同じgroupのK_floor／K_Aを一度評価した後のm比較は再利用できるが、
cache再利用・共通化だけで新規性や実用速度を主張しない。
Maxwellのcompact式は三次をO(L)個の**formal commutator**にするため、
word展開数とformal commutator数を混ぜて速度比較しない。

## 4. 同情報の強い対照と未証明事項

| 対照 | 共通条件 | 判別する差 |
|---|---|---|
| norm-only grouped BCH | 同じnative列、DF fragment norm、candidate domain、cost | 小行列の非可換構造を使う価値 |
| general compact BCH + 同じDF word backend | 同じG_i/N/access、canonicalization、budget | DF-specific変更adapterが一般法以上の情報／取得costを持つか |
| DF変更adapter | 共通K_floor、変動K_A、同じ列とbudget | m比較の構成と選択。既知対照が同じならmethod deltaなし |
| 小domain exhaustive oracle | 同じ列をI2で共通採点、評価専用 | I1選択のmiss/regret。oracle winnerを設計結果としない |

norm-onlyは`||[X,Y]||<=2||X||||Y||`で同じBCH構造を評価する案。
DF backendは上記ordered words。compact法でもそのbackendを使用できるため、
それとDF変更adapterが代数的に同じになる可能性を先に明記する。
一致した対照を外してnorm-only比較だけでGOにしない。

未証明／未実装：finite-T remainder、boundのtightness、sector semantic adapter、canonicalizationの
数値安定性、取得時間/memory、既知法との同値・差、I1からのaccuracy/shot予測、actual basis cost。
厳密な有限時間保証を新pilotの必須claimにするか、leading heuristicとoracle評価に限定するかはGPT review事項。
heuristicならその旨を仕様と結果へ明示し、途中でcertificateへ読み替えない。

## 5. 過去STOPとの差と非claim

P-D/B-Fのmodel・係数設計から、BM候補はnative実行列へ変更対象を移す。
ただし「特徴量を変えたoptimizer」だけならR3のSTOP理由と同じ。
FRのphase/radius結果は別evidenceであり、BMを自動で正当化しない。
coalescing、三層hybrid、importance sampling、相対basis fusion、BCH一般式は既知利用として記載する。
new algorithm、最良schedule、分子での改善、oracle-free保証、compiled gain、独立validationを今回は主張しない。
技術差が具体化できない場合の縮小／終了と論文着地点はGPT側へ戻す。
