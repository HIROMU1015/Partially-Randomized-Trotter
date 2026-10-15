# Track B／M1-S G0-T1：S0正式理論研究契約

2026-10-16 JST。**`M1S_G0T1_S0_FORMAL_CONTRACT_ADOPTED_THEORY_EXECUTION_NOT_AUTHORIZED_MANDATORY_STOP`**。

**ユーザーがG0-T1／第三案A′の限定研究契約を正式採択した。T1a/T1bの新規理論実施は未承認。** 本書は採択内容を固定する規範文書であり、数学的定理の成立・研究結果・次段階GOを認定するものではない。今回の新証明・科学計算・compileは0件。

## 1. ユーザーによる正式採択の根拠

今回のユーザー提供指示「Track B／M1-S G0-T1研究契約の正式採択・S0最終固定」は、[最新GPT科学レビュー][GPT]、特に§9に従い、**「M1-SのG0-T1研究契約を正式採択します。」**と明示した。また、正式契約の新規作成・commit/pushを許可し、T1a/T1bの証明・理論研究の実行を明示的に除外した。

採択主体はユーザー、科学的推奨の根拠はGPT、今回のCodex担当は**採択済み内容の文書固定・整合性確認・公開**である。原本レビューに記された「ユーザー採択待ち」はレビュー作成時点の履歴として保持する。今回の採択を理由に過去原本や候補契約の状態を書き換えない。

採択内容と実施未承認フラグを[採択記録][ADOPTION]、出典・byte identityを[証拠索引][INDEX]へ分離して保存する。採択記録は**研究契約の決定記録**であり、runnerのauthorizationや実行markerではない。

## 2. 採択対象・研究順序・除外範囲

正式採択する範囲は**G0-T1のみ**。T1aをQ1-core＋Q2-min、T1bをQ3-coreとして限定する。順序は**A′：T1a → Checkpoint 1 → T1b → Checkpoint 2 → 必要ならGate 0の判断**で固定する。

| 採択する契約内容 | 保持する境界 |
|---|---|
| T1a：完全乱択の平均化保証 | 固定有限次元・Pauli/Hermitian involution・孤立した対象固有state。主比較C-sec |
| Checkpoint 1 | T1aの数学的有効性・既知性・T1bへの可搬性を判断。自動T1b開始なし |
| T1b：部分乱択への先導次数の移送 | exact deterministic half＋中央Pauli/involution-tail、固定r・小stepの系列 |
| Checkpoint 2 | 重要GPT科学レビューが必須。ただしその時点のレビュー開始承認が必要 |
| Gate 0 | Checkpoint 2以降に必要性・対象・条件を別途判断。科学計算の許可なし |

一般guiding state、任意縮退/Jordan・全次元一様な一般理論、一般Hermitian/DF event、内部H_Dの実装PF、Q4/RPE保証、Q5/native総資源・forecastは初期scope外。固定step幅のr→∞、rと系サイズの同時増加も初期T1bの主極限から除外する。

Parent allocation D＋E、M1-Aの補助的位置付け、M2等の代替、旧科学分類・SOURCE・result・freeze・authorization・marker・STOPは保持する。M1-Aは必要に応じたbaseline理論整理の参照であり、今回の新理論実行許可ではない。

## 3. 固定資料・原本同一性・版の優先順位

Repositoryは `HIROMU1015/Partially-Randomized-Trotter`。本書の採択内容は今回のユーザー指示、[GPT採択レビュー][GPT] §9およびその説明、直前の[修正候補][CANDIDATE]を基点とする。衝突時は今回明示された採択内容を優先し、旧候補の「C-sec/C-log未選択」「joint path未選択」を本書へ持ち越さない。

| 資料 | 固定commit | 今回の役割 |
|---|---|---|
| 最新GPT採択レビュー原本 | `27df63ede3513ed3982ea4c341823a3f18fa2b83` | [科学的根拠][GPT]。§§4–7/9–11の選択・受理境界 |
| 直前S0修正候補・索引 | `0d84c2181de128e496bdc1e6161264cdf84eeba1` | [候補本文][CANDIDATE]／[候補索引][CANDINDEX]。旧版を上書きしない |
| 前回GPTレビュー | `d1039afae54520f15545177a7c852407830a0927` | [局所高次誤差・dressingの留保][GPTPREV] |
| G0事前契約 | `0f0952614dcf5c06d3d63c21b5d49091b8ecce37` | [旧Q1–Q5とデータ所在][PRECONTRACT]。広いscopeの一括採択ではない |
| G0-pre | `8b7666d0270f7fac01e430e195bef7b58ac0116c` | [先行研究・ordered law・費用の既存照合][G0PRE] |
| P0.5 | `ed3ddeacc88e998e7bc1d8c0db8163e106312791` | [既知定理と対象差、H6、費用fieldの既存監査][P05] |

最新原本は、ユーザーが本文で指定したrepository直下の `track_b_m1s_g0t1_s0_formal_adoption_gpt_scientific_review_20261016.md` から取得した。

- 保存先：`docs/research/track_b_m1s_g0t1_s0_formal_adoption_gpt_scientific_review_20261016.md`。
- **44,886 bytes**、SHA256 **`e54c8cd4ddabf97aca5fb37057d2af64fd3bc468fe528c94f06b9e12d4a278d0`**。
- Git blob：`d02a8432fa7765fbffcb6cd16faadbcd3332d377`。原本保存commitは上表の最新GPT欄。
- 保存前に対象pathと同一blobの全ref履歴を確認し、既存保存がないことを確認。byte copy、commit/push、GitHubからの再取得を照合した。本文・改行・当時の採択待ち記録を変更していない。

表の資料は候補索引を含む7ファイルを固定refから取得し、復号bytesとGit blobを照合する。既存の科学検討・文献検索・数値監査は再実行しない。過去レビューに登場するClaude原本を今回独立に受領・照合したとは扱わず、GPTの引用・評価の範囲を越えて補完しない。

## 4. M1-Sの正式研究目的

**Gate共有±を用いたdirectional randomized Hamiltonian simulationについて、coherent平均信号の減衰と位相に関する保証を構築し、その機構が部分乱択Hamiltonian simulationへ移送できる条件を明らかにする。**

研究対象は、完全乱択の長時間平均化、局所高次誤差・スペクトル結合、部分乱択への先導次数の移送、そのdressing/aliasing/共鳴条件、改善因子rhoの数学的意味と成立域である。

方向制御、qDRIFT、回文word、共有乱数そのものを新アルゴリズムの発明と主張しない。中心的な新規性**候補**は既存構成に対するcoherent signal保証と移送条件。T1aのみならtechnical note、T1bの有用な移送が得られれば限定理論claim候補となり得るが、契約採択の時点で成果・新規性・実用利益を認定しない。

## 5. T1aの正式対象・C-sec主比較モデル

### 5.1 固定law・state・厳密信号

完全乱択、決定論blockなし、固定有限次元のHamiltonianを対象とする。符号をP_jへ含めるPauli/Hermitian involutionの確率分解を使い、

\[
P_j=P_j^\dagger,\quad P_j^2=I,\quad p_j\ge0,\quad
\sum_jp_j=1,\quad h=\sum_jp_jP_j,\quad A_j=e^{i\theta P_j}
\]

とする。共通の線形角theta、同位置の二branchでlabel共有、異なるgate位置では同一lawのIID samplingを使う。右端先作用、時系列label `(j_1,...,j_N)` とし、

\[
B_1=A_{j_N}\cdots A_{j_1},\qquad
B_0=A_{j_N}^{\dagger}\cdots A_{j_1}^{\dagger},\qquad
\Phi_\theta(X)=\mathbb E_j[A_jXA_j].
\]

negative branchは**同順sign reversal**であり、全wordのliteral adjointではない。既存レビューが確認した `E[B_0^dagger B_1]=Phi_theta^N(I)` を出発点とする。Φを通常のCPTP channel `E[A_j X A_j^dagger]` と同一視しない。

第一の対象は、固定hの**孤立した固有state** `h|k>=e_k|k>`。厳密信号を

\[
z_N=\langle k|\Phi_\theta^N(I)|k\rangle
\]

とする。一般guiding stateの複素weightやRPEは初期主張へ追加しない。hの対象stateの孤立性だけで、演算子空間のpair-sum共鳴・小分母が消えるとは仮定しない。

### 5.2 正式に採択する比較信号

**C-secを主比較モデルとして固定する。**

\[
\mathcal L_1(X)=hX+Xh,\qquad
\mathcal C(X)=\mathbb E_j[(P_j-h)X(P_j-h)],
\]
\[
\mathcal D(X)=-\tfrac12\{I-h^2,X\}-\mathcal C(X),\qquad
\mathcal D_{\rm sec}=\sum_\omega\Pi_\omega\mathcal D\Pi_\omega.
\]

Π_ωは**Hilbert–Schmidt演算子空間**におけるL1の同一周波数 `omega=e_a+e_b` blockのスペクトル射影。厳密に同じ周波数の作用を残す。近共鳴clusterへ後からまとめ直すことは主比較対象の変更として扱い、黙って実施しない。厳密同周波数blockを定義することと、任意縮退・Jordanに関する一般保証の完遂は別である。

比較信号は

\[
z_N^{\rm sec}=\left\langle k\left|
e^{iN\theta\mathcal L_1}e^{N\theta^2\mathcal D_{\rm sec}}(I)
\right|k\right\rangle.
\]

**厳密なz_Nとこのz_N^secの差がT1aの主対象**。この定義・比較対象の採択は、誤差定理が既に証明されたという意味ではない。

## 6. C-logの補助的位置付けと契約改訂の境界

C-logは一stepの位相drift・三次以上の局所項の確認に使う**補助的な証明・検算手法**として保持する。今回その解析を開始しない。将来使用する場合、logの存在域・branch、theta=0からの連続選択、採用次数、C-secへの戻し方を明記する。

`Phi_theta^N=exp(N log Phi_theta)` という正確な再表示だけをT1aの完成結果としない。C-logへ主比較モデルを変更することは本契約の履行ではなく、科学契約の改訂である。必要性が判明した場合は理由・C-secとの差・誤差と受理条件への影響を記録し、ユーザー/GPTの科学判断へ戻す。

主targetを変えない補助的な証明手法の変更ごとに、同じ全面的なGPTレビューを強制しない。ただし、主要claimや保証の意味を変える選択をCodexだけで採択しない。

## 7. T1aの誤差保証対象・証明義務・受理条件

主評価座標は**複素coherent信号の絶対誤差**

\[
\varepsilon_{\rm sig}(N,\theta,k)=|z_N-z_N^{\rm sec}|.
\]

Nを自然数、`abs(theta)<=theta_0`、`N theta^2<=D_cap` とする**明示した長時間・有限減衰予算の対象族全体**で有効な上界を証明することが義務である。固定Nでtheta→0とするだけでは受理しない。theta_0、cutoff、重み汎関数、定数は証明で根拠を与える内容であり、今回恣意的な数値を選ばない。

| 必須の会計 | 将来の証明・報告で明示する内容 |
|---|---|
| 局所高次の累積 | 一step三次以上の剰余、使用norm、N反復と有限D_capに一様な制御。可換Pauli例に残る `O(D abs(theta))` 級の寄与を欠落させない |
| 有限角度の位相歪み | 主比較C-secとの差として扱う。log等で補正する場合もC-secとの関係を示し、局所誤差と二重計上しない |
| 非共鳴mode | 周波数差・coupling・小分母・重み。実質零の結合と小gapを区別し、bare energy gapを代入しない |
| 低周波・near resonance・pair-sum | 同一周波数blockへ残したものと残差へ出すもの、cutoff、実際に結合するweightを区別する |
| 対象stateのmode weight | 初期Iから励起され対象stateのmatrix elementへ寄与するmode。必要な左右mode・射影条件数を開示する |
| 依存性・安定性 | D_cap、gap、weight、次元、sampling law、normへの定数依存。全space normを使う場合はその強さと有効性を示す |
| 位相保証を付ける場合 | 例えば `abs(z_N^sec)>=z_min>0` とする**参照信号の下界**と、厳密信号がその近傍にあることを説明し、phase branchを固定する |

局所高次項の必要性は[前回GPT][GPTPREV] §4.3と[採択レビュー][GPT] §5.3の確認済み紙上検算を参照する。**今回その展開・反例を証明し直していない**。非共鳴・低周波項だけが小さい評価を、T1aの完成結果として受理しない。

受理に必要なのは、固定law/C-sec/stateで定義された誤差命題、全対象族で空でない有効域、仮定・定数・適用不能例、既知channel保証との意味のある対応、T1bへ渡せる周波数・平均化量・error budgetである。全次元一様、任意縮退・一般DF eventまでを初期完了要件にしない。

複素絶対誤差から位相へ進むのはsignal floor付きの追加保証に限る。近零では位相未定義／未判定と複素信号評価を分ける。位相保証をRPEのRMS energy error保証へ格上げしない。**会計・受理条件は採択済み、具体的な上界・定理の成立は未証明**である。

## 8. T1bのexact-half参照protocol

固定有限次元の `H=H_D+H_R`、固定 `lambda_R>0`、`H_R=lambda_R sum_j p_j P_j` を対象とする。H_Dはexact exponential、tailはPauli/Hermitian involution。符号・identity・constantをどちらへ配置したかをHamiltonian分解とphase台帳で保持する。

線形角と時間辞書は§9で固定する。前後halfを

\[
D=e^{+i\alpha H_D},\qquad D'=D
\]

とし、時系列label `(j_1,...,j_r)` に対して

\[
R=A_{j_r}\cdots A_{j_1},\quad
R^{\rm rev}=A_{j_1}\cdots A_{j_r},\quad
R_-=A_{j_r}^\dagger\cdots A_{j_1}^\dagger,
\]
\[
G_+=DRD,\qquad G_-=D^\dagger R_-D^\dagger
\]

と定義する。右端先作用で、前half→j_1からj_r→後half。negative branchは同じ位置labelを共有して同順にsign reversalする。R_-をR全体のadjointと置かず、primeもadjointと読まない。

gate位置間は同一lawのIID、step間も同じstep仕様でIIDとする。既存レビューがこの順序と独立性で確認した厳密な平均写像は

\[
\boxed{\Psi(X)=\mathbb E[G_-^\dagger XG_+]
=D\Phi_\theta^r(DXD)D.}
\]

s stepの枝干渉平均は固定lawならΨ^s(I)。stepや位置ごとに仕様を変えた場合に固定写像の乗算を流用しない。H_D=0ならD=I、Ψ=Φ^rの完全乱択へ戻る。空tail/λ_R=0は本ratio参照の外で、無理にrhoを定義しない。

controlled operatorはbranch polarityを明記した `diag(B_0,B_1)`、観測平均は `⟨k¦B_0^dagger B_1¦k⟩` を参照する。X/Y符号とsystem global phase・identity由来のcontrolled relative phaseを保持し、projective一致で代用しない。無条件平均とshot freshness・列再利用の統計保証は別契約である。

このΨ恒等式は出発点として既知であり、**減衰改善、有効mode、rho、RPE保証が成立したとの採択ではない**。内部H_DをPFへ置き換えた実装の同値・誤差・価格は初期対象に含めない。

## 9. T1bの正式joint scaling

次の辞書と**一つの主系列**を固定する。

\[
T_{\rm rel}=s\delta_{\rm rel},\qquad
\theta=\frac{\lambda_R\delta_{\rm rel}}{2r},\qquad
\alpha=\frac{\delta_{\rm rel}}4,\qquad
D_{\rm damp}=sr\theta^2\le D_{\rm cap}.
\]

- Hamiltonian、λ_R、次元、分割、sampling lawを固定する。
- **r≥1は固定の正整数**。rを系サイズやstep幅と同時に増やす一様保証は初期主張にしない。
- **delta_rel→0**、従ってtheta→0、alpha→0とする。
- sは正整数の系列で、**s delta_rel^2を有限範囲で管理**し、D_damp≤D_capを満たす。
- 非自明な減衰を扱う主系列では**D_dampが正の有限値へ近づく**条件を使う。D_damp→0は整合限界として区別する。

T_relは干渉信号の相対時間。理想±枝の各時間絶対値はT_rel/2であり、一stepの決定論halfは各delta_rel/4という規約である。alphaはexp(i alpha H_D)の指数係数で、係数を含めた個別gate角と区別する。

固定delta_relのままr→∞とする経路ではalphaが固定に留まるため、theta→0だけからdressing消失を推論してはいけない。**その経路は本契約の初期対象外**。有限alphaの理論、rと系サイズの同時増加は別の研究契約を要する。

T1aの完全乱択でも、H=lambda hという規格化を用いる場合の線形角の辞書 `N theta=lambda T_rel/2` を保持し、同じthetaだけを時間一致の根拠にしない。これは理想taskへの対応であり、実際の平均信号の厳密一致を主張する式ではない。T1bで小stepに伴う固有位相分離が縮むことを無視せず、摂動／分離の比と長時間累積を§10で扱う。

## 10. T1bの移送条件・dressing・outer bias

T1bの参照stateは**全Hの対象固有state** `H|k>=E_k|k>`。h_R固有stateとの一致を仮定せず、全HのstateとΨの左右有効mode・射影との関係を定義する。有効Hamiltonianを用いるなら、全Hとのmode差・位相bias・連続分枝を説明する。

必要な証明・分類項目は、有効mode、長時間の先導減衰係数、位相補正、dressingの絶対誤差、係数の相対誤差、pair-sum/alias、Dに伴う交差項、outer symmetric-step bias、rhoの適用域である。固定r・delta_rel→0・sの採択済み系列で定数・剰余を管理する。

### 10.1 状態ノルムと係数の誤差

`a=⟨k|H_D|k⟩` とすると、既存GPTレビューが確認した

\[
\|D|k\rangle-e^{i\alpha a}|k\rangle\|
\le |\alpha|\sigma_k(H_R)
\]

は状態ベクトルの上界として参照できる。これは今回の新しい証明ではない。次の層を別々に報告する：状態の変化、左右有効mode・射影の変化、減衰係数の絶対差、分母下界がある場合の相対差、rhoの誤差。

**状態の近さから減衰率やrhoの相対誤差が同程度と結論しない。** DXDはDXD^daggerという密度演算子の変換ではない。alpha sigma(H_R)やcommutator尺度の小ささだけで、coherent superoperatorのmodeやratio精度を保証しない。

### 10.2 共鳴・sector・outer splitting

同一周波数blockと、近共鳴・pair-sumへ結合するweight、小分母、有限stepのalias距離を明示する。Pauli-tailは粒子数/spinの目的sectorから出る可能性を残す。全Fock最小gapが小さいだけで自動STOPせず、実際の到達・結合を扱う。粒子数保存だけで到達可能空間の非縮退端点性を認定しない。

**H_Dをexactにしても、H_D/H_Rの非可換なouter splitting biasは消えない。** まず同じ分割の決定論的対称step参照に対する乱択の減衰・biasを扱い、次に真のHに対するouter symmetric-step biasを別項にする。これをQ4まで無視してよいとはしない。

GPT採択レビュー§6.3の局所O(delta_rel^3)・累積O(s delta_rel^3)等は、適切な条件で検討すべき**候補スケール**として参照する。具体的な係数・norm・一様剰余は未証明で、今回新しいboundを確定しない。T1bでは少なくとも存在・規模の分類を必要範囲で行い、RPE総error budgetや実費用への伝播はQ4/Q5に残す。

## 11. 改善因子rhoの扱い

完全乱択のh固有stateで `V_k=1-e_k^2>0`、`v_kk=Var_j⟨k¦P_j¦k⟩` とすると、既存レビューが支持する先導係数の代数は

\[
\rho_\infty=\frac{V_k+v_{kk}}{2V_k}
=\frac12+\frac{v_{kk}}{2V_k}\in[1/2,1].
\]

係数の式と、有限N/thetaで同精度を達成するイベント比は別である。V_k=0で比を定義しない。同相対時間・同減衰予算・位相条件・比較lawでの一様保証なしに、必要イベント数・shot数・総費用比へ読み替えない。

部分乱択の `rho_infinity^(H)` は**証明済み公式として採択しない**。旧資料の候補式、比較する両係数・normalization、全Hと有効mode、dressingと分母の正当性をT1bで調べ、出力を次の四つに区分する。

1. 成立。
2. 補正付きで成立（dressing補正版等を明示）。
3. 適用外。
4. 未解決。

零・極小分母、共鳴・alias、sector外結合で条件が破れる場合はdomain failure／未判定として保存し、比を便宜的に補完しない。有限時間rho(T)やsinc一次kernelを全次数の区間保証として受理しない。rho単独のGO/STOP、Gate 0の閾値、総資源削減率は採択していない。

## 12. Baselineと比較意味論

**比較するのは同じ理想Hamiltonian時間発展に由来する目標量であり、各方式の実際の平均信号が等しいとは前提しない。** 同じH、T_rel、初期参照state、分割・係数law（比較に必要な範囲）、理想的coherent phase目標を先に対応させ、各meanのbias・減衰を別々に扱う。

| 方式 | 対応させる意味論 |
|---|---|
| Ordinary qDRIFT | 片枝でfull relative timeを扱う。有限角度mean・減衰・biasを独自に評価する |
| 独立± qDRIFT | half-time二branchを独立draw。負枝順序・位相・sampling lawを明記する |
| Gate共有± M1-S | 同位置共有、同順sign reversal。採択したΦ/Ψの実際のcoherent平均 |
| 既存PR-RTE | unbiased numerator、normalizer、ordered PF、RPE側の保証を元のscopeと区別する |

full-wordを共有して負枝にliteral adjointを使う方式は別law。無条件平均とfresh-shot/reuseによる統計集中も区別する。理想物理時間／相対時間、複素信号／位相、減衰／位相bias、イベント数／必要shot数、controlled価格／総資源、oracle診断／実用predictor、厳密保証／探索proxyを分ける。

比較する減衰予算、精度・保証対象、情報アクセスを一致させる。実際のmeanや誤差が元から等しいとは仮定せず、各方式が同じ比較条件を満たすかを区別する。理論の対象stateはoracle参照であり、実機がexact groundやgap/weightを無料で知るという主張ではない。

既知qDRIFT/RTEの費用係数差、M1-Sイベント比、RPE期待総資源は別の層。control phase、precision、ancilla、HWP/catalyst、STAR、Clifford+T、compiled RZ/CXの座標を混ぜない。費用比較へ進む際は各方式へ公平な最適化機会を与えるが、今回は価格取得・最適化を行わない。M1-Sの全対象に整合するf_RはNOT_ESTABLISHEDで、固定加法モデル `S≈f_R(1−rho)` を実資源GO閾値にしない。

## 13. Checkpoint 1・2と非自動進行

**Checkpoint 1（T1a後）**は、複素信号の長時間一様保証、局所高次・非共鳴・近共鳴、既知理論との差、T1bへ渡せる構造を評価する。形式的な定理完成や技術PASSだけでT1bへ進まない。判断候補は進行推奨、範囲縮小、technical note、STOPである。

次の実施にはユーザーのT1b実行承認が別途必要。重要な新しい科学判断があればGPTレビューの必要性を判定し、その開始承認を得る。単なる文書・補助手法の変更ごとに同じ詳細レビューを反復しない。

**Checkpoint 2（T1b後）**は、移送、有効modeとdressing、joint scalingとalias、rhoの数学的意味、実分子への適用可能性、独立した科学的価値を扱う。**GPTの重要科学レビューが必須**だが、資料完成だけでは開始しない。ユーザーのその時点の明示的レビュー開始承認が必要。

Gate 0はCheckpoint 2以降に必要性・評価対象・条件を判断する。実施する場合は別の具体的な科学計算承認を要する。どのCheckpointからもGate 0、Q4/Q5、DF拡張へ自動進行しない。現在はCheckpoint 1/2とも未実施で、GOを記録していない。

## 14. 成功・失敗・縮小・STOPの記録規則

将来の理論結果は[GPT採択レビュー][GPT] §10に従い、`PROVED_WITH_DOMAIN`、`PARTIAL_SUFFICIENT_CONDITION`、`COUNTEREXAMPLE_IN_DEFINED_CLASS`、`UNRESOLVED`、`NOT_APPLICABLE`を区別する。これは**将来の出力区分の採択**であり、現在の結果classificationを変更するものではない。

| 将来確認する状況 | 判断への渡し方 |
|---|---|
| 固定law/C-secで有効な非空の一様域が得られる | 仮定・定数・適用不能例・既知保証との比較とともにCheckpoint 1へ |
| 固定有限次元で閉じ、全次元一様は未達 | 直ちに全M1-S失敗としない。依存性とT1bへの有用性を評価 |
| 局所高次誤差を欠く、またはexact-logの再表示だけ | T1aの完成結果として受理しない |
| 既知理論の直接系に留まり可搬性が乏しい | technical note化、縮小、STOPを検討 |
| 定義したclass内の解析反例・不能性が得られる | 根拠と対象を示し、単なる未解決/INCONCLUSIVEと区別 |
| T1bでratioが定義・制御できない | ratio claimを停止／限定。複素信号の絶対誤差保証の価値は別評価 |
| T1bが条件過大・未完了、適用域や費用が不明 | 範囲と未完部分を報告。未承認のGate 0/E0で即時補完しない |

縮小・technical note・M2等への転換はその時点の科学判断。限定的なM1-Sの失敗をTrack B全体の一般no-goへ拡張しない。今回新しい反例の構築、成功/失敗の判定は行っていない。

## 15. 採択済みの事項と未証明の事項

**採択済み**は、G0-T1/A′、T1aの固定state classとC-sec主比較・複素絶対誤差、C-log補助手法、T1bのexact-half/Pauli-tail/固定r・小step・有限減衰系列、証明義務・Checkpoint・STOPである。C-sec/C-logやT1b主極限は未選択ではない。

**未証明**は、T1aの局所高次・スペクトル結合を含む一様上界、定数・norm/重み・有効域の具体化、phaseへの条件付き換算、T1bのfull-H有効mode・先導減衰/位相、dressing/交差項/outer biasの定量制御、rho^(H)の成立・修正・不適用の判定である。

norm、theta_0、cutoff、weight functional等の具体値・式は、採択した主対象を証明する中で根拠付きで記述する内容であり、今回は数値固定しない。主比較対象、state class、joint pathを変える必要が出た場合は、単なる補助定数の決定とは区別して科学契約の改訂へ戻す。

**未確立かつ初期scope外**は、分子での改善量、一般DF event保証、RPE RMS、required shots、native単価、全体D/R費用割合、総資源優位、独立新規性・最終論文claimである。契約採択をこれらの成果認定と混同しない。

## 16. 今回承認していない作業・実施カウンタ

今回、T1a/T1bの新規証明・G0-T1理論研究の実行、Gate 0/E0、数値反例探索、Hamiltonian/行列評価・対角化・simulation、sampling/trajectory/operator guard、回路生成/compile/synthesis、新資源forecast/最適化、Q4/Q5、一般DF event研究は**すべて未承認・未実施**。

新しい数学的証明0、科学計算0、compile0、科学runner/科学実験テスト0。許可された文書の参照・構文・数式転記・hash検査を、科学的命題の検証や新しい重要科学レビューと呼ばない。

既存SOURCE/result/marker/authorizationを変更せず、過去の「未採択」記録も保持する。今回追加する採択索引には、**G0-T1契約が採択された事実だけをtrue**とし、T1a/T1b実施許可、Checkpoint GO、Gate 0許可、STOP解除をtrueにしない。

## 17. 次の担当と必要な実施承認

今回のCodex担当はS0正式版の作成・保全・公開までで、完了後停止する。**次の段階は、ユーザーによるT1aの新規理論実施の別途明示承認**である。正式採択をもう一度求める必要はなく、採択済み契約と実施許可を分ける。

その承認が出た場合に限り、採択したlaw/C-sec/state/誤差会計・scope・STOPに従って理論担当/Codex技術補助がT1aを行う。今回、その実施承認書・runner・実行markerは作らない。

T1bはCheckpoint 1の価値判断と個別実施承認後、Checkpoint 2のGPTレビューはその時点の開始承認後、Gate 0/Q4/Q5/一般DF拡張はさらに別の採択・実行承認後とする。ユーザーの実施承認がない間は、文書固定以外の科学的作業を開始しない。

## 18. 参考文献・固定GitHub資料・履歴の保持

本契約の具体的選択は[GPT採択レビュー][GPT] §9、その数理的説明は§§3–8/10–11、技術的基点は[直前候補][CANDIDATE]。局所高次誤差の既存検算は[前回GPT][GPTPREV] §4、旧データ所在は[旧事前契約][PRECONTRACT] §9を参照する。いずれも今回新たに科学検討をやり直していない。

既存一次文献との対応は[G0-pre][G0PRE] §3・文献locator、[P0.5][P05] §2を再利用する。

- [Günther et al., PR v2][PR]：partial-RTE、Appendix A.4の対称PF halving／独立±、Appendix B/EのRPE・費用。厳密対称性の記述を「誤り」として覆したとは主張しない。
- [Campbell, qDRIFT][CAMP]：既知のrandom compilerとcontrolled構成。M1-Sの新しいcoherent保証の成立を意味しない。
- [Kanasugi et al., early-FT QPE v2][FUJ]：partial randomizationとSTAR費用モデル。既知gamma差を新しいM1-S効果としない。
- COS、KGW、MLMC、SCU、CHEN、Gu/Li等は既存の固定監査の対象・norm・law・測定・費用の差を維持する。Kato/Daviesは背景の候補手法であり、完成済みの適用保証とはしない。

研究概要・日付付き研究ノート・manifestへ今回の正式採択と停止位置を追加する。旧候補、旧レビュー、旧manifest entry、科学判断の履歴を消さず、正式版への新しい参照を置く。

## 19. 公開確認とmandatory STOP

専用branch `track-b-m1s-g0t1-s0-formal-adoption-20261016` を直前候補の固定commitから作成する。原本を先にbyte保存・公開し、本契約・採択記録・索引を追加する。親worktreeのHEAD/index/既存dirty bytes、既存候補・原本・科学source/resultを保護し、文書以外の変更を混入させない。

文書検査と公開後のremote SHA、再取得bytes・SHA256の照合を[証拠索引][INDEX]および[公開receiptのdirectory][ARTDIR]に記録する。payload commitとその後のreceipt commitを区別し、自己参照hashを作らない。文書検査は科学理論の成立・CI再現を認定するものではない。

[共通指示][COMMON]と既存の性能・保全方針を維持する。将来の科学計算には別途承認と実行前のCPU/GPU/メモリ・支配処理・並列/thread/cache整合の準備が必要だが、今回はbackend/workerや科学計算予算を選定していない。新規計算契約に科学phase/total wall-time capを既定で追加せず、tool polling timeoutと区別し、既存凍結条件を変更しない。

**S0正式契約は採択・文書固定済み。数学的定理は未証明。T1a/T1bの理論実施、Gate 0等の科学実行は未承認。公開確認完了後はmandatory STOPし、T1aを自動開始しない。**

[GPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/27df63ede3513ed3982ea4c341823a3f18fa2b83/docs/research/track_b_m1s_g0t1_s0_formal_adoption_gpt_scientific_review_20261016.md
[CANDIDATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d84c2181de128e496bdc1e6161264cdf84eeba1/docs/tracks/algorithm_codesign/m1s_g0t1_s0_repaired_theory_contract_candidate_20261016.md
[CANDINDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d84c2181de128e496bdc1e6161264cdf84eeba1/artifacts/track_b_m1s_g0t1_s0_contract_repair/2026-10-16/v1/evidence_index.json
[GPTPREV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1039afae54520f15545177a7c852407830a0927/docs/research/track_b_m1s_g0t1_claude_gpt_independent_scientific_review_20261016.md
[PRECONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/docs/tracks/algorithm_codesign/m1s_g0_proof_and_gate0_precontract_20261015.md
[G0PRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md
[P05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d84c2181de128e496bdc1e6161264cdf84eeba1/docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md
[ADOPTION]: ../../../artifacts/track_b_m1s_g0t1_s0_formal_adoption/2026-10-16/v1/adoption_record.json
[INDEX]: ../../../artifacts/track_b_m1s_g0t1_s0_formal_adoption/2026-10-16/v1/evidence_index.json
[ARTDIR]: ../../../artifacts/track_b_m1s_g0t1_s0_formal_adoption/2026-10-16/v1/
[PR]: https://arxiv.org/abs/2503.05647v2
[CAMP]: https://arxiv.org/abs/1811.08017
[FUJ]: https://arxiv.org/abs/2603.22778v2
