# M1-S先行研究調査：技術比較と独立レビュー用資料

2026-10-17 JST。`LITERATURE_AND_TECHNICAL_COMPARISON_COMPLETE / NOVELTY_NOT_ADJUDICATED / MANDATORY_STOP`。

対象repositoryは`HIROMU1015/Partially-Randomized-Trotter`、科学資料の固定基点は
`81d6103ec89d485cb7336c3a90bd95dad66992aa`。
独立branch `m1s-prior-art-survey-20261017`、worktree `/tmp/prt-m1s-prior-art-20261017`で作成した。
ユーザーの今回の指示を文献取得・技術比較・Markdown作成・commit/pushの承認とし、
GPT科学レビュー開始、新規証明、科学計算の承認にはしない。

## 調査結果の読み方

M1-Sに関係する既知の構成は、共有乱数、対称積、確率的cross term、平均演算子の信号解析、
位相と減衰の分離、測定数と回路単価の最適化にまたがる。
特に**MLMC-qDRIFT、Childs–Ostrander–Su、Morisaki–Fujii、Güntherほか**は、
それぞれ異なる比較軸で近い。単一論文を「全M1-Sと同じ」と認定する証拠は本調査では揃えていない。
その逆の、同一手法の不存在や独立新規性も認定しない。

| 成果物 | 内容 |
|---|---|
| [文献一覧](bibliography.md) | 著者・年・DOI・固定arXiv版・読解箇所・取得水準 |
| [一次資料の定理・数式・構成](primary_evidence.md) | 重要な比較をページ／節／式へ結び付けた証拠カード |
| [技術比較表](comparison.md) | branchのlaw、順序、観測量、保証、費用、未照合事項 |
| [GPTレビュー資料索引](review_index.md) | T1a/T1b、有限step、γ依存V1、既存GPTレビュー、SOURCE・raw・監査への固定リンク |
| [取得記録](literature_retrieval.json) | 今回取得した一次PDFの版・bytes・SHA256、失敗した取得経路 |
| [内部資料identity](repository_evidence.json) | 固定commitの必須資料・実装・結果・監査のpath／blob／SHA256 |
| [調査範囲・保全と公開](audit.md) | 検索範囲、未確認事項、禁止作業未実施、公開照合の方法 |

本文・式番号は指定版に限定する。PDFページは表紙込みの1始まり。
「関連箇所確認」は全証明読了、定理の再証明、実験再現を意味しない。
一次PDF・HTML・抽出全文・紙面画像をrepositoryへ転載せず、リンクとidentityを残す。

## 1. 比較するM1-Sの固定定義

ここは新しい理論の導出ではなく、基点の[T1b §§2–5,10](../../tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md)の定義の転記である。
右端先作用、$H=H_D+H_R$、$H_R=\lambda_R h$、$h=\sum_jp_jP_j$、
$P_j=P_j^\dagger, P_j^2=I$。固定整数$r$、

$$
D=e^{i\delta H_D/4},\quad A_j=e^{i\theta P_j},\quad
\theta=\frac{\lambda_R\delta}{2r},\qquad
\Phi_\theta(X)=\mathbb E_j[A_jXA_j],\quad
\Psi_\delta(X)=D\Phi_\theta^r(DXD)D.
$$

同じchronological label列を正負branchで共有し、各gateの符号を反転する。
位置間・step間はIID。これはfull-wordのadjointを負枝へ置く定義と違う。
完全乱択で $U_+=A_{j_N}\cdots A_{j_1}$ に対して、M1-Sの負枝は
$U_-=A_{j_N}^\dagger\cdots A_{j_1}^\dagger$。
したがって干渉wordは $A_{j_1}\cdots A_{j_N}A_{j_N}\cdots A_{j_1}$ となる。
full-adjoint構成なら負枝は $U_+^\dagger$ で、干渉対象は $U_+^2$ 型になる。
inverseを直列にかけるmirrorの $U_+^\dagger U_+=I$ とも区別する。

$\Phi$は通常の$\mathbb E[A_jXA_j^\dagger]$というsystem channelではない。
ただし物理的なancilla込み回路のchannelとそのoff-diagonal blockを区別すべきで、
「非CPなblockを解析する」というだけで既存channel理論からの還元不能を意味しない。
この注意は[Abbottほか §4, Eq.(5)](https://arxiv.org/pdf/1810.09826v3)の実装依存性とも整合する。

## 2. 類似性が高い文献と根拠

以下の順序は読むための優先順であり、新規性への点数や科学的優劣の順位ではない。

| 比較軸 | 最優先資料 | 一次根拠 | 残る区別 |
|---|---|---|---|
| 量子的な二枝でlabelを共有 | [MLMC-qDRIFT v2](https://arxiv.org/pdf/2604.26865v2) | §2.2 Eqs.(14),(15)、§2.3 Eq.(30) | fine/coarse・異なる角度／列長、observable差。opposite-sign同深さのM1-Sとはlawが違う |
| 共有乱数から対称wordを作る | [COS v2](https://arxiv.org/pdf/1805.08385v2) | p3 Eq.(6)、p5 Lemma 1 | 全項permutationの共有。IID replacement、枝間のsign reversalとは区別 |
| mean operatorとHadamard信号の誤差 | [MF v1](https://arxiv.org/pdf/2609.36694v1) | pp3–4 Eqs.(22)–(27)、p6 Eq.(35) | 任意Hermitian項と分散bound。M1-Sのsecular参照・共鳴floorへの直接還元は未照合 |
| partial evolution＋directional control＋RPE | [PR v2](https://arxiv.org/pdf/2503.05647v2) | Appendix A.4 Eq.(A46)、B Theorem B.2、E | random RTEの正負枝は独立。決定論部分の費用半減をM1-S全体へ移さない |
| coherent cross termとshot overhead | [SCU v2](https://arxiv.org/pdf/2407.21095v2)、[WBC v2](https://arxiv.org/pdf/2110.12071v2) | SCU Fig.1/Eq.(2)、WBC Fig.1/Lemma 2 | 特定shared-label lawの定理ではない。normalizationの費用を含む |
| 位相安定性とmode解析 | [Gu v2](https://arxiv.org/pdf/2208.04100v2)、[Yi–Crosson出版本文](https://www.nature.com/articles/s41534-022-00548-w) | Gu Thm.1/SI、YC Lemma 3/Thm.2 | noisy unitary channel／決定論unitaryと、M1-S平均写像の対象・gapを対応させる必要 |

**今回の取得上の前進**：旧Checkpoint 2後監査では未取得だったYi–Crosson出版補遺を、
今回は出版社リンク経由で取得・関連箇所確認した。
[Supplementary Note 5, Eqs.(61)–(68)](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41534-022-00548-w/MediaObjects/41534_2022_548_MOESM1_ESM.pdf)
はoff-diagonal leading perturbationとenergy shiftの二次評価を扱う。
旧監査の未取得記録は改変しない。出版版全証明とpreprintの全式の同一性までは認定していない。

## 3. M1-Sの既存成果への対応

### 3.1 T1a/T1b：既存の解析的成果

[T1a](../../tracks/algorithm_codesign/m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md)は
固定有限次元・involution・有限$N\theta^2$の全step一様複素信号bound。
[T1b](../../tracks/algorithm_codesign/m1s_g0t1_t1b_partial_randomized_transfer_theory_20261016.md)は
exact deterministic halves、固定$r$、全Hのpair-sum射影を使った部分乱択への移送である。
[Checkpoint 2レビュー](../track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md)
の限定受理と、共鳴時の正の指数混合／Jensen下界を引き継ぐ。

$$
Q=-\frac{\beta}{2}\mathbb E(J_j-L_R)^2\preceq0,\qquad
\beta=\lambda_R^2/(4r),\quad Q_0=\sum_\omega\Pi_\omega Q\Pi_\omega,
$$
$$
|z_s-z_s^{H,\mathrm{sec}}|\le\sqrt n\left[
Ku|\delta|+uq_n+2|\delta|\kappa_f\{1+u(q+q_0)\}\right],\quad u=s\delta^2.
$$

分散恒等式・Duhamel・telescoping・spectral pinching・摂動解析という技法の既知性と、
このjoint lawの明示定数／量化／参照floorのpriorityは別問題である。
MF、COS、Gu、YCに加え、古典的な平均化・弱結合理論との対応も後続判断に残る。
Davies原論文は今回abstract／書誌確認に留まり、その定理で上式を置き換えてはいない。

単一pair-sum条件(SP)と$W_k>0$の下の既存式
$\rho=(W_k+v_k)/(2W_k)$は**定義された先導rateの比**。
等しい独立draw数ではindependent ±の各枝を半数にする比較が必要で、
per-branch $r$固定の比較、native価格、必要shotsの比と混同しない。

### 3.2 finite-stepとγ依存：保存結果の引用

以下の全行に共通する条件は **linear H4、1.00 Å、STO-3G、8 modes、保存DF rank 12、
生成順prefix $L_D=3,6,9$（G3/G6/G9）、保存数値sector-ground oracle、
$\delta=1/8,1/16\ \mathrm{Ha}^{-1}$、$R=2,4$、$T=1,4\ \mathrm{Ha}^{-1}$**。
Pauli-tailのshared／ordinary／independent比較であり、DF-native総費用比較ではない。
値・件数は各既存報告からの引用で、新規集計していない。

| 既存成果 | 保存された内容 | 文献との対応と限界 |
|---|---|---|
| [finite-step完了](../../tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md) | zero込み108行。G3/G6は各16絶対対照でvisibility higher、G9の16対照は絶対差未解像 | 平均複素信号の局所観測。phase bias増加と共存し、T1bのrate検証や資源GOではない |
| [γ依存V1完了](../../tracks/algorithm_codesign/m1s_v1_gamma_completed_results_and_stop_20261017.md) | G3/G6は各16対照で内部tradeoff境界。G9は形式的margin順位を保存しても絶対差未解像を保持 | Hoeffding＋信号円板幾何に基づく同一の連続十分shot設計の比較。最小必要shotsの定理ではない |
| [π/3限定gate完了](../../tracks/algorithm_codesign/m1s_gamma_pi3_completed_results_and_stop_20261017.md) | G3の16、G6の8が成立、G6の8が不成立、G9の16は未照合 | 採択routeの十分領域との交差。RPE全round成功・整数shot削減・native総資源を示さない |

この証拠は`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。
保存oracleの厳密固有状態性、platform coverage、全空間(SP)は未認証。
固定commitに収録されたことはCIや第三者の独立科学再現を意味しない。

## 4. 既知結果と未照合部分の境界

| 区分 | この調査での扱い |
|---|---|
| 既知原理・既存式の直接特殊化 | Hadamard X/Y平均、IID first momentの因数分解、Hoeffding有界変数boundと十分shot式、対称PF、共有乱数によるcoupling一般、費用×測定数の会計 |
| 既存レビューが受理したM1-S結果 | T1a/T1bの限定解析、共鳴floor、固定データのfinite-step visibility／phase tradeoff。今回の再証明・新規認定はしない |
| 還元可能性未照合 | MF/Chen/COSへのHSまたはancilla lift、Guへのcontrolled-channel対応、古典平均化からT1a/T1bの全定数・floorが直接従うか |
| 追加の証拠を要する科学問題 | 実RPEの精度域・state contamination・全round、整数shots、whole-wrapper費用、強いbaselineに対する同条件総資源差 |
| 今回判断しない | 新規性の最終認定、論文化方針・研究価値、継続／縮小／終了、新しい計算条件の採択 |

「還元をしていない」を「還元不可能」と読まない。
未確認の定理にM1-Sの非自明性を預けず、既知原理が共通という理由だけで全結果を同一視もしない。
後続GPT用の具体的な未解決項目は[資料索引](review_index.md)にまとめた。

## 5. 引渡しと停止

文献証拠・技術比較の作成まで完了。科学status、SOURCE、結果、契約、authorization、grant、ledger、
markerは変更しない。進行中shot・cost feasibilityのbranchへmergeも書込みも行わない。
本branchは基点commit時点の資料に固定し、並行作業の最新結果を推測して取り込まない。

**commit/push・GitHub identity・保全確認後にmandatory STOP。
GPTの新規性・研究価値の独立科学レビューは未開始。新科学計算へ自動移行しない。**
