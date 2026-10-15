# M1-S G0-T1／T1a：Checkpoint 1 Decision Packet

2026-10-16 JST。**`T1A_ANALYTIC_PROOF_SUBMITTED_WITH_DOMAIN_CHECKPOINT1_PENDING_MANDATORY_STOP`**。

**T1aの解析的証明を提出する。固定有限次元・有限減衰予算で、C-secとの差に全N一様のO(abs(theta))上界を得た。** これはCodexによる著者主張 `PROVED_WITH_DOMAIN` であり、独立科学レビューによる受理・研究価値・新規性の認定ではない。Checkpoint 1の判断、T1bの実施、数値計算は未実施。

主資料は[定理と全証明][PROOF]。まず§§1–6の定理・局所剰余・平均化評価を読み、§§7–9.1で位相系と非可換例・channel比較の意味を確認できる。[正式S0契約][S0]の主比較・law・state・対象極限を変更していない。

## 1. 今回の実施承認と固定source

ユーザーの「**次の作業に進んで大丈夫**」を、直前の完了報告が次段階として明示した**T1aの新規理論実施**の承認として受領した。記録は[今回の承認記録][AUTH]。S0契約採択を再度求めず、今回の承認を過去S0の未実施記録へ遡及させない。

- 基点：`373b8e4b9831f4d00bdc00ffc741ac61e03f0841`、正式契約33,099 bytes、SHA256 `4305a44ee690aa3c9d0d56aff7665a7c308a77109626cbe602d4eebd1d59f49e`。
- 採択科学レビュー原本：[27df63eのGPT原本][GPT]。44,886 bytes、SHA256 `e54c8cd4ddabf97aca5fb37057d2af64fd3bc468fe528c94f06b9e12d4a278d0`。今回再編集・新規レビュー扱いにしていない。
- 出典、byte identity、結果区分、実施カウンタ：[証拠索引][INDEX]／[解析claim一覧][CLAIMS]。
- 今回は紙上の解析的証明と引継ぎ文書。新しいHamiltonian/行列の数値評価、sampling、simulation、compile、科学runner/testは全て0件。理論作業を「新科学0」とは表記しない。

この承認をT1b、Checkpoint 1の科学GO、Checkpoint 2レビュー開始、Gate 0/E0、Q4/Q5、DF拡張の承認に使わない。作業記録はrunner markerではなく、成果の事前登録とも主張しない。

## 2. 提出する主命題

有限次元n、Hermitian involutionの確率分解 `h=E P_j`、同位置共有±・位置間IID・共通線形角を固定する。厳密写像は `Phi_theta(X)=E[A_j X A_j]`、`A_j=exp(i theta P_j)`。対象は孤立したh固有state|k>。

HS演算子空間で `L(X)=hX+Xh`、`D=−(E J_j^2−L^2)/2`、`J_j(X)=P_j X+X P_j`、`D_0=sum_omega Pi_omega D Pi_omega`。周波数は**pair-sum** `omega=e_a+e_b`。比較対象はS0と同じ

\[
z_N^{\rm sec}=\langle k|e^{iN\theta L}e^{\tau D_0}(I)|k\rangle,
\qquad \tau=N\theta^2.
\]

`b=norm(D)_{2→2}<=2`、`b_0=norm(D_0)<=b`。η≥0を分析用cutoffとして

\[
b_{\rm near}(\eta)=\left\|\sum_{0<|\omega-\nu|\le\eta}\Pi_\omega D\Pi_\nu\right\|_{2\to2},
\quad
\kappa_{\rm far}(\eta)=\sum_{|\omega-\nu|>\eta}
\frac{\|\Pi_\omega D\Pi_\nu\|_{2\to2}}{|\omega-\nu|}.
\]

全整数N≥0、実theta、abs(theta)≤1、tau≤D_capで

\[
\boxed{|z_N-z_N^{\rm sec}|\le\sqrt n
\left[18\tau|\theta|+\tau b_{\rm near}(\eta)
+2|\theta|\kappa_{\rm far}(\eta)\{1+\tau(b+b_0)\}\right].}
\]

固定モデルならη=0でb_near=0、κ_far(0)<∞なので

\[
\sup_{N:N\theta^2\le D_{\rm cap}}|z_N-z_N^{\rm sec}|
\le\sqrt n|\theta|[18D_{\rm cap}+2\kappa_{\rm far}(0)(1+4D_{\rm cap})].
\]

theta=0やN=0は厳密一致。θの上限1と定数18は証明中のnorm/Taylor評価に根拠を持つ解析域であり、分子実験の閾値採択ではない。ηはC-secを近共鳴clusterへ変更するパラメータではない。

## 3. 証明を支える構造と監査箇所

| 論点 | 証明の箇所・根拠 | 見落とすと変わる点 |
|---|---|---|
| 非CPTP写像の安定性 | 証明§3、D=−E(J_j−L)^2/2≤0、Φとexp(iθL+θ²D)のHS収縮 | CPTP性やΦの正規性を仮定してはいけない |
| 局所高次の累積 | §4、局所18 abs(theta)^3、telescopingで18 tau abs(theta) | 非共鳴項が零でも必要。長時間に残る |
| 平均化 | §5、interaction pictureとfar primitiveの積分評価 | 積分後の微分には全D(t)を用いる |
| 共鳴・近共鳴 | 同一ωはD_0、非零小gapはb_near、farは結合つきκ | hの孤立性をpair-sum非共鳴と同一視しない |
| 重み・norm | §6、入力IのHS norm√n、対象functional norm1 | operator/diamond normへ無条件に移し替えない |
| phaseへの変換 | §7、参照floor r_*とE<r_* | 位相の主値だけ。RPE/shot保証ではない |

Φの固有mode/Jordanやlog分枝に依存しない。特別な共通不変射影がある場合の平均化weight改善は証明§6.1に条件を記したが、**局所誤差の√nは一般には残る**。全state-specific weightを解決したとはしない。

## 4. 非空の保証域と比較の意味

既存レビューと同じ非可換二準位 `P_1=Z, P_2=X, p_1=p_2=1/2` を紙上で評価した。決定論blockなし、n=2、Pauli-tailのみ、DF rank・分子geometry/basis/L_Dは該当しない。角域は以下の解析域で、数値validation windowではない。

\[
z_N^{\rm sec}=e^{\pm i\sqrt2 N\theta}e^{-\tau/2},\qquad
|z_N-z_N^{\rm sec}|\le C_*|\theta|,
\quad C_*=18\sqrt2D_{\rm cap}+1+2D_{\rm cap}.
\]

任意の固定有限D_capに対して `0<abs(theta)<min(1,exp(−D_cap/2)/C_*)` は非空で、全対象Nに参照floor `r_*=exp(−D_cap/2)` がある。一般の条件付き位相系は

\[
|z_N|\ge r_*-E>0,\quad
|\operatorname{Arg}(z_N/z_N^{\rm sec})|\le\arcsin(E/r_*).
\]

tau→tau_*>0、theta→0ならこの例の位相誤差は零へ、理想unit-radius信号との複素距離は `1−exp(−tau_*/2)>0` へ近づく。

証明§9.1では同じlawのCPTP embedding `U_j=diag(A_j^dagger,A_j)` を用い、理想unitary channelとのdiamond normが少なくともこの複素距離となることも示す。channel保証の誤り・新定理への数値的な優越ではなく、**理想channelへの近さと、減衰を許した位相保証のtarget差**を明示したもの。通常の片枝qDRIFT channelをΦとして計算していない。

一般モデルでは共鳴blockでC-secが零に近づく可能性があり、主複素上界から無条件のphase保証は出ない。単一pair-sum条件下の先導係数 `V_k+v_kk` は証明§8で示したが、有限イベント比rho・半減保証ではない。

## 5. Checkpoint 1で判断が必要な点

ここは**判断の入力**であり、今回のCodexによる科学的推奨やGOの記録ではない。

1. **数学的有効性。** 散逸性のvariance恒等式、定数18、非可換telescoping、near/far積分の境界項、HS→signalの√n、phase diskの証明に誤りがないか。固定law/C-secに対する非空の全N一様域がS0の受理条件を満たすか。
2. **既知性と研究価値。** Taylor/Duhamel/有限次元平均化の標準道具から得たこの具体化は、既知の直接系を越える研究上の価値を持つか。非可換例のradial/phase分離とchannel embeddingは有用な整理か、technical noteで十分か。今回網羅的新規性調査や独立priority認定はしていない。
3. **定数と適用域。** √nとκは全spaceの量で、小gapや多数のcouplingで上界が空になり得る。最小bare energy gapや次元に依存しないoracle-free predictorを得たわけではない。一般分子への適用可能性は未判断。
4. **T1bに渡せるもの。** HS散逸性による安定性、local remainderの累積法、pair-sum block/near/far分解、floor付きphaseというproof構造が、正式なexact-half参照・固定r小step系列で有用か。**T1bの写像展開や移送の新証明は今回一切していない。**
5. **移送前に残る条件。** full-Hの固有stateと有効mode、dressingの係数誤差、alias・sector外結合、Dによる交差項、outer symmetric-step bias、ratioの分母とnormalizationを別途扱う必要がある。T1aの正則性だけでこれらの成立を認定しない。

科学判断の候補はS0のまま、進行推奨、範囲縮小、technical note、STOP。どれも今回選択していない。新しい重要科学判断・新規性判断は[共通研究指示][COMMON]に従う独立GPTレビューの対象であり、その開始はユーザーの別途承認後とする。

## 6. 実施済み・未実施の一覧

| 項目 | 状態 |
|---|---|
| T1aの解析的証明 | 著者が提出済み、独立科学的受理はpending |
| 非可換二準位例 | 同じモデルの手計算による解析例。数値実験ではない |
| 文書検査 | hash、参照、JSON/Markdown、旧資料保持、公開bytesを別記録で確認 |
| 数値的裏付け・CI・外部再現 | 今回なし。文書検査を科学検証と呼ばない |
| Checkpoint 1科学判断 | 未実施、GOなし |
| T1b新証明・移送 | 未承認・未実施 |
| Gate 0/E0、Hamiltonian数値、sampling、回路/compile | 未承認・未実施 |
| Q4/RPE、Q5/費用、一般DF event、最終総cost | 初期scope外・未実施 |

S0の当時の「T1a未承認・未証明」記録は履歴として保持し、今回の新しい承認と著者証明を別namespaceで追加した。Parent allocation D＋E、M1-Aの補助的位置、既存source/result/freeze/authorization/marker/STOPは変更していない。

## 7. 引継ぎと停止

T1aの主上界・全証明・条件付き系・非可換例を読み、**Checkpoint 1の科学判断を行うための資料**が揃った。次はその重要科学レビューの開始承認を受ける段階である。科学GOや新規性の認定、T1bの実行許可を本資料で代行しない。

正式契約§13の「**形式的な定理完成や技術PASSだけでT1bへ進まない**」、§17の「**T1bはCheckpoint 1の価値判断と個別実施承認後**」に従い、**ここでmandatory STOP**。今回のT1a承認を再度求める必要はない。後続の新規計算や移送証明は自動開始しない。

[PROOF]: m1s_g0t1_t1a_uniform_csec_signal_bound_and_proof_20261016.md
[S0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md
[GPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/27df63ede3513ed3982ea4c341823a3f18fa2b83/docs/research/track_b_m1s_g0t1_s0_formal_adoption_gpt_scientific_review_20261016.md
[AUTH]: ../../../artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/user_authorization_record.json
[INDEX]: ../../../artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/evidence_index.json
[CLAIMS]: ../../../artifacts/track_b_m1s_g0t1_t1a_theory/2026-10-16/v1/analytic_claim_inventory.json
[COMMON]: ../../research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md
