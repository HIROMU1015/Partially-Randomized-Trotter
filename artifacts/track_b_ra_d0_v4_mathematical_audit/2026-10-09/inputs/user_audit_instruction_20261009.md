# Track B / RA-D0 v4：数値構成アルゴリズムの独立数学監査

## 1. 今回の目的

Track B / RA-RTEについて、GPT側で設計したRA-D0 v4の数値構成アルゴリズムを、**独立した立場から数学的に監査し、最小限のsynthetic/off-domain検証を行ってください。**

今回の目的は、RA-D0 v3で発生したnumerical certificate failureに対して提案されたv4が、

1. 数学的に正しい十分条件を与えるか
2. 実装可能なdyadic samplerを一般的に構成できるか
3. B2/B3の比較対象を不正に変えないか
4. confidenceとresource capsを同時に保証できるか
5. 実装に進めるだけの技術的根拠があるか

を判断することです。

**今回はv4のproduction source実装、registered optimization、authorization、one-shot実行を行いません。**

数理監査と独立した人工データによる検証まででmandatory STOPしてください。

---

## 2. 固定された研究背景

リポジトリ：

`HIROMU1015/Partially-Randomized-Trotter`

主要commit：

| 内容 | Commit |
|---|---|
| RA-D0 v3 source S | `45cffb2aa10f9219b6cad929c3ade49fe7d36ca8` |
| v3 authorization A | `2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9` |
| v3 technical result R | `35f8b949079f15d0348bc082b916324870da7246` |
| T0 audit | `72192b3475d59f5c56370cb4068d5659789f0ef4` |
| T0.1 audit | `5cf56e4a5949d64c24eac127bac0c223d431df87` |
| T0.2 exact certificate | `d3a7cbb239487ddedf44699378f6c182c1fe5993` |

既存のsource、contract、results、authorization、consumed marker、R1/R1.5、Track A、過去STOPはすべて保護してください。

T0.2では、保存済み1点について、

- membership
- mean
- confidence
- workspace
- sampler validity

の全certificateがPASSしました。

ただし、事後的に選んだ1つのweight shiftによる結果であり、一般的な修復アルゴリズムの成立やB2/B3の資源的優位性はまだ示していません。

---

# 3. 現在の研究RQ

研究Bの中心RQは、

> 同じ有限Taylor平均を実現するランダムunitary ensembleについて、degree-localな係数配分の自由化が、既存representationの混合より実装資源を削減できるか。

です。

現在の比較class：

- B1：representationを固定し、合成精度配分を自由化
- B2：既存representationのwhole-ensemble mixtureを許す
- B3：degree-localな係数再配分まで許す

primary comparisonは引き続き、

\[
\boxed{
U_Q^{B3,\mathrm{certified}}
<
L_Q^{B2,\mathrm{outer,certified}}
}
\]

です。

今回の数値構成の修正によって、研究対象のTaylor target、candidate columns、precision、resource coordinates、accuracy、confidenceを変更してはいけません。

---

# 4. 監査対象となるv4設計

## 4.1 基本定義

固定有限Taylor targetを、

\[
t=(t_0,\ldots,t_m)
\]

とします。

各logical groupを \(g\)、合成precisionを \(p\) とします。

保存candidate tableには、各groupの理想係数、

\[
c_g\in[c_g^-,c_g^+]
\]

と、対応するdegree-local coefficient vector \(v_g\) が存在します。

定義：

\[
\bar c_g=\frac{c_g^-+c_g^+}{2}.
\]

保存されたD intervalsは、

\[
D_{g,p}\in[D^-_{g,p},D^+_{g,p}]
\]

とします。

まず、同じlogical groupに属するprecision columnsについて、理想D intervalsが同一であるという前提が必要か、また固定tableで実際に成立しているか監査してください。

以下ではその前提が成立する場合に \(D_g^\pm\) と表記します。

共通sampling denominator：

\[
N=2^{60}.
\]

数値mean許容幅：

\[
\delta_{\mathrm{num}}=10^{-12}.
\]

confidence parameter：

\[
e=\frac1{200}.
\]

既存の \(\ell_{\mathrm{up}}\)、\(\kappa_{n,\mathrm{up}}\)、resource accountingを維持します。

---

## 4.2 構造的parameterization

新しい変数を、

\[
u_{g,p}\ge0
\]

とし、

\[
q_{g,p}^{*}=\bar c_g u_{g,p}
\]

と定義します。

### B2

representation \(r\) の各groupについて、

\[
\sum_pu_{r,g,p}=z_r
\]

\[
z_r\ge0,\qquad
\sum_rz_r=y
\]

とする。

さらに、

\[
\sum_{r,g,p}\bar c_{r,g}u_{r,g,p}=1.
\]

このとき、

\[
\sum_pq_{r,g,p}^{*}
=
\bar c_{r,g}z_r
\]

が構造的に成立します。

### B3

degree-local representationの制約を、

\[
\sum_gv_g\left(\sum_pu_{g,p}\right)=yt
\]

とします。

同時に、

\[
\sum_{g,p}\bar c_gu_{g,p}=1
\]

を課します。

### 監査事項

以下を独立に証明・検証してください。

1. B2の係数関係が構造的に成立する。
2. B2においてdegree matchingが自動的に成立するか。不成立なら追加制約が必要か。
3. B3が本来のdegree-local自由度を維持する。
4. B1/B2/B3のnumerical nestingと矛盾しない。
5. \(\bar c_g\) によるapproximationをmean residualで正しく吸収できる。
6. inactive representation、zero-mass groupを未定義な除算なしで扱える。
7. \(y>0\) を線形制約の範囲で保証できる。
8. 変数の変更が、本来の候補classを不当に拡大していない。

特に、**v4のinner generatorが元のnumerical class全体を表現しているとは仮定しないでください。**

---

# 5. Mean residualの十分条件

保存されたdegree vector \(v_g\) を使い、

\[
\rho_g=
\sum_k
\max\left(
|\bar c_gD^-_{k,g}-v_{k,g}|,
|\bar c_gD^+_{k,g}-v_{k,g}|
\right)
\]

と定義します。

また、

\[
\Xi(u)=
\sum_g\rho_g\sum_pu_{g,p}
\]

とします。

提案された上界は、

\[
\boxed{
\xi(q^*,y)
\le\Xi(u)
}
\]

です。

### 監査事項

- interval endpointsの向き
- absolute valueの扱い
- degreeごとの和
- 非負変数の必要性
- 同一groupのprecision間でDが同一という前提
- B2/B3双方でdegree matchingが成立する条件

を明確にしてください。

**この上界が成立しない反例があれば、PASSとせず具体的な有理数反例を返してください。**

---

# 6. Hierarchical dyadic rounding

## 6.1 Group counts

丸め前のgroup massを、

\[
M_g=\sum_pq_{g,p}^{*}
\]

とします。

まずgroupごとにinteger counts \(K_g\) を求める。

\[
\sum_gK_g=N,\qquad K_g\ge0.
\]

largest-remainder法を用い、tieは固定index順。

提案上界：

\[
\left|\frac{K_g}{N}-M_g\right|
\le\frac1N.
\]

## 6.2 Precision counts

group totals \(K_g\) を固定して、

\[
\sum_pK_{g,p}=K_g
\]

を満たす整数countsをlargest-remainder法で生成。

最終law：

\[
q^N_{g,p}=\frac{K_{g,p}}N.
\]

\(y\) は元のnearest/half-up ruleで \(1/N\) 格子へ丸めます。

latent \(z\) はphysical samplerではないのでdyadic化を要求せず、representation sharesを保ちながら丸め後yへ再スケールします。

### 監査事項

- 全countsの非負性
- 総和N
- group mass保存
- precision rounding error
- inactive groupに正のcountsを誤って割り当てないこと
- zero-mass処理
- group/precision tie処理
- index順の固定
- 丸め順序による誤差の独立した上界

特に、**丸め前の分布がすでにnormalizedであることを保証してから誤差上界を適用**してください。

---

# 7. B2 membership guarantee

元のmembership allowance：

\[
\tau_g=
\frac{3+c_g^+/2}{N}
\]

を変更しません。

提案された保証は、

\[
\left|
M^N_{r,g}-z_r^Nc_{r,g}
\right|
\le
\frac1N+
\frac{c_{r,g}^+}{2N}
+
Y_{\max}\operatorname{rad}(c_{r,g})
\]

です。

十分条件として、

\[
Y_{\max}\operatorname{rad}(c_{r,g})
\le\frac2N
\]

を事前確認します。

### 監査事項

1. B2 structural equalityから導出できること。
2. y roundingがlatent zへ与える影響。
3. \(z_r/y\) のshare保持条件。
4. coefficient interval midpointとendpoint間の差。
5. この前提で元の\(\tau_g\)を超えないこと。
6. zero-mass representationでも成立すること。
7. 固定tableでpreflight可能な条件であること。

この十分条件が成立しない場合、tauを広げずに適用不能としてください。

---

# 8. 丸め誤差上界の監査

提案されたboundsを**証明する対象**として扱ってください。

正しいと仮定して実装しないでください。

## 8.1 Mean rounding reserve

\[
L_g=
\sum_k
\max(|D^-_{k,g}|,|D^+_{k,g}|)
\]

\[
\boxed{
\Gamma_\xi=
\frac{
\sum_gL_g+\|t\|_1/2
}{N}
}
\]

提案保証：

\[
\xi(q^N,y^N)
\le
\Xi(u)+\Gamma_\xi.
\]

## 8.2 Synthesis bias reserve

合成誤差上界 \(d_{g,p}\) に対し、

\[
\boxed{
\Gamma_d=
\frac1N
\sum_g\left(
\max_p d_{g,p}+\sum_pd_{g,p}
\right)
}
\]

提案保証：

\[
d^\mathsf Tq^N
\le d^\mathsf Tq^*+\Gamma_d.
\]

## 8.3 Resource reserve

resource \(Q\in\{T,CX,1Q\}\) について、

\[
\boxed{
\Gamma_Q=
\frac1N
\sum_g\left(
\max_p C_{g,p,Q}
+\sum_p C_{g,p,Q}
\right)
}
\]

提案保証：

\[
|C_Q^\mathsf Tq^N-C_Q^\mathsf Tq^*|
\le\Gamma_Q.
\]

### 重点監査

以下を明確にしてください。

- group counts丸めの誤差
- group内precision丸めの誤差
- 2つを合成した誤差
- bias/costの非負性が必要か
- precision columns間のD identityが必要か
- group数・precision数への依存
- alias columnの扱い
- inactive groupの扱い
- 端点値での等号成立
- 反例の有無

もし式が不十分なら、正しい上界と反例を提示してください。

ただし、**監査段階で新しい式をproduction契約として勝手に採択しないでください。**

---

# 9. Confidence保証の監査

丸め前のconfidence下界：

\[
H(u,y)=
ey-
\sum_{g,p}\bar c_gd_{g,p}u_{g,p}
-\Xi(u)
\]

とします。

rounding reserve：

\[
\Gamma_h=
\frac{e}{2N}+\Gamma_d+\Gamma_\xi.
\]

提案するLP条件：

\[
\boxed{
H(u,y)\ge
\kappa_{n,\mathrm{up}}+\Gamma_h
}
\]

### 監査事項

この条件が成立するなら、

\[
ey^N-d^\mathsf Tq^N-\xi(q^N,y^N)
\ge\kappa_{n,\mathrm{up}}
\]

を保証できることを独立に証明してください。

confidenceを保証するために、

- \(\kappa\) のroundingを不正に緩めていないか
- yの誤差が正しく含まれるか
- 合成誤差のupperが同じ意味で使われているか
- mean reserveとの二重計上・計上漏れがないか

を確認してください。

T0.2のconfidence修復成功を一般的な証明の代わりに使わないでください。

---

# 10. Mean cap / resource caps

Mean capに対する提案条件：

\[
\boxed{
\Xi(u)+\Gamma_\xi+
\frac{\delta_{\mathrm{num}}}{2N}
\le y\delta_{\mathrm{num}}
}
\]

Resource \(R\) の上限 \(b_R\)：

\[
\boxed{
2n\left(
\sum_{g,p}\bar c_gC_{g,p,R}u_{g,p}
+h_R+\Gamma_R
\right)\le b_R
}
\]

これらを課したexact continuous solutionから、量子化後の、

\[
\xi(q^N,y^N)\le y^N\delta_{\mathrm{num}}
\]

および、

\[
G_R(q^N)\le b_R
\]

が導かれるか証明してください。

1Qにのみ付く既存のfixed overhead \(h_{1Q}=5/2\) を保持し、T/CXのoverheadは元contractどおり扱ってください。

workspaceはexpected valueではなく、active-support peakで判定してください。

---

# 11. Fixed-n LPとしての成立

v4の設計問題について、

- normalization
- degree matching
- B2 structural membership
- mean reserve
- confidence reserve
- resource caps

がすべて**固定候補表・固定nにおいて線形制約**になるか確認してください。

目的関数の一例は、

\[
\min
2n\left(
\sum_{g,p}\bar c_g C_{g,p,Q}u_{g,p}
+h_Q
\right).
\]

この最小値が意味するものを明確にしてください。

特に、

> Margin付きinner generatorの最小値 = 元のB2/B3全体の最小値

とは主張しないこと。

修復costがどの程度増加したか、certified lowerとのgapをどのように記録できるかも検討してください。

---

# 12. B2/B3比較の公平性

ここは最重要項目です。

v4のinner generatorは、元の数値クラスの部分集合となる可能性があります。

したがって、inner generatorで得たB2の最小値を、元のB2クラスのlower boundとして使用してはいけません。

比較用lowerは引き続き、

**元のB2 numerical classを包含するouter relaxation**

から取得する必要があります。

primary witness：

\[
\boxed{
U_Q^{B3,\mathrm{certified}}
<
L_Q^{B2,\mathrm{outer,certified}}
}
\]

を維持すること。

また、certified B2 lawがB3の候補classに正しく埋め込めるか確認してください。

同一implementationのaliasを合算する場合、平均・cost・bias・workspaceが不変である証明が必要です。

B2だけに有利なnumerical repairや、B3だけに許す後処理を認めないでください。

---

# 13. Infeasibility classification

次の概念を必ず区別してください。

### A. Certified primal feasible

元のcertificateを全てPASSした実装可能law。

### B. Inner generator infeasible

margin付きの保守的LPに実行可能解が存在しない。

これは**元のB2/B3 classのinfeasibilityを意味しない**。

### C. Original outer-certified infeasible

元のnumerical classを確実に包含するouter problemについて、正しいinfeasibility proofを得た場合。

この場合に限り、元のclassの実行不能を結論できます。

### D. Solver/certificate failure

数値解・dual・Farkas等のcertificateが取得できない。

technical inconclusiveとして扱う。

### E. Resource cap

計算資源の上限に到達した場合。

technical inconclusiveとして扱う。

BとCを混同しないためのstatus設計を提案してください。

---

# 14. Exact rational LPの実装可能性

今回、数値の厳密性が問題になったため、exact-rational LPを候補としています。

ただし、**新しいsolverを自作することは今回の目的ではありません。**

既存のexact LP backend、例えばSoPlex等について、

- 利用可能性
- rational input
- exact primal取得
- exact dual/Farkas取得
- 独立したFraction verification
- 小規模LPにおける実行時間
- ライセンス・依存環境

を調査してください。

実際に利用可能なものがあれば、**synthetic/off-domain LPに限って**最小限の動作確認を行って構いません。

新規依存の無断導入、registered LP solve、production runtime変更は禁止です。

利用できない場合は`UNVERIFIED_BACKEND`として報告してください。

古いv3の2秒/LP、total runtime capを自動継承しないでください。

新backendでの実行capは別途設計・レビュー対象です。

---

# 15. 独立したsynthetic検証

数学的監査に加えて、固定seedまたは決定的に生成した小規模な人工有理数例で検証してください。

最低限：

1. 全representation active
2. 一部representation inactive
3. active group mass = 0
4. groupに複数precision
5. coefficient intervalの幅が0
6. coefficient intervalの幅が有限
7. membership preflightが成立
8. membership preflightが不成立
9. confidence marginが正
10. confidence marginが境界上
11. confidence marginが負
12. resource capが十分大きい
13. resource capが境界上
14. resource capを破る場合
15. inner generator infeasibleだが元class feasibleな場合
16. aliasを含む場合
17. 同一logical event内でprecision cost/errorが異なる場合
18. 極端に小さいgroup mass
19. largest-remainderのtie
20. \(N=2^{60}\) のinteger counts

を含めてください。

テストは提示した保証式を前提としてPASSさせるのではなく、**得られたlawを独立したFraction certificateで再評価**してください。

また、必要に応じて小さな分母で網羅的探索を行い、反例を探してください。ただし本番のdenominatorを変更したことにはしないでください。

T0.2の保存結果は既知の回帰テストとして利用して構いませんが、registered solverを再実行しないでください。

---

# 16. 数学監査で重要な姿勢

GPTが提案した数式を正しいものと仮定しないでください。

特に、次の観点で独立検証してください。

- B2/B3のrepresentationの意味が変わっていないか
- 十分条件が本当に十分か
- 有理数midpointへの変更が新しい誤差を作っていないか
- 各rounding boundに定数不足がないか
- 厳密な不等号・等号の扱い
- extreme caseの反例
- 実装でのrounding overflow / zero division
- コストが保守的すぎて最適化として無意味にならないか
- inner generatorの保守性によって比較可能な点が消失する可能性
- B2/B3のcertified lower/upperの整合性

問題が見つかった場合は、

1. 問題箇所
2. 数学的な反例
3. 必要な追加仮定
4. 修正候補
5. 修正した場合の比較条件への影響

を報告してください。

**修正候補を独断でproductionへ採用しないでください。**

---

# 17. 今回の作業範囲

許可：

- 数式の独立監査
- source/contractのread-only確認
- synthetic/off-domain LP
- 人工有理数例によるproperty test
- exact solverの利用可能性調査
- 小規模な技術検証
- 監査専用script/testの新規作成
- 監査結果・反例の保存

禁止：

- RA-D0 v3再実行
- RA-D0 v4 production source実装
- registered B2/B3 optimization
- 実際のbudget freeze
- 新authorization
- 既存markerの変更
- 新synthesis
- 新angle / precision
- IS / CTSへの拡張
- science / molecule / DF / NPZ
- circuit / matrix / trajectory
- GPU
- 既存の研究結果・STOPの再分類
- Track A変更

旧v3 source・結果・contract・authorization・markerは全て保護してください。

---

# 18. Branchと成果物

T0.2 commit：

`d3a7cbb239487ddedf44699378f6c182c1fe5993`

を基点に独立branchを作成してください。

推奨branch：

`track-b-ra-d0-v4-mathematical-audit-20261009`

監査用docs：

- `docs/tracks/algorithm_codesign/ra_d0_v4_mathematical_audit_20261009.md`
- `docs/tracks/algorithm_codesign/ra_d0_v4_gpt_handoff_20261009.md`

推奨artifacts：

`artifacts/track_b_ra_d0_v4_mathematical_audit/2026-10-09/`

最低限：

- `input_identity_v1.json`
- `theorem_audit_v1.json`
- `rounding_bound_audit_v1.json`
- `baseline_nesting_audit_v1.json`
- `infeasibility_semantics_v1.json`
- `synthetic_verification_v1.json`
- `exact_solver_feasibility_v1.json`
- `resource_cost_audit_v1.json`
- `evidence_manifest_v1.json`

必要な場合は監査専用scriptとtestsを追加して構いません。

元のsourceを改変する必要がある場合は、その時点でSTOPしてGPTへ報告してください。

---

# 19. GO / STOP基準

最終分類を以下から選んでください。

### `V4_MATH_AUDIT_PASS`

- 主要な十分条件の証明が成立
- B2/B3の意味が維持される
- independent synthetic verificationがPASS
- original numerical classに対するlower/upperの使い分けが正しい
- 重大な反例・論理的blocking issueなし

ただし、この判定だけではv4 production実装を認可しません。

### `V4_MATH_AUDIT_REVISION_REQUIRED`

- 数学的に不正確な保証式
- 重要な仮定漏れ
- B2/B3比較の不公平
- resource reserveの不足
- 反例が発見された

など。

### `V4_EXACT_SOLVER_FEASIBILITY_UNVERIFIED`

数学的監査は成立したが、実装可能なexact backendを確認できない場合。

数学的PASSと技術的未確認を区別して報告してください。

### `V4_TECHNICAL_INCONCLUSIVE`

入力不整合、技術的障害などによって監査自体が未完了の場合。

---

# 20. 完了報告

commit・push後、以下を報告してください。

- branch
- full SHA
- remote SHA一致
- worktree clean
- 最終audit classification
- 成立した定理・十分条件
- 成立しなかった定理・条件
- 見つかった反例
- synthetic tests数とPASS/FAIL
- exact LP backend候補と検証状況
- B1/B2/B3 nestingの評価
- membership保証の評価
- mean保証の評価
- confidence保証の評価
- resource cap保証の評価
- primary witnessの公平性
- inner/outer infeasibilityの扱い
- 旧v3/T0/T0.1/T0.2 protected hashes不変
- registered solver calls = 0
- new synthesis/science = 0
- v4 production implementation = 0
- blocking issue

最後に、

**「このv4 numerical algorithmを実装へ進められるか」**

について、証拠に基づくGO/STOP提案を出してください。

ただし、研究方針の最終判断はGPT側で行います。

## 21. Mandatory STOP

監査資料をcommit・pushしたらmandatory STOPしてください。

監査PASSでも、v4 production source、authorization、one-shotへ自動移行しないでください。

今回の作業は、**GPTのv4数理設計案を独立に検証するところまで**です。