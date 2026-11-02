# N3：保証・条件族・算法の統合claim map v1

**2026-10-13現在**：[成果確定claim・成立条件](n3_saved_attribution_v1_claim_and_observability_20261013.md)と[完了GPTレビュー](n3_saved_attribution_v1_scientific_review_2026-10-13.md)を参照。
T1–T9とS1の厳密分離を保持し、現行N3の追加検証系列を終了、主研究としての採択を見送る。以下は2026-10-12時点の定理・準備履歴であり、A1の未実装記述は当時の状態。

2026-10-12 JST。完了済み[科学レビュー原文](../../artifacts/n3_theory_task_integration_v1/2026-10-12/review_intake.md)第10.1節へのCodex技術追補。
旧報告、source、primary ledger、gate、STOPを変更しない。中心テーマ・新規性の採択は行わない。

## 対象と情報

H=Σ h_ij a_i†a_j+Σ_{i<j}J_ij n_i n_j、実対称h、非負unordered density J、一外部mode dを空へ射影、固定粒子数1≤ν≤m。
固定sectorで Hν=[[A,B],[B†,C]]、A=dΓν(h_A)+V_A、V_A≽0、BB†=dΓν(vvᵀ)。
係数から取得し証拠を持つ a=λmin(A)≤U、C≽cI、g=c−U>0を使う。真のa/E0/gapは運用入力にしない。
保証対象は0≤a−E0(Hν)≤δ。状態・時間発展・量子資源・統計誤差はこの保証に含まれない。
ν=mの一体射影集合も扱える。β=||v||=0では幅下端0なのでstrict NORM改善の議論から除外する。

## 定理と検査の対応

| ID | 主張／数学的根拠 | 実装・証拠と限界 |
|---|---|---|
| T1 | CARと外部空sectorからBB†=dΓν(vvᵀ) | 既存occupation数学監査。一外部・density classに限定。一般二体へ移植しない |
| T2 | σν(K)=min_{0≼γ≼I,trγ=ν}tr(Kγ) | Ky Fan/Fantope既知原理。[文献対応](n3_literature_correspondence_v1_20261012.md)。固有値縮退でも成り立つ |
| T3 | Z≽0、K−τI+Z≽0⇒σν(K)≥s=ντ−trZ | 任意rank-ν ΠでtrΠZ≤trZ。既存exact PSD checker。有限有理proposalの完全性はない |
| T4 | K=κh_A−vvᵀ、0≤κ≤D=g+δ、s≥κU−δD⇒射影誤差≤δ | 非負κV_A、Schur補元。下記証明。十分条件であり物理的不能判定ではない |
| T5 | NORM、FIVE、OLDの有効証拠をOCCへ埋込み可能 | NORM:κ=0,τ=0,Z=vvᵀ,s=−β²。OLD:τ=μ,Z=0,s=νμ。FIVE⊂OLD。埋込みは利益に数えない |
| T6 | ν=1で理想OCC族=理想OLD族。κ=0で理想OCC=NORM | γ≼Iの冗長性、rank-one vvᵀ。有限保存pool・探索coverageの同一性は導かない |
| T7 | E_forced>U iff 理想OCCにNORMより狭い幅が存在（β>0,g>0） | 既存[保存幅契約](n3_saved_width_v1_mathematical_contract_20261012.md)の片側微分証明。新しいE_forced固有値計算なし |
| T8 | 保存W≽0,trW=1,r=trW(h_A−U I/ν)≤0,b=vᵀWv>β²/ν⇒全旧OR下端δ_N | 今回のstdlib独立照合、次節。十分な分離条件であり、失敗時は未判定 |
| T9 | 固定証拠のδ受理は上方単調。c強化でも証拠を保持可能 | κ≤g+δ、κU−s≤δ(g+δ)。同じh,v,ν,U,κ,τ,Zが必要。別modeや別Uへの無監査cacheは禁止 |
| A1 | 幅指向の有限generatorと一mode選択task | [仕様v2](n3_width_reduction_v2_spec_20261012.md)。未実装・未実行・未採択。定理と算法の費用／coverageを分ける |

## T4の独立照合

σν(K)≥sとκV_A≽0から κA−BB†≽sI≽(κU−δD)I。
z=a−δ と置くと C−zI≽(c−a+δ)I≽DI≻0。
したがってSchur補元は

\[
A-zI-B(C-zI)^{-1}B^\dagger
\succeq A-aI+\delta I-BB^\dagger/D
\succeq(1-\kappa/D)(A-aI)+\kappa(U-a)I/D\succeq0.
\]

よってHν≽zI。変分原理からE0≤a、従って0≤a−E0≤δ。
c>U・κ≥0・V_A≽0・κ≤Dがそれぞれ使われる。densityの負符号、一般orbital rotation、外部多modeを無断で許すとこの証明契約を保てない。

## T8の全幅排除と登録対照下端

OLDのscalar条件よりμ≥μ0=(κU−δ(g+δ))/ν。μをμ0へ下げるとPSDを保つので、全κ/μの可行性は

\[
M_\delta(\kappa)=\kappa(h_A-U I/\nu)-vv^T+\delta(g+\delta)I/\nu\succeq0,
\quad0\le\kappa\le g+\delta
\]

に等価。δ_N=(√(g²+4β²)−g)/2。条件T8では、全0<δ≤δ_N、全κ≥0に対し

\[
\operatorname{tr}(WM_\delta(\kappa))
=\kappa r-b+\delta(g+\delta)/\nu
\le-b+\beta^2/\nu<0.
\]

PSD同士のtrace積が非負であることに反するためOLD不可行、FIVEも不可行。
NORMの受理域はδ≥δ_Nなので、同じ固定U,c,v,h_A,νでのNORM OR FIVE OR全OLDのinfimumはδ_N。
有理数出力で無理数δ_Nを達成できるとは言わず、infimumと安全な有理上端を区別する。
指定一点でのdual両端負だけではこの全幅補題は導けない。r≤0とstrict b条件を別に検査する。

## S1の保存値照合

source checker固定commit `370c42a6`。旧診断payload `86c69983544064d1ed7ee7910634c541c7b982b7`、公開base `26ed4b959bf5b59d23e2131d8be785ee701f0bc2`。
star、一体＋非負density、scale1.25、full10→active9、ν4、一外部mode8、元δgoal=.005。化学geometry/basis/DF rank/policy/L_D/PF時間窓は適用なし。
今回δは各保存OCCの正の有理上端、共通U=−1951/1064、c=−233352716459809/183068686024704。

| 入力 | r（表示） | b−β²/ν（表示） | 登録旧OR下端に対する保存OCC達成幅の改善（表示） |
|---|---:|---:|---:|
| S1_edge_pair | −0.118915673192 | 0.000190972222222 | 20.4240870225% |
| S1_distributed_three | −0.00904487648179 | 0.0000297799294903 | 0.343339962134% |

全判定・改善区間は[saved_analytic_checks.json](../../artifacts/n3_theory_task_integration_v1/2026-10-12/saved_analytic_checks.json)のexact Fraction値。
NORM rootは保存済み区間を多項式で検査しただけ。新root/δ scan/κ/W/固有値を生成していない。
独立に元行列から計算したr,b,qと改善区間など14項目がレビュー付録と一致。
OCCのZ PSDとU,c妥当性は既存source-bound監査の不変bytesを継承。この小さいcheckerは歴史的全proof replayの代替ではない。
同じS1の事後的developmentであり、元8入力の追加保証0／2-base未達を変更しない。
改善は保証上端の達成値と登録旧ORのinfimumの比較。OCC最適幅、OLD単独最適幅、真の誤差、量子資源改善ではない。

## 技術検証

30 rational synthetic tests：r=0、b境界、ν1、β0、単一点の分離が全幅条件を満たさない反例、非PSD/零trace/形状、float/bool/oracle、root囲み、証拠継承、保存identity/幅/q/trace改変を検査。
finite failureは不可行性にせずSUFFICIENT_LEMMA_NOT_ESTABLISHED。既存science producer/NumPy/solverをimportしない。
これらはsource-bound local検証。immutable CI・外部環境での科学再現と表示しない。

今回の帰結はレビューで導出済みの内容の独立照合。元の一点診断報告の最適幅未判定記述は当時の履歴として保存し、この追補を新しい根拠として参照する。

## 2026-10-13 成果固定と後続実装の位置

A1はその後のv2 generator・selector・同値SDP・固定SOS比較で実装され、[固定6H pilot](n3_width_comparison_v2_results_20261013.md)と[保存帰属事後評価](n3_saved_attribution_v1_results_20261013.md)を一回ずつ完了した。
実装・限定観測の完了を、主研究としての採択・最適性・同機能同品質の速度優位とは扱わない。
同じ前提と情報の理想条件族ではNORM/FIVE/OLD⊆OCC=同値SDP⊆固定density-aware SOS。左側のS1分離と右側の指定query分離は異なる保存例であり、同一queryで全層のstrict差を示していない。
FULL_H_LOWERを含むv2 CHEAP_OR全体は、この包含鎖の左端に置かない。
[小さい証拠索引](../../artifacts/n3_achievements_closure/2026-10-13/existing_evidence_index.json)から、旧族全幅下端、同値契約、SOS正/OCC負と他mode正、元task結果へ追える。
文書反映で新たな証明・PSD/eigen/幅計算を行わず、T1–T9の成立条件と保存証拠を保持した。
研究方針は追加benchmark・帰属診断系列終了、主研究採択見送り、N1停止/N2保留、Mandatory STOP。[再開条件](n3_saved_attribution_v1_claim_and_observability_20261013.md#再開条件と作業完了)を満たす具体的な用途または理論差分が提示された場合のみ再検討する。
