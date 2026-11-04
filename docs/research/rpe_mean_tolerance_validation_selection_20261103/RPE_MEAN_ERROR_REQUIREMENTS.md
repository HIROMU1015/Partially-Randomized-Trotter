# RPE測定平均の必要精度：数式上の条件と未評価の数値

2026-11-03 JST。技術調査・静的導出・人工検証のみ。新しいH4数値評価、保存値の数値再解析、
sampling、state-action、build/compile、費用評価、独立した重要科学レビューは実施していない。
親はrun02結果commit `80616e6450a63c77d5e76276739a2322111adda4`。
本書の式をH4保存値へ代入した新しい許容差・標本数・成功率は計算していない。

**結論：保存coordinate epsilonを、そのまま実装平均の追加許容差として使えない。**
既存予算を保持する保守的な十分条件は、後述の「epsilonから固定Nの統計半幅を引いた残余」による。
その残余の具体値は未評価であり、必要精度を0.01等に決めていない。
この十分条件が唯一の必要条件、または最大限許される差だとも主張しない。

## 固定研究条件と一次資料

linear H4/隣接1.00 Å/STO-3G/legacy DF12、8 system qubit、4電子sector70、保存physical ground state。
主対象はPartial L_D3、δ=.02の既存18round、q=2^m、m0..17、保存選択r/K・N/α/β。
Full L_D12のV1/V2結果は背景として保持し、新しいFull検証を提案範囲へ混ぜない。

- [V1報告](../pr_rpe_v1_e0_state_radius_20261103/REPORT.md)：commit e520e6e396f7e0def9e522ea1cff58eb10ddfa8f。
- [V2報告](../pr_rpe_v2_branch_success_20261103/REPORT.md)、[round CSV](../pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv)、
  [確率JSON](../pr_rpe_v2_branch_success_20261103/SUCCESS_PROBABILITY_RESULTS.json)、[branch JSON](../pr_rpe_v2_branch_success_20261103/BRANCH_RESULTS.json)：
  commit 78c49d0b5f1d71e63b5959e78813237923bbc648。
- [run02報告](../rpe_fresh_iid_measurement_a_run02_20261103/REPORT.md)と
  [RESULT](../../../artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run02/RESULT.json)：親commit固定。
- carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`の
  [rpe_resource_accounting.py](../../../src/trotterlib/rpe_resource_accounting.py)：`evaluate_rpe_round_candidate`, `_hoeffding_shots`。
  [research_direction_decision_cost.py](../../../src/trotterlib/research_direction_decision_cost.py)は
  `_evaluate_profile`で`beta_stat=beta_rpe-beta_pf-beta_rte`を構成する。

これらは同じ保存入力の系譜であり、carrierを元科学実行のclean SOURCEと認定しない。
run02の実行SOURCEは別の`d0b1da1b04c266a1f7abf8e43966ca6e289a8695`。

## 1. 誤差を置く位置

各roundの保存attenuated event-meanをμ_m=x_m+i y_m、実装の真の周辺平均をν_mと置く。
差d_m=ν_m−μ_mが新しい検証対象である。hatν_mは有限shotの推定値。

| 誤差 | 既存予算／意味 | 新しい平均照合との関係 |
|---|---|---|
| PF | β_PF、物理referenceとPF信号のphase差 | dへ重複計上しない。経験的係数・状態仮定を保持 |
| finite-RTE | β_RTE、corrected finite operator対PF。正normalization減衰は半径へ入る | 保存μは既にfinite-RTEの信号。dはfinite truncation誤差そのものではない |
| 実装周辺平均差 | d=ν−μ | sampler law/normalization/作用順序等の接続差。既存の独立予算はない |
| 有限shot | hatν−ν | 固定N/αとβ_stat/epsilonで扱う |
| 数値 | 保存μと再構成値の数値誤差・演算誤差 | 物理的dと区別し、比較時の不確かさに含める |
| branch | nearest-branchの誤選択 | 全roundのphase事象から別に制御。平均残差の小ささだけで認定しない |

U(t)=exp(−itH)、signal=〈ψ|U|ψ〉、base phase φ=−Eδ、E_hat=−φ_hat/δ。
scalar/extracted identityは既にsignalへ含まれ、再加算しない。
Hadamard sineは既存Sdg→Hの+Im規約。normalization補正済み信号を±1測定平均に使わない。

## 2. 複素座標からphaseへの静的な十分条件

μ≠0、ρ=|μ|とする。複素摂動eが|e|≤D<ρなら、原点を含まない半径Dのdiskの接線から

    circular_phase_distance(μ+e, μ) ≤ asin(D/ρ)

を得る。座標別上界D_c,D_sではD≤sqrt(D_c²+D_s²)。
ρが小さいほど同じ座標誤差への感度が上がる。ρ=0、D≥ρ、非有限値ではこのphase保証は使えない。
principal Argの単純差ではなく円周距離を使う。数値的なnear-oneを厳密なρ≥1に置き換えない。

既存SOURCEは、モデルのρ_lb>0と0<β_stat<π/2に対し

    ε_m = ρ_lb,m sin(β_stat,m) / sqrt(2)

を使う。両座標の**保存平均に対する合計誤差**がε_m未満なら、正しい半径下界という条件の下で
phase誤差をβ_stat以内へ制御できる。
V1の物理状態・半径の制約は残り、観測ρやモデルρを新しいrigorous lower boundにしていない。
V2には、固定Nの格子上で座標successからphase条件への含意違反0という別の数値証拠がある。
これは固定保存μでの格子検査であり、μを動かした確率評価や連続領域の保証ではない。

## 3. 既存N/αを変えないときの平均差の残余

±1 fresh-IID周辺測定では、各軸の真の平均νに対してHoeffding半幅は

    s_m,a = sqrt(2 log(2/α_m,a) / N_m,a)

である。|d_m,a|≤τ_m,aと数値比較の不確かさu_m,aを別に考えると、保守的な十分条件は

    τ_m,a + u_m,a + s_m,a < ε_m,a .                         (1)

すなわち、保存epsilonの中を「追加系統差＋数値＋有限shot」で共有する。
両軸同じ許容差なら

    R_m = min_a [ ε_m,a − s_m,a ] ,
    τ_m + max_a u_m,a < R_m .                              (2)

R_mは残余の定義であり、今回H4で計算していない。Nのceilは余裕を生み得るが、
余裕の存在・大きさや実装差を吸収できることは未確認。R_m≤0ならこの十分条件では正の追加許容差を得られない。
元軸処理は`abs(sample_mean-mean)>=epsilon`を失敗とするため、境界はstrictに扱う。
導出に使うHoeffdingは真のν中心、V2の保存格子の座標事象はμ中心であり、dを無視して混ぜない。

別の書き方は、dと数値誤差の合成phase上界b_implを用いて

    β_PF,m + β_RTE,m + b_impl,m + b_shot,m ≤ β_RPE

とすること。既存β_statをそのままb_shotに満額使い、b_implを無償で追加することはできない。
実装差をfinite-RTE予算へ含めるなら、既存finite-RTE誤差との合成上界を示す必要がある。
β_RTE全額が空いていると扱わない。geometryで式(1)へ合算したdを、さらにphase予算へ二重加算もしない。

**より緩い許容差を得る別案**は、固定N/αのままμ近傍のνに対する二項phase失敗確率を上から評価すること。
νの長方形領域内で失敗率のsupを制御する新しい数値再解析が必要であり、保存点の確率や
中心点での含意違反0だけでは代用できない。端点だけでsupになるとは一般に仮定できない。
今回は行わず、式(2)で情報が不足する場合の後続案に留める。

## 4. branchとenergyへ接続する条件

[元reconstructor](../../../src/trotterlib/rpe_four_round_phase_validation.py)はE0を引数に取らず、qが毎回2倍になる候補のうち
前roundの推定に円周距離で最も近いbranchを選ぶ。
真の共通φに対するround phase誤差上界をb_mとすると、隣接候補の間隔は2π/q_m。
正しい候補と前推定の距離は最大(2b_(m−1)+b_m)/q_mなので

    2 b_(m−1) + b_m < π

が一意選択の十分条件。全round同じbならb<π/3となり、既存のstrict条件と一致する。
観測平均に曖昧性がないことと、有限shot下でこの条件が保たれることは別である。
保存されたbest-runner-up gapを2で割り、そのまま普遍的なnoise許容差にもしない。

全18roundで共通referenceへのphase条件が成立する事象を、失敗確率のunion boundで結ぶ。
union bound自体はround間独立を要しない。各軸の二項法則・cos/sin productにはそれぞれ独立測定の仮定がある。
初期alias windowは外部条件であり、最終phase代表元が境界を越えない局所guardも必要。
正しいbranch/aliasの下でenergy誤差はβ_RPE/(q_max δ)へ接続する。

以下はV2報告からの既存値の転記で、新規評価ではない：
β_RPE=.4、q_max=131072、δ=.02、nominal energy resolution=.000152587890625 Ha、
保存暫定目標=.00015936001019904 Ha。
Partialの保存モデル下のphase union bound=2.8153520984407623e-6、
条件付きenergy成功下界の数値値=.9999971846479015。
この成功下界はμを真のmarginalとした保存モデルに対するもので、未知dを含む実装へ移送済みではない。
単一roundの追加検証だけで全18roundの条件を認定しない。

## 5. 不一致検出、同等性、標本設計

検証用family失敗率ηと対象軸数Jに対しh=sqrt(2 log(2J/η)/n)を用いると、
推定差d_hatの同時区間は[d_hat−h,d_hat+h]となる。μ自体に数値不確かさがあれば区間を拡張する。

| 判定 | 条件／主張 |
|---|---|
| 零差への不一致検出 | 0が区間外。所定の許容差超過とは別 |
| 所定τ内 | 区間全体が[−τ,τ]内：abs(d_hat)+h≤τ。信頼区間に基づく有限失敗率の主張 |
| 所定τ外 | 区間が[−τ,τ]と交わらない：abs(d_hat)−h>τ |
| 未判定 | 上のどちらも認定できない |

真の差の設計上限D<τを仮定するなら、D+2h≤τとなるnは同等性判定の保守的設計になる。
例としてn≥8 log(2J/η)/(τ−D)²という式を使えるが、H4へ数値代入していない。
二項のexact区間を用いる案もある。±1平均をp=(1+ν)/2へ変換し、区間をνへ戻す。
観測結果を見てτ・η・N・対象を選ばず、powerと同時評価の設計を先に固定する。
検証ηは既存RPEのαではない。追加の検証confidenceとRPE energy失敗率を結ぶ場合も、別の失敗事象として明示する。
検証区間が正しい事象を前提に後続RPEを評価するなら、全体の失敗率の主張では原則η+Σαを管理し、
保存α_totalを維持したまま追加のηを無償で0扱いしない。

fresh shotの最終±1出力は周辺平均νを持つBernoulliであり、trajectory選択と条件付き測定の揺らぎを含む。
最終測定に対する上記hへ「trajectory誤差」を機械的にもう一度足さない。
条件付きsignalだけの平均を使うRao–Blackwell型照合は測定noiseを省けるが、samplerの未知系統差を消すものではない。
未観測eventの確率質量pが不明なら未観測を0と扱わない。±1/単位diskの測定平均では、
質量pの部分を任意に置換した平均差は最大2pという粗い上界を持つが、そのpの評価は別に必要。

run02の半幅約.156/.165、各N510/454、NO_DISCREPANCY_DETECTEDは既存報告の引用。
これから式(2)の残余内や任意の小さいτ内への同等性を認定できない。
raw1,928件をpool/再集計した新しい区間・p値は本作業では作成していない。

## 今回確定したこと／次に数値評価が必要なこと

確定したのは式(1)–(2)、phase/branchへの十分条件、誤差予算の使い方、方式の適用条件。
未確定は対象roundのR_m、比較の数値不確かさu_m、許容差の具体値、必要ならrobust二項許容領域と追加n。
これらを得る科学的再解析は[次段計画](NEXT_STAGE_DECISION.md)の別許可範囲とする。
幾何・branch境界・独立event平均・判定の違いは[人工8 tests](test_design_identities.py)で確認した。
H4結果やV2の既存statusを変更する検査ではない。
