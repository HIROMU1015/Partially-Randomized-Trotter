# 局所mean保証から18round branch・energyへ：証拠と未保証条件

2026-11-04 JST。H4/隣接1 Å/STO-3G/legacy DF12、Partial L_D3/δ=.02、保存physical state/4電子sector70。
固定保存値とSOURCEの静的整理であり、新しい半径・phase・成功率・energy値を計算していない。

## 1. 三つの保証水準

| 水準 | 現在の証拠 | 未保証・今回の範囲 |
|---|---|---|
| ① 指定ideal categoricalの各round測定平均 | m17のみ96-bit包含で両軸の固定N/α/εのstrict十分条件確認済み。m0..16はSOURCE/入力/人工経路を準備 | 未実行の17セルの成立はUNKNOWN。uniform53 IID、exact ideal未compile gate、保存state rayというモデル限定 |
| ② 18round branch・energyの条件付き成功 | V1/V2は保存physical平均に対する数値的支持。V2は`CONDITIONAL_SUCCESS` | ①全36軸だけでは半径・共通E0位相・alias・decoder数値実装まで認定できない。確率の新しい保証付き移送も未実施 |
| ③ 実装された量子測定全体 | Bの少数固定trajectoryとq1/q8のideal controlled/Hadamard照合、旧fresh-IID仮想診断 | 全support float64 guard、固定PRNGのIID、全18round controlled回路、compile/noise、実state preparationは未保証 |

①→②→③は別の接続である。①を実施しても`RPE_tolerance_certified=false`、全体UNDETERMINEDを維持する。
元float64 guard証明系列は直前の[打切り判断](../rpe_m17_invariant_radial_preflight_20261104/REPORT.md)のまま保留し、今回再開しない。

## 2. 固定証拠と限界

- [V1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md)：physical H ground stateとPF有効固有状態controlを区別。Partialのfingerprint・保存PF/RTE残差は数値的に整合するが、PF固有状態そのもの・厳密単位半径下界を証明しない。
- [V2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md)：保存平均のnearest-branchは全18round一意、別oracle診断と一致。decoderへE0を渡していない。保存平均を真のIID marginalと仮定したfloat64二項和であり、外向き包含ではない。
- [m17包含](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34840ce9bf81d985b32764d9bc9d77645ed52e59/docs/research/rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md)：ν_catと保存μ0の座標差を保証する局所結果。実PRNG/仮想float/compile/noise/state-preparation bridgeはnull。
- [平均必要精度](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/be0757472ffa6bed60ecbeef4df27f1985046045/docs/research/rpe_mean_tolerance_validation_selection_20261103/RPE_MEAN_ERROR_REQUIREMENTS.md)：固定epsilonの中で平均差＋数値包含＋shot半幅を共有する。保存epsilonを新しい系統差へ満額追加しない。
- [Bのphysical state・固定request](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e22d7718c47358ff650dbc04e8e72f6ab1c8bf41/docs/research/rpe_fixed_request_measurement_b_20261103/REPORT.md)：Partial r1/K2・q1/q8の強制profileを含む6request/12axisの局所ideal照合。分布samplingでも、長q全supportの証明でもない。Bの明示stateを元Full18round診断のmissing fingerprintへ代入しない。

以下のV2数値は報告からの転記のみ。Partialの保存共通E0 phase失敗率union bound=`2.8153520984407623e-6`、条件付き成功下界=`.9999971846479015`。
これは真の平均を保存μ0に固定した解析モデルのfloat64結果であり、新ν_catの全区間上に一様な確率上界ではない。
V2の「座標successからphase条件への含意違反0」も同じ保存点の有限格子の数値検査である。
新しいν_catの二項確率に中心点のV2確率をそのまま代入できない。

Full L_D12はV1の7点のmodel radius不一致（m6,7,8,9,10,11,14）、診断自身のstate fingerprint欠測、δ=.02のPF固有状態control欠測を保持する。
今回の新包含対象には含めない。

## 3. 物理state・信号半径・系統誤差

保存raw vectorのnormalized rayと、厳密な物理Hの基底固有vector、PF有効固有state、実際に準備したstateは同義ではない。
既存固有残差・high overlap・fingerprint一致は対応の数値的証拠だが、厳密基底stateのfidelity/gap証明やstate preparation保証ではない。
保存E0=`-2.1663874486347616` Haも数値referenceであり、真のsector基底energyとの保証付き誤差は別に必要。

資源SOURCEはPF有効固有状態における基準半径を仮定する。元H physical stateで観測したnear-one radiusを厳密なρ≥1へ昇格しない。
Partialのattenuated radius・PF control radius・model lower boundを区別し、normalizationを取り除いた信号を測定平均へ混ぜない。
経験的PF係数Cをrigorous上界と呼ばない。保存β_PF/β_RTEが共通E0に対して正当な上界であることも独立の条件である。

①が全軸で成立すれば、ideal IID ±1測定について

    Pr(|hatν_m,a − μ0_m,a| >= ε_m,a) <= α_m,a

が得られる。ここでU_catを既にepsilon内に入れているので、phase予算へ同じU_catをもう一度加えない。
ただしこの座標事象を共通E0 phaseへ結ぶ幾何・系統誤差が未保証である。

静的な十分条件の一例は、保存μ0をexactな中心とし、

    D_m = sqrt(ε_cos² + ε_sin²),  ρ0_m = |μ0_m|
    D_m < ρ0_m
    b_m = d_circle(arg μ0_m, q_m φ) + asin(D_m/ρ0_m)

を保証付きで評価すること。φは全round共通の物理reference位相である。
ここではρ0は固定中心の半径であり、実ν_cat自体の半径とは異なる。ρ0/phase/referenceの計算も未実行。
別案はν_catの矩形全体で系統位相と正のradiusを評価してshot geometryを結ぶこと、または矩形全体の二項phase失敗率supを保証すること。
後者は新しい科学再解析であり、端点だけが最大になると仮定しない。まず単純な幾何で足りるかを別SOURCEで準備する。

## 4. branch・alias・energyへ接続する定理条件

[既存decoder SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_four_round_phase_validation.py)の`reconstruct_rpe_phase`はqを倍増し、前推定に円周距離で最も近い候補を選ぶ。
理想数学decoderでは、共通φへの各round phase誤差上界b_mに対し

    2 b_(m-1) + b_m < π

が一意選択の十分条件。同じbならb<π/3。保存β_RPE=.4という数値だけで各roundのこの事象を証明したことにはならない。
実コードはfloat64 wrap/angle/候補除算/距離/argminを用いるため、境界からのstrict marginと丸めの接続も別途必要。
観測平均のbest/runner-up gapをそのまま普遍的noise許容差へ変換しない。

初期m0はbranch0、phase window `[-π,π)`。U(t)=exp(-itH)、φ=-Eδ、E_hat=-φ_hat/δ。
energy window `(-π/δ,π/δ]`に対象が入ることは外部条件である。
最終代表元の境界crossingを除くmarginも必要で、円周誤差だけから実直線上のenergy誤差へは移れない。
scalar/identity位相は既に含まれ、新たなenergy shiftを加えない。

条件を満たす全round事象の下では、理想decoderの誤差はb_17/q_17、energyはb_17/(q_17 δ)へ接続する。
V2の保存nominal値は`.000152587890625` Ha、保存暫定目標は`.00015936001019904` Ha。
CA/10を今回正式採択しない。保存reference E0の不確かさ、DFモデルから真の分子energyへの誤差も別であり、nominal resolutionに暗黙に含めない。
δのexactな対象は固定SOURCEのbinary64 parameterで、表示decimalとの換算を無断で変更しない。

## 5. union boundと最小の後続検証

全36軸の成功事象をunion boundで結ぶ場合、失敗率はΣ_m,a α_m,a以下。
union bound自体にround間・軸間独立は不要だが、各軸Hoeffdingにはshot内のfresh独立性と±1有界性が必要。
V2のcos/sin二項productには軸実験の独立性も必要である。
固定seed非重複、少数trajectory照合、samplingのPASSから数学的IIDを証明しない。

保存α_total=.05は宣言値。保存decimal αをexactに足した和が宣言値以下かは、今回再計算していない。
将来の全体証明ではexact和／外向き和を評価して実際の値を示し、丸め許容差やbudget変更で黙って.05へ丸めない。
決定論的包含自体にsampling confidence ηはなく、新たなηを架空に加えない。

次はまず①のm0..16限定包含（この公開SOURCEに対する別許可）が最小。
その結果が揃ってから、別の技術準備でμ0/radius/reference・branch strict margin・alias/reference誤差・α和を結ぶSOURCEを固定し、新しい科学評価は再び別途許可を得る。
実装量子測定全体の③、Full、価格優位はその計画にも自動では含めない。
