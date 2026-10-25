# 研究B：固定24条件のshot加重native資源評価

2026-10-25 JST。**固定科学実行COMPLETE／科学的資格INCONCLUSIVE**。科学SOURCE `9bebdc46ebd60039f315da9b6c6e57bd9c004b85` を変更せず、許可された保存値解析を一回だけ実行した。Runtimeは `COMPLETED_DESCRIPTIVE_MANDATORY_STOP`、科学的状態は `INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS`。u_num／gammaと未観測rare費用が未取得のため、保証付き同精度達成・無条件期待費用順位・総資源優位は認定しない。

## SOURCE・条件・入力・保存

[実行準備・固定scope](research_b_outer_shot_weighted_execution_ready_20261025.md)、[数学式・設計](research_b_outer_shot_weighted_resource_design_20261025.md)をそのまま使った。結果branchは `codex/research-b-outer-shot-results-20261025`。科学SOURCEと結果公開commitは区別する。

Synthetic 2-mode one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix、n_D=1、T=±0.18、ε=1e-8／1e-9。分子geometry／AO basis／分子DF rank探索は非該当。B3=Yoshida4/A2/r2K2、D1=Yoshida4/A3/r2K2、R2=Yoshida4/A3/r4K2、R4=Yoshida4/A3/r2K4、B4=SABA2/A2/q2/r2K2、F1=Refined-S2/A2/q2/f8/r2K2。全候補q2。

単価はB1のfull prepared X/Y Hadamard wrapper。State preparation、control、evolution、X/Y終端、measurementを含み、保存Qiskit1.3.0・rz/sx/x/cx・opt1・seed17・approximation1・qubits_initially_zero=False・topologyなし・angle-preserving native v5を維持する。今回は回路生成・compile・新sampling・新Hamiltonian/operator/signal計算・quantum shot実行を行っていない。

| 固定入力 | 結果commit | 用途 |
|---|---|---|
| 22行 | `33715848bfb83d49ec4e54d6eef481012a7441e4` | 共通corrected physical complex signal直接bias、normalization、全R occurrence |
| A0 | `31662b49157b15f2d1c03eef1c2faa07e2f905e7` | B3/F1/R4両Tの補足。主biasへ選択的に差し替えない |
| B0 | `0db9b3ed189eb6cd9d66e070993dbd0e72d29104` | Native profile・input provenance |
| B1 | `bef63edbd8cc564444b44e4e8cf23baa90f05fe5` | 保存n256のpaired費用、mean／variance／SEとsampling law |
| Rare | `cae3cd784dbdfcf2fe96902345379aca4b9606c7` | B3/B4/D1/R4の保存8確率・未知費用の扱い |

[原本9file・STOP](../../artifacts/research_b_outer_shot_weighted/20261025/saved-once/)、[SHOTS.csv](../../artifacts/research_b_outer_shot_weighted/20261025/saved-once/SHOTS.csv)、[RESOURCES.csv](../../artifacts/research_b_outer_shot_weighted/20261025/saved-once/RESOURCES.csv)、[COMPARISONS.csv](../../artifacts/research_b_outer_shot_weighted/20261025/saved-once/COMPARISONS.csv)、[要約](../../artifacts/research_b_outer_shot_weighted/20261025/SUMMARY.json)、[process環境・preflight](../../artifacts/research_b_outer_shot_weighted/20261025/execution_logs/)を公開する。Rawは科学checkoutとbyte一致、13固定入力blobは実行後も不変。正確なSOURCE・各入力path/blob・scope・Python環境は[provenance.json](../../artifacts/research_b_outer_shot_weighted/20261025/saved-once/provenance.json)に保存した。

実行command（使用済みoutputで再起動しない）：

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-shot-science-20261025/scripts/run_b2d_outer_shot_weighted.py \
  --execute --source-commit 9bebdc46ebd60039f315da9b6c6e57bd9c004b85 \
  --output /tmp/prt-research-b-outer-shot-science-20261025/artifacts/research_b_outer_shot_weighted/20261025/saved-once
```

## 実績・数値correctness

| 登録境界 | 完了／登録 |
|---|---:|
| `shot.bias` | 24／24 |
| `shot.bind` | 12／12 |
| `shot.compare` | 24／24 |
| `shot.count` | 48／48 |
| `shot.joint` | 72／72 |
| `shot.normalization` | 12／12 |
| `shot.qualification` | 24／24 |
| `shot.rare` | 4／4 |
| `shot.resource` | 144／144 |

全12保存結合がnormalization計算に先行し、48整数shot要求・72 joint moment・144資源行・24比較・24資格が完了した。88 occurrence・200 short-step因子を全て保持。全ceilが一意に確定し、scope超過・未完了・失敗call・保存失敗は0、events728件。実行前の人工45件PASSは準備の技術実績であり、今回の科学的資格とは区別する。

保存normalizationをN_used、signed binary64 τに対する90dps outward scalar区間を[N_lo,N_hi]、shot用の上界をU=max(N_hi,abs(N_used))と分けた。全normalizationが正で保存値との整合gateを通過した。Scale差の複素bias上界を別保存し、統計rangeを広げればbiasも消えるとは扱わない。

| Candidate（両T） | N_used（表示丸め） | Rescaling bias scalar上界（表示丸め） | p_rare |
|---|---:|---:|---:|
| B3 | 1.0003658930326542 | 1.1078772E-15 | 1.82895756403E-4 |
| D1 | 1.0003658930326542 | 1.1078772E-15 | 1.82895756403E-4 |
| R2 | 1.0001829324167670 | 1.4715214E-15 | 9.14535898003E-5 |
| R4 | 1.0003658935224375 | 9.5653663E-16 | 1.82896245917E-4 |
| B4 | 1.0000279276482560 | 1.3668368E-17 | 1.39635269131E-5 |
| F1 | 1.0000069818525916 | 9.9916852E-16 | 3.49090794495E-6 |

R2/F1の両Tについて追加4件のp_rareを保存B1 p0の積から評価した。B3/B4/D1/R4の8確率は旧解析の保存値を再利用した。符号Tで確率とshot数は一致した。表示丸めは上界性を保証せず、保証付き演算を再利用する場合はJSONのexact Fraction端点が正本。Scalar enclosureはモデルのnormalization評価に限られ、signal全体のu_num、PRNG bit-law、gammaのcertificateではない。

## 条件付きの十分shot数

s=2ε/5、alpha_X=alpha_Y=1/40、eta=s/sqrt(2)とし、fresh IID full RTE trajectoryごとのbounded outcome ±1を仮定して `m_a=ceil(4*U²/s²*ln(2/alpha_a))` を評価した。各(candidate,T,ε)のX/Y union boundが95%であり、全candidate/T/εのfamilywise保証ではない。最小shot数の最適解や実backend費用を主張しない。

X/Y・両Tでmは同一。下表は**各axisあたり**のexact integerで、X+Yの合計ではない。

| Candidate | m_axis at ε=1e-8 | m_axis at ε=1e-9 |
|---|---:|---:|
| B3 | 1096308481839662540 | 109630848183966253904 |
| D1 | 1096308481839662540 | 109630848183966253904 |
| R2 | 1095907502690109238 | 109590750269010923789 |
| R4 | 1096308482913175258 | 109630848291317525775 |
| B4 | 1095567849372165468 | 109556784937216546703 |
| F1 | 1095521956053880197 | 109552195605388019638 |

大きなshot整数・gate費用との積をfloatに落とさず、Python int／Fractionで保持した。48式のoutward endpointのceil一致とUの上界条件を公開時にも照合した。これらはモデルに対する保守的な十分数で、quantum shotをこの回数実行したものではない。

## Bias予算と未取得資格

共通biasは22行のcorrected physical complex signalへの直接絶対差。全12caseでA0補足の有無を区別し、補足のあるcaseでもA0値へ差し替えなかった。予算はb+u_num≤ε/2、gamma≤ε/10、statistical≤2ε/5。

| ε | 観測bだけがε/2以内 | 観測bだけがε/2を超える | 保証付きdeterministic／implementation／end-to-end |
|---|---|---|---|
| 1e-8 | 全6候補・両T（12条件） | なし | 全て未確定 |
| 1e-9 | D1/R2/R4・両T（6条件） | B3/B4/F1・両T（6条件） | 全て未確定 |

A0対22行の観測予算flag不一致は出なかった。しかしA0の経験的一致、correctness atol、B0/B1の有限標本operator差をu_numやgammaへ読み替えていない。u_num=null、gamma=null、全24行のend_to_end_qualified=nullを維持する。Budget外の候補にも参考shot費用は表示するが、同精度適格候補として認定しない。

## S0条件付きshot加重資源

各値は `G_S0=m_X*mean_X+m_Y*mean_Y`。全て既存256 paired trajectoryに基づく**S0条件付き参考値**であり、未知rare費用を含む無条件母平均ではない。各metricのW_i=m_X*C_iX+m_Y*C_iYの不偏分散と同値なcovariance式をexact整数raw momentsから評価し、SE=sqrt(var(W)/256)をinterval保存した。
X/Yを512独立標本にせず、費用MCのSEとquantum measurementのconfidenceを分離する。SE欄は上端の表示丸めでconfidence intervalではない。

| ID | T | ε | G_S0 RZ | SE_S0 RZ | G_S0 CX | SE_S0 CX | 観測b点予算 |
|---|---|---|---:|---:|---:|---:|---|
| B3 | +.18 | 1e-8 | 8.9829204E+21 | 3.50411E+18 | 6.1148661E+21 | 2.77232E+18 | 以内 |
| B3 | +.18 | 1e-9 | 8.9829204E+23 | 3.50411E+20 | 6.1148661E+23 | 2.77232E+20 | 超過 |
| B3 | −.18 | 1e-8 | 8.9871344E+21 | 3.73782E+18 | 6.1184634E+21 | 2.97045E+18 | 以内 |
| B3 | −.18 | 1e-9 | 8.9871344E+23 | 3.73782E+20 | 6.1184634E+23 | 2.97045E+20 | 超過 |
| D1 | +.18 | 1e-8 | 6.0129523E+22 | 3.62906E+18 | 4.1159908E+22 | 2.83709E+18 | 以内 |
| D1 | +.18 | 1e-9 | 6.0129523E+24 | 3.62906E+20 | 4.1159908E+24 | 2.83709E+20 | 以内 |
| D1 | −.18 | 1e-8 | 6.0128657E+22 | 3.67431E+18 | 4.1158778E+22 | 2.88641E+18 | 以内 |
| D1 | −.18 | 1e-9 | 6.0128657E+24 | 3.67431E+20 | 4.1158778E+24 | 2.88641E+20 | 以内 |
| R2 | +.18 | 1e-8 | 6.0380231E+22 | 5.09956E+18 | 4.1356501E+22 | 3.96073E+18 | 以内 |
| R2 | +.18 | 1e-9 | 6.0380231E+24 | 5.09956E+20 | 4.1356501E+24 | 3.96073E+20 | 以内 |
| R2 | −.18 | 1e-8 | 6.0384169E+22 | 5.22345E+18 | 4.1359994E+22 | 4.04025E+18 | 以内 |
| R2 | −.18 | 1e-9 | 6.0384169E+24 | 5.22345E+20 | 4.1359994E+24 | 4.04025E+20 | 以内 |
| R4 | +.18 | 1e-8 | 6.0131141E+22 | 3.74425E+18 | 4.1161245E+22 | 2.95830E+18 | 以内 |
| R4 | +.18 | 1e-9 | 6.0131141E+24 | 3.74425E+20 | 4.1161245E+24 | 2.95830E+20 | 以内 |
| R4 | −.18 | 1e-8 | 6.0129129E+22 | 3.17450E+18 | 4.1158983E+22 | 2.53873E+18 | 以内 |
| R4 | −.18 | 1e-9 | 6.0129129E+24 | 3.17450E+20 | 4.1158983E+24 | 2.53873E+20 | 以内 |
| B4 | +.18 | 1e-8 | 4.5747062E+21 | 3.11914E+18 | 3.1028536E+21 | 2.48265E+18 | 以内 |
| B4 | +.18 | 1e-9 | 4.5747062E+23 | 3.11914E+20 | 3.1028536E+23 | 2.48265E+20 | 超過 |
| B4 | −.18 | 1e-8 | 4.5778987E+21 | 2.95961E+18 | 3.1052159E+21 | 2.33441E+18 | 以内 |
| B4 | −.18 | 1e-9 | 4.5778987E+23 | 2.95961E+20 | 3.1052159E+23 | 2.33441E+20 | 超過 |
| F1 | +.18 | 1e-8 | 2.3875728E+22 | 5.54536E+18 | 1.6277745E+22 | 4.43497E+18 | 以内 |
| F1 | +.18 | 1e-9 | 2.3875728E+24 | 5.54536E+20 | 1.6277745E+24 | 4.43497E+20 | 超過 |
| F1 | −.18 | 1e-8 | 2.3869635E+22 | 5.84962E+18 | 1.6272917E+22 | 4.61507E+18 | 以内 |
| F1 | −.18 | 1e-9 | 2.3869635E+24 | 5.84962E+20 | 1.6272917E+24 | 4.61507E+20 | 超過 |

全144行にはRZ/CX depth、total depth、circuit sizeも含む。Shot加重depthは逐次work proxyであり並列wall timeではない。Exact G分数、variance、SE区間、元のB1軸別mean／variance／SEは[result.json](../../artifacts/research_b_outer_shot_weighted/20261025/saved-once/result.json)へ保存した。公開時にも全144行でexact mean／covariance分散式とSE平方の包含を照合した。保証付き同精度順位・Clifford+T・physical qubit費用は対象外。

## 未知rare費用とshot加重比較

各axis/metricのunknown u_X,u_Yを残し、`G_hat(u)=(1-p)*G_S0+p*m_X*u_X+p*m_Y*u_Y`。真のGにはさらにS0母平均と経験平均との差が残る。Unknown rare mean、variance、unconditional_G／SE／winnerはnull。B1 INCONCLUSIVE_RARE_EVENT_COVERAGEと既存rare CONDITIONAL_SENSITIVITY_ONLYを維持する。

比較の正側P／負側QをA:B3/B4、B:R4/D1、C:R2/D1とし、M_i=m_iX+m_iY、u_i^m=(m_iX*u_iX+m_iY*u_iY)/M_iを定義する。中央S0経験mean下の境界は

`u_Q^m = I + slope*u_P^m`。

同じu_P^mに対しu_Q^mが境界より下ならQ低費用、上ならP低費用。Rare両axis同費用を仮定したものではない。軸切片Iは他方を座標上0とした数学的交点であり、実費用を0補完した仮定ではない。未知rare費用の上界がなく、費用非負性だけでは無条件順位を保証しない。

以下はε=1e-8の**RZ**中央境界の表示。ε=1e-9も丸め桁では同じだが、ceilの違いを含むexact係数は各条件で別に保存した。

| 比較(P/Q) | T | I central | Slope | I at ±2SE矩形 | S0経験寄与の差の感度 |
|---|---|---:|---:|---|---|
| B3/B4 | +.18 | 1.4402660E+8 | 1.31069607161E+1 | 1.4359369E+8 … 1.4445950E+8 | Q低費用側の差が±2SEで正 |
| B3/B4 | −.18 | 1.4405996E+8 | 1.31069607161E+1 | 1.4362221E+8 … 1.4449771E+8 | Q低費用側の差が±2SEで正 |
| R4/D1 | +.18 | 4.0359583E+3 | 1.00000267744E+0 | -3.2730086E+4 … 4.0802003E+4 | ±1SEで符号未確定 |
| R4/D1 | −.18 | 1.1745373E+3 | 1.00000267744E+0 | -3.2976131E+4 … 3.5325205E+4 | ±1SEで符号未確定 |
| R2/D1 | +.18 | 6.3883046E+5 | 4.99848340320E-1 | 5.9530402E+5 … 6.8235689E+5 | Q低費用側の差が±2SEで正 |
| R2/D1 | −.18 | 6.5080730E+5 | 4.99848340320E-1 | 6.0643743E+5 … 6.9517716E+5 | Q低費用側の差が±2SEで正 |

B4対B3はS0参考費用に大きな差が残った。R2対D1ではR2のnormalization低下によるshot減少は、保存S0単価の増加を相殺せず、固定S0経験寄与ではD1が低い方向を±2SE感度でも維持した。D1対R4は±1SE矩形で符号が変わり識別不能。これらは全24比較に保存したconditionalな感度で、無条件母平均順位や95%のwinner保証ではない。候補間独立性・共分散を仮定せず、各候補のcost SEを足した矩形を用いた。RZ/CXの全比較式・unknown係数・±1/2SEはCSV/JSONが正本。

単価だけの旧境界を転記せず、各候補のshot数で係数・切片を更新した。追加sample・rare circuit取得や巨大な仮想費用gridは生成しなかった。

## 環境・性能・保全・次の判断

Python3.11.9、mpmath1.3.0、Linux x86_64の既存環境を使用。Preflightで32 CPU affinityと `MemAvailable: 55193548 kB` を確認した。小規模な保存値走査・scalar演算として直列実行し、B_j／joint momentのcacheを再利用した。CPU／RAM／worker／wall timeに人工cap、依存install・native build、共有環境や他job操作は0。

Whole-process wall **1.05秒**、user CPU0.88秒、system CPU0.07秒、peak RSS114,252KiB、swap0、exit0。Evaluator wall0.757秒はprocess起動・最後の保存までの時間とは異なる。速度向上率の対照測定は行っていない。

固定SOURCE変更・中断・修正・retry／resume・追加attemptは0。Input/correctness/ceilが通過した技術的実績を、signal全体の数値誤差・実装誤差やsame-accuracy資格へ格上げしない。旧A0/B0/B1/22行/rareのSOURCE・raw・STOPを保全する。

結果を公開しremote内容照合後**mandatory STOP**。次は研究Bのユーザーが、条件付きモデル会計の採択、u_num／gammaやrare費用保証を強める必要性を判断する。Codexは新しい明示的なscopeと許可なしに、追加MC／A1／gamma評価／quantum shot／新候補・新精度・全RPE評価を開始しない。
