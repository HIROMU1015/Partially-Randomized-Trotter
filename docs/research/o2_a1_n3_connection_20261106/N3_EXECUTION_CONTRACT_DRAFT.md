# A1 N3限定測定・科学実行契約草案

2026-11-06 JST。**推奨案／未採択／実科学実行は未許可。**
ユーザーが今回許可したのは技術実装・人工/mock検証・本草案・公開まで。本書の数値や方法は既存の正式条件を自動変更しない。

## 1. 固定対象と前提

保存linear H4/1.00 Å/STO-3G/legacy DF12、8 system qubits、全4電子number sectorのN2保証ground、L_D3、m0/q1/δ=0.02/r1/K2、coefficient_atol=0.0、X/Y、理想gate・保存DFモデル限定。
N1=744e1a09661ebba6841fe5faf456f88c41c47780、N2=5e9bd08068738cdf1070bdc43bee8194b113ae33、engine=fdf00cee3c87d804845497afc89ac6a71f3b3f59、target契約=2b2e6c1bc3b2d201d1a6d1238172187d68aea6ab。
一次targetはN1 actual finite-grid Qとselected full列によるattenuated複素mean μ_X/μ_Y。w=W=1。元未近似H、algebraic/full-basis信号、energy/RPE phase精度へ読み替えない。
N1 preparation/CDF/ν/policy、N2 state/witness、engine gate proofを再利用し、これらの認証を再実行しない。

## 2. 推奨する採択事項

| 項目 | 今回の推奨案 | 理由・限界 |
|---|---|---|
| 座標精度 | ε_X=ε_Y=1/8 | 最初の1-round mean診断として有限に区切る。旧1/4を無条件継承せず細かい座標目標を提示。energy精度や検出力の保証ではない |
| 同時失敗確率 | α_diag=1/20、4方式×2軸全体 | 全方式に同じ保証を与える。18round配分を流用しない |
| Control集合 | 既存g1/g2/g3をそのまま保持 | feature変更と手法評価を混同しない。新feature選択なし |
| A1学習 | 各axis、5層それぞれ64 fresh conditional bits、計320。既存intercept付きOLSを一回 | 希少controlの無条件出現待ちを避け、designを被覆。これは旧1024-shot無条件学習からの**未採択の変更案** |
| B0学習/pilot | 0 | rawに不要な学習費を負担させない |
| B1-O/B1-OL pilot | 各正層・axisに64 conditional bits（train stageとして記録） | 配分用full価格平均を取得。A1の学習データとは共有しない |
| Calibration | 全方式の全正層/axisに512 fresh bits、A1はunconditional Qで512 | 固定有限数。calが十分狭いvariance上界を与える保証はなく、Hoeffding routeも認める。旧η規則の無条件継承ではない |
| Production | 独立calから十分数を一度凍結、方式/axis合計≤8192 | capは科学callの有限scope。足りなければ未完了STOP。結果後の増量・精度緩和なし |
| 条件付き取得 | actual countによる直接条件付き取得 | 全正層を保持。希少path rejectionやfloat再正規化を使わない |
| Wordモデル | 独立uniform53-bit。実装候補PCG64/SeedSequence、master seed=20261106を一回固定 | deterministic PRNGの独立性を証明した意味ではない。seed選別/変更なし |
| Path word上限 | 1 trialあたり64、測定wordは別domainで1 | 超過はSTOP。CPU/RAM/worker制限ではない |
| CV range guard | exact D_CV=2+Σ|λ_j|≤4 | 実際の凍結係数からbox range。sample min/maxで縮めない、clip/再fit/feature削除なし |
| Engine allowance | 保存ENGINE_ERROR_CERTIFIEDの**厳密な有理数上界**をそのまま使う | 表示≤7.8079008e-7。固定exact engine/理想gate/wordモデルを採択する条件。1/32を正式閾値にしない |
| 主資源 | 同selected full no-prep interrogationのcompiled RZ count | 以下の同compiler/profile・取得範囲を採択する必要。実機価格やTを代用しない |
| 費用の主報告 | 全段階の実現campaign RZと古典費vector | 期待費用改善の保証とは区別する |

上記はコードのdefault実行条件ではない。採択する条件をユーザーが選び、同時に新規科学callの範囲を明示許可して初めて実N3を起動できる。別の承認証明書・grant発行を新設する必要はない。

## 3. 学習可能性と5層baseline

N1に保存された固定対象で、g1≈0.04983、g2≈3.39e-6、g3≈1.89e-9（既存結果の引用のみ）。**1024 unconditional shotsでg2/g3を十分学習できるとは置かない。**
無条件trainでの未観測確率は一般に(1−ν_j)^n、全未観測を避けるのに必要なnはν_jと要求確率に依存する。今回はこの式へ実科学値を代入した新しい学習予算計算・rankingはしていない。
全0または重複列ではintercept込みrank不足となる。元minimum-norm OLSが数値を返しても識別可能性を示さない。本adapterはrank<4ならSTOP。

推奨の条件付き学習は次の5つのcontrol patternを被覆する。

| 取得層 | (g1,g2,g3) |
|---|---|
| n0,L0=a | (1,0,0) |
| n0,L0≠a | (0,0,0) |
| n2,L0≠a | (0,0,0) |
| n2,L0=a,L1≠b | (0,1,0) |
| n2,L0=a,L1=b | (0,1,1) |

a=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`。全5正層を含めるとdesignの4種類の行はintercept＋3controlでrank4を持つ。これはresponseの正確な学習・真のQでの最適係数・CV改善の保証ではない。
条件付きtrainはQ-IID学習ではないが、独立trainで任意λを固定してからfresh unconditional Q productionを取り、exact νでcenterするため、productionの不偏性は保持される。target-Q最適OLSと呼ばず、weighted OLSや重要度再重み付けも今回追加しない。
A1のcal/prodを5層balanced平均へ無断変更しない。異なる取得lawはtrainだけの提案で、全train full RZを課金する。

現3controlは5層内で定数なので、飽和した5層平均にCV定数を足しても既知質量で合成すると相殺する。B1-OLは必須の強い対照で、Raw/order-onlyへの勝利をCV独自改善としない。
5層直接取得は希少層の無条件待ちを要しない。一方、正層quota・全full価格・cal費を要し、exact engineの実際の実行時間・memoryは未取得。高速・安価・実campaign完走を認定しない。

旧1024無条件案を選ぶならrank/未観測をそのまま保存し、失敗時STOPする。Control削減、学習予算の追加、別fit/seedへの切替えは、その案を別途採択しなければ行わない。

## 4. 取得・独立性と同じtarget

- B0：canonical actual Qのraw mean。B1-O：n0/n2の全正層。B1-OL：N1保存の全5正層。A1：上記3-controlを学習後、unconditional Qでcal/prod。
- 全方式で同じH/state/sector/order/phase/Q/selected policy、同じX/Y符号・精度・αを共有する。
- Integer count conditional lookupはQ(ω|h)そのもの。B1は保存p_hによる層平均合成だけを行う。未観測の正層を削除・再正規化しない。g3の層も残す。
- method×axis×stage×stratum×path/bitで独立domainを先に固定。全X/Y別fresh shot、全train/pilot/cal/prod別fresh data。比較方式間の共通乱数・データ共有はこの案では用いない。
- bit0→+1、bit1→−1、event phaseを測定後に再適用しない。CV側に渡すのはbitと必要なpath metadataだけ。
- word capやrange/price失敗の未完了campaignも保存する。完走したseedだけを採用せず、confidenceは成功条件で選別したcoverageとして述べない。

## 5. 統計保証・有限配分

λはtrainだけで一回fitし、binary64 slopesをexact Fractionへ凍結。exact-Q νとexact補正を使うため本モデルのcentering追加誤差は0。違う丸め実装を使う場合はb_center=Σ|λ_j|e_jを別途必要とし、未取得値を0にしない。
B0/B1はrange D=2、A1は凍結されたbox D_CV。独立fixed-n calの不偏sample varianceをs²、cell failureをδ_calとし、

```text
V_U = min(D²/4, (sqrt(s²)+D sqrt(2 log(1/δ_cal)/(n_cal−1)))²)
e   = ε − b_engine − b_center > 0
N_B = max(1,ceil(log(2/δ_prod)*(2 V_U/e²+2D/(3e))))
N_H = max(1,ceil(D² log(2/δ_prod)/(2e²)))
N   = min(N_B,N_H)
```

既知のMaurer–Pontil v1 Theorem10とbounded Bernstein/Hoeffdingを利用する。[一次資料](https://arxiv.org/pdf/0907.3740v1)。calとprodは独立で、λ/層取得法/rangeはcalより前に固定する。sample variance=0を真variance=0とせず、sqrt/log/ceilは既存mpmath.iv100dps＋外向き256bit helperで評価する。
4方式×2軸の各方式軸へα_diag/8、その半分をcal、半分をprodへ。H正層なら各々δ_cal,h=δ_prod,h=α_diag/(16H)。全方式両軸・全正層のunionはα_diag以下。価格保証は付けないためprice用αを流用していない。

B1有限ε配分は、等ε_h=eと、正整数j_hの和J=8に対するε_h=e j_h/(J p_h)から選ぶ。全正層でN_h≥1、Σp_h ε_h≤e、ΣN_h≤8192を要求し、pilot価格のmax(1,Ĉ_h)を使う予測ΣĈ_h N_hが最小の候補を一つ選ぶ。tieはε/quotaのtuple順。
このJ=8は新しい有限候補の提案で、旧草案J=64を自動継承しない。真の最適Neyman配分・期待費用最適性とは呼ばない。
calの各V_Uが同時に有効な事象に条件付け、独立prod前にε_h/quotaを凍結する。pilot/calで選んだ候補をprod outcomeで選び直さない。
N1 lawの全supportに一様b_engineが適用され、B1もΣp_h=1なので共通engine項を一度だけ加える。A1のexact補正にも同じengine平均誤差が使え、λでstate項を増幅する別項はない。

## 6. 価格と全段階費用

Qiskit1.3.0、basis=(rz,sx,x,cx)、opt_level2、seed17、backend/coupling/layout/routing指定なし。保存selected logical full列をそのまま使い、D half steps・scalar/identity/event位相・制御・ancilla・X/Y readoutを含む。状態準備だけ除外する。
取得した全train/pilot/cal/prod path・axisをpriceする。price cache hitもshot RZは課金し、compile実費だけをcache hitとして別記録する。新run専用cache、他axis・未使用path・full-basis対照・全K2 pathの価格列挙は行わない。
compileされた回路でengine測定を置換しない。今回のengine認証は理想logical gateのもの。native合成/実機誤差を保証しない。

全段階の実現RZ、取得数、hit/missとcompiled metricを方式別に残す。共通setupを一方式だけに押し付けず、ν/層質量等の既存setupの扱いも共通に明示する。
古典generation/条件付き整数棄却、fit、補正、compile、cache、setup、engine simulation、peak memoryを分離記録する。未知項はUNKNOWN。RZと秒/bytesを換算率なしに加算しない。engine simulation費は量子algorithmのonline estimator費とは分ける。
`collect`のTrialCost内の古典0は既存型への接続用placeholderだけであり、正式な古典費はexternal計測vectorを使う。`summarize_costs`は欠測なら総古典秒/peakをUNKNOWNにする。

実現campaign差は記述的比較。期待価格/期待総費用の保証には、全pathで有効な価格rangeと独立価格cal・精度/δ等をoutcome前に別途固定する必要がある。今回はUNKNOWNを維持する。
Nがcal依存なのでE[N]E[C]を無条件に掛けない。完了時のΣ実価格と、条件付きproduction予測・期待費用未保証を区別する。

## 7. 明示許可が必要な次回科学scope

**推奨は上記一条件・4方式×X/Yの一campaignだけ。** 許可には本草案の採択項目、固定commit、科学実行範囲を指定する。今回の公開はその許可ではない。
採択後、同APIを呼ぶ薄いdriver/serializerを作り、人工検証・SOURCE/依存環境を実入力前に固定・公開照合する。承認原文の保存や発行手続きは要求しない。

| 次回のcall | 有限範囲 | STOP |
|---|---|---|
| input/service binding | 保存N1/N2/proof復号1回、同じstateをbind。profile/hash照合 | identity、環境、unsupported gate、証明matrix不一致。qualification/solveへfallbackしない |
| RNG | 事前domainへ一回生成、seed固定、trial単位word記録 | reuse・word cap・不正wordでSTOP。retry/seed変更なし |
| train/pilot | 上記固定quota。A1は5×64/axis、B1はH×64/axis、B0は0 | rank<4、range>4、quota欠落。無条件へ切替えない |
| cal | 全方式/全正層/axisそれぞれ512、事前fixed-n | range/独立性不成立、欠測。追加calなし |
| 配分凍結 | 各方式axisで一回、calとpilotだけを入力 | 十分Nがcap超過、headroom≤0、正層欠落 |
| prod | 各方式axis合計≤8192、凍結quotaを一回 | quota変更・path/Born/range/stream不一致。追加productionなし |
| real-state/Born | 取得trialごとに同じexact engineを1回。X/Y別bit | gate count/key/matrix/norm/[0,1]検査。float kernelへ置換なし |
| full回路/価格 | 各取得trialのselected fullのみ。cache missだけcompile | 価格/profile/回路不一致・UNKNOWN価格を0にしない |
| 保存・公開 | bit/path/stream/quota/λ/rank/cal証拠/N/cost/失敗理由を独立directoryへ | 出力衝突。成功でも公開照合後mandatory STOP |

driverは処理前にpath/trial/callをjournalし、bit取得直後に保存、price失敗時も取得済bitを残す。SOURCEにはactual call数と各段階の予定/実施quota、全trace・計測範囲を記録する。
vector・Born値・真signal・E0/gapはCV入力/公開学習テーブルへ含めない。保存stateの原ファイルは保全する。

新partition/preparation/CDF、N2 solve/spectral、engine再認証、新molecule/round、別controls、full-basis対照探索、native費用、期待費用保証、18round、C1解凍はこの案にも含めない。
実行時の空き資源と他ユーザー非干渉を確認し、CPU/RAM/workerに人工上限を新設しない。正確なFractionの性能は未測定で、cache/並列化しても数値kernel・word/同一科学条件は変更しない。

**現在は技術成果・本草案の公開後mandatory STOP。採択と別の明示科学実行許可が必要。**
