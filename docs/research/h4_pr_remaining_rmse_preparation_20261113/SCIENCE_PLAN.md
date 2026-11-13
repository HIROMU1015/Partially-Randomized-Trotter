# S1/S2の有限科学実行案とS3の実行前条件

2026-11-13 JST。**未実行・新科学承認待ち。** [PLAN.json](PLAN.json)のexact configuration、10 physical candidate、SOURCE/law/入力locatorを正本とする。今回はmetadataと人工試験だけ。最も重複の少ない次実行はS1＋S2の一括scopeであり、roundごとの追加承認や別runner作成を繰り返す計画ではない。

## 共通SOURCE・入力・固定条件

- 本REPORTを含む公開commitを新`h4_pr_rmse_completion.py`、`pr_rpe_power_mean.py`、runnerの科学SOURCEとして固定する。実行時`--source`で同定し、旧科学SOURCEへ偽装しない。新しい承認証明書・grantは不要。
- B3216d27a69ac8532d2a22e78f5f94491d202d238の正本configuration ID `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。Aの受入済みm1 packet 3e035c0d61386010f59ad7ea792bc7a5cefd18acを参照。
- H4 DF10/linear1Å/STO-3G/sector(2,2)/36・保存数値state・P0・S2/prefix1/K2/ΔPF=.1/h=.1/M12・S0 exactκ・minkowski/finite_Bを維持。εE=.00159360010199040 Ha、両budget=.00079680005099520 Ha。入力H/state/snapshot/law/order/scalar/compiler/prior/decoderを変更しない。
- 保存snapshot/receiptとtail contextの正確なpath/hashはPLAN.context_source。既存H4 loaderのhash/sector/H/state/completion検査を再利用し、SCF/DF fit/state生成は0。数値state energy/residualは既存receiptの値を用い、ground保証とは分ける。
- 共通scalar gate/geometryとR-aware式は科学7c3e1e5d…、reference checkerはP1 71ba49bb…、reference/actionは既存D2、K2 lawは既存rte.py。旧高確率N solverは呼ばない。旧m0/m1/m6の64/60/38はcopyのみ。
- complex128/元binary64 schedule、元reference≤1e−10、norm/vector/scalar closure≤1e−9、数値allowance u floor1e−9・logB margin floor1e−12・元R-aware event/stage式を維持。これらは数値gate/empirical allowanceで、rigorous誤差上界ではない。

## S1：m3/m4の保存meanを補完して診断

| 項目 | 有限scope |
|---|---|
| 対象 | m3/T=.8/q8/R16/ID045843b2429bece6f29b、m4/T=1.6/q16/R32/ID3df0cfd3356c358aeda8 |
| 保存再利用 | 155529e2…long_time signal_recordsの2 mean、exact-tail対照、logB、raw/corrected/bias/最大block境界norm/closure。元endpoint ε=.0005をselectorにしたu/margin。旧Nは不採用 |
| 元数値SOURCE | e417ffc9…`h46_long_time_signal.evolve_batch`、polynomial/spectral block。元SOURCE hashとcarrier bindingを照合済み |
| 新計算 | T=.8/1.6の二経路full-H reference **各2 action、合計4 action**。保存mean2候補のscalar gate/bias/phase/radiusとnominal X/Y **2組** |
| 新mean/normalizer | **0**。内部対照は保存値を使う。保存logBのexp表示等のscalar算術はS1に含む |
| norm証拠契約 | `observed_error=null`を維持し、返却前の元SOURCE gate≤1e−9を条件付き診断の根拠にする。これは提案・承認対象で、観測値やrigorous資格への昇格ではない |
| reference整合 | 保存referenceを中心に保ち、新pair2経路の元値も保存。3者の差を元gateで比較し、その最大をallowanceに反映。中心の取り替えでgateを通さない |
| allowance | 元R-aware logical event式を使い、`max(新u,過去u)`、`max(新margin,過去margin)`。旧backendのblock境界normを用いる違いを明記 |
| 並列化 | reference2時点は独立、診断2候補は独立。H/state contextを共有し、同じfull-H作用を重複取得しない |
| 成功条件 | exact identity/SOURCE/law、保存成功record、reference一致、finite norm/closure、bias分解・元allowance条件・Decimal ceilが成立。結果はnominal/UNQUALIFIED |
| STOP | source guardの根拠が不足、観測scalarが必要と判明、旧endpoint不一致、reference/closure失敗、非有限値・丸め未解決。新meanや閾値変更へfallbackしない |
| B成果物 | m3/m4のmean/ref/allowance provenance、SOURCE guardと観測欠測の別欄、局所diagnostic、nominal/null資格と全canonical identity |

**追加meanの条件付き別scope**：SOURCE guardを条件付きempirical診断に使えない場合に限り、同じm3/m4のmeanとexact-tail対照を各1候補（計2候補）取得する別案が必要になる。今回は自動分岐を実装していない。S1実行中に判明したら部分結果を保存してSTOPし、科学範囲を勝手に拡張しない。保存meanの再計算を先に要求する計画ではない。

## S2：欠測8 meanの取得とnominal batch

| 項目 | 有限scope |
|---|---|
| 対象 | m2,m5,m7,m8,m9,m10,m11,m12。T/q/R/event/physical IDはPLANのS2 rowsに固定 |
| reference再利用 | 89c91c692…H4 batch `scope_observed.references`のT=.4,3.2,12.8,25.6,51.2,102.4,204.8,409.6、**8 pair**。新reference **0** |
| 主計算 | SOURCE e417ffc9…由来spectral powerによる **8 finite meanと8 exact-tail対照**。3columnを同一head列へ通すが、corrected/rawは独立に伝播。対照は内部共有で、別の16 jobを意味しない |
| normalization | K2有限distributionを固定signed event timeで各候補評価し、元logBの積和。今回S2のS2公式は一pieceなので通常 **8個のevent normalizer**。同じdistribution callbackを通じone-ULP log感度も記録 |
| 高速前処理 | 同じ保存contextのhead2 blockとtail1 block、合計 **3 spectra**を既存spectrumで構成・共有。full-H ground eigensolveなし。再構成/isometry残差を保存。head/tailの積順を変えない |
| 実系同値性 | 最小候補 **m2だけ**を既存event逐次actionでも **1 mean＋1 exact-tail対照**として取得。raw/corrected/exact-tailの差が両経路の同recipe allowance内、logBが一致することを検査。この直接経路には同じK2 normalizerを **1回追加**使用（合計9 event-normalizer評価）。新候補探索ではない |
| 局所評価 | 主8候補のnumerical gate/allowance/phase/radius、経験的nominal X/Y **8組**。直接経路はgate/disk比較のみで、別Nや別候補を最適化しない |
| 精度 | 元gate・R-aware u/stages/event数を保持。過去allowanceとnormalizer感度を下回らないmargin。b≤1/非有限値/underflowをB=1で救済しない |
| 高速化 | tail polynomialのR乗でevent loopを除去、outer qとhead順は維持。最大m12でq4096、R8160、logical events33,423,360。spectra共有・独立vector・8候補並列。独自CPU/RAM/worker capなし |
| 成功条件 | 同H/state/Tの保存pair、K/law/order/scalar/normalization一致、元norm/closure gate、bias/allowance/ceil条件とm2照合が成立。長時間の各候補にも独自の数値gateを要求 |
| STOP | binding/保存pair不一致、normalizer精度/非有限値/closure破綻、direct-vs-power不一致、spectral再構成失敗。R/K/M/閾値やアルゴリズム変更、科学retryへ進まない |
| B成果物 | raw/corrected/exact-tail/logB、観測normとspectral残差、reference/mean SOURCE、局所disk/nominal/未資格、m2同値性、full identity/law |

S1＋S2を一括承認する場合の主scopeは、**保存mean診断2＋新reference4 action＋新高速mean8/対照8＋m2直接mean1/対照1＋新nominal X/Y10組**。実系数値同値性はこの一候補に限定し、さらに別roundを追加しない。m2の一致は全長時間roundのrigorous保証ではなく、長時間側のnorm/closure/sensitivity/allowanceも個別に報告する。

## 一括実行と結果保全

既定CLIはmetadataだけ。

```bash
python3 -B scripts/run_h4_pr_rmse_completion.py --out <fresh-PLAN.json>
```

S1/S2の科学実行は、ユーザーが上記有限scopeとnorm証拠契約を明示承認した後だけ行う。

```bash
python3 -B scripts/run_h4_pr_rmse_completion.py --execute --stage S1_S2 \
  --source <今回公開された固定科学commit> --out <fresh-run-directory>
```

`S1`又は`S2`だけの段階実行も可能だが、既存m0/m1/m6は常にcopyだけ。段階を分ける場合も、保存された新model recordsを`join_records`で同一configurationへ結合し、meanやNを再計算する必要はない。重複round・他SOURCE/identity・旧policy・十分N混入は拒否する。今回の旧m1 packetを再作成するCLIではない。

各roundのmean/refは局所gateより前に独立fileへ保存、診断は完了ごとに保存する。並列で開始済みの有限候補は元入力のまま完了/失敗を記録する。失敗を理由に既存・他担当のprocessを停止しない。群にFAILがあれば新しい合格packetを出さず、部分結果と理由を残してSTOP。retry/resume/候補変更を自動実行しない。

最終packetには新scope/元packet locator/roundごとの診断SOURCEを付け、m0/m1/m6の値と元SOURCEを保持する。十分N_X/Y、actual N、global RMSE boundはnull、全体UNQUALIFIEDを維持する。B3216d27aのm1専用validatorへ偽装せず、B担当へ新scopeを渡す。

## S3：全13round energy RMSE（未決定・今回実行0）

### 必要な入力と統計モデル

全13roundの候補・whole-schedule ID、raw mean `z_m`、SOURCE/数値allowance、axis別の正整数nominal N、固定h/prior/decoder/targetが必要。実meanがない時点で仮Nを埋めない。S1/S2完了後に、保存済み3roundと新10roundを同じcanonical contextで結ぶ。

ideal finite lawのfresh-iid Hadamard試行なら、X/Yの確率は `p_X=(1+Re z_m)/2`、`p_Y=(1+Im z_m)/2`。各試行で正の結果数を独立binomialから取り、座標`2K/N−1`を元decoderへ渡せる。同じtrajectoryを多shotで再利用するlawとは別。raw meanをcorrected meanへ置換しない。確率が範囲外/数値未解決ならclippingせずSTOP。

再利用候補は`pr_rpe_rmse.decode_supplied_coordinates`→`rpe_four_round_phase_validation.reconstruct_rpe_phase`。既存`_marginal_branch_monte_carlo`はbinomial生成とvectorized reconstructionのSOURCEを提供するが、旧β/phase failure/参考oracle branchという目的である。その返値をenergy RMSEに読み替えず、将来の最小変更で全energy lossを保存し、scalar decoderとの人工一致を確認する。oracleのE_stateで実decoderのbranchを選ばない。今回はsimulation runnerを追加・実行していない。

### 未解決trialと出力範囲

元decoderはzero X/Yやnearest-branch tieをenergy=nullとする。これらのtrialを捨てたRMSEは成功条件付きに限る。ユーザーは次のどの評価を行うかを科学的に決める必要がある。

1. 未解決事象へのSOURCE付き出力規則を含む完全なenergy推定器を定義して、全trialのenergy RMSEを評価する。元decoderを無断で変更しない。
2. decoderを維持し、未解決確率とresolved条件付きMSEを別々に報告する。この場合、全trial energy RMSEの資格は未取得のまま。
3. 明示的な失敗lossを導入するなら、それはまずそのlossに対するriskの評価であり、出力energyが定義されない限りenergy RMSEとは呼ばない。

resolved時の元phase wrapからenergyは`[−π/h,π/h]`内だが、これはpriorへのclippingやalias資格ではない。未解決の扱いを含む有効なloss範囲を定義できるか確認する。元priorが保存DF-H groundを含む証拠と、`h·prior幅<2π`のalias条件も別に保持する。

### 統計精度・信頼度と有限trial範囲

恣意的な10000trial等を先に採用しない。新検証の信頼水準 `1−δ_val` とMSE絶対精度 `η_MSE`（Ha²）を、何を判断したいかからユーザーが指定する。旧αtotal=.05はこの新δではない。例えば予算との比較なら、MSE上側信頼限界がεE²以下か判断できる精度が必要で、境界なら「未解決」と報告し試行を自動追加しない。

全trialの二乗lossを`0≤L≤L_max`と保証できる契約を定めた場合、固定iid試行数の一つの保守案は

`n_val ≥ ceil[L_max² log(2/δ_val)/(2 η_MSE²)]`。

これはHoeffdingを**科学検証のMSE標本平均**へ用いる一般式であり、旧round別高確率Nの流用ではない。`L_max`は二乗lossの上限（Ha²）。推定器の出力範囲とtarget範囲が定まらなければn_valも未定/nullでよい。CI `[L_hat−η,L_hat+η]∩[0,L_max]`からRMSE区間を平方根で変換する。未解決率を別推定するなら、その二項区間と同時信頼度の配分も先に決める。別の有効な分散利用型/逐次CIを選ぶこともできるが、手法・停止規則を固定してから実行する。

n_valが決まれば、13round×X/Y=26個のbinomial countを1trialとして生成する固定有限scopeになる。これは量子shot数と別の古典検証反復数である。実際の回路sample/trajectory/compiled価格はこの統計pilotのscopeに含めない。パラメータ探索・N最適化も別scope。

### target・系統bias・Minkowski

保存DF-Hの真のsector ground energyが最終target。保存E_stateを検証truth proxyに使う場合は、保存数値state条件付きという証拠水準と残差/ground未同定を明記する。有効PF groundとの一致を仮定しない。元分子HとDF-Hの差まで保証しない。

一般にmeanエネルギーbiasと統計分散の両方を評価する必要がある。局所phase biasをTで割った値やraw error diskをεtrotとして全体へ足す根拠はない。Rがroundごとに変わるfinite-K meanが、同じ有効Hermitian Hamiltonianの全round信号を厳密に表すとも限らない。

Minkowski契約を使うなら、定義した決定論基準energyとtargetの差、および同じ確率モデルでの推定誤差のL2上界を説明できる必要がある。全meanの無限shot decoderを基準にする案もあり得るが、zero/tie/aliasがないこと・成分が元εqpe/εtrotの意味に合うことを科学判断してから採択する。今回基準energyを新計算せず、この分解も採択しない。

S3の最初の結果はideal finite mean/保存state条件付きの**経験的**RMSEと統計区間になり得る。実samplerとの差・実装誤差・厳密ground誤差を0にしない。十分Nやglobal rigorous boundを供給するには、信号/decoder/誤差契約の適用根拠がさらに必要で、nominal配分やMonte Carlo点推定だけでは認定しない。

## 次にユーザーが承認・決定する最小事項

- **次の科学実行**：上記固定H4/S1_S2の全10round、m3/m4のSOURCE gateに条件付けたempirical診断、4 reference action、8高速mean/対照、m2直接照合1候補、10nominal pairまでを一括承認するか。S3/価格/Gは含めない。
- **その後のS3判断**：全mean取得後、未解決trialの契約・target/state/lawの証拠水準・Minkowski成分・検証MSE精度/信頼度を決定し、導出された有限反復数とSOURCEを別承認する。今回任意の値は補わない。

**A**は承認されたscopeだけを実行、**B**は新scopeを未資格診断として受入、**ユーザー**は科学条件と次のscopeを判断する。今回公開照合後mandatory STOP。
