# 最小の追加証拠を得る有限案（全て未実行）

2026-11-13 JST。今回の承認は調査・条件整理まで。以下は**新しい科学実行の承認ではない**。固定H4 DF10/S2/L_D1/K2/ΔPF=.1/h=.1/M12/exactκ/N配分・既存decoderを保全し、必要な段階だけ別途選ぶ。

## 0. 実計算前の決定・技術準備

| 決定 | 指定が必要な理由 |
|---|---|
| 主張水準 | plug-inモデルのconfidence付きrisk、解析的ideal-HDFのrisk、actual-law／数学的global boundを区別する |
| 全事象loss | UNDEFINEDのままでは全RMSEがない。未解決energy補完family又は固定lossを定義する。decoder/fallback変更は別採択 |
| 経路 | 直接全riskを示すか、既定εqpe/εtrotの各成分も示すか。新budgetは勝手に設けない |
| 精度・confidence | 上側riskのη、統計confidenceとtarget包含幅・raw許容誤差を、所望の判定不等式に合わせて事前指定 |
| 新SOURCE | 既存B ab12・A bba/169d・C d72を参照し、新しいreader/認証手法の公開commitを固定する。今回未実装のsolverを既存品として扱わない |

既存SOURCEからsupportの全域条件、上側CI又は解析的tail法、exact/interval計算の方法を人工入力で確定するのは技術準備。実H4値への適用は下記の科学scopeとなる。

## B1：保存S3の上側risk評価（最初の統計候補）

- **対象・入力**：2084791の固定run01に保存された43,819 trialの1群、PRIMARY/EXECUTIONと同じproxy・N・decoder。原本を一回だけ新目的で解析する。元結果は上書きしない。
- **目的**：全事象loss又はenergy補完familyの契約の下で、解決分population MSEと未解決loss寄与を含む有効な上側riskを得る。条件付き点RMSEへの誤った昇格を避ける。
- **有限call**：固定1群のreader1回、事前固定した上側評価法1つ、同じ群の零/未解決/alias診断参照。**新binomial draw・decoder再simulation・mean/reference/normalizer/shot変更は0**。
- **再利用**：B rpe_decoder_statistics.summarize／Clopper–Pearson／loss会計を使用可能。現Hoeffdingだけでは粗いことが既に分かっている。分散利用法・tail法を選ぶ場合は技術検証とSOURCE固定を先行し、元CIに後付けで都合のよい方式を選ばない。複数法の比較・選択を行うならconfidence会計と有限法集合を別scopeにする。
- **成功条件**：全trialを会計し、support・未解決loss・数値誤差・confidenceを満たした上側riskが得られること。目標未達も保存可能な結果。**既存43,819回だけで有用な上界が得られる保証はない**。
- **STOP**：SOURCE/hash不一致、loss未定義、support証明不足、未許可の新推定器が必要、数値精度未解決。trialを捨てる、追加反復・再seed、CI法を替えてretryしない。

この段階は保存値の新科学再解析であり別承認が必要。極端なloss域を使った単純Hoeffdingの必要反復数を、新たに数値算出・自動追加する案ではない。

**B側の代替技術案**：全loss域が粗くB1では判定不能と見込まれる場合、zero/tie/誤branchの確率massを解析的に囲む方法を、先に人工入力で検証する。zeroはREPORTの有限Binomial式を再利用できる。branch依存部はdecoderの逐次状態に沿ったmass集約や保守的tail上界が候補だが、既存exact_countsを13roundへ直積適用するだけでは解決しない。新方式が技術的に確立した場合の科学scopeは、固定1 configuration／固定26軸count law／全13round／固定Nについてのrisk上界評価1回、新simulation0とする。方法・誤差精度・SOURCEを確定してから別承認を求め、Aでsolverを重複開発しない。

## A1：同じ保存DF-Hのground energy包含

- **入力**：既存snapshot/receipt、hamiltonian hash `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`、snapshot `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`、state `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`。元loaderのsector(2,2)/36・signed weights・completion・fragment順・scalarを保全。
- **目的**：EproxyとE0,DFの距離を認証する。小さい残差だけのground同定を避ける。
- **方法候補**：事前指定したdreqについてL=Eproxy−dreqを固定し、同じ正確な保存係数でHDF−L Iの正定値性をexact-rational又はverified interval LDLで証明する。保存stateの正規化Rayleigh商の認証上界Uも得て、[L,U]で最低固有値を囲む。U≤Eproxy+dreqが成立すれば所望のproxy距離が得られる。これは方法の候補で、dreqの値は今回選んでいない。
- **有限call**：**保存sector行列の認証組立1個（36 occupation列）、保存stateのRayleigh商1個、固定Lの正定値判定1回**。新eigensolve/SCF/DF/state generation・signal/reference/normalizer・S3は0。既存occupation表現／binding検査を再利用し、binary64行列を無根拠にexactなHDFとみなさない。
- **精度**：exact-rationalなら保存binary64係数を正確な数として扱う。intervalなら事前指定precisionと外向き丸め・行列組立誤差を含む。単なるfloat LDLの正pivotを認証にしない。保存stateはnormを新規変更せず、Rayleigh商の分母も包含する。
- **成功条件**：同HDFに対するL≤E0,DF≤Uと指定幅を証明。gap/overlapはこの直接target証拠の必須出力ではない。元ground_state_certified=falseを改稿せず、新SOURCEで達成範囲だけを報告する。
- **STOP**：binding/sector不一致、Hermiticityの認証不成立、interval pivotが正と判定できない、指定dreqを満たさない。新state・閾値緩和・precision増加・端点探索・追加eigensolveへ自動移行しない。

既存priorと元eighによる数値証拠は再利用するが、これらだけで新包含を得たとは扱わない。入力はDF-Hに限定し、DF前Hmolのenergyをこの1問題から保証しない。

## A2：raw数値enclosure（数学的globalを求める場合のみ）

直接riskの統計モデルを解析的ideal finite-Kへ移送するには、13rawの数値誤差が必要。経験的allowanceの仮定で限定診断を続ける選択肢と、認証enclosureを取得する選択肢を区別する。

- **有限候補**：同じ1 H/state、固定13候補だけのverified finite-K mean batch。2個の共有head half-stepのverified指数、各roundの同じK2 polynomialのR乗・outer q乗とscalar phase、**13raw enclosure／13normalizer enclosure**を取得する案。大きなevent loop・新trajectory・新reference/exact-tail対照は直接riskのためには必須でない。
- **再利用**：c76ffe5のpower-meanの同じ分割・順・scalarと保存結果を比較対象に使用。新verified算術は先に人工・SOURCE検証が必要で、元kernelへ黙って置換しない。今回その実装も数値適用も行っていない。
- **精度・判定**：事前指定したηm,aの包含幅、正しいnormalization、保存値との対応を検証。m3/m4の新enclosureが得られても元観測normの欠測はそのまま。観測normだけ2値を追加することをglobal数値認証の代用にしない。
- **STOP**：指定精度未達、closure/normalization/候補不一致。R/K/δ/M・候補・科学retryを変更しない。

これは13roundの新科学計算なので別承認が必要。まずBが必要なrisk精度を確定し、raw数値認証が本当に必要なclaimか判断する。全13meanを最初に取り直す指示ではない。

## C F0とjoint-law（actualへの移送を求める場合）

C d72dc4の`finite_k_sampler_law_alignment_20261113/SCIENCE_PLAN.json`をそのまま使用する。対象は1 preparation、保存324 componentと13 order分布、P/Q/F/Gのexact TVと条件付きevent/path上界。Aでhelper・samplerを複製しない。新sampling/compile/価格/N/RMSE計算はそのF0案の範囲外。

F0だけでは解析重み→source重み、event作用、実PCG64 seed ensembleのjoint性を保証しない。idealモデルだけのclaimでは、その仮定を明示して条件付きで進められる。actual-lawを主張する場合はC/Bが全shot joint lawへの根拠を別に示す。固定seedや同一blobを根拠にΔ=0を採用しない。

## 既定Minkowski予算を個別検証する場合だけ追加するもの

ユーザーが基準Z0（厳密単一位相かsaved-state referenceか、B減衰・Nの扱い）、同じ全事象出力規約・couplingを選択した後、Bが**固定1 configuration／固定N／基準1モデル**のQPE riskと、実近似モデルとの差L2を評価する有限案を固定する。参照modelの数値適用も別承認。今回比較モデル数・反復数を任意採択せず、新simulationは0。

全risk直接評価が通ってもεqpe/εtrotの個別達成とは呼ばない。target/law/数値誤差を既定budgetへ組み込むときは、それぞれの包含先と同時confidenceを固定する。追加budgetを無断で作らない。

**終了**：各承認scope完了又は未解決で結果を保存してSTOP。十分Nは固定N配分・指定claimに対する資格としてのみ扱い、最小N／最適化・Hmol／actual保証へ拡張しない。今回の調査公開後はmandatory STOP。
