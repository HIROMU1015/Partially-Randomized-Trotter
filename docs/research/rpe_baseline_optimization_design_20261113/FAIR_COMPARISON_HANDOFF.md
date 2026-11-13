# PR・RPE統計・価格担当への公平比較HANDOFF

[REPORT](REPORT.md)／[探索計画](BASELINE_OPTIMIZATION_PLAN.md)／[次の有限案](NEXT_SCIENCE_PLAN.md)。値の新規評価・新canonical ID発行・実G/候補選定はしていない。Cはbaseline固有の設計を担当し、他担当のsolver/controller/compilerを複製しない。

## 1. 共通task contract

同じ保存full DF-Hのground energy、H/state/snapshot/sector、幾何・basis・DF表現、energy precision、prior/alias/decoder、lossと全outcome処理、state仮定、P0、価格resource/unitを使う。referenceはfull-H targetへ結ぶ。旧finite-time精度/HP型Nと今回のRMSE Nを区別する。

最初のshellは保存h=.1/M12。h/Mを比較対象にするなら、各方式に同じ採択範囲を与える。profileにはcompiler hashだけでなくcontrol・native policy・lowering ID・measurementを含める。旧generic-controlled ZZ価格と新CX-CRZ-CX価格を別contextとして保持する。旧価格を上書きせず、特定方式だけ新最適化を使った差はその実装scenarioの差として報告する。

## 2. candidateと全roundの受渡し

既存`pr_rpe_rmse_controller.generate`、Dの`full_baseline_v2_requests`、Aの`h4_four_method_screen`のmetadata mappingを再利用する。以下は必要な意味の一覧で、新しい共通schema/API実装の要求ではない。

|層/情報|必須の意味|
|---|---|
|task|H/state/snapshot/sector、target Hamiltonian/energy、ε/claim、prior/decoder/alias/loss/risk policy、P=0|
|configuration|方式、formula、requested Δ/global p/B0又はB0I、h/M、finite-K/κ又は明示layout、共通profile/Source|
|round|m、T_m、q_RPE、q_PF又はpure grouping q、実幅T/q、R、K、signed weights・順序、整数allocation、総event位置、scalar policy|
|physical identity|元physical candidate ID、H/preparation/split/retained-H/one-body/sampling law binding、数値backend SOURCE|
|B identity|Bが生成したcanonical configuration/round ID。physical IDと別field。新IDをCが捏造しない|
|signal入力/出力|同target reference/source、raw/corrected complex mean、logB/B/attenuation、signed bias、radius/phase、数値error/allowance/資格|
|統計入力/出力|全round mean、state/energy/prior仮定、全事象loss、N_X/Y・配分、feasible/UNKNOWN、不足条件。実際のshotと十分数を区別|
|価格入力/出力|同candidate/axis/law/full wrapper/profile/lowering、DIRECT/MODEL/extrapolation/missing、law mean又はfinite cohort、区間/共分散/systematic/source|

B既存specはFull PF/TRUNCATED PFにglobal formula/Δ/L_Dを持つ。Full PFのmethodはPF、split.deterministic_scope=FULL_H。truncatedはTRUNCATED_H、retained_hamiltonianにconstruction/source/prefixと元target Hを保持する。physicalのFULL/B0/B0IとBの比較方式名は別fieldに置く。標準Y4はformula='4th'、M5は'4th(m5_best)'で別のcanonical formula IDとsigned tupleを使う。

Full-Hはmethod=FULL_H_RTE、formula/Δ_pf/κ=None（κなしlayoutの場合）、L_D0、one_body_policy=RANDOMIZED、deterministic_blocks0、all DF fragments、full-H λ/source、抽出identity scalar一回、有限偶数K。physical legacy row内method=PRでもcomparison_method/family=FULL_H_RTE/FULL_H_PURE_RTEを保持し、DF B3/prefix0と誤結合しない。

全roundでTとq/R/Kを明示する。pureの総s=qRが同じでもphysical ID/grouping・CDF/seed reset・compiler対象を保持する。保存signal共有は同値性/数値scopeに限定し、価格の共有を意味しない。

## 3. 担当と既存実装の境界

- **PR候補/signal担当**：既存PR探索を管理。baselineが同prefixに拘束されるわけではない。同prefix比較用には同retained/tail/sourceを渡す。
- **baseline物理signal担当（A等）**：元snapshotを再利用し、PFは直接expmのsigned macro-step、pure RTEは元finite-K polynomial/raw/normalizer経路を使う。保存mean/referenceの再利用と未完screenを管理。Cがその実行中worktreeを読む/変更する必要はない。
- **RPE統計担当（A/B等）**：採択した全事象energy riskとN政策を評価する。signal局所値だけからCが十分N/全RMSEを再実装しない。N最適化が価格に依存する場合の再接続もこの担当が行う。
- **価格担当（D等）**：同一lowering/compiler/full wrapperのdirect/model区間を供給。Full PF affineの検証、pureのlaw平均とrare/残差を管理。Cはcompiled価格取得やfitを重複しない。
- **B**：physical schedule→canonical→統計/価格受入→全Gの既存flow。実会計・正式選定は別科学許可後。未測定round/N/Cを0や旧値で埋めない。
- **C**：今回の探索自由度・公平性・SOURCE索引。将来必要なbaseline固有metadata差分だけを技術提案する。

現在のglobal Δ/formula/p/K以外に、round別q/p/formula/Kを探索したい場合は、まずBが明示scheduleの版と再現条件を管理する。Dの物理builderが任意qを受け取れることと、Bの全schedule資格APIがその可変性を扱うことは別である。共通部品の変更をC branchで先取りしない。

## 4. 科学的な比較条件

Truncated PFはB0/B0Iともfull-H target。cutoffの補正やtarget包含の不明量を0にせず、保持Hのgroundや自分の最適化状態へstate/targetを替えない。B0Iの元scalar規則はphase/controlにも反映し、signal側だけ補正して価格側でscalarを抜かない。

Full PFは回路の細かさとNの競合、pure RTEはfinite-K/attenuation/総eventとNの競合を全roundで扱う。同じq、同じN、同じformula、同じK、同じ古典compile予算を強制することが公平性ではない。どの自由度をどこまで探索/未探索としたか、精度達成と計算努力を別に表示する。

価格誤差はcdf6a806のdomain/同時区間契約。PRのE[S]_idealとDのsampler残差、Full-H専用law、fixed seed PCG64のIID仮定はd72dc4e6の区別を保つ。Sやevent数はcompiled lower boundではない。NとCを掛けられることと、同taskの適格な全RPE費用になることを区別する。

CLAUDEの戦略レビューは検討材料で、候補削減・下界・検証省略の正式採択を代替しない。今回その提案を自動採択していない。

## 5. 取込みとSTOP

成果はこのdocs一式と既存SOURCEへの参照だけ。新production code/runner/共通APIは0、branch全体のmergeや他担当branchの変更は不要。必要な設計文書だけを参照・選択取込みできる。

次は**ユーザー**が有限scopeとclaimを判断し、担当がNEXT_SCIENCE_PLANの対象SOURCE/ID/上限を固定する。本納品自体は科学実行許可ではない。Cはremote SHA/tree/主要成果物照合後**mandatory STOP**し、新探索・再解析・compile/Gへ進まない。
