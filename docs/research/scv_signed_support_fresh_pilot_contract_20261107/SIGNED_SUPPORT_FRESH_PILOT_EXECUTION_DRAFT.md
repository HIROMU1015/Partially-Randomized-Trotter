# Signed support fresh pilot：有限科学実行契約の第一候補

2026-11-07 JST / SERVER1。**全条件は提案・未採択・未実行。今回の文書作業は実装/人工検証/科学実行の許可ではない。**
固定科学対象・推定器・統計的限界は[科学設計](SIGNED_SUPPORT_FRESH_PILOT_SCIENCE_DESIGN.md)、必要な接続差分は[activation計画](SIGNED_SUPPORT_REUSE_AND_ACTIVATION_PLAN.md)を本草案の一部とする。

## 1. 第一候補：一条件・一campaign・最大448 physical shots

保存linear H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、N1 actual Q/selected full、N2保証ground、既存認証済みexact Fraction engine、X/Y、full no-prepの探索的paired診断だけとする。B0/B1-O・calibration・保証付きproduction・他roundは含めない。

| 設定 | 第一候補（採択待ち） | 根拠・位置付け |
|---|---|---|
| campaign | `SCV-SIGNED-H4-H1-PAIRED-20261107-v1` | 旧224/752/metadata実行と別identity |
| purpose | `SCV_SIGNED_SUPPORT_H1_FRESH_PAIRED_DIAGNOSTIC` | acquisition tagと推定器の目的を区別 |
| h1 fresh train | X64、Y64 | scalar slopeの探索的学習と有料trainの負担の折衷。成功/係数精度を保証しない |
| 共通fresh eval | 各軸h0…4=(8,128,8,8,8)、160/軸 | h1の追加効果を中心に、全正層の診断を保持 |
| physical総上限 | train128＋共通eval320＝448 | B1-OL/SCVで共通評価を二重取得しない |
| 係数 | h1のみ軸別、他4層0 | 既存signed scalar設計に限定 |
| guard | exact D1=2+2abs(λ1)≤6 | 補正幅≤4の探索的安定性guard。旧indicator式を使わない |
| master seed | `20261110` | 旧SCV224=20261109、752=20261108と別。生成していない候補 |
| word/stream | PCG64/SeedSequence、12 domain、24子stream | 既存n3.pcg64_streams/Streamsを再利用 |
| word上限 | path64＋bit1 / trial、総≤29,120 | 既存direct conditional Q・有限word規約を維持 |
| 主指標 | full compiled RZ | Qiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed_transpiler17 |
| paired集計 | full＋固定half A/B、各軸3、計6回 | 同じ320評価recordの補助診断。追加shot/価格なし |
| retry/追加campaign | 0 / 0 | 失敗時のseed変更・refit・追加shotなし |

低費用の代替はtrain32/軸・同じevalで384 physical shots。[科学設計 §4](SIGNED_SUPPORT_FRESH_PILOT_SCIENCE_DESIGN.md)のトレードオフを踏まえ、今回は448案を第一候補とする。両案を連続実行する契約ではない。保存momentだけでtrain最適量が決まるとはしない。

## 2. Stage A：preflight・保存metadata復元・独立namespace

1. 新しい実行SOURCE、環境/依存/native profile、N1/N2/engine証明のbytes、fixed selected policy、compilerを実入力前に照合する。b23b3c0/996e434eと認証済みengine SOURCEを読み取り再利用し、O1/O2の別kernelへ置換しない。
2. b4512675のMETADATA_RESULTをbyte hashで確認し、保存metadata/control identity、feature version、全5層p/ν/m₂/Σ、ELIGIBLE、STOPを読む。旧metadata attemptを再開/再実行せず、次の独立campaignの入力として参照する。
3. 新しいsigned Metadataを既存binderで固定LAW/catalogへ一回bindし、同じmetadata identityを確認する。保存momentを持つtyped signed controlを薄い復元adapterで一回復元する。**SignedControlのmoment計算コンストラクタは0回**。具体的な型/API制約は[再利用計画](SIGNED_SUPPORT_REUSE_AND_ACTIVATION_PLAN.md)に従う。
4. 新output候補 `artifacts/scv_signed_support_fresh_pilot/2026-11-07/campaign_run01/`、manager候補 `manager_run01/`、同output/cacheを固定。既存directory/STOP/旧cacheの衝突ではSTOPし、run02へ変更しない。今回directoryを作成/起動していない。
5. 既存二窓CPU idle admissionと最大30分の有限待機を候補として再利用し、実空きCPU・affinity・memory/pressure・非干渉を記録する。admission不成立ならstate/bundle decode・PRNG・shot前STOP。自分のcron entryと既存運用の持続経路が利用できなければSTOPし、他のcrontabや共有設定を変えない。
6. saved service bundleをparentで一回decodeし、certified proofをattachする。saved samplerとmetadata samplerのlaw/a/b/pattern/bank等の一致後、同一samplerへ接続する。engine内部state/Born/真の信号/E0/gapはCV fitへ渡さない。
7. stream bookを一回構築し、全domain/child mapping、quota、guard、feature/control identityを先に保存する。評価domainの乱数をtrain中にadvanceせず、取得はStage Cまで禁止する。

parentで一回decodeしたserviceを既存Linux fork poolへ引き継ぐ設計とし、workerごとの再decodeを行わない。fork以外の方式・worker再生成が必要なら条件変更なので実入力前に判断へ戻す。shared runtime/manager/engineの再認証は行わない。

## 3. Seed・domain・trialの規約

- train domain：`Domain('A1',axis,'train','1')`、axis=X,Yの2個。A1は取得tagだけで、旧3-control推定器を使う意味ではない。実際のpath取得はB1-OL/h1への直接条件付きQ。
- eval domain：`Domain('B1-OL',axis,'prod',h)`、axis=X,Y、h='0'…'4'の10個。prodは既存schema名で、保証付きproductionではない。
- n3.pcg64_streamsにより**token辞書順**で12 domainを並べ、SeedSequence(20261110).spawn(24)の子2iをpath、2i+1をbitへbindする。PCG64は24個。固定順とspawn keysを保存する。今は生成しない。
- 各domainのtrialは0起点で予定quota内のみ。一claim、path最大64 word、bit1 word。workerのPinnedSamplerは保存pathとbit wordを再検査するだけで、新しいPRNG取得ではない。
- 全体のtrial identityは `(campaign_id, domain.token, trial_id)`。旧helperのticketやMeasurement.keyはoutput内で有効なlocal IDであり、campaignを越えて一意とは呼ばない。source/envelope/journalのcampaign・purpose・control・seed/spawn来歴を含めて混入を拒否する。
- pathとbit、trainとeval、XとY、全条件付き層を分離する。対応のある2推定器は**意図して同じeval record**を読み、独立method campaignとは呼ばない。

独立uniform53-bitは統計モデルの仮定。異なるSeedSequence子や非重複IDだけで、PCG64の数学的独立性を証明したとはしない。旧bit/word/係数/seed/stream/cacheの再投入なし。

## 4. Stage B：全train取得→X fit→Y fit→両軸凍結

順序は **X h1 train64→Y h1 train64→X fit/guard→Y fit/guard**。各trainのbitとfull価格を保存・検証してから次の取得へ進む。旧OrderedDriverの「両軸trainをfitより先に取得する」順序を再利用する。Y trainを再取得しない。

各軸でsigned.fit_h1を一回呼び、intercept付きOLS、rank2/finite、binary64→Fraction freezeを維持する。既存kernelのrank検査はOLS後であり、失敗をminimum-norm解で救済しない。fit sinkで実際の係数・rank・range・全signed feature countを先に保存する。

外側でactual frozen widthをFractionの6と比較する。D1>6、rank不足、nonfinite、scope不一致は全campaign STOP。clipping・係数丸め・refit・追加trainはなし。X fitが不合格ならY fit/Stage Cを開始しないが、既に取得した両軸trainと全費用は残す。これはguard成功履歴だけを選別して安く見せる設計ではない。

両軸PASS後に全λ・scope/control identity・quota・guard・train provenanceを凍結保存する。評価bit/価格を学習や配分選択へ使わない。inactive4層は常にλ=0。

## 5. Stage C：全5層の共通fresh評価と診断

Xの全5層160→Yの全5層160。各軸でh0…4を既定quota順に取得し、各層内ではtrial ID順を保存する。並列完了順が係数/乱数mapping/固定二分を変えないようにする。X評価中の異常では新しいY評価を始めない。

signed.paired_evaluateへ明示counts=(8,128,8,8,8)を渡し、同じrecordから両平均・全層raw/residual sample variance・paired covariance・全5層推定量分散点指標・平均差のsample-based varianceを取得する。全positive層を保持し、正層を落とさず元p_hで加重する。

補助二分は各層trial IDの前半/後半（h1は各64、他層は各4）。同じ凍結modelで各halfを既存paired_evaluateへ渡す。再学習・追加bit・追加価格はない。full/halfを独立campaign・独立性能再現とは呼ばない。

### 事前診断規則案

軸bのδ_b=V̂_OL,b−V̂_SCV,b、half scaleとH_bは[科学設計 §6](SIGNED_SUPPORT_FRESH_PILOT_SCIENCE_DESIGN.md)に従う。exact診断値で以下のflagを分け、正式な科学GO/NO-GOにしない。

| 条件 | 記録する分類 |
|---|---|
| rank不足/nonfinite/guard超過 | 学習不適格STOP。評価は未取得、改善比較UNKNOWN、全train費を保持 |
| 完走でδ_X≤0かつδ_Y≤0 | 両軸で追加point改善を観測しない。無相関の証明ではない |
| 正のδが一軸のみ | 一軸のpoint改善傾向。全complex signal優位に一般化しない |
| 正のδが両軸 | 両軸のpoint改善傾向。下の不確かさflagも併記 |
| full δ>0だが片half非正、またはδ≤H | `INCONCLUSIVE_SPLIT_DIAGNOSTIC`。経験的変動に対して小さい/方向が不安定な点差 |
| active h1のfull/half raw・residualがall-equal、raw V̂=0、active feature多様性不足 | `INCONCLUSIVE_DEGENERATE_OR_SPARSE`。母分散0や比率0を認定せず、割算不能はUNKNOWN |
| point改善があり、学習込みpoint scoreがbaseline以上 | 分散点改善はあるが、学習費回収を支持するpoint材料なし |
| point scoreも低い | 費用込みpoint傾向として報告。期待費用/精度保証や正式GOではない |

inactive層のall-equalは層別の不確かさflagとして残すが、それだけでactive h1の方向診断を自動失敗させない。いずれも層を除外せず、sample variance=0を母分散0にしない。

一つのattemptで統計的不確かさを十分定量化したとはしない。HはSE/CIではない。全結果に `statistical_significance=NOT_ASSESSED`、`precision=NOT_CERTIFIED`、`expected_cost_guarantee=UNKNOWN` を付す。平均差varianceをvariance削減量のCIに使わない。差が小さい/未判定でも追加shotを自動投入しない。

## 6. 有限call・予算案

正常完了の予定値/上限。途中STOPでは開始/RETURN/EXCEPTION/未完了を区別する。内部呼出しと外側APIを二重課金しない。

| 処理 | train X | train Y | eval X | eval Y | 全campaign上限 |
|---|---:|---:|---:|---:|---:|
| fresh physical measurement/trial | 64 | 64 | 160 | 160 | 448 |
| actual conditional path取得 | 64 | 64 | 160 | 160 | 448 |
| exact probability / Born / bit threshold | 各64 | 各64 | 各160 | 各160 | 各448 |
| SavedEngine.selected_circuit | ≤128 | ≤128 | ≤320 | ≤320 | ≤896 |
| RTE build_sequence / Hadamard wrapper内部build | 各≤128 | 各≤128 | 各≤320 | 各≤320 | 各≤896（selected_circuitの内部であり別trialではない） |
| SavedPrices.price | ≤64 | ≤64 | ≤160 | ≤160 | ≤448 |
| transpile/compile | ≤64 | ≤64 | ≤160 | ≤160 | ≤448、cache missだけ |
| 新53-bit word | ≤4,160 | ≤4,160 | ≤10,400 | ≤10,400 | ≤29,120（path≤28,672、bit≤448） |
| OLS fit | 後段X1 | 後段Y1 | 0 | 0 | 2、refit0 |
| paired_evaluate | 0 | 0 | 3 | 3 | 6、うち4は固定half補助 |

selected_circuitが2/trialになるのは、測定engineと価格側でそれぞれ同じfull列をbuildする既存SOURCEのため。cache hitでも価格側のbuildは先に行う。未取得pathのprice-only build/compileは0。数値gate適用は既存certificateのmax_gate_count=955を各probabilityが検査し、正常全体でapply_exact≤427,840。gate certificate/intervalを再生成しない。

| setup/その他 | 上限案 |
|---|---:|
| 独立campaign / parent / 科学pool | 各1 |
| 保存signed結果読込・typed復元 | 各1 |
| signed Metadata binder | 1（新しいsigned moment有限和は0） |
| saved bundle decode / SavedEngine構築 / proof attach | 各1、parentのみ。fork worker再decode0 |
| PRNG stream book / SeedSequence root | 各1、子24、domain12 |
| worker用PinnedSampler/word検査book | 各≤448。新PRNG sourceではない |
| worker initialize/compiler/cache handle | 各≤W、pool一回。worker自動再生成0 |
| 科学retry / 追加campaign / 追加shot | 0 |
| N1再preparation/CDF、N2 solve/eigh、再認証 | 各0 |
| metadata-only SignedControl moment再計算・群選択 | 各0 |
| B0/B1-O、cal/probe/十分N、別control/round | 各0 |

Wはadmission時に実空きCPU・affinity・memory/pressure・stage内残task数に基づき決めて記録し、恣意的な低い固定worker上限を設けない。1 stageの有用task数は最大160であり、それを越えるworkerを起動する必要はない。これはCPU資源を低く固定する制限ではない。各stageのquota/全call上限を優先し、並列化で448を越えない。

静的上限は実測値と別fieldで保存する。kernel内のbasis登録・gate行列identity検査等を「全て1回」と記述しない。実装時にこの経路と異なる追加scienceが必要なら、SOURCE固定前に判断へ戻し、勝手にcall予算を拡張しない。

## 7. 価格・費用・古典負荷

SavedPrices.priceと既存compiler/cacheを使用する。Qiskit1.3.0、basis=(rz,sx,x,cx)、optimization_level=2、transpiler_seed=17、backend/coupling/layout/routingは保存profileと一致。状態準備のみ除外し、D blocks・selected RTE・phase・ancilla/readout/measurementを含むfull interrogationを毎shot課金する。

各軸train RZ、共通eval RZ、SCV参照=train＋eval、B1-OL参照=eval、全campaign physical=sum(train＋eval)を分離する。cache hitも全量子価格を計上し、参照費を足して二重課金しない。同quotaなら実RZの追加分はtrainだけであり、point variance×実現RZの改善を期待費用保証とはしない。

以下は別timer/metricで保存する。nested/inclusive区間と加算可能区間を区別する。

- 保存結果・bundle decode、親path生成、OLS/fit、full/half paired集計、parent elapsed wall。
- worker probability inclusive wall、測定側/価格側selected circuit build、exact apply/Born arithmetic、gate matrix検査、compile、cache/lock/lookup。分離不能な項はUNKNOWN。
- worker wall区間和、実workerのuser/system CPU（取得できれば）、終了code/signal、peak RSSと取得可能なPSS、実際のcgroup/memory/pressure。RSSを共有page込みで無条件加算しない。
- manager/outer elapsedとsetup/finalizer overhead。worker区間和をCPUやparent wallと呼ばない。

既存 `price_seconds−compile_seconds` は価格側build等も含むため、純cache時間と表示しない。旧TrialCostのclassical=0 placeholderを実測0にしない。失敗groupの費用はenvelopeだけで完備とせず、authoritative worker journal/reconcileで全in-flight tailまで集計する。

## 8. STOP・in-flight回収・保存

SOURCE/input/metadata/control/feature/law/stream/word/price identity不整合、rank/nonfinite/signed range/guard失敗、重複trial、call超過、worker/保存異常、非干渉不成立は全campaign STOP。STOP_REQUESTEDを先にdurable保存し、次のstage/task/科学callを受理せず、同じseed・条件でも再実行しない。

既存Runtime/ManagedPoolを再利用し、prepare_trial/submit/worker開始・engine/Born/price/compileの境界でSTOPと予算を検査する。**CALL_STARTとSTOP遷移だけを短い同一job内の共有排他で直列化するadapter**を提案する。長いengine/compile本体はlock外で並列に動かす。これによりSTOP後の新しいCALL_STARTを拒否し、STOP前に受理・記録済みのcallだけをin-flightとして回収する。これは次回人工検査が必要な差分で、今は保証済みとしない。

pendingで未受理/未開始taskはcancel。開始済みは共有STOPを既存gate境界等で確認し、既に取得したbitは価格より先に保存する。STOP後に新しい価格処理を開始して欠測を埋めない。進行中compile等は許可済みin-flightとして回収し、price未RETURNはUNKNOWN。worker/parentが異常消失した場合は既存のown UID・PID/start time/session検査を通す回収方式だけを使い、共有processへsignalを送らない。再起動・再測定はしない。

部分結果は `BIT_AND_PRICE_DURABLE` / `BIT_DURABLE_PRICE_MISSING` / `ATTEMPT_NO_DURABLE_BIT` / planned未開始等に分類し、未取得bit/価格/時間を0としない。全体COMPLETEでも最後にSTOPを保存する。途中STOPを安い比較結果やguard成功率の推定と読み替えない。

保存候補：SOURCE/input/profile、SAVED_CONTROL_FROZEN、STREAMS、TRIAL/WORD/worker journal、RECEIVED各stage、X/Y FIT・凍結係数/guard、X/Y full/half paired結果、費用/古典timer、call上限/実績、ATTEMPTS/未完了、REPORT/STOP、原本保全と公開照合。全て新namespaceでatomic/fsync保存。bit先行、STOP先行、exact mean/係数/質量を維持する。

## 9. Stage D・必要な採択と承認

追加条件を曖昧にしたまま実行しない。ユーザーは第一候補のtrain64/軸、eval quota、448上限、seed20261110/stream規約、D≤6、full/half診断、価格profile/call上限/STOP、saved-control復元と既存manager接続を採択するか判断する。

採択後、SERVER1 Codexが最小adapter/activationと人工検査を準備し、SOURCE/input/env/compiler/managerを固定公開する。実行SOURCE commitはその時の新SHAで、b4512675やこの文書commitを実行SOURCEと偽らない。公開照合・全preflight PASS後に**一campaignだけ実行する別途明示許可**が必要。実装/人工検査と一回科学実行を同じ次回指示で明示許可してもよいが、今回の設計依頼から推定しない。

完了/中断で全結果・既知費とUNKNOWN・STOPを公開し、remote SHA/本文/blob/treeを照合してmandatory STOP。精度保証付きcal/production、追加campaign、別control、詳細科学レビューへ自動進行しない。新しいgrant/承認証明書を要求しない。
