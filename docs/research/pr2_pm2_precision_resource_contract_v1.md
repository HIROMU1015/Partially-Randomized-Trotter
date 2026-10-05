# Track A PM-2 精度と測定込み資源境界の事後解析契約

2026-10-05 JST。利用者のpost-PM-1方針reviewと「その方針で進めて」を受け、
研究主張と保存値解析の契約を固定する。今回は契約、schema、入力identity、候補台帳、停止条件の準備だけ。
**PM-2解析実行は未認可、未実施。** 新しい科学計算も認可しない。

準備statusは `PM2_PRECISION_CONTRACT_FROZEN_ANALYSIS_NOT_AUTHORIZED`。
これはlocalで内容を固定した状態であり、source commitやimmutable CIに固定済みという意味ではない。
原review `pr2_post_pm1_research_redesign_20261005.md` はworkspace rootにある利用者資料で、移動・編集しない。

## 研究の主張と完成条件

主RQは、固定DF Hamiltonianの残差をdiscard、deterministic保持、canonical finite-RTEでrandom補完する
どの登録構成が、同じcoherent-signal精度で低い測定込み資源を持つか、その境界をbias、normalization、
必要shot、軸別full-wrapper費用でどこまで説明できるか、である。
「部分ランダム化が勝つこと」や新しいalgorithm、selector、sampling分布の開発を達成条件にしない。

PM-0の帰属訂正を維持する。旧S2からの方式判断変化をqだけへ帰属せず、旧selectorのprimary regretを
frontier損失と区別する。PM-1のB0 rank4/5はaccuracy適格だが、適格性だけから低資源とは言えない。
discard/PFの純bias分解は欠測のまま、q依存を誤差相殺で説明したことにはしない。

完成条件は、同じ候補domain内の精度依存、必要shotと回路費用の寄与、共通状態準備費用による点推定境界、
移送の限定範囲を再現可能に説明し、未登録条件と未解決機構を明記すること。
非自明なcrossoverの存在や特定methodの勝利は要求しない。既知trade-offの限定再現でも、その範囲で閉じる。
PM-2後の研究価値は人のreviewで判断し、runnerは研究判断・次段認可を出さない。

## 保存証拠と比較集合

基準evidence commitは `194cc604b90c56a0e7e949b91b064a4bcfc846da`。
明示した4 JSONだけを読み、bytes/SHA-256とそのcommit blobの同一性を要求する。
入力を一件でも変更・欠損した場合は代替データや再計算で救済しない。

| 入力 | SHA-256 | 使用範囲 |
|---|---|---|
| M1-A result | `1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086` | 全210候補の保存axis bias、normalization、基準shots |
| M1-B1 compile map | `71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4` | 全210候補の軸別6指標meanと保存paired cost samples |
| PM-1 result | `9305857873602d6bc4f45fbc78c4903911d083156620df01e9b23f00e7fdf05b` | 新B0 rank4/5の全8候補 |
| M2 result | `f41a92beb57e59cddc8c063b061c40acd4da50cb76ac0698efc2bce004937931` | 元の固定5構成だけの別集合 |

正確なpathとfile identityは[入力台帳](../../artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/input_identity_v1.json)、
候補fingerprintとsignal/costの独立したrow indexは[候補台帳](../../artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/candidate_inventory_v1.json)に固定する。
JSON内のNPZ/runtime/cache pathは辿らない。

developmentはH4 linear 1.00 Å、STO-3G、DF rank12、8 system qubits、固定状態、T=0.8、
二次DF-prefix PF、Qiskit1.3.0 opt1、basis rz/sx/x/cx、seed17、topologyなし、状態準備を除く測定付きwrapper。
M1のL_D=0/3/6/9/12、q=1/2/4/8、delta=T/q=0.8/0.4/0.2/0.1、登録済みr/Kだけに、
PM-1のL_D=4/5、同じq/delta、r=K=0を追加した**全218候補**である。
元ε=0.05で不適格だった4件も保持する。基準適格数は214で、候補の追加・除外・置換をしない。

M2はH4 linear 1.30 Å、同じbasis/rank/T、B2 rank3/q1/r4/K2、B2 rank3/q1/r8/K2、
B0 rank6/q1、B1 rank12/q1、B3 rank0/q8/r32/K4の5件だけ。
PM-1 rank5をheld-outへ移したとはせず、development全218件とのmethod optimum比較をしない。
両geometryで比べる場合は元の共通5構成へ限定し、M2の正式transfer statusを再判定しない。
H4 1.30 Åは使用済みであり、fresh blind evidenceではない。

## 精度表示とshot式

解析ラベルは常に **POSTHOC_SAVED_VALUES_ONLY**。
解析前に設定を固定しても、元の科学データ取得前の事前登録や独立確認試験にはならない。

表示範囲はcomplex-signal ε=0.005〜0.1。元の0.05に対して10倍厳しい〜2倍緩い要求を調べるtask由来の範囲で、
方式順位を見て選ばない。301個の対数等間隔点を
`0.005*(0.1/0.005)**(i/300), i=0..300` と定義し、両端を正確に置き、ε=0.05を明示追加してsort/deduplicateする。
最大302点。解析結果による範囲変更やadaptive点追加はしない。
各candidateの `epsilon_min=sqrt(2)*max(axis_bias)` は解析的な適格境界として別保存し、
この有限表示点だけから連続ε域の厳密採用境界を主張しない。

α_real=α_imag=0.025、軸対応はreal→cosine、imag→sineに固定する。
各candidateで `ε>epsilon_min` かつ全軸 `s_a=ε/sqrt(2)-b_a>0` のときだけaccuracy適格。
境界の等号は不適格。該当軸のshotと全matched workをnullにし、CSVではMISSINGを使う。
適格なら `N_a=ceil(2*B^2/s_a^2*log(2/0.025))`、`N_total=N_real+N_imag`。
これは現行対称軸配分とcorrected Hoeffding十分shot式の下での適格性であり、
一般の推定法の最少shot数や、methodとしての精度達成不可能性を判定しない。
αは一つのsignal taskの失敗確率配分であり、218候補×ε表示点のfamilywise winner保証ではない。

ε=0.05で元のaxis eligibility、整数shot、既存6指標matched workを再現することをfuture解析gateにする。
整数とeligibilityは完全一致、work照合はrel=1e-12 / abs=1e-6のbookkeeping tolerance。
これを達成できなければ停止し、thresholdや式を結果後に変更しない。
準備ではこの再評価をせず、保存fieldとidentityのcoverageだけを照合する。

## 費用と不確かさ

各metricについて `G(ε,0)=N_real*C_cosine+N_imag*C_sine` とする。
**Cは保存された軸別compile平均**であり、元εのshot-weighted C_eff=G/Nを固定流用しない。
primaryはcompiled rz_count、secondary point Paretoはrz_count/rz_depth/cx_count/cx_depth/total_depth/circuit_size。
Paretoは全成分<=かつ一成分<のstrict dominanceで、同値点のtiesを保持する。新しい10% GO閾値を導入しない。

random candidateは元の32 paired trajectoryだけを使い、Re/Im covarianceを保持する。
各metricのsample variance/covarianceをs_cc/s_ss/s_cs、n=32とし、分母n-1で保存paired標本から計算する。
`SE(G)=sqrt((N_real^2*s_cc+N_imag^2*s_ss+2*N_real*N_imag*s_cs)/n)`。
点±2SEはengineering intervalでありformal CI、winner認定、multiple-comparison保証ではない。
deterministic費用にはsampling varianceがないが、欠損random varianceを0へ置換しない。
区間重なりはwinner未確定と明記し、追加trajectoryで自動救済しない。

Pは全candidate共通の非負RZ-equivalent状態準備費用/shotという仮想感度。
primaryだけを `G_RZ(ε,P)=G_RZ(ε,0)+N_total(ε)*P` とし、各固定εで直線交点から全P>=0のlower envelopeを得る。
P-grid、実際の状態準備compile、candidate別準備法は行わない。Pによるsampling SEの増加はない。
交点tieと無限端を保持し、JSONの無限上端はnull。P感度を他5metricの準備費用へ無条件変換しない。
現εで新8discardが既存B2よりNとG双方で大きい部分は代数で扱い、追加P-gridを作らない。

## 成果物と停止

future成果物はsummary JSON、全candidate×固定εのledger CSV、適格境界CSV、P envelope CSV、
method別点最小tiesを全て残した寄与分解CSV、claim audit JSON、report、bytes/SHA manifest。
fieldは[contract settings](../../artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/contract_settings_v1.json)に固定する。
不適格、欠測、未登録、winner未確定を0やmethod不可能へ読み替えない。
主図を作る場合もこれらの保存解析値から作り、図の見栄えでε/P域や候補を変更しない。

future解析はCPU1 process、BLAS1、最大302×(218+5)=67,346 candidate-ε records。
NPZ/NPY/pickle、旧runtime/cache/registryのresolve/stat/hash/load、signal、trajectory、回路build/compile、
ground-state solve、量子shot、GPU、全repository testsは0。旧M1/M2/PM-0/PM-1 result/source/status/authorization/manifestは不変。

future terminalは `PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW` または `IMPLEMENTATION_GATE_FAILED` のみ。
どちらもmandatory STOP、next_stage_authorized=false、research_decision=null。
人のreviewで限定resource studyとして完成させるか、追加一点の情報価値があるかを判断する。
rank4/5 exact truncated-H signal、strong synthesis/higher-order PFは別監査・別認可が必要。
PM-3、H5/H6/H12、別geometry/分子、長RPE、追加trajectory、Track B統合を自動認可しない。

## 準備実装と次のbarrier

- [契約と入力inventory module](../../src/trottertracks/resource_applicability/pm2_precision_contract.py) はstdlib-only。
- [準備runner](../../scripts/resource_applicability/run_pr2_pm2_precision_contract.py) はstdoutのみ。解析commandを持たない。
- [contract tests](../../tests/tracks/resource_applicability/test_pm2_precision_contract.py) と
  [guard付き限定test入口](../../scripts/resource_applicability/run_pr2_pm2_preparation_tests.py) は合成・保存JSONだけ。
- [準備artifact](../../artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/) は新しい独立directory。

reserved result schemaはfuture summaryの型と停止規則を予約するもので、全CSV/値/manifestを
検査する完成済みanalysis validatorではない。解析実装時にsource-boundな値・coverage検査を追加する。
準備検査は専用21 local tests、fail/skip0。pre-import guardは診断用でOS-level sandboxではない。
正確なcommand、Python identity、時刻、stdoutとprotected-access counterは準備test auditへ保存する。

次はこの契約/schema/input coverageの確認、その後に保存値解析module/runner/testsを実装してsourceを固定する。
解析の実行は別の利用者指示を必要とし、今回の準備runnerやschemaは実行認可ではない。
commit/pushも今回の指示からは行わない。原reviewと既存Markdown整理・Track Bの変更を混ぜない。
