# 文書索引

## 2026-10-11 Track A H6精度一致比較 v1：固定run一回の認可・実行前seal

利用者が公開12183dbの固定契約に従う科学run一回を明示認可。[新grantと実行前seal](research/track_a_h6_matched_accuracy_execution_seal_v1.md)。
linear H6/1.00Å/STO-3G/tol-only rank19/sector400/保存state/T0.8、B0/B1 S2・S4/B2の92候補・4精度を変更しない。
source057ba97・211件/親33件/環境/planを再照合。CPU [0,2,5,6]、費用2 process、phase/total wall cap=null、資源guard維持。
grant/source/入力/sealを先に公開・remote照合後一回launch。実行後は原結果と監査を公開してmandatory STOP。旧pilot STOP/partial証拠を保持。
H8・追加探索・入力再生成は未認可。retry/resumeなし。一般のH6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、next_stage_authorized=falseを維持。
以下は実行前までを含む各段階当時の履歴。



## 2026-10-11 Track A H6精度一致資源比較 v1：準備・未認可STOP

[結果前実行契約](research/track_a_h6_matched_accuracy_execution_contract_v1.md)とsealed manifestを次の入口とする。
linear H6/1.00Å/STO-3G/tol-only rank19/sector400/T0.8、保存state不変。B0共通prefix5/10/15、B1 S2/S4、B2 q/r/Kの92有限候補と4精度。
signal先行、全適格候補の費用探索8 trajectory、B2精度別上位2の独立確認32 trajectory。empirical u/shot-inclusive RZ/paired統計・rare-order限界を保存する設計。
コンパクトstageとsector/prefix cache、2 disposable cost process。phase/total wall cap=null、メモリ・出力・call/instruction guardは有限。
synthetic/injected/dummy検査159件PASS。準備で新H6 signal/sampling/compile0。旧7cell/32wrapper・原STOP・欠測・凍結sourceを保全。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION。新grant未発行、別の明示認可後に一回実行。次の主要GPTレビューはH6 map/確認後、H8設計前。
以下は各段階当時の履歴。



## 2026-10-11 Track A H6 pilot後レビューの採用と精度一致比較の準備

[GPT独立レビュー](research/track_a_h6_technical_pilot_post_review_2026-10-11.md)を利用者の方針として取り込み、
[H6精度一致資源比較の準備契約](research/track_a_h6_matched_accuracy_preparation_contract_v1.md)を追加した。
linear H6/1.00Å/STO-3G/tol-only rank19/sector400/T0.8の同じ採用DF・保存stateを維持する。
旧q1/q2、prefix10/full19/0のcorrectness7/7・wrapper32/36は登録範囲の部分証拠として採用し、原STOP・欠測7件は変更しない。
B3 replica1・残り4 wrapperの取得を主検証設計の一律前提にしない。RQ-R主軸、RQ-P1補助、H6はdevelopment。
次はB2とB1 S2/S4の精度一致比較契約、q/共通partitionを優先した有限候補規則、empirical u/shot/費用統計、
signal先行評価とprimary費用取得の分離、高速化・並列化とwall capなしの新版準備。grid・source・実行sealは未固定。
ZIP hashと7 cellの主要scalar/16 primary wrapper費用の転記一致を静的確認。レビュー内u=0の算術は正式shot予算ではない。
今回の変更は文書・metadataのみ、新科学計算/signal/sampling/回路build/compile0、ローカル未commit。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、旧N/Gnull・u未認定・UNDETERMINEDを保全し、追加launch認可なし。
新計算は別seal/明示認可、実行後公開・監査・mandatory STOP。次の主要GPTレビューはH6 map/確認結果後、H8設計前。
以下は各段階当時の履歴。


## 2026-10-11 Track A：固定H6技術pilot v2結果・GPTレビュー待ち

最新入口は[H6 pilot v2一次結果・監査・欠測・レビュー論点](research/track_a_h6_technical_pilot_result_v2.md)。
linear H6/1Å/STO-3G/rank19、sector400、prefix10/full19/0、T0.8、q1/q2（delta0.8/0.4）。
原status `H6_TECHNICAL_PILOT_STOP`、correctness7/7・wrapper32/36、missing7。
明示認可されたscience worker一回、retry/resume0。全rawと原statusを保全、保存bytes/source/input/environmentを照合。
これは技術的整合性と個別回路費用の証拠。matched accuracy・期待費用・PR winner・最終total-costを認定しない。
旧v1 STOP/凍結source/旧結果/dirty/untracked/Track Bを保持。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u/ground-state未認定・UNDETERMINED、next_stage_authorized=false。
一回認可消費済み、mandatory STOP、GPT独立科学レビューへ戻す。以下は段階当時の履歴。


## 2026-10-10 Track A：H6技術pilot v2一回実行認可・実行前seal

最新入口は[固定pilot一回grant/source/予算](research/track_a_h6_technical_pilot_execution_seal_v2.md)。ユーザー「認可するので作業を進めて」＋annotation 1を直前説明のH6 pilot v2一回へ結合。
準備source 0b04886/manifest/保存snapshotを変更せず、新grantを保存。linear H6/1Å/STO-3G/rank19/sector400/T0.8、7 cell・36 wrapper。
CPU IDs [0,2,5,6]・Numba4/BLAS1、wall上限7200秒/AS8GiB/output512MiB。実行前段階でpilotの科学結果はまだない。
remote preflight後一回起動。原STOP/欠測も保存し、公開/監査/remote照合後mandatory STOP/GPT引渡し。retry/resume/source救済なし。
旧source/freezes/入力/結果/dirty/untracked・Track Bを保全。N/Gnull・u未認定・UNDETERMINED、H6本検証/H8は未認可。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、next_stage_authorized=false。以下は段階当時の履歴。


## 2026-10-10 Track A：H6 pilot v2 coverage接続準備・未認可STOP

最新入口は[登録時間の固定/実runtime接続synthetic/新seal](research/track_a_h6_technical_pilot_preparation_v2.md)。linear H6/1Å/STO-3G/rank19/p10/sector400/T0.8。
旧v1 source/STOP/rawを保全し別versionを追加。登録validation timesを全field込みで照合し、actual bounds/差分をgate前保存する。
156 local synthetic（v2 70/旧pilot37/coverage49）PASS。実runtime actual_boundsをsynthetic metadataで直接接続。real H6数値PASSではない。
7 cell/36 wrapper・245 keys/735 probes・科学plan/政策/seed/caps不変。CPU [0,2,5,6]/Numba4/BLAS1、total7200秒/AS8GiB/output512MiB。
science206/validation3 source・親33ファイル・環境/新outputをseal。実H6 decode/prepare/signal/sampling/build/compile0、新grant未発行。
旧v1 grantは消費済み。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定・UNDETERMINED、next_stage_authorized=false、mandatory STOP。
次は別の明示認可後にv2一回run。科学判断を自動承認しない。以下は段階当時の履歴。


## 2026-10-10 Track A：H6技術pilot v1・coverage schema STOP

最新入口は[H6 pilot原STOP/欠測/監査/静的診断](research/track_a_h6_technical_pilot_stop_v1.md)。linear H6/1Å/STO-3G/rank19/p10/sector400/T0.8。
science worker一回、ACTUAL_PRIMITIVE_COVERAGEでinput-reference段階STOP。correctness0/7・wrapper0/36、参照/probe/sampling/compile全counter0。
actual_boundsのregistered_validation_times_v2追加fieldとfixed coverageのschema差を静的確認。actual bounds/準備receiptは未保存、後から再計算で代用しない。
raw14件・保存bytes監査STOP記録、原source/STOP/旧入力/科学結果・dirty/untrackedを保全。初回infra未起動も保存、startup2/science1/retry0。
一回認可消費済み。新source修正・新科学計算・再実行なし。H6のPR/精度/費用結論はまだない。次の候補は別versionのinterface修正と再準備。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定・UNDETERMINED、next_stage_authorized=false、mandatory STOP。
以下は段階当時の履歴。


## 2026-10-10 Track A：H6技術pilot一回実行認可・実行前seal

最新入口は[固定pilot一回grant/source/予算](research/track_a_h6_technical_pilot_execution_seal_v1.md)。ユーザー「次の作業に進んで」を直前説明のH6 pilot一回へ結合。
準備source a643220/manifest/保存snapshotを変更せず、新grantを保存。linear H6/1Å/STO-3G/rank19/sector400/T0.8、7 cell・36 wrapper。
CPU IDs [0,2,5,6]・Numba4/BLAS1、wall上限7200秒/AS8GiB/output512MiB。実行前段階でpilotの科学結果はまだない。
remote preflight後一回起動。原STOP/欠測も保存し、公開/監査/remote照合後mandatory STOP/GPT引渡し。retry/resume/source救済なし。
旧source/freezes/入力/結果/dirty/untracked・Track Bを保全。N/Gnull・u未認定・UNDETERMINED、H6本検証/H8は未認可。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、next_stage_authorized=false。以下は段階当時の履歴。


## 2026-10-10 Track A：H6技術pilot実行前固定・未認可STOP

最新入口は[保存snapshot接続/7 cell・36 wrapper準備](research/track_a_h6_technical_pilot_preparation_v1.md)。linear H6/1Å/STO-3G、rank19/p10、sector400、T0.8。
新weighted policy・signed generation order・tol-only1e-8/cutoff0を維持。新port/gate/runnerを追加し旧source/結果/STOPを保全。
local synthetic129 passed（新37/旧51/旧41）。登録245 times×3 probes=735回。actual prepared representation/instruction boundsは将来のpre-action gateで検査し今回は未計算。
CPU IDs [0, 2, 5, 6]・Numba4/OMP4/BLAS1、worker1、total7200秒/AS8GiB/output512MiB、compile36/trajectory4/occurrence8の上限をseal。
準備だけでH6 signal/sampling/量子回路build/compile0。N/Gnull・u未認定・UNDETERMINED。別pilot grantなし、旧入力完成grant消費済み。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、next_stage_authorized=false、mandatory STOP。
以下は段階当時の履歴。


## 2026-10-10 Track A：H6保存DF入力完成の並列一回実行完了・mandatory STOP

最新入口は[並列一回結果/source/raw/監査](research/track_a_h6_saved_df_completion_parallel_result_v2.md)。linear H6/1Å/STO-3G、tol-only1e-8/cutoff0、全19 signed fragments/order保持、sector400。
新policyの構造・重み付き予算・独立係数/summary照合を通過、PASS_ENGINEERING。state/snapshot保存・loader roundtrip・保存bytes監査PASS。
CPU IDs [0, 2, 5, 6]・Numba4/OMP4/BLAS1、solver1/matvec41、wall約3.9秒。新SCF/DF/signal/sampling/量子回路compile0、retry/resumeなし。
旧source/入力/結果/STOP/freezes・dirty/untrackedを保全。一回認可消費済み。H6_input_accepted=trueは新工学政策での入力完成だけを表す。
u/ground-state未認定、N/Gnull・UNDETERMINED。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、next_stage_authorized=false、mandatory STOP。H6 pilotは別認可。
以下は段階当時の履歴。

## 2026-10-10 Track A：保存DF入力完成の並列一回実行seal v2

最新入口は[並列source/認可/seal](research/track_a_h6_saved_df_completion_parallel_execution_seal_v2.md)。ユーザーの並列計算開始指示を保存DF受理＋state/snapshot一回へ結合。
旧source/freezes/入力/STOPを保全し別versionを追加。CPU IDs [0, 2, 5, 6]・Numba4/OMP4/BLAS1、worker1。旧CPU2は一論理CPUのID2指定。
local synthetic92 passed（新51/旧41）、toy serial/parallel一致。科学政策・19fragment・tol-only1e-8/cutoff0維持。
total1080秒/AS8GiB/output32MiB/matvec10000、retry/resumeなし。実行前段階でH6入力受理/state結果はまだない。
結果公開・remote照合後mandatory STOP。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定・UNDETERMINED。H6 pilotは別認可。
以下は段階当時の履歴。

## 2026-10-10 Track A：保存DF入力完成source・synthetic・実行前固定完了

最新入口は[新policy入力完成の準備/認可対象](research/track_a_h6_saved_df_completion_preparation_v1.md)。
旧sourceを変更せずweighted gate/独立再構成/read-only importer/policy-bound loader/port/runner/auditorを追加。
19fragment・signed lambda/order・tol-only1e-8/cutoff0維持。追加予算1e-10 Ha、判定余裕1%、工程PASSと厳密certificateを区別。
ローカルsynthetic122 passed（新41/旧38/旧43）。real H6の数値評価・受理/state/sampling/compileは未実行。
新source/parent/environment/CPU2/exclusive outputと60/120/900秒・total1080秒/AS8GiB/output32MiB/matvec10000を固定。
新grantなし。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定・UNDETERMINED、mandatory STOP。
次は新対象への明示認可後に一回入力完成。その後公開・remote照合・STOP。H6 pilotはさらに別認可。
以下は段階当時の履歴。

## 2026-10-10 Track A：GPT独立Hermitizationレビュー取り込み・新policy準備へ

最新入口は[独立レビュー採用・次のCodex準備契約](research/track_a_h6_weighted_hermitization_preparation_contract_v1.md)。
19fragment・lambda/order・tol-only1e-8を保持し、構造＋重み付き変更予算＋独立係数再構成を採用方針とする。
projection追加予算1e-10 Haはレビュー由来の新工程政策。PASS_ENGINEERINGと厳密certificateを区別する。
新policy/gate/raw importer/input-completion/policy-bound loader・synthetic・新source/caps/sealを一作業単位で準備する。
旧loaderにも無重み1e-10 gateがある。旧adapter/loaderを迂回・上書きせず、新versionで接続する。
今回はレビュー原文copyと静的identity/source監査だけ。実装/新検査/real配列数値decode・state/pilot/sampling/compile0。
旧STOP/診断raw/source/freezes・dirty/untracked・Track Bを保全。新grantなし、H6入力受理false。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定・UNDETERMINED。入力完成/pilotは各別認可、mandatory STOP。
以下は段階当時の履歴。

## 2026-10-10 Track A：保存integrals H6 DF診断一回・GPTレビュー待ち

最新入口は[診断結果・raw/監査/source索引](research/track_a_h6_df_diagnostic_result_v1.md)。
別grant/固定source ff24de4で一回実行し、原status H6_DF_DIAGNOSTIC_RECORDED、保存bytes監査PASS。
linear H6/1Å/STO-3G/tol-only1e-8、actual rank19、元Hermitization許容1e-10違反index15–18。
raw25件・要約欠測0。lambda/g非Hermiticity/weighted係数差を別保存。parent wall約2.1秒、再試行なし。
これは新runの診断証拠で、旧未保存rawの復元・H6入力受理・政策PASSではない。
旧source/integrals/STOP/freezesとdirty/untracked・Track Bを保全。DF政策変更・state/pilot/sampling/compileなし。
grantは一回消費済み。N/Gnull・u未認定・UNDETERMINED、H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION。
mandatory STOP。根本原因/政策の科学判断はGPT独立レビューへ戻す。以下は段階当時の履歴。

## 2026-10-10 Track A：H6 DF診断準備・未実行

最新入口は[Hermitization STOP後の診断仕様/source/seal](research/track_a_h6_df_hermitization_diagnostic_preparation_v1.md)。
既存48,794 artifact paths/配列116件を検索。失敗rawは回収できず、旧integrals/STOP/source identityを確認。
保存integralsだけでtol-only1e-8の一回診断を準備。全raw先保存、lambda/非Hermiticity/係数整合性を分離。
新source ff24de4・43 local synthetic tests pass、CPU2/worker1/BLAS1、total480秒/AS8GiB/output32MiBをseal。
今回runner起動・実DF call0、旧source/integrals/STOPとdirty/untracked・Track Bを保全。
新grantなし・実行未認可、H6_DF_DIAGNOSTIC_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION。
診断後も別identity/保存監査/公開/mandatory STOP。DF政策変更・H6 GOはGPT判断へ戻す。
以下は段階当時の履歴。

## 2026-10-10 Track A：H6入力生成・DF検査STOP v1

最新入口は[H6一回実行のSTOP・一次証拠・欠測](research/track_a_ax2b_h6_input_generation_stop_v1.md)。
linear H6/1Å/STO-3G、tol-only1e-8、source67312f3/seal affa3f3、CPU2/worker1/BLAS1。
integrals保存後HERMITIZATION_POLICY:fragment_15でSTOP（parent wall約1.7秒）。DF receipt/state未生成。
integral1完了、DF adapter1試行/0完了、matvec/signal/sampling/build/compile0。actual rank/失敗raw/差は未保存。
原STOP/raw20件・保存bytes監査・静的source監査を公開。旧source/H4/freezes/dirty/untracked・Track Bを保全。
閾値/rank救済・source修正・再実行なし。GPTへ早期差戻し、mandatory STOP。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u/ground-state未認定・UNDETERMINED。
以下は段階当時の履歴。

## 2026-10-10 Track A：H6入力生成一回実行のseal v1

最新入口は[H6入力生成seal・認可・確認先](research/track_a_ax2b_h6_input_generation_execution_seal_v1.md)。
ユーザーの継続指示を入力生成一回に結合。source67312f3・tol-only1e-8・sector400と旧plan/capsを維持。
CPU2/worker1/BLAS1、phase900/300/900秒・total2100秒・AS8GiB・output128MiB。
新sealed manifestと専用grantを固定し、remote照合後に一回起動する。この段階では実入力未生成。
入力生成後は保存監査・公開・mandatory STOP。H6 pilotは別認可、H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION。
N/Gnull・u/ground-state未認定・UNDETERMINED。旧source/freezes/結果・dirty/untracked・Track Bを保全。
以下は段階当時の履歴。

## 2026-10-10 Track A：H6入力生成準備 v1（未実行）

最新入口は[H6入力生成契約・source/合成検証索引](research/track_a_ax2b_h6_input_preparation_v1.md)。
既存tol-only adapter/solver/matrix-free/H6 loaderを再利用し、入力生成専用gate・watchdog・snapshot・stdlib保存監査を追加。
87 local synthetic/metadata tests pass（新38・既存49）、実H6入力/state/signal・sampling/circuit build/compile0。
source67312f3、183 science/3 validation freeze。予算・新outputを固定、CPU null・unsealed・新grantなし。
H6_INPUT_GENERATION_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定、mandatory STOP。
入力生成の別指示後にresource/seal/grantを固定し、一回生成・保存監査後STOP。H6 pilotはさらに別認可。
旧H4 source/freezes/結果・既存dirty/untracked・Track Bを保全。以下は段階当時の履歴。

## 2026-10-10 Track A：H4補完実行 v1・STOP

最新入口は[H4補完結果・source/raw/監査索引](research/track_a_ax2b_h4_supplement_execution_v1.md)。
別grantでEVENT_CONTROL 4群、S4 2 correctness/4 MPを一回ずつ完了。旧6 correctness/12 MP/STOPを保持し、複数runのunionを記録。
各reference36/primitive537、control100（event単位）、sampling/compile0。保存監査PASS、補完欠測0。
source67aa6bb・seal/grant5bbebb4、CPU1/worker1/BLAS1、予算・精度・stage/probe変更なし。
N/Gnull・UNDETERMINED・u未認定、H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
H6入力・pilotは別指示を要する。旧証拠・dirty/untracked・Track Bを保全。以下は段階当時の履歴。

## 2026-10-10 Track A：H4補完準備 v1（未実行）

最新入口は[補完準備・source/予算索引](research/track_a_ax2b_h4_supplement_preparation_v1.md)。
[GPT独立レビュー](research/track_a_ax2b_h4_limited_stop_independent_review_2026-10-10.md)に従い、旧6 correctness/12 MP/STOPを保持。
cell/dps-local MP cache、S4 2 cellとexplicit 4群の独立単位、atomic進捗を新versionで準備。
122 local synthetic/metadata tests pass。分子load/signal/sampling/circuit build/compileは今回0。
source 67aa6bb、180 science/2 validation freeze。対象・予算・新outputを固定、CPU/grant/sealは未確定。
H4_SUPPLEMENT_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
N/Gnull・UNDETERMINED・u未認定。旧source/freezes/結果・既存dirty/未追跡・Track Bを保全。以下は段階当時の履歴。

## 2026-10-10 Track A：H4限定v3一回実行・GPT引渡し

最新入口は[H4 v3実行結果・レビュー索引](research/track_a_ax2b_h4_limited_execution_v2.md)。原status H4_LIMITED_STOP、reason PHASE_WALL_CAP:correctness。
correctness6/8、MP12/16、explicit event0/4。coverage一致・actual全体保存。
source b228f23、seal d2b1511、別認可a699a74を結果前に公開。一回grant消費済み、retry/resumeなし。
原結果/欠測・保存監査を引渡し、科学GO/STOP・u/shot/総費用・H6をCodexは承認しない。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・UNDETERMINED、mandatory STOP。
旧source/freezes/STOP・既存dirty/未追跡・Track Bを保全。以下は各段階当時の履歴。

## 2026-10-10 Track A：coverage serialization準備v3

最新入口は[coverage修正・準備v3](research/track_a_ax2b_h4_coverage_preparation_v3.md)。旧v2/source/freeze/STOPを保持し新versionを追加。
tuple/listだけ正規化し、数値・型・順序・coverage変更を拒否。actual/差分をbounded保存。
49 local metadata/mock tests pass、分子load/prepare/signal/sampling/build/compile0。
同じH4 8 cell/capsの新固定・別認可付き一回実行を今回指示の範囲とする。旧grant/output再利用なし。
この段階は準備で分子PASSではない。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定。
実行後はmandatory STOP。既存dirty/未追跡・Track B・旧証拠を保全。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4限定一回実行・coverage interface STOP

最新入口は[H4限定実行報告](research/track_a_ax2b_h4_limited_execution_v1.md)。固定source/manifestの一回実行はACTUAL_COVERAGE_CHANGEDでSTOP。
correctness0/8、reference/primitive/control/sampling/compile counter0。input/native準備は制御フローから推論。
保存schedule list対runtime tupleの静的interface差を確認。actual bounds全体は未保存。
source/manifest/旧証拠を編集せず、raw STOPと欠測・保存監査を別inventoryへ公開する。
grantは消費済み、retry/resume0。次の修正・新manifest/科学実行は今回未実施。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull・u未認定、mandatory STOP。
既存dirty/未追跡・Track Bを保全。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4限定science manifest固定・seal

最新入口は[H4限定seal報告](research/track_a_ax2b_h4_limited_seal_v1.md)。保存再監査済みboundsと旧8 cell・capsを固定する。
science source/input/env・CPU3/worker1/BLAS1・専用future outputをmetadataとして結合。
35合成tests pass。新科学計算/array decode/native準備/probe/sampling/build/compile0。
execution_plan_sealed=trueは条件固定だけ。science_authorized=false / launch_allowed=false。
H4_LIMITED_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、mandatory STOP。
旧source/freezes/STOP/結果・dirty差分を保持。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P保存read gate v2・再監査

最新入口は[H4-P保存再監査](research/track_a_ax2b_h4_native_receipt_reaudit_v2.md)。新gate/runner・36合成testsを追加。
原16MiB aggregate budget内で4MiB超JSONを読める保存専用経路。凍結v1は変更しない。
元source/STOP/receiptのbytesを保持し、実行時sourceと新audit sourceを別に固定する。
新分子計算/native準備/signal/probe/sampling/wrapper build/compile0。H4-P再実行なし。
science manifestのseal/launchなし。`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P一回実行・親監査STOP

最新入口は[H4-P実行報告](research/track_a_ax2b_h4_native_receipt_execution_v1.md)。保存H4 load1/native準備8とreceipt保存を実施。
親はB3 JSON4,443,419 bytesを4MiB読込gateで拒否しSTOP。総output9,821,513 bytesは16MiB以内。
保存JSONのstdlib補助監査は一致。原STOP・source・結果のbytesを維持する。
signal/probe/sampling/wrapper build/compile0。retry/resume/source修正/science sealなし。
`H4_NATIVE_RECEIPT_STOP` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4-P runner・実行前固定 v1

最新入口は[H4-P準備契約](research/track_a_ax2b_h4_native_receipt_preparation_v1.md)。専用source・runner・48合成testsを追加した。
CPU3・900秒・AS8GiB・output16MiB・load1/prepare8を未来の計画へ指定する。
今回の実分子load/native準備/signal/sampling/wrapper build/compileは0。
H4-P取得planのsealは認可ではなく、H4 science manifestは未sealのまま。
`H4_NATIVE_RECEIPT_NOT_AUTHORIZED` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED`。
`DRAFT_NOT_AUTHORIZATION`、mandatory STOP。以下は各段階当時の履歴。

## 2026-10-10 Track A：H4実行前契約・metadata固定 v3

最新入口は[H4契約・metadata preflight](research/track_a_ax2b_h4_prelaunch_contract_v3.md)。
保存入力/source/旧8 cellを照合し、179 primitive-time組/537 probesを固定した。
専用metadata tests12 passed。新科学計算/array load/sampling/circuit/compile0。
native instruction receipt、CPU実割当、別grantは未固定。manifestは未sealを維持。
`H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
以下は各stage当時の履歴。旧結果・sourceと既存dirty差分を保全する。

## 2026-10-10 Track A：H4/H6 backend接続準備 v2

最新入口は[接続・実行gate・合成検証の報告](research/track_a_ax2b_bound_ports_preparation_v2.md)。
H4独立MP/stage/event port、専用H6 sector/native backendと別grant必須launcherを追加した。
新39＋前回49の88 local synthetic/mock tests pass。分子の正しさ・総u・CI証拠ではない。
source固定のみ。actual input/coverage、CPU/別認可は未seal。新科学計算/sampling/circuit/compile0。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。旧証拠・既存dirty差分を保全。
以下は各stage当時の履歴。

## 2026-10-10 Track A：独立レビュー後の準備

[GPT独立レビュー](research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)を受け、[準備追補](research/track_a_ax2b_post_independent_review_amendment_v1.md)と[H6準備契約 v2](research/track_a_ax2b_h6_pilot_preparation_contract_v2.md)を追加。
H4-N/A/E/Mの限定計画、独立small reference・u-aware会計・tol-only adapter、別H6 controller/caps/watchdogを準備した。
専用49 local synthetic tests pass。分子H4/H6検証の新結果・総u認定ではない。
H6 molecular backend/science launcher、H4全stage検証port、input/CPU/別認可は未完了。
旧source/results/freeze/manifestと既存dirty差分を保全。`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。
以下は各stage当時の履歴。

2026-10-06のTrack Aは[H4正式契約準備指示](research/gpu_server_track_a_h4_geometry_contract_preparation_prompt.md)と
[scope JSON](research/track_a_h4_geometry_contract_preparation_scope_v1.json)。
利用者確認済みの追加6距離・最大12 workers、将来74,784-wrapper上限を契約準備へ反映する。
checkpoint数値回路identity修正とzero-compute検査まで。本計算・science port・新benchmarkは未認可、旧証拠不変。
利用者指示で今回の契約準備資料だけをcommit/pushする。以下は各milestone当時の履歴。

最新のTrack Aは、利用者の新しい意向により原稿作成を保留し、
[H4 geometryと要求精度の全候補map](research/track_a_geometry_precision_extension_proposal_v0.md)を準備する段階。
設計案・予算・未確定条件は[別JSON](research/track_a_geometry_precision_extension_proposal_v0.json)へ分離した。
[GPUサーバー側準備指示](research/gpu_server_track_a_h4_geometry_resource_preparation_prompt.md)は既存server環境を優先し、
環境inventory・synthetic CPU benchmark・契約草案まで。本計算はこのhandoffだけでは起動しない。
距離・実行場所・worker・新science source/authorizationは未固定で、本計算は未実行。
旧原稿・証拠・STOPは保存する。以下は各milestone当時の履歴。

Track Aの現在の入口は[原稿v0.1・Supplement・claim audit・review依頼](manuscripts/README.md)。
固定artifactから4主図と補足図1を生成し、日本語通し原稿を作成した。科学計算0、科学STOPを維持する。
原稿のscope監査は執筆者によるもので、独立投稿可能性reviewは次の段階。以下は各stage当時の履歴。

最新のTrack AはPM-2後reviewを採用した[主張・証拠対応表](research/track_a_post_pm2_claim_evidence_map.md)と
[原稿・主要4図の設計](research/track_a_post_pm2_manuscript_design.md)。
追加計算をせず限定case studyとして原稿化する。現在は設計までで、図生成・通し原稿は未実施。
根拠path・固定commit・証拠階層・先行研究との差分・非claimを整理した。
既存result/status/manifest不変、科学計算STOP、Track B変更0。以下のreview待ち等は各stage当時の履歴である。

最新のTrack Aは[PM-2精度・資源境界結果と照合](pr2_pm2_precision_resource_result_validation.md)。
保存JSONだけの302点・67,346行、元ε再現、paired uncertainty、P envelopeを照合し、研究方針review待ちSTOP。
新しい科学計算0、利用者指示でresult commitへ収録するPOSTHOC local evidence。以下は各milestone当時の履歴として読む。

現在は[PM-2保存値解析実装](research/pr2_pm2_precision_analysis_implementation.md)と
[source固定監査](../artifacts/resource_applicability/pr2_pm2_precision_implementation/2026-10-05/source_freeze_v1.json)の段階。
62 local synthetic tests合格、source `324435d77b6642dbd44e8d1f178420daf62e77ed`、本解析は未実行。
研究契約は不変で、別の明示解析指示後も全statusでmandatory STOPする。

最新のTrack A準備は[PM-2精度と資源境界契約](research/pr2_pm2_precision_resource_contract_v1.md)。
全218 development候補とM2元5構成を別集合で固定し、schema・input identity・成果物・STOP条件まで準備した。
保存JSON/input coverageだけを照合し、精度解析は未実施・未認可。旧結果は変更しない。

Track Aの最新は[PM-1実行結果・照合](pr2_pm1_discard_result_validation.md)。
H4 1.00 Å、STO-3G、DF rank12、T=0.8、B0 rank4/5 × q=1/2/4/8の8件がaccuracy適格、
8 signals/16 wrappersを一回完了。pre/post201 local tests passed。
新B0最小rank5・q1は保存B2 r4のprimary点推定の1.75659倍、旧rank6 discardより9.34%低い。
`PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、mandatory STOP、研究判断null。
結果と監査は利用者指示によるresult commitへ収録するlocal evidence。旧draft/preparation/finalizationの未実行記述は当時の履歴として保持する。

最新のTrack A claim限定は[PM-0 POSTHOC証拠帰属・機構解析](research/pr2_post_m2_evidence_attribution.md)。
元のM1/M2結果と区別し、同一domain、selector指標別regret、N×cost、same-R、欠測の表を辿る。
後続の[PM-1近接discard契約・実装](research/pr2_pm1_nearby_discard_contract_v1.md)は
8 signal＋16 wrapperのfuture上限とsealed planを準備した段階。本計算・authorizationは未実施。
[GPTへのPM-1準備bundleレビュー依頼](research/pr2_pm1_preparation_external_review_request_fd7552e.md)から
PM-0の根拠、固定source、plan、監査へ辿れる。

このディレクトリには、研究方針の正本、実装規約、検証報告、発表資料の案内が共存する。
研究全体を初めて読む場合は、先に[`../PROJECT_MAP.md`](../PROJECT_MAP.md)と
[`research/研究概要・現状.md`](research/研究概要・現状.md)を読む。

PR-2別系列の最新結果は
[`M2 held-out結果照合`](pr2_matched_accuracy_m2_transfer_result_validation.md)。
固定5構成、196 wrappersの一回実行は`TRANSFER_SUPPORTED`、研究方針全面review待ちで停止している。
developmentの根拠は[`M1-B1結果検証`](pr2_matched_accuracy_m1_b1_result_validation.md)。
旧S0 STOPとS2結果を保持した
S2後reviewでは、[`matched-accuracy先行研究gate`](research/pr2_matched_accuracy_prior_art_gate_v1.md)、
[`M1前resource-map契約`](research/pr2_matched_accuracy_resource_contract_v1.md)、
[`M1実装契約`](research/pr2_matched_accuracy_m1_implementation_contract_v1.md)、
[`M1前最終amendment`](research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md)を固定した。続いて
[`M1-A validation`](pr2_matched_accuracy_m1_a_validation.md)でdevelopment 1.00 Åの210候補を評価した。
64 proxy-frontier候補中52件が16-cell cap外に残り、`SELECTION_LIMITED`でcompile 0のまま停止した。
外部review後、[`M1-B1 bounded compile契約`](research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md)で
accuracy適格random 194 cell×32 trajectory×2軸とB0/B1 16 cell×2軸、計12,448 wrapperの有限grid、
cache identity、B1後STOPをzero-compute固定した。
実行前外部reviewの修正要求は、[`M1-B1 execution contract amendment v2`](research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md)で
科学実行sourceをauthorizationより先に固定し、runnerのterminal statusをcompile map完成review待ちまたは
implementation failureだけに限定した。
actual execution source commit `33f436b`、source-bound plan v2、
[`M1-B1 execution authorization v1`](research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md)を固定して
12,448-wrapper mapを完了した。検証後の判断は`CONTINUE_RESOURCE_STUDY`で、その時点ではheld-out未認可だった。
後続M2だけを別契約・source・authorization・最終reviewと利用者指示に従って実行した。追加96、S3は未実行・未認可。

## 研究方針と現在地

- [`research/研究概要・現状.md`](research/研究概要・現状.md)：最新の短い全体要約
- [`research/prevalidation_catalog_evidence_map.md`](research/prevalidation_catalog_evidence_map.md)：事前検証カタログの実施IDと文書・artifact・testの対応
- [`research/README.md`](research/README.md)：研究文書内の索引
- [`research/研究目的・研究課題.md`](research/研究目的・研究課題.md)：目的と研究課題
- [`research/研究方法・解析手順.md`](research/研究方法・解析手順.md)：採用する解析手順
- [`research/数値実験・評価計画.md`](research/数値実験・評価計画.md)：検証と評価の計画
- `research/研究ノート/`：時系列の判断記録。現在の仕様ではない

## 現在の主な検証文書

### PF係数

- [`pf_delta_validation.md`](pf_delta_validation.md)
- [`pf_c_system_size_validation.md`](pf_c_system_size_validation.md)

### finite RTE

- [`rte_conventions.md`](rte_conventions.md)
- [`rte_truncation_budget.md`](rte_truncation_budget.md)
- [`finite_rte_signal_validation.md`](finite_rte_signal_validation.md)
- [`research/finite_rte_phase_amplitude_contract.md`](research/finite_rte_phase_amplitude_contract.md)：FR-0の補正後演算子・平均演算子、位相・信号半径境界、比較契約
- [`research/finite_rte_phase_amplitude_prior_art.md`](research/finite_rte_phase_amplitude_prior_art.md)：finite RTEと近接解析のscoped先行研究監査
- [`research/finite_rte_phase_amplitude_fr1_preregistration.md`](research/finite_rte_phase_amplitude_fr1_preregistration.md)：FR-1の非可換toy入力・gate・停止規則と実行後status
- [`finite_rte_phase_amplitude_validation.md`](finite_rte_phase_amplitude_validation.md)：G0/G1/G3/G4通過、G2不通過、mechanism-only停止結果
- [`fr_revision_fr1a_posthoc.md`](fr_revision_fr1a_posthoc.md)：既存FR-1を正scalarと強いbaselineで再解析し、scalar-only説明となった事後監査
- [`fr_revision_nonuniform.md`](fr_revision_nonuniform.md)：非一様4×4でFR固有の境界改善を確認したが、固定予算の選択差がなくmechanism-onlyで停止したFR-R1b結果
- [`research/fr_revision_scalar_structure_contract.md`](research/fr_revision_scalar_structure_contract.md)：FR-R0の正scalar分離、情報層、強いbaseline、FR-R1事前登録要件
- [`research/fr_revision_fr1a_posthoc_plan.md`](research/fr_revision_fr1a_posthoc_plan.md)：既存FR-1を再分類しない正scalar事後解析計画
- [`research/fr_revision_nonuniform_preregistration.md`](research/fr_revision_nonuniform_preregistration.md)：非一様4×4の固定grid、状態、比較、GO/STOP
- [`research/fr_research_claim_and_manuscript.md`](research/fr_research_claim_and_manuscript.md)：FR-R1b後の中核主張C1/C2、条件付きC3、先行研究監査、証明義務、完成判定
- [`research/pr2_s0_s1_execution_amendment_v3.md`](research/pr2_s0_s1_execution_amendment_v3.md)：S0実行と、S0通過時だけのS1 correctness実行を許可し、S1 summary後のmandatory STOPを固定
- [`research/pr2_codex_validation_policy_d3e1723.md`](research/pr2_codex_validation_policy_d3e1723.md)：Codexが実装・実行してよいS0/S1範囲とS2/S3禁止を定める方針
- [`research/pr2_s0_reproduction_stop_c644925.md`](research/pr2_s0_reproduction_stop_c644925.md)：development hash不一致による`STOP_INPUT_REPRODUCTION_MISMATCH`、S1未実行、証拠hashと再試行条件
- [`research/pr2_s0_external_review_request_c644925.md`](research/pr2_s0_external_review_request_c644925.md)：S0 terminal STOP後に、終了または新しい結果前amendmentの要否をGPTへ確認するレビュー依頼
- [`research/pr2_v4_s2_development_authorization_v5.md`](research/pr2_v4_s2_development_authorization_v5.md)：V4 correctness、development-only S2、S2後mandatory STOPを結果前固定
- [`pr2_v4_s2_development_validation.md`](pr2_v4_s2_development_validation.md)：V4/S2の実行結果、B2/B3 frontier、rank control、方針review判断
- [`pr2_matched_accuracy_m1_a_validation.md`](pr2_matched_accuracy_m1_a_validation.md)：210候補のcompile-free signal/selector、52未選択frontier、`SELECTION_LIMITED`、全compile counter 0を記録するM1-A結果
- [`pr2_matched_accuracy_m1_b1_result_validation.md`](pr2_matched_accuracy_m1_b1_result_validation.md)：12,448 wrapper、全checkpoint/cache再集計、actual B2 rank-3 frontier、旧selector監査、fixed-q=8比較、状態準備感度と`CONTINUE_RESOURCE_STUDY`を記録するM1-B1結果
- [`pr2_matched_accuracy_m2_transfer_result_validation.md`](pr2_matched_accuracy_m2_transfer_result_validation.md)：固定5構成のH4 1.30 Å transfer、196 wrapper照合、`TRANSFER_SUPPORTED`、primary ratio/paired uncertainty、資源・pre/post testsと研究方針reviewへのmandatory STOP
- [`research/pr2_matched_accuracy_prior_art_gate_v1.md`](research/pr2_matched_accuracy_prior_art_gate_v1.md)：M1前のclaim-level先行研究比較と`PROCEED_RESOURCE_STUDY`判定
- [`research/pr2_matched_accuracy_resource_contract_v1.md`](research/pr2_matched_accuracy_resource_contract_v1.md)：matched-accuracy baseline、可変q correctness、compile選抜、held-out前freezeを定める研究契約
- [`research/pr2_matched_accuracy_m1_implementation_contract_v1.md`](research/pr2_matched_accuracy_m1_implementation_contract_v1.md)：M1の候補identity、seed、selector、schema、zero-compute guardと科学計算未承認を固定する実装契約
- [`research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md`](research/pr2_matched_accuracy_m1_preexecution_amendment_v2.md)：2026年の近接研究二件との最終claim照合と、M1-A limited時にcompile job 0で停止するhard barrierを追加する現行amendment
- [`research/pr2_matched_accuracy_m1_execution_authorization_v1.md`](research/pr2_matched_accuracy_m1_execution_authorization_v1.md)：development-only M1-Aの入力、source、最大212 signal、compile 0、held-out access 0を結果前固定する実行承認
- [`research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md`](research/pr2_matched_accuracy_m1_execution_authorization_v1_1.md)：v1のresult未作成停止後、固定KのRTEConfig self-consistencyだけを修正して同じM1-Aを再認可
- [`research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md`](research/pr2_m1_a_selection_limited_external_review_request_3c1831e.md)：commit `3c1831e`のM1-A結果を固定し、compile上限拡張・technical note・停止の三択をGPTへ依頼するレビュー文
- [`research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md`](research/pr2_matched_accuracy_m1_b1_bounded_compile_contract_v1.md)：194 random＋16 baseline cell、12,448 wrapper上限、cache/checkpoint identity、B1後STOPを固定し、科学実行を未承認に保つ契約
- [`research/pr2_m1_b1_preexecution_external_review_request_1228168.md`](research/pr2_m1_b1_preexecution_external_review_request_1228168.md)：source commitとzero-compute planを固定し、result-prior M1-B1 authorization作成前のGPTレビュー項目と回答形式を定める依頼文
- [`research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md`](research/pr2_matched_accuracy_m1_b1_execution_contract_amendment_v2.md)：execution source先行固定と、compile map完成後に研究判断を外部reviewへ戻すterminal status修正
- [`research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md`](research/pr2_matched_accuracy_m1_b1_execution_authorization_v1.md)：source commit `33f436b`、plan v2、12,448 wrapper、6 workers、2 terminal status、held-out禁止を結果前固定する一回限りの実行認可
- [`research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md`](research/pr2_m1_b1_execution_authorization_external_review_request_8fc2400.md)：authorization bundle commit `8fc2400`を固定し、本計算開始前の最終GPT reviewと三択回答形式を指定する依頼文
- [`df_rte_tail_extraction.md`](df_rte_tail_extraction.md)
- [`df_rte_event_circuit_api.md`](df_rte_event_circuit_api.md)

### コンパイル後回路コスト

- [`random_circuit_cost_validation.md`](random_circuit_cost_validation.md)
- [`rte_boundary_cost_validation.md`](rte_boundary_cost_validation.md)
- [`rte_boundary_pair_validation.md`](rte_boundary_pair_validation.md)
- [`hierarchical_cost_validation.md`](hierarchical_cost_validation.md)
- [`rte_connected_cluster_cost_validation.md`](rte_connected_cluster_cost_validation.md)
- [`rte_compiled_cost_validation_summary.md`](rte_compiled_cost_validation_summary.md)
- [`rte_compiled_event_cost.md`](rte_compiled_event_cost.md)
- [`df_partial_s2_compiled_cost.md`](df_partial_s2_compiled_cost.md)
- [`df_partial_s2_repeated_compiled_cost.md`](df_partial_s2_repeated_compiled_cost.md)

### RPEへの接続

- [`rpe_resource_accounting.md`](rpe_resource_accounting.md)
- [`rpe_hadamard_interrogation.md`](rpe_hadamard_interrogation.md)
- [`rpe_round_cost_connection_validation.md`](rpe_round_cost_connection_validation.md)
- [`rpe_hadamard_failure_validation.md`](rpe_hadamard_failure_validation.md)
- [`rpe_hadamard_proxy_resource_validation.md`](rpe_hadamard_proxy_resource_validation.md)
- [`rpe_allocation_sensitivity_validation.md`](rpe_allocation_sensitivity_validation.md)
- [`rpe_four_round_accounting_validation.md`](rpe_four_round_accounting_validation.md)
- [`rpe_four_round_phase_validation.md`](rpe_four_round_phase_validation.md)
- [`rpe_target_round_horizon_validation.md`](rpe_target_round_horizon_validation.md)
- [`rpe_delta_round_schedule_validation.md`](rpe_delta_round_schedule_validation.md)
- [`rpe_delta_compiled_cost_validation.md`](rpe_delta_compiled_cost_validation.md)
- [`research_direction_prevalidation.md`](research_direction_prevalidation.md)
- [`research_direction_ablation.md`](research_direction_ablation.md)
- [`research_direction_pf_sensitivity.md`](research_direction_pf_sensitivity.md)
- [`research_direction_gate_s1.md`](research_direction_gate_s1.md)
- [`research_direction_structure_pilot.md`](research_direction_structure_pilot.md)
- [`research_direction_sequence_policy.md`](research_direction_sequence_policy.md)
- [`research_direction_full_scope.md`](research_direction_full_scope.md)
- [`research_direction_full_scope_extension.md`](research_direction_full_scope_extension.md)
- [`research_direction_decision_cost.md`](research_direction_decision_cost.md)
- [`research_direction_late_round_proxy.md`](research_direction_late_round_proxy.md)
- [`research_direction_compiler_transfer.md`](research_direction_compiler_transfer.md)
- [`research_direction_uncertainty_break_even.md`](research_direction_uncertainty_break_even.md)
- [`research_direction_wp11_synthesis.md`](research_direction_wp11_synthesis.md)
- [`research_direction_full_opt2.md`](research_direction_full_opt2.md)：WP11選択M06-Fの事前固定条件、51/51完全性監査、direct-RZ測定、coherent opt2再最適化、A0 proxy-lineage再照合
- [`research_direction_signal_weight_pilot.md`](research_direction_signal_weight_pilot.md)：P-Bのenergy bias・target weight・q別signal再解析とテーマ選定判断
- [`research_direction_geometry_energy_difference_pilot.md`](research_direction_geometry_energy_difference_pilot.md)：P-Cのgeometry依存signed PF error、未使用geometry/delta、差分bias予測
- [`research/pc_geometry_tracking_breakdown_preregistration.md`](research/pc_geometry_tracking_breakdown_preregistration.md)：P-Cの8 geometry、追跡規則、blind region、7 gate、停止規則を計算前に固定
- [`research_direction_geometry_tracking_breakdown.md`](research_direction_geometry_tracking_breakdown.md)：追跡prefix不変、stretch予測破れ、固定gateによるcurrent H4 P-C停止
- [`research/pd_energy_tail_pareto_preregistration.md`](research/pd_energy_tail_pareto_preregistration.md)：P-Dの固定5公式、development/blind、7 gate、停止規則
- [`research_direction_energy_tail_pareto.md`](research_direction_energy_tail_pareto.md)：energy-onlyとtail-aware選択のblind逆転、P-D条件付き候補、次のsigned-time/internal-H_D gate
- [`research/pd_realization_go_no_go_preregistration.md`](research/pd_realization_go_no_go_preregistration.md)：P-D現実化の負時間finite-RTE、fragment内部`H_D`誤差、fresh `L_D=5`、Go/No-Go停止規則
- [`research_direction_pd_realization.md`](research_direction_pd_realization.md)：D1--D3全通過、P-D正式候補化と研究再設計停止点
- [`research/pd_primary_research_contract.md`](research/pd_primary_research_contract.md)：P-D S0の主RQ、比較契約、Case A--D、強制停止
- [`research/pd_prior_art_and_baselines.md`](research/pd_prior_art_and_baselines.md)：既知absolute-tail-time modelと新規性候補の境界
- [`research/pd_s1_fair_comparison_preregistration.md`](research/pd_s1_fair_comparison_preregistration.md)：固定時間・位相予算、B0/B1a/B1b/B2/B4、K4・境界規則
- [`research_direction_pd_fair_comparison.md`](research_direction_pd_fair_comparison.md)：B1b/B2/B4一致、Case C/D不成立、B1a境界未解消のS1停止結果
- [`research/pd_s1_posthoc_reanalysis_plan.md`](research/pd_s1_posthoc_reanalysis_plan.md)：固定S1 artifactだけを使う事後再解析の入力、5%近傍、解釈規則、停止条件
- [`research_direction_pd_s1_posthoc.md`](research_direction_pd_s1_posthoc.md)：一次Case Bを保存した主baseline再解釈、B1a診断、nested/native内訳
- [`research/r3_prior_art_and_minimal_contract.md`](research/r3_prior_art_and_minimal_contract.md)：広いR3の重複、R3-S0不通過、実行しない条件付き最小検証契約
- [`../pd_s1_review_5c331f0.md`](../pd_s1_review_5c331f0.md)：S1 snapshotに対する外部レビュー。正式方針ではなく事後再解析の入力資料
- [`research_direction_joint_synthesis_pilot.md`](research_direction_joint_synthesis_pilot.md)：P-Aのinterval-aware DF回路列合成、強いbaseline、未使用列holdout
- [`research_direction_theme_selection.md`](research_direction_theme_selection.md)：P-B/P-C/P-A比較と後続停止点を含む選定履歴。現行判断はA/B/Cに確認済み主題なし
- [`research/pa_joint_synthesis_prior_art_audit.md`](research/pa_joint_synthesis_prior_art_audit.md)：P-A v1のscoped prior-art audit、限定novelty statement、v2境界
- [`research/pa_joint_synthesis_blind_validation_preregistration.md`](research/pa_joint_synthesis_blind_validation_preregistration.md)：H5 physical transferとH4 opt2 compiler transferの事前登録、compile前task manifest、固定gate
- [`research_direction_joint_synthesis_blind_validation.md`](research_direction_joint_synthesis_blind_validation.md)：P-A v1のH5 physical transferとH4 opt2 compiler transferの完了結果、固定gate、判断、scope
- [`research/pa_joint_synthesis_v1_formalization.md`](research/pa_joint_synthesis_v1_formalization.md)：P-A v1の有限候補、DP最適性・計算量・同値性条件、一区間退化と次の機構識別
- [`research/pa_joint_synthesis_mechanism_validation_preregistration.md`](research/pa_joint_synthesis_mechanism_validation_preregistration.md)：明示的一区間baseline、forced support変化、order 2 stream、固定gate・停止規則の事前登録
- [`research_direction_joint_synthesis_mechanism_validation.md`](research_direction_joint_synthesis_mechanism_validation.md)：P-A非退化mechanism検証の0 split・0 plan差・0 RZ改善とP-C復帰判断

## 実行・運用

- [`server_parallel_validation_execution.md`](server_parallel_validation_execution.md)：共有CPU/GPUサーバー向けのbounded実行、checkpoint、resume、dry-run
- [`pr2_s2_parallel_execution.md`](pr2_s2_parallel_execution.md)：固定PR-2 S2のcell-level CPU並列化、段階barrier、persistent compile cache、serial同値性test
- [`examples/parallel_validation_h4_q1_manifest.json`](examples/parallel_validation_h4_q1_manifest.json)：H4 q=1のdry-run用manifest例

## 発表資料と参考文献

- [`presentations/README.md`](presentations/README.md)：発表資料・構成案の位置づけ
- [`references/README.md`](references/README.md)：同梱した論文PDFの位置づけ
- [`rte_source_versions.md`](rte_source_versions.md)：RTE一次資料の版管理

## 状態の読み方

個別文書に数値があっても、それだけで現在利用可能とは判断しない。
再現可能性、失効、成果物の有無は[`../VALIDATION_STATUS.md`](../VALIDATION_STATUS.md)と
[`../artifacts/validation_manifest.json`](../artifacts/validation_manifest.json)で確認する。

## M2 usable B2契約修正 v2（2026-10-04）

外部reviewの修正要求を[amendment v2](research/pr2_matched_accuracy_m2_transfer_contract_amendment_v2.md)へ反映した。
Pareto supportとprimary ratioは共にaccuracy-eligibleかつprimary重大underestimateのないB2だけを使う。
v1証拠・固定5構成・seed・196-wrapper上限を維持し、科学実行とheld-out accessは未認可である。
moduleは`src/trotterlib/pr2_matched_accuracy_m2_transfer_contract.py`、runnerは
`scripts/run_pr2_matched_accuracy_m2_transfer_contract.py`、testは
`tests/test_pr2_matched_accuracy_m2_transfer_contract.py`、schema/planは
`artifacts/pr2_matched_accuracy_m2_transfer_contract/2026-10-04/`から辿れる。

## PR-2 M2科学実行コードの入口

[実装資料](research/pr2_matched_accuracy_m2_transfer_execution_implementation.md)に、固定5構成・196-wrapper
上限、usable B2、paired-axis covariance、one-shot停止、source-bound planと別authorizationを記録した。
sourceとsynthetic検証を固定する段階であり、held-out開封・科学実行・次段階は未認可である。

## PR-2 M2最終実行前レビュー

- [実行authorization](research/pr2_matched_accuracy_m2_transfer_execution_authorization_v1.md)：actual source/plan、固定5構成、196 wrappers、最大5 workers、一回限りを固定する。
- [最終review依頼](research/pr2_m2_execution_authorization_external_review_request_90a9f24.md)：最終承認と利用者の実行指示までheld-out未開封・本計算未実行で停止する。
- `artifacts/pr2_matched_accuracy_m2_execution/2026-10-04/authorization_audit_v1.json`：local zero-science gateとtimed tests。科学結果・immutable CIではない。
