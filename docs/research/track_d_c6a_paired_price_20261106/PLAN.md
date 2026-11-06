# Track D C6a：q2/q4 paired価格校正pilot — 実行前PLAN

2026-11-06 JST。利用者のC6a指示に基づく一回の新規実行。C5公開 `9a6920b8355643701ab7436a4fd3b34862e9e3bd` を基点とする専用branch `codex/track-d-c6a-paired-price-pilot-20261106`。本PLAN・SOURCE・人工test・seed表をcommit/pushし、remote tree/bytesを照合してからsamplingする。新grant／承認証明は作らない。

## 固定科学条件・上限

linear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=1/10。元H厳密ground仮定・状態準備費除外。元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、input packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。scalar/identity、signed atom、normalization/attenuationを維持し、後二者をunitaryへ掛けない。

| q | whole-path標本 | fresh h-step occurrence | 固定cut | 最大P0 wrapper |
|---:|---:|---:|---:|---:|
|2|16|32|complete step1後|64|
|4|16|64|complete step2後|64|

計32 path、96 occurrence、各32 canonical events、最大128 full wrapper。q2_sample00..15、次にq4_sample00..15、各step昇順の結果非依存規則。保存32 cohortの巡回・反復・追加標本・seed/cut変更は行わない。

## sampling接続と実行境界

既存 `configure_rte → sample_occurrence → iter_sample_rte_events`、NumPy1.26.4 PCG64を再利用する。master=2026110606。q別domainは `track_d_c6a_q2_q4_paired_price_pilot_v1:calibration:q2`／`:calibration:q4`。既存 `trajectory_seed((master,domain,C5_base), 'proposed', endpoint, log2(q), 'C6A_PAIRED_PRICE_ONLY', sample, occurrence)` を使う。全96 seedは[固定表](../../../artifacts/track_d_c6a_paired_price/2026-11-06/technical/seed_plan.json)に保存する。method/axisはseedに入れず、同じpathを両方式・X/Yへ渡す。

q8用domain `track_d_c6a_q2_q4_paired_price_pilot_v1:independent_holdout:q8:unused` は予約文字列だけ。seed/event/回路を生成しない。有限bit実装をideal IID証明とせず、ideal lawとの距離・joint stream資格は未評価。q別・sample別・occurrence別の非重複seedは擬似独立の設計であり数学的独立性の証明ではない。

各event列は結果を得る前に一回だけ生成し、32 pathすべてをSOURCE/seed/sample IDとともに保存してhashを固定する。その後にだけ回路へ渡す。event列そのものを新samplingの前に既知とはしない。生成順序・seed・lawは本PLANで先に固定し、event保存を回路実行前の境界とする。保存済みeventを再samplingで置換しない。

## 最小実装・再利用

- C3 `prepare_ordered` のsaved-file読込と、同じprimitiveを使う回路組立を `prepare_steps` へ分離する。既存C3/C5入口・固定cohort検査は残す。fresh入口だけが既存canonical event列・独立reference・ordinary diagnostic rawを渡す。
- C5 `wrapper_for` の同request/method/axis確認とfull-wrapper価格検査を共用する。旧 `price_record` の固定q2 guardは保持し、fresh pilot入口を別に限定する。SOURCE/input/law/seed/event digest/cutをrequestに結ぶ。
- 既存 `car_bilinears/raw_base/raw_frame`、B2 `reference_context/trajectory_reference` で全256 CAR参照を作る。Gaussian回路からtargetを逆生成しない。head指数とsigned atomを一度cacheし、raw workerで再利用する。元H ground vector・ground solve・DF fitを呼ばない。
- ordinary診断は既存 `build_kernel`、強い対照とReferenceはC3/C5の組立・StrongBuilder・Reference gadget・actual inverse、raw checksはStage Aの22項目＋paired X/Y、compile/postcheckはC2/P0とstate-action、最終paired/χはC3を再利用する。
- 新規はfresh request/seed/event接続、32 pathの作業配分と保存、ddof1 paired統計だけ。汎用sampler・Gaussian合成・compiler・state simulatorは新設しない。探索的affine fitも既存generic metric fitterを呼び、legacy DF datasetのidentityを偽装しない。cosine=X/sine=Yを明示する。

## correctness・公平性・compile

人工・回帰testとSOURCE/input/law preflight後、全pathのrawを検証する。ordered独立CAR、AB=V、Aのactual adjoint、unitarity、vacuum ray、analytic χ、scalar/identity各step一回、direct controlled-V、Reference complex coherence、X/Y誘導observable、global phaseを既存精度で照合する。全raw PASS後にだけ各pathの最大4 wrapperをcompileする（各pathのPASS前にはそのcompileをしない）。

strong対照はuncontrolled Gaussian、directional control、head relative fusion、tail同basis共有、q−1の合法なhalf-sweep step境界融合、scalar/identity位相集約。Referenceは一つのwhole-path gadget、A†は同じAのactual inverse、analytic χ、内部q−2境界融合とglobal cutによる1境界融合損失、追加register/CSWAPを含む。head-tail間の新しい融合・節約値の加算は行わない。

P0：Qiskit1.3.0/optimization_level1/seed17/rz,sx,x,cx/approximation_degree1.0/qubits_initially_zero=false/topology-free/位相保持。component abs1e−10、observable Frobenius2e−10、rtol0、global phase fitなし。compile後のCAR/raw/paired X/Y、χ、reference leakageを照合する。失敗後に閾値・phase・fixture・compiler条件を変えない。

RZ/CX count、各depth、total depth、circuit size、peak logical qubitsを個別保存する。RZ→T換算・routing・depth→実機時間換算は行わない。

## pilot統計・探索的モデル

標本単位はwhole path、各q n=16。X/Yを独立2nと数えない。両方式・各axis・各資源の有限標本平均、sample SD(ddof=1)、min/max、paired Reference−symmetric差の平均/SD/min/max/全16差、方式間sample covarianceを保存する。method×axis×7資源の全共分散行列（28列、rank≤15）も保ち、共有pathによる依存を落とさない。peak qubitsは幅として別扱いし、roundへ加算しない。

予定32本が揃わない場合、欠損/失敗IDと完了IDを報告し、complete-caseによる有利な平均は作らない。n16を母平均精度保証とせず、SE/CI・law support・uniform誤差上界は作らない。min/maxは観測範囲のみ。

必要に応じq2/q4の方式別平均へ既存uniform affine fitterを呼ぶ。2点2係数なので残差自由度0で、fit残差0は長q妥当性を示さない。boundary/laneモデルも候補として不足特徴を整理し、未確認の長q価格・確認的PASSは出さない。独立holdoutは混合しない。

C6bの絶対/相対価格幅・古典信頼度・q8独立holdout数・資源別規則についてpilot変動と校正共分散を根拠に提案する。q8分散やモデルbiasが未知なら必要数を無条件に断定せず、条件付き数式・判断に必要な不足を示す。提案は採択/実行許可ではない。

## 運用・保全・STOP

実空きCPU/RAMを既存observerで確認し、独立path/compileをprocess並列化する。並列数はtask数・実空きCPU・既存q4の実測peak memoryで決め、過去のworker capを引き継がない。小行列BLASの過剰thread重複を避け、重いstate-actionに残りのthreadを配分する。共有OS/venv/scheduler/他job/他ユーザーは変更しない。一時的なpressureだけで正常workerをkill/restartしない。自分のジョブは独立session/stdin切離し/local logで起動する。

namespace `artifacts/track_d_c6a_paired_price/2026-11-06/run01/`。未加工ログ・CPU/RAM/runtime・CAR cache・元回路とcheckはprivate `local/`へ保存する。公開科学viewはoperational欄（wall/RSS/thread）を除き、科学check/phase/price値は変えない。公開可能なraw/compiled/参照payloadとevent記録はlossless archive／hash付き索引で保全し、内部絶対path・資格情報・非公開input rawは公開しない。

SOURCE/input/law/seed/phase/correctness/公平性/compile/価格scope不一致で依存科学処理をSTOP。途中成果・失敗IDを保全し、無断retry/resumeなし。旧Stage A〜C5/B4-1/B4-2のSOURCE/input/result/grant/ledger/marker/STOPを変更しない。本新SOURCEの明示範囲は旧STOPの上書きと区別する。

B4-2 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` の条件付き13-round十分shot総数14738は参照だけで、N再算定・費用乗算をしない。B2/B4-2、q8 holdout/q>4分子回路、新endpoint/ground/DFfit、追加標本、Born shots、RPE実成功率/全13round期待価格/総費用、詳細科学レビューは対象外。

成果を専用branchへcommit/pushしremote SHA/tree/file bytes照合後mandatory STOP。C6b・長q移送・追加科学計算・詳細科学レビューへ進まず、次の判断をユーザー／GPTへ返す。
