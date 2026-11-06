# Track D C6b：独立q8 whole-path探索的paired価格検証 — 実行前PLAN

2026-11-07 JST。基点C6b-0 `c93f3c2373b158f8eff3800a62fe1a94cce5c3b0`、専用branch `codex/track-d-c6b-q8-exploratory-20261107`。利用者のC6b指示により、q8 fresh whole-path32本・raw correctness・PASS pathだけ最大4 wrapperずつ（全128）・postcheck・paired統計・凍結C6aモデルの探索的診断を一回実施する。新grant・承認証明は作らない。SOURCE/本PLAN/人工結果/seed表/凍結モデルをcommitし、公開tree/bytes照合後samplingする。

## 固定条件・先行STOP

linear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q8。元H厳密ground仮定・状態準備費除外、ground vector/ground solve/DF fitなし。元science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`、input packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。signed atom・frame順・sector・scalar/identity・normalization/attenuation・測定規約を保持し、weightをunitaryへ掛けない。

C6a結果 `e9d4e2f602d5f9e014507ca7e391719bab017b88`、実行SOURCE `6b6667026ec8b670f610e41141d195883a9eb513`、各q2/q4の16 calibrationと既存STOPを保全する。C6aのresult/seed表でholdout未生成・予約domain一致を確認する。C5 `9a6920b8355643701ab7436a4fd3b34862e9e3bd`、C3完了 `cc554cf6daea08df6c20ea8e1e92405259ce49b5`、Stage A等の旧SOURCE/input/result/grant/ledger/marker/STOPは変更しない。今回の明示許可は新C6b実行だけで、旧runのretry/resumeではない。

## Samplingと事前固定

cohort `track_d_c6b_q8_exploratory_paired_price_v1`、q8_sample00..31を昇順、各occurrence0..7を昇順。各step32 events、計256 canonical occurrence/8192 events。method/axisはseedに含めず同pathを共有する。

予約domain `track_d_c6a_q2_q4_paired_price_pilot_v1:independent_holdout:q8:unused` を今回初めて使用する。文字列中unusedは既存予約identityとして変更しない。masterは2026110606、完全master tupleは `(2026110606, domain, '9a6920b8355643701ab7436a4fd3b34862e9e3bd')`。既存 `trajectory_seed(master_tuple, 'proposed', endpoint, 3, 'C6B_PAIRED_PRICE_ONLY', sample, occurrence)`。SHA256(repr(tuple))先頭8 byte/little endian、NumPy1.26.4/PCG64。全seed/keyと順序を[seed表](../../../artifacts/track_d_c6b_q8/2026-11-07/technical/seed_plan.json)に固定し、C6a96 seedとの非重複を検査する。分離はideal IIDの数学的保証ではない。

既存 `configure_rte → sample_occurrence → iter_sample_rte_events` を呼ぶ。全32本のevent列を一度だけ生成・ローカル保存し、SOURCE/seed/ID/ordered event digest/file hashを固定してからCAR・回路・価格を評価する。旧cohort巡回・C3既知fixture・短path反復なし。価格による標本選択/除外/差替えなし。未使用性・binding不一致ならSTOP。

## モデル・family・価格幅・診断の固定

[凍結モデル](../../../artifacts/track_d_c6b_q8/2026-11-07/technical/frozen_model.json)にC6a保存affine係数、48係数共分散、校正JSON identity、q8予測・価格幅・family・zを保存する。再fitはしない。一次familyはRZ/CX count、RZ/CX depth、total depth、circuit sizeの各X/Y、計12量。両方式の絶対価格・qubitsも保存・記述するが絶対価格精度PASSは対象外。

各量のS_gはC6a保存strong symmetric q8予測規模、tau_g=1+.05 S_g。paired差を分母にしない。古典alpha=.05、二側normal z=Phi^-1(1−.05/(2×12))、固定n32、SDはddof1。いずれも探索的・名目的で、有効な95%同時coverageとは呼ばない。観測後にfamily/幅/標本数を変更しない。

一次の対象は**この保存モデルの固定予測値**とq8 paired母平均の隔たり。観測差e=mean(Reference−symmetric)−frozen prediction、SE8=s8/sqrt32を記録し、`abs(e)+z SE8 <= tau`を満たす量をEXPLORATORY_WITHIN_WIDTH、満たさない量をEXPLORATORY_OUTSIDE_WIDTHとする。後者はこの名目的余裕規則の未達であり、真の誤差がtau外という確認的棄却ではない。全12量が幅内なら全体も探索的幅内とする。

別欄でpopulation校正線の診断 `abs(e)+z sqrt(SE8^2+saved vcal)` を示す。vcalはC6a保存提案のpaired校正分散をそのまま使う。これは一次へ校正分散を重複加算しないための区別で、第二の確認的family・保証ではない。方式別モデル予測も保存するが精度合否へ追加しない。

前提はnormal近似・sample SDと擬似独立設計。有限bit law/PRNG joint-law、tail/support、coverage、affine bias/domainは未資格。全体の統計資格はSTATISTICALLY_UNQUALIFIED。価格幅内でも確認的モデルPASS・長q資格・全law母平均保証・資源優劣の確認的familyを宣言しない。価格のalphaをRPEのfailureと混ぜない。

## 再利用・最小差分・人工回帰

C6aのFreshRequest/seed/statisticsはq2/q4かつn16固定なので、そのガードと旧ファイルは保持し、q8/n32専用module/runner/testを追加する。短いrequest/event ingestionと統計のdomain選択だけが新規で、同じ機能の新sampler/compiler/simulatorは作らない。CAR context/description helperはC6a、head/atom cacheと独立CAR targetは既存B2、prepare_steps/StrongBuilder/Reference gadgetはC3/C5、wrapper/価格検査はC5、raw harnessはStage A、P0/postcheck/state-actionと最終pairedχはC2/C3を呼ぶ。

人工・回帰はseed/binding、q8/cut4の2-mode符号付き合成path、AB/actual adjoint、同じ組立の測定operator、価格資格、paired covariance、frozen条件付き対population診断、dispatch CPU余り/memoryを確認する。分子sampling/compile/Aerは人工テストで呼ばない。人工PASSを分子性能の証拠としない。

## Whole-path・公平な対照・精度

V=U7…U0=A B、B前4step・A後4step、cutはcomplete step4後。whole Vにgadget一つ、A†は同じAのactual inverse、analytic全pathχを補正。strongはuncontrolled Gaussian、directional中心、head relative fusion、tail同basis共有、7 step境界融合、scalar/identity位相集約を維持。Reference内部6境界融合・global cutの1境界損失、追加8 qubits・16 CSWAP・χ補正を含む。弱いbaseline・結果依存cut・節約値の単純加算なし。

rawは全256入力の独立CAR、AB=V、actual inverse、unitarity、vacuum ray、χ/scalar、strong direct controlled-V、complex coherence、X/Y observable・位相とχ省略negative controlを既存checkerで検証。**各pathのraw PASS後だけ同pathの4 wrapperへ進む**。raw全体barrierをなくし、独立pathとcompile/postcheckをpipeline化する。

P0：Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1.0/qubits_initially_zero=false/topology-free/位相保持。component abs1e−10、observable Frobenius/signal2e−10、rtol0、global phase fitなし。compile後CAR/raw/paired X/Y/χ/leakageを同harness・state-actionで照合。17-qubit全Operatorは作らない。失敗後の閾値緩和・phase fittingなし。RZ/CX count、各depth、total depth、size、peak logical qubitsをfull-wrapper単位で分離保存する。

## 並列化・保存・STOP

実空きCPU/RAMとself cgroup制約を観測し、C3保存q8 Reference/q4 directのprocess peakをmemory見積りにする。direct q8のru_maxrssは前taskの高水位を継承しており、一律の重task上限にしない。CPU slotとtask memoryでdispatch、CPU余りをheavy taskに分配し、完了slotは以後のtaskへ再配分する。共有環境を変えず、実行中taskの再起動・SOURCE hot-editなし。小行列BLASの過剰thread重複を避け、重いstate-actionにthreadを配分する。恣意的な低いworker capは設けない。

出力 `artifacts/track_d_c6b_q8/2026-11-07/run01/`、元event/QPY/NPZ/check/log/cacheと資源観測はlocalのみ。公開はSOURCE、科学scalar checks、全価格/paired差/共分散/探索的診断、seed/event hash索引、STOP。conditions/fusion内event payload・raw archive本体・内部絶対path・CPU/RAM・資格情報は公開しない。rawの外部公開は今回自動承認とせず、必要なら具体的対象をユーザーへ確認する。

SOURCE/input/law/seed/未使用性/binding、AB/phase/actual adjoint/vacuum、raw/compiled X/Y、対照公平性、P0/precision/scope不整合なら依存処理をSTOP。進行中taskの証拠を保存し、欠損/失敗IDを明記する。予定32本が欠けたcomplete-case平均を作らない。失敗後の無断retry/resume、追加sampling/compile、seed/cut/model/family/幅変更なし。

B4-2 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` の条件付き14738は保存結果の参照だけ。B5 `0a8b953aea1be20b110784960763d7f36ebccac8` のfinite-bit/joint-law/全path全round実装誤差の不足は残る。N再算定・価格との乗算、B6介入、Born shots/actual成功率、新ground/DFfit、q2/q4追加校正、q>8、全13-round費用、routing/T合成/noise、詳細科学レビューは禁止。

完了または不成立後、SOURCE/input前後照合・partialを含む結果・STOPを保全し、専用branchへcommit/公開、remote SHA/tree/公開bytes照合後**mandatory STOP**。次担当ユーザー／GPTへ研究判断を戻し、後続科学計算を自動開始しない。
