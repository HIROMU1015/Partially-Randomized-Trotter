# Track A PM-2 保存値解析の実装と停止条件

2026-10-05 JST。利用者が承認した「解析module・runner・testsの実装とsource固定」までを扱う。
[固定契約](pr2_pm2_precision_resource_contract_v1.md)は変更しない。
実データのprecision sweep、費用順位、P envelope、ε=0.05での数値再現検査はまだ実行していない。
状態は `PM2_ANALYSIS_SOURCE_FROZEN_AWAITING_USER_LAUNCH`。追加科学計算も未認可である。

## 実装の範囲

[解析module](../../src/trottertracks/resource_applicability/pm2_precision_analysis.py)はstdlib-only。
保存bias、normalization、軸別6 compiled metricsの平均と元のpaired cost標本だけを投影し、
corrected Hoeffding十分shot、matched work、点Pareto、method別点最小の全ties、共通仮想Pの直線envelopeを計算する。
新しいsignal、RNG sampling、circuit、compiler、分子・ground-state計算経路を持たない。
NumPy/SciPy/Qiskit/CuPyや旧science moduleをimportしない。

対象domainはdevelopment全218候補とM2元5構成を分離したまま。
H4 linear 1.00 Å / 1.30 Å、STO-3G、DF rank12、8 qubits、T=0.8、二次DF-prefix PF、
登録L_D/q/r/Kだけで、deltaは0.8/0.4/0.2/0.1。M2でrank4/5を評価したことにはしない。
compile値は元のQiskit1.3.0 opt1、basis rz/sx/x/cx、seed17、topologyなし、状態準備なしの保存値である。

ε=0.005〜0.1、301対数点と正確な0.05の最大302点、軸別α=0.025を固定する。
再評価には軸別平均を使い、元εのshot-weighted C_effを固定流用しない。
適格境界の等号、不適格・欠測はnull/MISSINGで、methodの達成不可能性へ一般化しない。
SEは元32 paired標本のcovarianceを保持し、数値的負分散を避ける等価な中心化paired-square式で計算する。
点±2SEとその重なりを表示するが、formal CI、familywise保証、精密winnerは主張しない。
Pは共通RZ-equivalent費用/shotの仮想感度だけで、実状態準備回路や他5metricの準備費用ではない。

## 入力と実行barrier

[future runner](../../scripts/resource_applicability/run_pr2_pm2_precision_analysis.py)は
完全40文字の`--source-commit`と、別の明示指示を反映する`--execute-saved-analysis`を必要とする。
後者を省略すると、source Git検査、保存入力、output作成より前に拒否する。
フラグは運用barrierであり、利用者指示や外部reviewを独立認証する仕組みとは主張しない。
今回このrunnerを実データへ起動していない。

source gateはmodule・runner・test入口・tests・契約の6 blob同一性、基準evidenceからの系譜、
固定準備manifest SHAと8準備fileのbytes/SHA/blob、settings一致を要求する。
基準evidenceは`194cc604b90c56a0e7e949b91b064a4bcfc846da`。
準備manifest SHAは`da30a1d3b91f44dbe4847a158e86abb1302f0ea14160ca1f77c780e0cabc488e`。
別rootや移行runの認可を与えず、固定sourceを使う既存Track A worktreeから起動する。

起動後も明示4 JSONだけを基準commit blobと照合し、候補inventoryのfingerprintを照合する。
ε=0.05のaxis eligibilityと整数shotsを完全一致、6指標workをrel1e-12/abs1e-6で再現できた場合だけ、
精度走査や順位計算へ進む。今回この実データ数値gateは未検証であり、後の実行で失敗した場合はSTOPする。
再現不一致を閾値緩和や代替入力で救済しない。

fixed outputは`artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/`。
今回作成していない。既存outputがあれば上書きせず拒否し、resume/retry commandを設けない。
CPU1、BLAS各1、最大67,346 candidate-ε行、追加候補・science・量子shot・GPUは0。
Python-level guardはNPZ/NPY/pickle、旧runtime、registryのfile/metadata accessを拒否する診断で、OS sandboxではない。

## 成果物とfailureの扱い

契約の8成果物を生成する。reserved result schemaに加え、ledgerの全domain×ε coverage、重複、
適格・欠測、境界台帳、CSV field set、completeの入力4件・出力集合・基準再現PASSを検査する。
reportはpoint mapとuncertaintyの説明までで、研究四分岐を自動選択しない。

成功は`PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW`、
実装・identity・数値gate failureは`IMPLEMENTATION_GATE_FAILED`。
どちらもmandatory STOP、next_stage_authorized=false、research_decision=null。
起動gate拒否はoutputを作らない。起動gate後のfailureは理由をsummaryとmanifestへ保存し、
partial outputは科学結果として使わない。failureを科学的なnegative resultへ読み替えない。

## 今回の検査とsource固定

[専用tests](../../tests/tracks/resource_applicability/test_pm2_precision_analysis.py)と
[guard付きtest入口](../../scripts/resource_applicability/run_pr2_pm2_implementation_tests.py)は合成値だけを使用した。
全62 local tests passed、fail/skip0。real saved evidence、NPZ/runtime/registryのaccess試行0。
mockによる起動gateとpipelineの検査は、実データ解析や実データpositive numerical gateではない。
証拠種別はlocal synthetic implementation testであり、immutable CI・外部再現ではない。

正確なcommand、Python3.11.0rc1、UTC時刻、stdout、counts、source hashは
[実装監査](../../artifacts/resource_applicability/pr2_pm2_precision_implementation/2026-10-05/)に保存する。
source commitの40文字identityは固定後の監査とhandoffで示す。自己参照hashをこの文書に埋め込まない。
準備artifactのlocal-uncommitted表記は作成時点の履歴として不変に保つ。

今回はsource固定のlocal commitまでで、pushも本解析も行わない。
次は利用者の明示解析指示後に保存値解析を一回行い、結果直後に研究方針reviewへ戻る。
PM-3、別geometry/分子、strong synthesis/higher-order PF、追加trajectory、energy/RPE、Track B統合は認可しない。
