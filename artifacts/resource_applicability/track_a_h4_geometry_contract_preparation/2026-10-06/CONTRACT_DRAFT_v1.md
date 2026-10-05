# Track A H4 geometry contract draft v1

`H4_GEOMETRY_CONTRACT_REQUIRES_DECISION_SCIENCE_NOT_AUTHORIZED`

2026-10-06 JST。利用者handoff commit
`2a80f1d5d5e5734e51d970b2b6822cd2543fd596`に従うlocal契約準備。
baseは公開準備commit `c2ab34fed49bb1fb104d39fe83b36858a2c92c2a`。
正式契約**案**であり、生成・資源の4判断を閉じるまでは契約完成としない。
science_execution_authorized=false、execution_plan_sealed=false、
actual source/snapshot/master seed/authorization/output identityはnull。
契約条件の確定と実行planのsealは別のbarrierである。

## 確認済みのscope

距離はcanonical Angstrom string `0.70/0.80/0.90/1.10/1.40/1.60`の6点。
原子IDをi=0..3の順で固定し、座標はH_i=(0,0,(i-1.5)d) Å。
decimal距離から一度だけbinary64へ変換し、生成後のexact coordinate bytesは後でfreezeする。
H4 linear/STO-3G、requested/actual DF rank12、8 system qubits＋ancilla8、
T=0.8、二次DF-prefix PF、canonical finite-RTE、delta=T/q。
参照はgeometry別DF Hamiltonianの固定sector内の最低固有状態。
exact untruncated chemistry energy、chemical-accuracy QPE、fresh blindの主張はしない。

保存inventoryと同じ218 template/点を維持する。B0=20、B1=4、B2=145、B3=49、
うちrandom194、baseline24。B0 L_D=3/4/5/6/9、B1=12、B2=3/6/9、B3=0、
q=1/2/4/8、regular random r=1/2/4/8/16/32、K=2/4、
既登録r64はB2-rank3-q1-r64-K2とB3-rank0-q8-r64-K2だけ。
selector/r64選抜・adaptive候補変更を再実行しない。
accuracy-ineligibleもsignal/compile監査に保持し、shots/matched-workはnull。
旧geometry-bound candidate fingerprintはコピーしない。

random cellは32 trajectories、二軸は同trajectoryを共有しwrapper keyは別。
1点194×32×2＋24×2=12,464 wrapper、6点74,784 logical records、
signal slotsは6×218=1,308、actual science transpile reservationsも74,784以下。
cacheでlogical sample/weightを削除しない。追加96、救済seed/geometry、GPU、実shotsは0。
外側spawned workersは最大12、BLAS/OMP/MKL/NumExpr/Rayon各1、
Qiskit内部process1、snapshot/solver段も内外二重並列なし。
12は上限であり予約ではない。将来launch直前に負荷・許可CPU・available RAM・pressureを再確認し、
不足ならworkersを下げるかSTOP、他人のjob/priority/affinityを変えない。

## 生成・state・sourceの判断待ち

詳細は[review_decisions_v1.json](review_decisions_v1.json)。

| 判断 | source根拠と推奨案 | 未固定の判断 |
|---|---|---|
| D1 SCF/DF | H4既存sourceはcharge0、multiplicity1、run_pyscfのRHF経路。明示conv_tol=1e-9/max_cycle=50、spin_basis=True/final_rank=12を推奨 | 初期guess、DIIS/積分/MO規約、全収束条件。implicit defaultsを確定済みとしない |
| D2 order/solver/gates | 旧S0/M1は生成順を保持。OpenFermion生成orderはl1由来weight、汎用rank関数はFrobenius重みで再ソートする。旧M1 scopeを保つ生成順継承を推奨 | weight tieとDF/eigenvector縮退の固定規約、sector/solverとthreshold承認 |
| D3 seed | SHA-256 domain separation、軸共有、duplicate時STOP。master seed候補20261006 | scientific master seed自体はnullのまま |
| D4 memory/wall/output | 下記静的見積りと停止上限案 | 正式memory/wall cap、run ID、fixed root、log/output cap |

D2の推奨sectorはNalpha=Nbeta=2、4電子、dimension36。旧sourceのfrozen layoutに36要素があるという
静的根拠であり、snapshotを開いて確認した結果ではない。
solver案はdense Hermitian eigh、最小固有値を選び、gap<=1e-10 HaならSTOP。
縮退から結果依存に別状態を選ばない。状態phaseは最大絶対sector成分、
同値なら最小sector basis indexの成分を実正にする。
norm誤差<=1e-12、参照residual L2<=1e-9 Ha、relative Hermiticity<=1e-12、
imaginary energy<=1e-11 Haをreview用提案とする。rank不足・SCF不収束・gate不通過に
rank padding、別geometry、別seed、別state、閾値変更で救済しない。
生成orderのtie明示は旧sourceへの修正ではなく、将来の別sourceで承認する規約である。

既存Python3.12.3と依存45件のversion/RECORD metadata、Qiskit plugin metadataは公開準備に一致した。
compilerは全options/defaults/pluginsの公開identityを参照し、basis/opt/seedだけの一致とは区別する。
元wheel archive・全binaryの独立同値検査は未実施。旧Python3.11 guardは不変。
新server-native science source/serializerの実装・round-trip意味論gateは後続の別指示で行う。

## 静的memory/output案

complex128 256×256 dense matrixは1 MiB、512×512 wrapper operatorは4 MiB、
36×36 sector matrixは20,736 bytes、8^4 spin-orbital tensorは65,536 bytes、
12×8×8 DF G stackは12,288 bytes。26個の256-square arrayなら26 MiB。
これはshape由来の内訳だけで、SCF/solver/compilerの一時buffer、Python circuit object、process RSSの
完全上界ではない。旧costやsyntheticのAS4 GiBから科学memoryを固定しない。
候補案はworker AS8 GiB×12＋driver8 GiB=104 GiB、host available>=64 GiB、
output cap10 GiB、wall停止72時間。wallは停止予算案であり完了ETAではない。
74,784 wrapper recordを8 KiB/recordと仮置きすると約584 MiB、
1,308×302=395,016 precision rowsを3 KiB/rowと仮置きすると約1.13 GiB。
serializer/object storageは未実装なのでrecord-size仮定も承認前に検査が必要。
観測空きRAMを予約と扱わず、共有cgroup/OS設定を変更する案は採用しない。
正式memory/wall/outputはnull、D4未解決でSTOPする。

## 数値identityとcheckpoint v1

[checkpoint_schema_v1.json](checkpoint_schema_v1.json)にwrapper_keyとnumerical_circuit_fingerprintを追加。
REGISTEREDの未build値はnull、COMPLETEは64桁SHA-256を必須とする。
baselineのseed/indexは**両方null**、random indexは0..31、seedはuint64。
wrapper keyはgeometry/H/DF/state/template/candidate/axis/seed/index/source/compiler/environment/semanticsを結ぶ。
candidate fingerprintは同geometry/H/DF/state/template/source/compiler/environment/semanticsから作る。

key encodingはUTF-8 sorted-key compact JSON。整数はJSON integer、binary64はcanonical
`{"real64_hex":float.hex()}`、complex128は実部/虚部hexの順序pair。
signed zeroは保持、非有限・非canonical hex・symbolic parameterを拒否する。
scalar phaseをmod 2piで丸めたり、角度を丸めて統合しない。
plan fingerprintだけは公開draftと同じsorted compact JSON/finite scalar方式を使い、
circuit/seed/keyのtagged encodingとは区別する。

数値fingerprintはcompiler投入前のordered full circuit declaration：
global phase、全exact角度/complex数、ordered qubits/clbits、ordered instructions/operands、
operation identity、conditions/control state/custom definitions、測定axisとmeasurementを含む。
今回のコードは合成JSON declarationのidentity検査であり、実Qiskit serializerや
物理operator同値性の検証ではない。custom-operation serializationの閉包・round-trip・
signal/cost意味論は将来science source gateで検証する。

trajectory seedはdomain=h4-trajectory-v1とmaster seed、geometry/H/DF/state/template/source/compiler/
environment/semantics/indexのSHA256先頭8 bytesをunsigned big-endianで読む。
axis/epsilonを入れない。step/occurrenceは別domainでparent seed、outer_step、
short_step、occurrence、draw_kindを結ぶ独立stream案。duplicate trajectory seedはSTOPで救済なし。
今回のtest master=7は架空値で、science master seedではない。

record全体のdigestはdomain=h4-completion-record-v1で作り、
**外部completion ledger**に保存する。record自身にdigestを埋めない。
意味論validatorは独立expected identity、numerical registry、外部ledgerを照合する。
JSON schemaだけでこれらの一致を保証したとは呼ばない。全registryを書き換える主体に対する
暗号署名・アクセス制御はこのin-memory契約検査が提供する保証ではない。

再利用は同geometry/candidate/axis/H/DF/state/source/compiler/environment/semantics、
完全な数値回路fingerprint一致のCOMPLETE非cache ownerに限定する。
owner link、外部record digest、compile reservation、metricsを照合し、
各random logical sampleのweight1/32（baseline1）を保持する。
cross geometry/cell/axis、RESERVED owner、owner欠測、自己参照cache chainを拒否する。

将来のatomic protocol：one-run registryをexclusive登録、compile前に予約slotを消費しfsync、
record temp→fsync→atomic rename、外部ledgerをlock下でcommitしdirectoryをfsyncする。
recordとledgerは二fileなので中断窓を消せない。未解決予約・orphan recordはAMBIGUOUSとして
消費済み予約数に残しSTOP、暗黙retry/resume/旧runtime移送をしない。
今回atomic IO、実cache/registry/科学checkpointは実装・作成していない。

## precisionと報告

新6点の結果を取得する将来stageでは、各geometryでfreezeした同じH/DF/stateのsignalとcostだけを結合する。
その保存値から0.005*(0.1/0.005)^(i/300)、i=0..300とexact0.05の302表示点を評価する。
旧geometryのbias/costを新点へコピーしない。各表示点に追加sampling/compileをしない。
alpha_axis=0.025、headroom=epsilon/sqrt(2)-bias_axis>0をstrictに要求、
N_axis=ceil(2B^2/headroom^2 log(2/alpha_axis))。
primaryはN_real E[C_cosine,RZ]+N_imag E[C_sine,RZ]、
secondaryは6 compiled metrics、共通P>=0はprimary RZだけのaffine感度とする。

paired32、ddof1のcovarianceを保持し点±2SEをengineering intervalとして表示する。
formal/familywise CI・厳密winner・method最適性ではない。点tiesは正確に保持、
displayのtemplate-ID sortを科学的tie-breakにしない。欠測/ineligibleはnull、
random variance欠測を0にしない。point Paretoはeligible complete coverageだけで計算する。
epsilon_minはsqrt(2)*max(axis_bias)、等号は不適格。cost switchは表示gridのbracket、
geometry境界は6点上の観測だけで補間/外挿した厳密境界を作らない。
結果依存の新GO/winner/materiality閾値を作らず研究判断はnull。

## freeze barrierと停止

契約判断閉鎖→別指示でnew science source/tests→actual source commit→source-bound計画と
別result-prior authorization→最終review→利用者明示launchの順序を守る。
生成前には新input hashは存在しない。将来の生成stageは承認済みsource/生成規約にだけ結合し、
生成後にH/DF/state/sector/order/coordinate bytesを凍結するまでsignal/costへ入らない。
実行input-bound sealはfreeze後にしか成立しない。生成段とsignal/cost段の認可・seal順を
future source reviewで明示し、null hashのままinput-bound sealedと名乗らない。

旧1.00 Å218候補と旧1.30 Å5構成は別identity layer。新1.00 Å anchor、1.30 Å全候補化、
8点化、追加96、H6/H8/H12、高次PF/strong synthesis/energy/RPE/Track Bは含めない。
future terminalはGEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEWまたはIMPLEMENTATION_GATE_FAILED、
どちらもmandatory STOP、next_stage=false、automatic research decision=false、research_decision=null。
今回は新契約案と合成record検査だけでSTOP。commit/pushも未認可。
