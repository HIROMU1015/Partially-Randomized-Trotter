# Track A AX-0：計算経路・予算・実装可能性

2026-10-09。静的source監査と環境の読取のみ。benchmark、科学runner、library import、GPU simulation、compileは行っていない。[実験仕様](track_a_ax0_benchmark_protocol.md)と[証拠目録](track_a_ax0_evidence_inventory.md)に従う。以下のpilot/実装は将来の別認可対象である。

## 1. 量子資源と古典計算の台帳

| 台帳 | 記録するもの | 記録しない意味 |
|---|---|---|
| quantum resource | one-shot full wrapperの各gate count/depth、期待cost、N_a、B、ΣN_aC_a、state preparation sensitivity | simulator CPU時間やcompile wall timeを量子runtimeとしない |
| classical evaluation | DF/状態前処理、signal/reference、sampling、build/transpile、fitのCPU秒・wall秒・peak RSS・matvec数・cache hit・worker/thread・disk | N_aが大きいから古典側でもN_a回sampling/compileする必要があるわけではない |

主taskはideal native gate accounting。FT physical qubit/runtime、magic state、T/Toffoliは現時点でN/A。cost sample数32/128と必要quantum shot数を別fieldにする。

## 2. 読取できた環境と未確定の割当

AX-0観測：logical CPU32、process affinity32、`/proc/meminfo` MemTotal=65,522,480 kB（約62.5 GiB）、観測時MemAvailable=54,790,848 kB（約52.3 GiB）。filesystemの観測時availableは約486 GiB。これは専有予算や今後の空き保証ではない。

`nvidia-smi`は存在するがqueryはexit9でdriverと通信できず、GPU型・VRAM・利用可能性は確認できない。cgroup v2のmemory.max/current、cpu.maxは今回のpathから確認できない。無制限・GPU使用可能と推定しない。環境変更・driver修復・GPU allocationは今回のscope外。

| 未知量U07 | 確定方法 | 確定段階／blocking範囲 |
|---|---|---|
| 使用してよいcore、同時job数、BLAS thread | user/queueの割当とprocess affinity、cgroupまたはjob設定を照合 | AX-1 saved解析の前に小さい上限、AX-2 science前に正式上限。32 workersを自動採用不可 |
| assigned RAM、reserve、worker RSS cap | job/container limitと共有利用を確認。AX-2でpeak RSSを測る | worker並列数・H6/H8 science不可。host MemTotalを予算にしない |
| GPU model/VRAM/driver、占有時間 | 読取可能なnode inventory/queue情報を確認。driver問題は別判断 | GPU経路は未採用。CPU経路の正当性監査は進められる |
| total wall/CPU hours、per-task timeout | AX-1/2/3/4別の上限を実行判断で指定 | 上限なしのrun不可。過去実行時間から予約量を断定しない |
| disk quota/temporary circuit size | assigned quotaと保存policyを確認、pilotでbytes/RSSを観測 | checkpoint/failure artifactの上限を確定してからmain run |
| source/library/compiler identity | 実行環境のversionとsource hashを記録 | sourceをimportしてbenchmarkしない今回とは別段階。旧server metadataを現環境versionと呼ばない |

旧server準備にある12 worker/BLAS1は別H4 scopeの設定であり、この計画の承認予算ではない。

## 3. 次元とdenseメモリ

H_n、STO-3G、2n spin orbitals、n electronsの設計上の次元。spin-balanced sectorはNα=Nβ=n/2を保つときに限る。表はcomplex128の単一arrayの算術であり、測定benchmarkではない。

| 系 | full dimension | number-sector dimension | balanced-spin dimension | full vector | full dense matrix | spin-sector dense matrix |
|---|---:|---:|---:|---:|---:|
| H4 | 256 | 70 | 36 | 4 KiB | 1 MiB | 約20 KiB |
| H6 | 4,096 | 924 | 400 | 64 KiB | 256 MiB | 約2.44 MiB |
| H8 | 65,536 | 12,870 | 4,900 | 1 MiB | **64 GiB** | 約366 MiB |

number-sector dense matrixはH6約13.0 MiB、H8約2.47 GiB。matrix-freeでもstate vectors、excitation tables、workspace、temporary buffers、solver Krylov vectorsを要する。vectorサイズだけでpeak RSSを見積もらない。Hadamard ancillaをfull statevectorに含めればvectorは2倍、dense matrixは4倍。q反復はHilbert dimensionを増やさないが、演算回数・回路長を増やす。

M1はfull many-body dense block、spectral exponent、norm検査を作る。H6ではmatrix一枚が256 MiBであり、L個のblock/eigensystemを持つと複数GiB、temporary/worker複製も増える。**H8 full-space dense matrixは禁止**。spin sectorのdense行列もfragmentごとの複製は避ける。H4 denseを小規模correctness oracleとして残す。

## 4. 既存機能と接続が必要な部分

| 経路 | 既存sourceと能力 | 新Track A runnerへの不足 |
|---|---|---|
| H4 signal | `pr2_matched_accuracy_m1_execution.py::_dense_block_operators/_random_signal_record` | 小規模参照として再利用。8-qubit guardを緩めるだけのH6/H8対応は不可 |
| DF/sector | `df_hamiltonian.py::PhysicalSector/df_linear_operator/solve_df_ground_state`：number/spin sector、Numba/chunk/fused/streaming、workspace cap | generic target/state adapter、tail identity policy、実primitiveのsector保存性とdiagnosticsを接続 |
| symbolic tail | `df_rte_tail.py`：normalized symbolic I/Z/ZZ events、identity extraction、small dense adapter | finite polynomialをLinearOperatorへ接続。symbolic準備にmany-body matrixは不要 |
| exact reference/tail | `pf_c_system_size_validation.py`：`expm_multiply`、PF行列なしのstate-action、sector solve | 旧exact-tail経路とfinite平均を区別。旧identity-in-tailと現行extracted phaseを揃える |
| repeated PF | `df_partial_s2.py`、`df_partial_s2_repeated.py`：q反復、phase、boundary optimization、controlled builder | 平均作用・reference経路に同じordering・scalar・one-bodyを渡す。CPU gate-actionのsector適合を確認 |
| GPU | `df_gpu_statevector.py`：statevector/batch/template/phase helper | sector LinearOperator/finite polynomialのGPU実装が完成しているわけではない。helper import自体にもCUDA初期化等がありAX-0では行わない |
| sampling/compile | `df_partial_s2_repeated_cost.py`、`df_rpe_hadamard_compiled_cost.py`：canonical event sampler、seed hierarchy、full wrapper、MC metadata | generic-size候補・checkpoint・追加batch規則・failure budgetの接続。新event列を保存する場合はschemaに明記 |
| proxy | `df_partial_randomized_pf.py`、`rpe_hadamard_compiled_cost_proxy.py`とcluster/stratified/boundary系 | analytically availableな量とlocal compiled/calibrated入力を分離し、H6/H8 test oracleを排除 |
| strong baseline | 高次PF formula、P-D負時間/control/inner-H_D検証、deterministic費用機能 | 新scopeのfull wrapper・state-actionまでの統合は必要。既存機能がないと誤認しない |

### sector保存性U05

H自身のN/Nα/Nβ保存と、circuit primitiveや多項式の中間作用がそれらを保存することは別である。fermionic Givensはnumberを保つが、一般のorbital diagonalizationがspinを混ぜると中間作用はbalanced-spin sectorを出る。net fragmentがspinを保つだけではprimitiveごとのprojectionを正当化できない。

AX-2でsourceのspin ordering/rotation/blockを確認し、H4でfull作用とsector作用を照合する。証明・検査できるprimitiveはspin sector、numberのみならnumber sector、必要な中間作用はfull **vector**へliftする。保存しない中間状態をsectorへ黙って切り捨てない。既存PF state-actionはground/tailをsectorで扱い、Qiskit half-actionでfull vectorを経由するので、このfallbackは新概念ではない。

## 5. finite平均・phase・referenceの数値契約U06

M1の平均はH̄_R=H_R/λ_Rに対するdegree K+1のTaylor polynomial Pであり、τ=λ_R T/R、1 occurrenceのcorrected作用P^r、全q occurrenceのnormalization B=b_K(τ)^Rである。H6/H8ではpolynomialをstateへ反復作用し、many-body polynomial matrixやdense eigensystemを作らない。Horner/再帰は将来の実装選択であり、source意味論を変えない。

`DFHamiltonian.select_blocks`はfull one-body/constantを残す。tailとしてそのまま使うとone-bodyを二重に含む。既存`pf_c_system_size_validation._random_tail_hamiltonian`のzero one-body/constant構成を参照し、現行Track Aのextracted identityを別途整合させる。B0 discard、B2/B3 tail、targetのscalar phaseを同じpolicy tableで検証する。

correctedとrawは独立に意味を持つrecordとし、B z_raw=z_corrを検査する。global phaseを無視した一致ではなく、controlled relative phase、constant、identity、event RTE phaseを含むcomplex一致を確認する。log Bを保存し、overflow/underflow・巨大中間normをfailure/unsupportedとして記録する。安定なcorrected直接作用があってもraw復元が数値的に不能なら、そのfieldの欠測を表示する。

参照はnormalizedψのexact DF signal actionを主とする。verified eigenstateの場合、Rayleigh energy Eとresidual ρから`|z(T)−exp(-iET)|≤Tρ`を数値予算に含めてenergy phaseを利用できる。小さいρだけでground-state性は証明しない。一般状態、近接準位、長時間診断では`expm_multiply`等のstate-action参照を使い、toleranceのtighteningと別小規模参照でu_aを評価する。

新ε_min=.001に対し主数値目標はu_a≤.01 ε_min。これは必要なheadroomであって厳密保証の代用ではない。AX-2でH4 dense対state-action、corrected/raw、τ→0・empty tail、scalar/identity、正負時間（高次baseline）、reverse/forward ordering、state norm、複素signalとfull wrapperを小さい同一条件で検証する。mean作用は非unitaryなので中間norm=1を要求しない。数値目標に届かなければanchorを緩めて別scopeにするかSTOPし、.0001へ進まない。

## 6. 演算・compile予算とpilot項目

naive finite polynomial作用は概ねR(K+1)回のtail matvecと2q回のdeterministic half actionを要する。matvecはDF block数L、orbital数、sector dimension、chunk/workspaceで変わる。reference `expm_multiply`/eigensolveの反復数は別に記録する。ここからCPU秒を推測せず、AX-2のbounded profileで測る。

回路長はq、R、Taylor order、basis/control、boundary optimizationに依存する。古典compileのメモリはstatevectorサイズとは異なり、gate DAGとtranspiler workspaceが支配しうる。主wrapper予算はrandom cell数M_R、deterministic cell数M_D、cost samples nについて、cacheなし上限
`W_base=2 n M_R + 2 M_D`。追加batch、strong baseline、correctness、diagnostic、confirmationのwrappersを別に加える。

旧H4 M1-B1では194 random×32×2＋16 deterministic×2=12,448 wrapper、実compile12,128/reuse320、6 workers・BLAS1、約21,021秒wallという保存記録がある。これは旧H4の古典評価費用であり、H6/H8の見積もりや量子実機時間ではない。q/Rが大きい候補へのlinear extrapolationは禁止する。

AX-2 profileはまずH4接続correctness、次に登録した小さいH6/H8技術cellをsingle worker/BLAS1で実施する計画とする。主science結果を見る前に、短/長deterministic、典型/最大登録event-length random、二axis wrapper、finite平均、referenceのwall/CPU/RSS/gate長/matvec/diskを測る。pilotで見たbias/Cはdevelopmentと表示するか、凍結モデル評価へ流入しない隔離を記録する。

並列worker上限は `min(割当worker cap, affinity/割当CPU, floor((assigned RAM−reserve)/worker RSS upper estimate))`。p95 RSSだけではtail-memory安全上限にならないので最大観測値・headroom・per-task RSS capも使う。CPU compileとCPU solver/matvecを同時に全coreへ張り付けない。GPUは確認できた場合にのみstatevector gate-action accelerationの候補とし、CPU poly/solverとtransfer費用も計測する。

| 段階 | 計算の範囲 | 起動前に固定する上限 |
|---|---|---|
| AX-0 | 文献、source、schema/header/hash、環境読取、5文書 | science/sampling/build/compile/GPU/test/fit=0 |
| AX-1 | 明示allowlistの保存値解析・小さいfitのみ | CPU/wall/RAM/output上限。science/sampling/build/compile=0 |
| AX-2 | correctness＋少数bounded profile | task list、worker/BLAS、task timeout/RSS、wrapper/matvec/disk/時間上限 |
| AX-3 | 登録H6 direct集合＋許容追加batch | lattice/direct quota、32→128条件、strong baseline、confirmation枠、total CPU/wall/RAM |
| AX-4 | 凍結H8比較。新モデル探索なし | H8 quota、seed、same uncertainty rule、total予算。overflow時の閉じ方 |

未知上限を便宜的な数値で埋めない。AX-1の上限はレビュー時、science上限はAX-2 profileと割当を照合して確定する。cost predictionで安く見える候補だけを残すのではなく、実験仕様のmodel-independent selectionを予算に収める。

## 7. checkpoint、failure、GO/STOP

新outputは旧結果/registry/runtime/cacheと別directory。resume keyはsource/environment/model、Hamiltonian/state/sector、candidate/T/δ/q/R/K、identity/coefficient tolerance、numerical algorithm/tolerance、seed/axis、compiler/control/scope、sample batchを含むfingerprintとする。適合keyの**完了**taskだけを再利用し、atomic write・input/output hashを記録する。angle-invarianceや別geometryのcache再利用は検証なしに行わない。

failureにはtask、stage、exception/timeout/OOM、部分wrapper数、CPU/wall/RSS、matvec、seed、completed sample数を残す。failed taskを黙って平均から落とさない。不完全batchは不完全と表示し、同一keyでの再実行かfailureのまま閉じる。欠測をzero costにしない。

予算消費は成功・失敗・追加batchを合計する。事前quota順に実行し、登録total上限やtask capを超える前に停止する。途中で得たPR優位を根拠に延長せず、negative結果を避けるため候補を追加しない。新たな延長判断は別protocol/versionとする。

AX-2→AX-3のGOは、sector/phase/target/mean/wrapper一致、数値headroom、保存・failure audit、direct集合とstrong baselineのscope、割当予算が成立すること。AX-3→AX-4は、H8で識別する未解決の問い、凍結予測、予算が成立すること。PR非優位、モデル一致、強いbaseline勝利はSTOP理由ではない。実装不整合、数値判定不能、予算不足、情報漏洩は対象縮小または限定結論での終了理由となる。

本書の計画を作成した時点で、AX-1のfitもAX-2のpilotも開始しない。独立研究レビューを受けた後、段階別の実行可否を決める。
