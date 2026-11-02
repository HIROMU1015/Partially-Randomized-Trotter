# H4 run07 STOP原因監査：8GiB AS制限が強く支持される

最も有力な原因は、大きな回路のcompile/計測処理が各workerの8GiB virtual address-space上限に達し、割当失敗またはそれに続くworker終了を起こしたこと。
直接記録された症状はworker response pipeのEOFであり、それ自体が根本原因とは扱わない。
元のnative stderrとworker終了コードが残っていないため、最後のexception/signalと厳密な失敗関数は断定できない。
source・科学条件/compiler/capsを変更せず、同条件での本体再実行や追加compile campaignは行わない。

## 直接証拠

使用SOURCE1dbdd1be2133a59ab81c7b81a6074af0880f72a3、artifact84cea869cfb03b3312b52e23946256bd7ab62ba1。
SOURCE62 pathsのGit/checkout/hashを再照合して不変を確認。
H4 linear/STO-3G、凍結DF fragments12、8system+ancillaの9qubits、geometry0.70 Angstrom、B0 deterministic L_D3、T0.8。
成功はq1/delta0.8のcosine/sine2wrapperだけ。compiled circuit_size2732010/2732009、signal1/1308候補。
完成recordの外部completion digestとledger chainを照合。cache再利用なし。全map成功・最終科学結論・旧compiler出力完全同一性にはしない。
固定plan baseline identityのhashだけから6予約を照合し、失敗serial2/worker2994886はscience-000003、q2/delta0.4/cosineと特定。
実NPZ/科学array・回路build・新seed・追加transpileは使わない。q2のゲート数は未記録で推定しない。

last worker AS8589639680B=7.999725GiB、設定上限8589934592B=8GiB、残294912B=288KiB。
AS>=7.9GiBの観測は66件。最後の観測からfirst STOPの通知まで0.645299秒で、1秒観測の間の厳密peakや退出時刻は未記録。
RSSは7.648457GiBで8GiB以内。host/effective available471050002432B（438.6995GiB）、PSI全scope0、OOMカウンタ0/0/1で開始時と一致。
STOP後にも同一cgroupのOOMカウンタ0/0/1を確認。host/cgroup OOM killを示す記録はない。
RLIMIT_ASは物理RAM空きとは別に仮想address-spaceの拡張を制限し、上限超過では割当がENOMEMになる。
[Linux getrlimit(2)](https://man7.org/linux/man-pages/man2/getrlimit.2.html)に対応する。
worker数の削減は同時消費を減らすが、一つの回路に設定された8GiB上限を増やさない。

## 大きな回路を生むsource経路

circuits.gaussian_basisは8orbitalのGaussian変換を256x256 dense JW matrixへ展開し、8qubit UnitaryGateに包む固定fallback。
installed Qiskit DefaultUnitarySynthesisの>2qubit経路はqs_decompositionを使い、generic Quantum Shannon decompositionの回路/DAGを生成する。
q1だけで約273万operationとなった実保存結果は、この表現とcompile経路が重いことを示す。
build_evolutionは隣接Gaussian basisの境界共有を既に実装している。各Pauli項ごとに無駄なbasisを入れる古い実装が現在の原因だとは扱わない。
q2はPF stepを増やすためさらに大きな回路を処理するが、failed q2の実ゲート数や正確なメモリ要求は不明。
2件の成功を未完成q2/q4や全random mapへ一般化しない。メモリリークの存在もこの証拠だけでは断定しない。

## 元のerrorが失われた二つの経路

OwnedPoolはstderr=subprocess.DEVNULLでspawnし、worker側redirect_stderrはPython sys.stderrだけを差し替える。
FD2へ直接出るnative runtimeのfatal diagnosticは保持されない。Qiskit accelerate binaryにはRust allocation-error handler関連文字列が存在するが、
文字列の存在は今回そのhandlerが動いた証明ではない。
[Rust handle_alloc_error公式仕様](https://doc.rust-lang.org/alloc/alloc/fn.handle_alloc_error.html)では標準設定でstderrメッセージ後abortする経路がある。
今回のnative signalをSIGABRTと補完しない。

workers.exception_textはmessage取得失敗を扱う一方、traceback.extract_tb/format_listやencodeに伴うMemoryErrorを保護していない。
workerのerror報告側もOSError/ValueErrorしか抑制しないため、二次MemoryErrorで応答を送れず終了し得る。
人工2caseで、通常のMemoryErrorはerror frame1を送るが、formatterへMemoryErrorを注入するとerror frame0となってMemoryErrorが外へ出ることを確認した。
実worker/科学compile/メモリstressを起動せず、worker AS設定・dispatch・IPCをmockした単一processの検査。
AS256MiB/RSS64MiB/wall10秒/output128KiB、実child/affinity/GPU0。wall0.031253秒、peak RSS28,311,552B。
これは保存経路の欠陥を示し、今回のhistorical native退出を再現したものではない。
observed_exit_codeは元first errorでnull。後のdriver exit143をworkerの元終了コードとみなさない。

## 結果保存・次の対処

6actual消費/予約（2COMPLETE、4RESERVED）、signal1、charge4,263,870,068Bを今回の履歴へ固定。
全6owned identitiesの退出を2回ABSENT確認し、raw18files・journal・ledger・record digestを照合。原本/partial結果/one-shotは保持、次attemptへ混合・再利用しない。
利用者承認の次attempt carry0、pressure<1%条件、4worker CPUsubset、17GiB/74805/72h/role8GiB/5秒等は不変。
対処対象はbounded native stderr/終了証拠の確保、割当失敗でも壊れないerror報告、および一回路のメモリ消費。
8GiBを超える予算やdense Gaussian表現/compile-policyの変更は今回採用・認可・実行しない。より大きなcapで全mapが完走する保証はしない。
本計算は停止中。今回の監査は新SOURCE/launch認可を発行せず、軽量資料だけを固定する。
[証拠・結論・独立review入口](../../artifacts/resource_applicability/track_a_h4_run07_stop_cause_audit/2026-10-10/README.md)。

独立原因review: PASS_STOP_CAUSE_AUDIT。根拠と推定の分離、SOURCE不変、停止receiptを合格確認。
これは原因監査資料の合格であり、修正済SOURCE・新launchの技術合格/認可ではない。
