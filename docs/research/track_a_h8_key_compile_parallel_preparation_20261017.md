# H8主要3構成・6軸：準備工程の並列化（2026-10-17）

**実装・人工32検査を完了。現在のrunは旧SOURCEのまま継続。新実装の科学起動・切替は0。**

SOURCE：`7f8d74660b2914a1223d0ac7990dde8d4645032e`。
branch：`track-a-h8-key-compile-parallel-preparation-20261017`。
[監査bundle](../../artifacts/resource_applicability/track_a_h8_key_compile_parallel_preparation/2026-10-17/README.md)。
基準REVIEW `472189d4e624c5196b248adf112c1d3a75d143c6` から、home配下に独立cloneを作成した。既存worktreeの登録・更新は行っていない。

## 実装した並列化

| 工程 | 旧SOURCE | 新SOURCE |
|---|---|---|
| 構成のnative回路構築 | 1構成ずつ | 3構成を並行構築 |
| left/right control | 1構成の2 worker | 各構成2 worker、全体最大6 worker |
| cosine/sine wrapper生成 | 親で順番に生成 | 各軸のworkerで2並列生成 |
| wrapper検証・fingerprint・QPY保存 | 1構成の2 worker | 各構成2 worker、全体最大6 worker |
| compile | 軸ごとの判定を挟む直列 | 同じ固定順・直列、投機compileなし |

[準備scheduler](../../scripts/h8_key_compile_preparation.py)、[control・wrapper](../../scripts/h8_key_compile_controls.py)、[接続adapter](../../scripts/h8_key_compile_adapter.py)、[owned process管理](../../scripts/h8_key_compile_parallel.py)、[契約](../../scripts/h8_key_compile_contract.py)。snapshotと保存preparationはdriverが一度だけ読み込み、forkのcopy-on-writeで渡す。大回路をIPCで転送せず、各workerが所有領域へ保存したQPYの小さなdescriptorを返す。

driverの下に最大3つの構成coordinator、その下に各2つ・計最大6つの準備workerを置く。したがって準備childは最大9 process。coordinatorは自身のCPUペアの先頭coreをworkerと共有し、内部worker実行中は待機する。計算配置案はdriver1＋準備6の7物理coreで、CPU番号は未指定。これは今回の独立タスク数に対応する配置であり、他の研究に対する恒常的な資源上限ではない。

全3構成の準備・control・native終了・回収・結果SHAが合格した場合だけcompileに進む。準備中の構成が不合格なら、同時実行中の自分の準備processを回収し、compileは0のままSTOPする。準備は前倒しするため、後続構成が不合格になるまでに他構成の準備呼出しを消費し得る。

## 会計・終了・binding

親driverが各case forkの前にglobal reservationを保存する。case-bound ticketはSOURCE/config/grant、case ID、CPUペア、予約fileのbytes SHAを結合する。子coordinatorは割り当て済み呼出しだけを消費し、使用記録を別fileに残す。欠測・二重消費・compile等の権限外呼出しはSTOPし、予約を返却しない。forkしたdictの更新を親の会計として扱わない。

call capsは旧実装と同じ：snapshot1、preparation decode1、prefix view2、native evolution3、measured wrapper6、probe copy6、control12、QPY dump/load各6、compile/model check各6。新signal/reference/scalar計算、SCF/DF/state、sampling、fit、費用再会計、追加compileは0。

own PID/start/PPID/UID、pidfd、PDEATHSIG、TERM→必要時KILL→wait4を維持する。構成coordinatorのTERM猶予10秒は内部workerのcleanup用で、計算wall上限ではない。launcherのdriver cleanup猶予30秒は維持。階層化したworkerの最初の失敗原因をnative receiptへ伝え、子・孫processを全回収してからSTOPする。

plan/config/grantはV2、新scopeは `H8_THREE_SELECTED_CANDIDATES_SIX_NATIVE_AXES_PARALLEL_PREPARATION_V2`。旧grantを流用できない。plan SHAは `31ffca4d9c8f8bedaf185593e87f795fb1ed56128bbe8f378f8258cc6cae77ca`。SOURCE closureは22→24 files、新schedulerと人工testを追加した。[旧/new blob・SHA](../../artifacts/resource_applicability/track_a_h8_key_compile_parallel_preparation/2026-10-17/source_closure_diff_v1.json) に変更理由を残す。

## 科学条件・資源・限界

Linear H8/隣接1Å/STO-3G、16 spin modes、sector4900、保存DF rank28、T=.8、signed generation order、symmetric-directional control、非merge S4 Yoshidaを維持。候補はB0I p15/q4 S4、B0I p11/q1 S2、B0 p15/q4 S2の各cosine/sine。元scalar/B0I自身のprefixの保存scalarを一度適用し、referenceは回転しない。

固定B1モデル・予測、各軸 `abs(predicted-measured)/measured < 1/50`、control誤差1e-8未満、compiler options・profileを変更していない。provider SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、compiler `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。入力binding fileは旧SOURCEとbyte一致、保存11入力のstreaming SHAを照合した。NPZの科学arrayは読んでいない。既存venv/provider/runtimeは変更していない。

新しいCPU/RAM/AS/RSS/CPUtime/wall上限は設けず、OSの制御をそのまま継承する。入場見積りは旧64GiB/caseを3倍した192GiB＋headroom16GiB＝208GiB。これは未実測の保守的なestimateであり、実消費・予約・RSS上限・最大値保証ではない。出力は従来どおりQPY6件等で初期2GiB/1024 inodeを見積もり、監視・stdout/stderrの継続成長を別途観測する。新方式で起動する際は7物理coreの非重複、実効空きメモリ、filesystem/block/inode/quotaをfreshに確認する。

重複fingerprint走査の省略やcanonical形式変更は行っていない。同じ固定native APIを別processで呼ぶ変更である。実H8での新方式end-to-end、実QPY/canonical byteの旧/new比較、短縮率、peak RSSは未測定。最大6 workerで全体が6倍速になるとは主張しない。各軸のモデル不一致2%以上は証拠を保存して次compile前にSTOPし、refit・追加候補・費用再会計へ進まない。

## 人工検証・現run保全

[人工tests](../../tests/test_h8_key_compile_preparation.py) と既存限定回帰を合わせ **32 test methods PASS、2.195秒、skip0**。3 coordinator・6内側workerのbarrier同時到達、wrapper生成が2つのchild内で重なること、人工spool bytes/fingerprint、予約/使用会計、改変・欠測、native非zero/未回収、各6位置のmodel不合格、入れ子失敗時の全回収と最初の原因を確認した。実transpile・実H8 build/control・sampling・fit・再会計は0。主担当自身の確認であり、別担当の独立reviewではない。subagent0。

利用者の「RUNはそのまま」を守り、実行中の旧SOURCE `352581103b7dbd4fa58e28af7a6ca65570d36e8e`、config、grant、one-shot lockの計25 immutable filesは作業前後でSHA一致。旧runのSOURCE/grant/output/ledger/marker・CPU設定を変更せず、停止・再起動しなかった。21:34:28 JSTの読取snapshotでは同じdriverが稼働中、control予約4・compile予約0、terminalなし。これは旧runの途中観測であり、新方式の性能結果でも完了証明でもない。

## 固定状態と次の担当

source/testsを上記SOURCEへ固定し、軽量plan/binding/無効grant・test結果・本文・索引を別REVIEW commitへ保存する。local commitのみでpushは実施しない。NPZ、実QPY、runtime/checkpoint/cache、private config、有効grant、認証情報はstageしない。

新方式は `approved=false`、`runtime_authorization=false`、`execution_sealed=false`、`allowed_cpus=[]`、`absolute_launch_command=null`。有効grant発行0・新規科学launch0。旧runの既存承認と稼働はこの未起動状態とは別である。

次は **Codex** が現在の旧runの結果・終了状態を確認する。新方式への切替は今回行わない。**ユーザー** が後で切替を指示した場合に、Codexが保存結果の再利用可否と新SOURCEへのbinding、fresh資源・未使用領域を確認する。費用再会計は別の実行範囲として扱う。
