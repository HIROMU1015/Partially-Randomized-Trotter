# H4 D：discard preparation整合性guardの技術停止と修正

2026-10-19 JST。主担当Codexによる技術確認。AI subagent 0。
状態：人工・契約39件PASS。新しいD契約はDRAFT、科学再実行は未承認。
修正code SOURCE：`8a82b9272ee24896e7d7cbb7400d9b8c12d8f804`。

## 並列化・高速化の確認

元SOURCE95fdfc8cfa34a26ae1b603405f8d2d0967f4e484には、候補間spawn process並列、
worker内prefix preparation cache、cos/sin間のevolution共有、fingerprint再利用がある。
保存実測16候補の再利用と全対象endpointで保存不適格の33候補の除外により、
未取得88候補の全面compile176軸に対し、固定D計画は55候補・110軸である。
除外は保存経験的適格集合内の論理であり、真のbiasの厳密証明ではない。

ユーザーの「次の作業に進んで大丈夫だが、計算の並列化と高速化はされているか」を
提示済みD契約への進行承認として記録した。freshな64物理coreの約61 core相当のidle、
約470 GiB実効空きRAM、既存cgroup制御の観測に基づき55 workerを設定した。
CPU affinity・OS・scheduler・cgroup・共有環境・他jobは変更していない。

今回追加したclassical最適化：親が実行前後に検証する大きなruntime/source/input manifestを、
各spawn workerへ複製しない。workerにはloaderが使うstage/output/native_configだけを渡す。
Pの場合は固定cellsとA adapterも保持する。親の承認・SOURCE・input・runtime照合は削除していない。
同じ停止run specを比較用入力としたpickle protocol4のpayloadは7,510,670 bytesから315 bytes。
これはclassical設定dataのserializing量で、実測速度倍率や量子費用の削減ではない。
将来のrun path長でbytesは変わる。新SOURCEによる実H4 benchmarkは行っていない。

## 失敗runと科学call

旧runは55 worker context・55 task entryまで進み、入力ロードとpreparation処理後、
`ValueError:PREPARATION_BINDING`でTECHNICAL_STOPとなった。
driverは親witnessがnative exit1を回収。wallは13.273866819217801秒。
既知55 worker PIDは読取確認時に残存0。個別worker exit codeの明細は保存されていないため補完しない。

新native evolution/control/wrapper compile/決定論費用会計はいずれも0。
全taskにcalls_ledgerがなく、source上の最初の予約より前のguardで停止したことを照合した。
preparation再構成の正確な回数は未保存で、最大55試行として履歴に残す。
入力ロード・preparationが行われた事実を「実H4処理0」とは報告しない。
科学的な点費用・勝者・PR比較結果は作成されていない。S_STATS/Pも開始していない。

## 原因と修正の根拠

固定provider SOURCE217c971dd537b1cc7fbc13742bc0f6c263ff7f45の
`pr2_matched_accuracy_m1_execution.py`では、`_prepare_discard(ham,p)`は
`truncated=ham.select_blocks(tuple(range(p)))`を作り、そのtruncatedをprepare_df_partial_s2へ渡す。
`DFPartialS2Preparation.hamiltonian_hash`はこのtruncated Hamiltonianのcanonical hashである。
一方、旧新規adapterはそのfieldと元full Hamiltonianのbindingとの一致を要求していた。
これはCodexが追加したguardの意味の取り違えで、元builderや保存inputの破損ではない。
原SOURCEの関連定義とSHAは[固定source証拠](../../artifacts/resource_applicability/track_a_h4_d_binding_fix/2026-10-19/fixed_source_evidence.json)に記録した。

修正は、saved full inputの同一性チェックを維持し、そのfirst-prefix viewをsourceどおりに作り、
builder呼出し前に同じcanonical関数・元WEIGHT_RULEでexpected truncated hashを計算する。
その値と返却preparationを照合し、prefix/fragment順・system数・scalar・空のrandom tailも検査する。
観測されたhashを期待値へコピーしてguardを通したり、保存Hamiltonian/state/scalarを変更したりしない。
Pのfaithful preparationにはこのdiscard修正を適用しない。

新worker初期化payloadの縮小も、同じcandidate task・scientific kernel・compiler・call上限を保持する。
元runの封印済み6 source file・grant・one-shot・STOP・旧byte記録は書き換えず、別worktreeで修正した。

## 検証と承認境界

既存人工29件に加え、新guard・cache・worker payloadの人工10件、計39件がPASSした。
うち2件は固定builderを実際に使う2-mode/3-fragmentの人工DFで、
full/truncated hashの差、正しいguard受入、同prefixの一回準備、元full inputの不変を検査する。
実H4 snapshotの読込とqiskit.transpile入口を禁止した。人工Gauss blockの構築は行うが、
実H4 native evolution/control/compile・raw標本統計・n計算・費用再会計は追加していない。
詳細結果は[technical_validation.json](../../artifacts/resource_applicability/track_a_h4_d_binding_fix/2026-10-19/technical_validation.json)。
作者自身の検証であり、独立科学reviewや外部再現とは呼ばない。
新DRAFTを実際のrunner/witnessの2入口へ渡し、両方とも`DRAFT_UNAUTHORIZED`で拒否した。
新output/lockは未作成、native driverは未起動。固定SOURCEのGit blobと実file、
元のtasks/input/call/runtime契約の不変、57記録と保全copyのSHA一致も確認した。
結果は[readiness_and_mandatory_STOP.json](../../artifacts/resource_applicability/track_a_h4_d_binding_fix/2026-10-19/readiness_and_mandatory_STOP.json)に記録した。

旧契約は「自動retryしない」、one-shotは消費済みなので、新SOURCE・新scopeの実行承認が必要。
新しいD契約はDRAFT_UNAUTHORIZED、grantはINVALIDとして保全し、科学再起動しない。
対象55候補・110 compile・最大220 control・決定論会計1 pass、33保存不適格、元input/compilerは維持する。
旧runのpreparation試行・停止・消費記録は返却/resetしない。
再実行承認後にはfresh資源とruntimeを再照合し、今回停止runと別のoutput/lockを使用する。

成果物：[停止projection](../../artifacts/resource_applicability/track_a_h4_d_binding_fix/2026-10-19/stopped_run_projection.json)、
[修正D契約案](../../artifacts/resource_applicability/track_a_h4_d_binding_fix/2026-10-19/D_corrected_contract_DRAFT.json)、
[classical payload計測](../../artifacts/resource_applicability/track_a_h4_d_binding_fix/2026-10-19/worker_payload_metadata.json)。
private保全manifestは57記録を含む。private path・grant原本・user receipt・PID一覧は公開projection外。

次の担当：ユーザーが修正SOURCEによるDの新規one-shot実行を承認するか判断する。
Codexは承認前に再起動しない。S_STATS/Pは引き続き別承認。修正はlocal commitまで、pushはしない。
