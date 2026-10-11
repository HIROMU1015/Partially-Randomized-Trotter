# H6共用サーバー Phase A v1 — dependency不足でmandatory STOP

2026-10-12 JST。資料入口は[Phase A結果bundle](../../artifacts/resource_applicability/track_a_h6_server_phase_a_result_v1/2026-10-12/README.md)。
利用者の明示承認に基づき、index447 `B1_p19_q8_4th`だけを一度起動した。
原statusは`H6_SERVER_PHASE_A_STOP`、worker exit1。新完了group0、保存wrapper0、transpile未到達。
成功・失敗後mandatory STOPの契約に従って終了し、retry/resume・Phase B・独立32確認・H8を実行していない。

## 起点と承認

Repository `HIROMU1015/Partially-Randomized-Trotter`、branch `track-a-h6-server-migration-preparation-20261011`。
Science SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12`、migration SOURCE `61cadec5b5b2c8f1138828f43f7780d36319ccc3`、
起点REVIEW `e51f19b649a7feb3f7f84d1e22cf49fc57e0e412`。利用者がpushしたorigin REVIEWを再取得し、一致を確認した。
source/環境profile/manifest/旧reuse/inputとcommitted準備bundleのbytes/SHAを再確認した。

新grant原bytes SHAは`c420691bf857ef2e860b64c6724e41770af16eaabde7be7459879e25c4fa80d0`。
grantとその解決済みSOURCE/profile/task/reuse/resource/output bindingは[authorization bundle](../../artifacts/resource_applicability/track_a_h6_server_phase_a_authorization_v1/2026-10-12/authorization_binding_v1.json)へ保存した。
旧grantとdraftは上書きしていない。manifest digestは`43672daed236ff47033458d7c7de1771ae2607c62a7ca83e6b07031b83207403`。
seed17585593421502146214、replica0、prefix19、q8、四次PF、cosine/sineの2 wrapper対象。

利用者は当初の運用規則確認STOP条件を変更し、「利用規則やschedulerは特にない」「他利用者に著しく迷惑を掛けない範囲で使用可」と説明した。
これをoperational clarificationとして保存し、CPU/RAM使用の承認と資源guardを維持した。
空きRAMを予約済みとして扱わず、scheduler・新cgroup・共有設定は作成/変更していない。

## 実行とSTOP原因

run IDは`h6-server-phase-a-447-launch_server_v1`。起動要求時刻は2026-10-12 01:22:47.983967 JST。
coordinator PID3844746、worker PID3844945。独立したhome socketのtmuxで自身のsessionを起動した。
科学workerは1回、one-shot lock消費済み。既存H4のlauncherやcacheは流用していない。

固定science portの`cost_task`で保存H6 inputを使用し、`_prepare`→Gaussian block→`U_to_qiskit_ops_jw`まで到達した。
専用環境にqiskit-natureがないため`BogoliubovTransform`をimportできず、旧sourceのdense fallbackへ入った。
そのfallbackは8 system modesを上限とし、H6の12 system qubitsに対して次のImportErrorを出した。

```text
qiskit-nature is required for fermionic Gaussian decomposition when num_qubits > 8.
```

原worker terminalと`failure_diagnostics.json`にexception type/message、32 frame上限のtraceback、unwind後memoryを保存した。
evolution build/control validation/full-wrapper/transpileのstageは未到達。compile attempt record0、wrapper0、cost result未生成。
これは環境dependency不足であり、AS32 GiBでのmemory/resource failure、方式不適格、量子資源的劣位の証拠ではない。

repositoryの固定requirementsには`qiskit-nature==0.7.2`がある。
準備時には指定environment gate9 package等の一致を確認したが、12-mode loweringに必要なoptional dependencyを確認し切れていなかった。
小規模人工fixtureは8-mode以下のfallbackを通るため、この不足を検出できなかった。準備の依存関係確認が不足していた。
qiskit-nature/qiskit-algorithms/qiskit-aerのpackage metadataは専用環境でabsentだった。
今回の直接原因はqiskit-natureであり、他packageが実H6経路で必要かは別途依存sourceと照合する必要がある。
科学source・専用環境を今回は変更せず、dependency installや再実行を行っていない。

## 適用された制限と実測

| 項目 | 原記録 |
|---|---|
| worker CPU/thread | [20,21] / 1 |
| coordinator CPU/thread | [22,23] / 1 |
| worker AS soft/hard | 32/32 GiB |
| coordinator AS soft/hard | 2/32 GiB |
| aggregate RSS上限 / job RAM予算 | 40 / 64 GiB |
| worker peak RSS/VmHWM | 378368000 bytes（約0.352 GiB） |
| worker peak VmSize/VmPeak | 812392448 bytes（約0.757 GiB） |
| sampled aggregate peak RSS | 434388992 bytes（約0.405 GiB） |
| worker監視wall | 4.0121200531721115秒（起動前binding検査を含まない） |
| worker exit / Resource STOP | 1 / first resource STOP reason=null |
| transpile時間・compiled metrics | 未取得 |

worker final memoryも上記peakと一致する。coordinatorの個別RSS/VmSize/VmPeak/VmHWMは原monitor schemaに保存されておらず、退出後に補完しない。
coordinator起動から終了までの正確なwall/exit codeも別記録されていない。worker監視wall・worker exitと混同しない。
compiler設定fingerprintは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`を維持したが、actual compiled wrapperはないため実compiled identity/費用を認定しない。
AS32 GiBの継承は実workerでも成功した。q8 full-wrapperが32 GiBで完走できるかは依然UNDETERMINED。

起動時はhost/effective available約465 GiB、host/全可視cgroup PSI0、OOM counter増加なし、filesystem約406.7 GiB/inode約2.20億/quotas disabled。
新tmux scopeを含む実launch cgroupでもfresh gateを通過した。これは予約ではない。
coordinatorがworkerとは独立して観測し、aggregate RSS/AS/headroom/PSI/OOM/output/欠測/stalenessのguardを適用した。
supervisorは未reap identityを保持してown worker groupをretireし、wait/reap・log/journal終了を確認した。
cleanup errorなし、coordinatorとworker双方の/proc entryが消失し、残存0を確認した。

## 既存結果・H4・監査

旧449/592 group・902 wrapper・92 signal・84 B3と224科学source/33 input/5220 rawをpost-STOP binding監査で再照合した。
新受理group0のためmissingは143のまま。index447と他142 groupを区別し、旧449と未取得結果を混合しない。
H4はread-only観測で稼働継続、第一STOP理由なし。H4 worker CPU2/4/5/6、driver16、observer18とH6 CPU20–23は交差しない。
H4/既存venv/他jobを変更・停止していない。これらの観測から他利用者への影響が完全に0と断定するものではない。

旧限定検査は固定証拠として保持し、今回は必要な継承/CPU/thread/grant gateの3検査だけを実施してPASSした。
初回test driverのFSIZE hard16 MiBがfixtureの必要2 GiBを下回ったため1検査が失敗した原記録も保全した。
test driverだけsoft16 MiB/hard2 GiBへ修正した同じ3検査がPASS。production source/制限変更、science retryではない。
新人工compile・stress campaignは行っていない。

raw、原grant bytes、failure diagnostics、ResourceTrace、launch/preflight、cleanup、byte inventory、post-STOP auditを別結果commitへ固定する。
実runtime cache・新NPZ・credential・内部SSH接続情報はstageしない。SOURCEと準備REVIEWは変更しない。

## 次段階へ戻す判断

次は専用環境に必要なdependencyを旧requirements/native loweringと一致させ、profile・bindingを再固定する準備が必要。
共有環境/H4 venvは引き続き変更せず、qiskit-natureの欠落を固定gate/import検査へ加えることを提案する。
再実行は今回の消費済みgrantで行わず、新しい明示的な一回認可が必要。今回自動retryは実行していない。

q8の実測peakが得られていないため、他142 groupの資源を今回の約0.35 GiBから見積もらない。
Phase Bの最大2 workers/各AS32 GiB/合計RSS72 GiB/job RAM96 GiBは引き続き未承認候補。
q8を正常に測定してからPhase B予算を再判断し、全592探索後の独立32確認にも別認可を要する。
科学的winner・PR優位・研究方針を自動確定せず、mandatory STOPで利用者/GPTへ判断を戻す。
