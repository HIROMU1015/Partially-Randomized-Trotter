# H6 full m5 CT01：未実施sine一軸の一回実行を起動

2026-10-19。**利用者の「次の作業に進んで大丈夫」を受け、準備済みの未実施sine一軸を一回起動した。終了・最終raw検査・単価の受理はまだ未監査。** 処理は独立supervisorとworkerで継続し、完了または不一致時にmandatory STOPする。費用再会計と次Phaseは今回の範囲に含めない。公開pushなし。

- SOURCE：`885bbd7199c9ada26704b7db75b99767734e92e7`
- branch：`track-a-h6-m5-ct01-sine-run01-20261019`
- 承認plan SHA256：`32849ff8539758dff5ed1502168cff1e537f62eae10de929cfae1d8ebf1d0928`
- 承認config SHA256：`a8542db4372bf626d1d893aa2fc0ea1f07d7a2a0bc4f74f4afb602e36f6afbf9`
- 起動日時：2026-10-19 22:47:16 JST（13:47:16 UTC）
- 起動：supervisor PID4185973、launcher PID4186040、worker PID4186191。

[準備報告・人工検証](track_a_h6_m5_ct01_sine_readiness_20261019.md)、[起動summary](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01_start/2026-10-19/summary.json)、[時刻付きsnapshot](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01_start/2026-10-19/startup_snapshot.json)。snapshotはその時刻の観測であり、正常終了の証明ではない。

## 固定対象と今回の変更範囲

H6直鎖・隣接1 Å・STO-3G・保存tol-only1e-8/observed DF19、full p19/q2、`4th(m5_best)` / `PF_M5_BEST_20D64C2_BINARY64_V1`、T=.8、P=0、13qubit/ancilla12、symmetric-directional control、scalar一回、full Hadamard sine。

元Qiskit1.3.0・rz/sx/x/cx・preset opt1・seed17・NumPy1.26.4・Python3.12.3、provider SOURCEc31ef424…を維持する。代数的1qubit縮減・再合成無効のcompilerは用いない。新SOURCEのcode bytesは前回準備から変更0。

保存cosineのraw差1.504043707445665e-9と831662 RZを、CT01の1e-7未満基準による別受入として再利用する。元1e-9未満基準でのFAIL・旧STOP・quarantine・grant/ledger/markerは保持した。元のnative/occupation/Hadamard等5チェックは再実行しない。

新規はsine一軸のcompile一回とcompiled回路の状態作用一回のみ。保存cosine wrapperに末尾Sdgを一つ挿入し、保存未compileベクトルにancillaのH Sdg Hを作用させて期待ベクトルを得る。新しいnative evolution・signal/oracle・SCF/DF/state/reference・samplingは生成しない。全未compile回路の再作用とcosineの重複compileを省く。

sine期待ベクトルのHadamard値と保存imagは1e-9未満、compile前後raw vector差はfiniteかつ1e-7未満を要求する。phase alignment・位相補正・再normalizationは行わない。失敗時の自動retry・追加compile・再fitは0、主表採用AD03は保留する。

## 承認結合・fresh gate・並列化

旧readiness commit81990c6c…のDRAFT・INVALID grant・prepare-only receiptを保持し、新しい実行承認receipt、承認plan、承認configと有効grantを別ディレクトリへ保存した。科学scope・SOURCE・入力・runtime・資源政策・pipeline identityが旧readinessと一致することを検査した。新承認結合と保存先だけを分離している。公開資料はprojectionで、有効grantや実行config自体を含まない。

起動前9項目、SOURCE/input/runtime全2025file、入力13 role、CT01 protocol Git object、保存cosine受入、未使用run/marker/supervisorを照合した。旧SOURCE・結果・grant・ledger・marker・STOPとreadiness成果等2816fileはSHA不変。既存58受理軸と旧STOPを保全し、他worktreeを凍結していない。[gate](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01_start/2026-10-19/prelaunch_gate.json)、[保全](../../artifacts/resource_applicability/track_a_h6_m5_ct01_sine_run01_start/2026-10-19/preservation_at_launch.json)。

起動前の実効空きRAMは503227494400 bytes、idle CPU相当61.64、disk空き481009111040 bytesで、一jobの収容条件を満たした。launcherでも改めてfresh gateを実施し、その実観測をsnapshotに結合する。RSS/disk proxyは恒久的なruntime capではない。

一つのsine回路のcompile→作用に依存があるためproductive worker一つと独立監視を使う。保存値再利用・末尾変換のベクトル化で重複を減らし、既存compilerのnum_processes=1を維持する。host/上位cgroup PSI・他worktreeの通常更新だけではkillせず、自身の異常・OOM・継続不能・入力破損・科学的不一致を記録する。3段階台帳と非返却markerが同じgrantでの再実行を拒否する。

## 現在の証拠と未完了事項

実行前の人工13＋回帰15件PASS、metadata16項目・改変拒否10種・構造40項目PASSは、前回の固定SOURCEの証拠を再利用した。作者自身の検査であり独立レビューではない。

現在のstage・workerのCPU時間・資源監視・各gateの実施状況は時刻付きsnapshotに記録する。最終sine raw差・実測単価・結果SHA・native exit・worker回収は終了後に別途監査する。未確認値をPASS/0として補完しない。旧pilotの全60軸正常終了への遡及変更や、今回の一軸から主比較の全受入を認定することはしない。

起動資料の構造検査34項目もPASSした。直近snapshotの処理段階・数値gateはJSONを正本とし、入力復元中をcompile開始と記載しない。

**次の担当：Codex**。自分のworkerの終了後、結果と各stage台帳、全artifact SHA、SOURCE/入力/runtime、native exit・worker回収、旧保全を照合して結果報告をlocal commitする。科学的または技術的STOPなら、その記録を保持し自動retryしない。正常終了でもmandatory STOPし、費用再会計や主比較へ自動移行しない。
