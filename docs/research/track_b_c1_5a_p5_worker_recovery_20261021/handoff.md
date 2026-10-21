# C1-5A-P5R handoff（mandatory STOP）

P5は1024/1024 whole-path・4096/4096 microstepsを完走し、accepted=1024、valid zero=0。全32 logical batch完了。経験的weight二乗平均=1.1105279232736538、観測max=1.0538158868007512。logical instructions/path=284.766601562、depth/path=76.6455078125。

P5 normalizer/completion CPU 283.99316秒、prefix setup CPU 0.493558755秒。Producer online wall 430.542884秒、全job wall 731.998624秒。全job hierarchy CPU 998.71687秒。旧CPU/wallとは負荷・cache・監視・起動方法の条件が異なる。

科学実行 attempt **1**、retry **0**。旧Ordinary・General Fullの再実行・旧raw再解析なし。旧C1-5Aの失敗・SOURCE・grant・marker・STOPは保持した。今回の完走によって旧failureの原因を確定したとはいわない。

固定モデルはLinear H4 1.00 Å / STO-3G / DF rank12 / R1 LD3 / 8 system qubits / 13 bases・324 labels。T=4/5, q=1, r=4, m=5, sigma=+1, microstep1/5。H=2048、root4096、eta/rho/angle tau10^-12、angle root256・terms64、seed20261021と旧bit規則を保持。model SHA256 `e96ea0faea31bc6a784dfb8dc6f7cf4856789b0256e4a1c53b63ce087c86dc49`。

## 公開・固定identity

Repository `HIROMU1015/Partially-Randomized-Trotter`、branch `track-b-c1-5a-p5-worker-recovery-20261021`。

- BASE/旧final `b022f2ea421649015a4c21ea3c904fe619d0c198`
- 新SOURCE `2fbd83a44a0deb76c6444ed7920741c6b5e2e748`
- 新AUTH `bf32b3b3455ee42e1039e7bebbb275eb73351969`（SOURCEの直接子、新authorizationのみ）
- Contract SHA256 `72887cb9e073922ad711940565b1a841bda97aef01e5341cf9b9e8fda1cff669`
- 科学参照 `243eea8e840f3c472753f7935e510391626fd325`

最終公開commitと全公開fileのbytes/SHA256/Git blob照合は `publication_verification.json` を参照。このhandoffを含む最終receipt自体は、その子commit公開後に外部の最終照合でも確認する。

## 障害調査と修正

旧workerの直接のexit/signalが保存されていないため、原因はUNRESOLVED。OOM、segfault、fork/native状態、OS kill、serialization、同時負荷のどれとも断定しない。旧setup 104,977群・max RSS1,770,984 KiB、初期化後fork21worker、巨大law反復走査を静的に確認した。[調査記録](WORKER_FAILURE_ANALYSIS.md)。

修正は既存P5Closed数学を保持し、群CDFを既知の累積整数表へ、巨大stateをone producer所有へ、logical providerをfresh execへ変更。各pathをfsync保存し、PID/exit/signal/wait4 CPU/RSS、PSS/private、cgroup前後差、partialを残す。失敗taskを再投入しない。人工14テストPASS。別査読agentは使っていない。[技術修正](TECHNICAL_REPAIR.md)。

単一producerはPython巨大stateのfork継承・重複cache・重複表を避ける設計選択であり、sampling自体に逐次依存があるとの主張ではない。最適な並列P5を実現したとはいわない。observed CPU stageで残るsampling bottleneckも費用として報告する。logicalは32 fresh processで並列化。共有環境・他job・他worktreeに操作しない。

## 証拠と解釈の範囲

[terminal](result/terminal.json)、[summary](result/summary.json)、[比較Markdown](comparison.md)、[比較CSV](comparison.csv)、[費用・monitor補足](RESOURCE_REPORT.md)、[保全確認](preservation_verification.json)。比較ファイルは完走時のみ。旧Ordinary/Fullは保存summary/setup/terminal引用に限定する。

Raw batch、sampling、event、angle、exit/monitor/journal/stdout/stderrは[archive index](raw_archives/index.json)の全member bytes/SHA256付きで保存。`P5R_raw.tar.gz.partNNN`を番号順に連結するとgzip tarになる。展開先は新しい任意のdirectoryにし、封印済SOURCEや既存結果へ上書きしない。直置きsummary/terminalとarchive内memberの対応もindexに記録。元の新rawは当job専用runtime領域にもbyte同一で保管した。

旧source依存・旧namespace58ファイル、R1入力、runtime選択ファイル、非公開レビュー原本2件のbytes/SHA256を開始前後で照合。原本本文は公開しない。旧source結果を変更せず、追加は新namespace内のみ。runtime manifestは選択したexecutable/package/source identityであり環境全体の完全固定を保証するものではない。

P5はUNCERTIFIED_EXPLORATORY。TAIL_ONLY_PLUS_SCALARの観測で、canonical一様bias/range certificate、IID、native T/CX/1Q、deterministic halves、state preparation/readout、量子shots、coherent estimator、十分N、全round RPEを取得していない。未知項目を0にしない。logical/weightの僅差は有限1024path・異なるmethod bit keyであり、統計的優位を認定しない。

旧Ordinary/Fullは21fork workers/方法・3方式同時、今回P5は単独・one cache・prefix・32fresh provider・local BLAS threads1。古典CPU/wall/storageの境界と実装条件の差を明示し、資源winnerや原著価値を決定しない。Fullでlogical改善が小さい理由について保存summaryが示すのは、ほとんどがempty parentで、4 child rotations/pathが共通に残り、zeroを観測せず、Fullのpacket検査費用を全4096step支払ったことまで。

## 次の担当・必要な承認

|担当|次に判断・作業すること|承認境界|
|---|---|---|
|ユーザー・GPT|今回の観測と制約を読み、C1を継続・縮小・終了するか、資源優位/論文化を目指す価値が残るか判断する。|科学的な最終判断をCodexは採択しない。詳細レビューを始める場合はユーザーが開始範囲を承認する。|
|ユーザー・GPT|追加投資するなら、selector/cache/費用境界を揃える比較設計、canonical保証、native接続のうち、どの最小証拠が判断を変えるかを先に選ぶ。|新proof/実装/科学計算は各範囲の別途承認。自動で新grantを作らない。|
|Codex|今回の公開・保全照合をもってSTOP。承認後に指定範囲の技術作業だけ行う。|追加H4、旧Ordinary/Full再実行、native synthesis/shots/Full-N/Pilot/RPEへ進まない。|

**mandatory STOP。**
