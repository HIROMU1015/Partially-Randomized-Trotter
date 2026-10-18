# Track A H6：D・B・Aの限定並列検証

利用者の「DBAの検証をまずH6で行って」「できるだけ並行して」を実行承認として記録する。
linear H6/1Å/STO-3G、保存tol-only DF19/sector400、T=.8、P=0。
既存 SOURCE・結果・grant・ledger・markerを変更せず、専用cloneと新しい実行領域を使用する。

- D：B0+I p11/q4 S4、cosine判定後sineの最大2軸。保存予測は各1289041/5 RZ。
  相対誤差の分母は実測値、各軸1/50以上なら即STOP。再fit・追加compileなし。
- B：p5/q8/r2のK2 s2_multiとK4 s4、各n8、計16 trajectory/32軸。
  各群の先頭control合格後に残り7を並列化。条件付き抽選だけを変更し量子event重みは変更しない。
  K4 s2_multi、他q、他候補の母平均費用は未解決として残す。controlは別に計8 probe。
- A：p3/4×q2/4/8×r4×K2/4の12構成、p5/q4・8/r8/K4、p5/q16/r2/K2、
  p4/q8/r8/K4、計16構成の正式signal/normalization/経験的適格性。
  保存済みouter診断・preparationを再利用する。通常/非相殺、既存5精度、倍率1。
  このstageはsampling/compile/ランキング/費用再会計を行わない。

各stageは独立SOURCE closure/grant/one-shot lock/outputで実行し、それぞれmandatory STOPする。
正常jobを他stageのSTOPで一括終了しない。入力・compiler等の共有依存が破損した場合は関連stageを停止する。
一時的PSIやCPU競合は観測のみで、実OOMや科学的不一致に対する保護を維持する。
fresh affinity/RAMと既存類似taskの実測から並列度を選び、旧worker/RAM上限を恒常的に移さない。
共有環境や他利用者のprocessへ変更を行わない。AI subagentは使用しない。

実行前の実装検証は作者自身による人工検証であり、独立レビューではない。
実測・モデル推定・有限標本pilot・経験的allowanceを区別し、厳密bias/母平均保証を付けない。
実行結果は終了後に追記する。現時点は実行準備で、既存H8系列を再実行しない。

[機械可読契約・入力来歴](../../artifacts/resource_applicability/track_a_h6_dba/2026-10-18/)。

## 実行状況：A正常終了、D/B稼働snapshot

SOURCEはD `579d03c9…`、B `0e50f9de…`、A `c507ec64…`。
各段階でSOURCE・230 provider files・34,273 runtime files・入力hash・compiler/fresh資源gateを通過した。
独立cloneのHEAD更新を実行中SOURCEの改変と混同せず、実際に使用するfile bytes/Git objectを検査する。
既存計画cloneのHEAD `41ba315b…`・cleanも保持した。

**Aは16/16正常終了・native exit0・全16 signal worker回収・最終gate PASS、mandatory STOP。**
修正版runのwall72.629秒、native CPU129.570秒、driver peak RSS479,940,608 bytes。
保存Hamiltonian/state/DFとfull-H referenceを使い、SCF・DF・state solver・新full referenceを実行していない。
新しいnative Gaussian操作列を生成せず、封印済みfull preparationのblockをprefix viewとして再利用した。
保存p3/4のexact-tail診断、p5/q4・q8の元診断を再利用し、q16のouter診断だけを追加した。
通常/非相殺を別々に保存し、倍率1での適格構成数は以下。費用順位・最良PRは選んでいない。

| ε | 通常signed /16 | 経験的非相殺 /16 |
|---:|---:|---:|
| .05 | 16 | 16 |
| .01 | 16 | 16 |
| .005 | 16 | 16 |
| .001 | 12 | 12 |
| .0005 | 7 | 7 |

厳密bias certificate、母平均の優位、全変数の最適性は認定しない。
Aの新sampling・native compile・費用再会計は0であり、A_COSTを自動起動しない。
[A結果gzip・SHA・終了証拠](../../artifacts/resource_applicability/track_a_h6_dba/2026-10-18/A_execution_receipt.json)。

**D/Bはsnapshot時点で稼働中。** Dの保存cosineは258,653 RZ、予測1289041/5に対する
軸単体誤差0.326615%でPASS。sine・stage全体の正常終了確認はまだなく、全2軸PASSとはしない。
Bは2群の先頭trajectoryがnative controlと独立algebraic event/X/Y検証を通過し、
paired実測完了後に残り14件を最大8workerで開始した。全n8の統計・母平均はまだ認定しない。
[時点付き稼働証拠](../../artifacts/resource_applicability/track_a_h6_dba/2026-10-18/execution_snapshot_v1.json)。

CPU割当はD1枠、B8worker枠、A16worker枠を重ねずに選んだ。Dは各軸の2%判定を挟む直列。
Bはcount表cache・paired event列とevolution再利用、Aはprimitive/spectral/tail行列の共有と16候補の並列化を使う。
各workerの内部threadは1。旧CPU/RAM/wall上限の恒常的な移送、共有設定変更、他jobの停止は行っていない。
serial baselineや高速化倍率は測定していない。

### 技術STOPの保全と修正

- B初回は旧pilot seed照合のbytes→整数変換で、seed登録・sampling・worker・compile前にSTOPした。
  後続witnessも既存logへの排他的書込を拒否してPopen前に退出した。
  旧PID不在・native wait4完了・output/lock未取得・科学dispatch0を照合し、新SOURCEと独立領域で未開始実測を起動した。
  起動操作への自動承認審査の拒否1件も保存し、重複実行がない証拠を確認後に再審査を通過した。
- A初回はq16 oracleのnested sequenceをndarrayへ変換していない不具合でSTOPした。
  16候補signalとq16 outerは未開始で、新時間primitive6worker・36 probeは既にPASSしていた。
  型変換を修正し、旧STOP・grant・lockを保持してPASS済みprimitiveをSHA結合で再利用した。
  新runでprimitiveを再実行せず、未開始部分だけを取得した。
- 科学的不一致による再fit・追加候補・標本追加は0。STOPを遡及的にPASSへ書換えていない。

人工18件が固定H6 runtimeでPASS。独立多項分布全列挙、2量子bitのK2 multi/K4 native event oracle、
prefix操作列の再利用、Horner/独立forward、配列型、旧seed、grant範囲とcosine失敗時のsine拒否を検証した。
作者自身の検証であり、追加AI reviewer/subagentは0。

次の担当はCodex。D/Bの終了証拠・2%判定・n8完了・SOURCE/runtime最終gateを回収し、完了資料を固定する。
観測器は10秒ごとに自分のprocessと既知fileだけを読み取り、結果保存後に終了する。
新しい科学stageを起動したり、PSIや他worktree更新で正常workerを停止したりしない。


## 完了追記：D・B・Aすべて正常終了

3 stageともnative exit0・同じ親のwait4回収・最終SOURCE/入力/compiler/runtime/fresh gate PASSで
mandatory STOPした。前節のD/B稼働記述は過去のsnapshotとして保持する。
[終了結果・最終gate受入receipt](../../artifacts/resource_applicability/track_a_h6_dba/2026-10-18/completion/)。

DのB0+I p11/q4 S4はcosine/sineとも258,653 RZ、119,156 CX。
保存予測257,808.2 RZに対する各軸の誤差は4224/1293265（約0.326615%）で、2%未満PASS。
これは登録2軸の検証であり、未測定候補全体の費用モデル精度は認定しない。
本体wall1174.115秒、witnessを含むwall1192.282秒。

BはH6 p5/q8/r2の登録2群各n8、計16 trajectory・32軸を完了した。
保存済みの条件付きpilot集計は以下。cosine/sineはpairedで、今回の両群では同じRZだった。

| 群 | n | 条件付き標本平均RZ/軸 | 標本SE | 最小–最大 |
|---|---:|---:|---:|---:|
| K2 s2_multi | 8 | 97,173.5 | 444.8895 | 95,154–99,461 |
| K4 s4 | 8 | 97,839.625 | 589.9994 | 95,339–100,501 |

本体wall1264.432秒、witnessを含むwall1265.017秒。これは固定候補・条件付き分布の小標本で、
母平均の保証値やPRの無条件費用として扱わない。K4 s2_multiと他qの費用は未解決のまま。
既存確認n32への混入・未観測費用の0補完・PR費用再会計・新順位は行っていない。

Aは前節の16構成・通常/経験的非相殺の適格性をそのまま正本とし、再計算していない。
修正版runの本体wall72.034秒、witnessを含むwall72.629秒。
3 stageのSOURCEは前節と同じで、旧STOP・grant・lockを保持した。
今回の終了資料の作成は保存値とmetadataの照合だけで、新しい科学的数値解析ではない。

次の担当はCodex。H8 Dは別clone・別SOURCE/grantで先行起動済みであり、その終了証拠を回収する。
H6の実測費用統合、Aの費用取得、候補追加はここから自動実行しない。H4は利用者指示で保留する。
