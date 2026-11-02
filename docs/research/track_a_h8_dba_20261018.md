# Track A H8 DBA：Dの独立2構成・4軸を先行実行

利用者の「実行準備ができたらそのまま実行してよい」と、その後の「H8の計算は先に始められるか」に基づき、
H6の完了を開始条件にせず、既存の固定計画のH8 Dを実行する。準備中にH6 DBA全stageの正常終了を確認した。
H4は利用者指示で保留。H6のSOURCE/結果/grant/ledger/marker・他worktreeを変更しない。

| 構成 | 保存モデル予測RZ/軸 | 軸 |
|---|---:|---|
| B0 p15/q8 S2 | 3117248/7 | cosine・sine |
| B0+I p16/q4 S4 | 4973552/7 | cosine・sine |

linear H8/1Å/STO-3G、保存tol-only DF28、16 spin modes、sector4900、T=.8、P=0。
SOURCE provider c31ef424…、Qiskit1.3.0、rz/sx/x/cx、optimization1、seed17、topologyなし、
symmetric-directional、元scalarまたは元scalar＋自身のprefixの保存tail identityを一度だけ加える。
full-H reference・signal・allowanceは保存値を維持する。予測は6軸反映後の保存会計から読み取り、新しいfitや予測評価はしない。

H8専用configを使い、2構成を独立grant/lock/outputで並列化し、各構成内はcosineの2%判定後にsineへ進む。
controlのleft/right、X/Y・fingerprint・QPY保存は各2workerで実行する。
検証済みQPYを1軸ずつloadし、大回路を同時に保持しない。既存の内容検証付きfingerprint cacheを再利用する。
各軸でabs(保存予測−実測)/実測が1/50以上なら即STOPし、同じモデル仮説を使う待機軸も起動しない。
技術異常は当該契約をSTOPし、無関係なB/AやH6を一括停止しない。retry・再fit・追加compileはしない。

各構成につきsnapshot1、保存preparation復元1、native evolution1、control4、wrapper2、compile2、
QPY dump/load各2、model check2まで。新signal・sampling・SCF/DF/state・費用再会計は0。
各構成終了時mandatory STOP。Bの希少層4群/64軸とAの境界16構成はadapter未sealで、今回のDから自動起動しない。

人工7件PASS。2%境界、H6/別stage grant拒否、scalar一回加算と保存比率のbinary64丸め、
synthetic 2量子bitでordinary/directional/X/Y・QPY往復・cache有無の実compiled RZ/CX一致を検証した。
これは作者自身の技術検査で、追加AI reviewer/subagentは0。実H8のPASSや性能保証とは区別する。

fresh gateで固定SOURCE・入力hash・265 provider files・34,273 runtime files・compiler/profile、
CPU affinity・RAM・FSを確認する。CPU/RAMの旧恒常上限は移送しない。
各構成の80GiBは起動時の余裕を含む見積りでありRSS上限ではない。一時PSI/CPU競合は観測に留める。
共有OS/他利用者の環境・processに変更を加えない。

[固定plan・保存値の結合](../../artifacts/resource_applicability/track_a_h8_dba/2026-10-18/)。
SOURCEをcommitして個別grantをsealし、一回ずつ起動する。起動状況と終了証拠は後続追記する。

次の担当はCodex。Dの2構成の安定稼働を確認し、B/Aの独立adapterを既存計画の範囲で接続する。
PR母平均保証、厳密bias、全候補実測最良は認定しない。実測と費用統合再会計を分離する。


## 起動追記：Dの2構成を並列で実行中

SOURCE `49879a9c73477c49081a3059cec70e4f4c6a5924` に対し、構成別の有効grantと
専用lock/outputを一回ずつ使用した。両構成ともSOURCE・入力・compiler/runtime・CPU/RAM/FSの
fresh gateをPASSした。6個の異なる物理coreを選び、各構成のcontrolは2workerで実行する。
共有configと28 SOURCE fileは封印時のbytes/Git objectに一致し、起動後に変更していない。
人工検証は7件PASSで、新しい実H8の結果と区別する。

起動準備中にH6 D/B/Aは全て正常終了し、完了資料を別branchのcommit `648184eb` に保存した。
H6とH8の科学processが同時に走ったという記録ではない。H6の完了をH8の起動gateにはしていない。

観測時刻：2026-10-18T13:41:20+09:00。
B0はdirectional controlを終え、cosine/sineのcontrol・fingerprint・QPY保存へ進んだ。
B0+I S4は回路生成を終え、left/rightのdirectional controlを並列実行している。
これは稼働snapshotであり、実測RZ・各軸2%判定・全4軸PASSはまだ認定しない。

[起動・資源・SOURCE証拠](../../artifacts/resource_applicability/track_a_h8_dba/2026-10-18/running_receipt_v1.json)。
`scripts/observe_h8_dba.py` は自分の既知のfile/PIDだけを10秒ごとに読み、進行と終了証拠を保存する。
科学jobへのsignal・gate変更・retry・次stageの起動は行わない。監視scriptは科学SOURCE closureの外にある。

次の担当はCodex。Dの各軸の2%判定とnative exit・最終gateを回収する。
B/Aはadapter未seal・未起動であり、独立した既存計画の準備完了後に扱う。
費用統合再会計、追加候補、H4はここから自動開始しない。

## 完了追記：追加Dの2構成・4軸すべてPASS

両構成ともSOURCE `49879a9c73477c49081a3059cec70e4f4c6a5924`、native exit0・同じ親wait4回収・最終SOURCE/入力/compiler/runtime gate PASSでmandatory STOPした。

| 構成 | 実測RZ/軸（両軸同値） | 保存モデルとの軸別相対誤差 |
|---|---:|---:|
| B0_p15_q8_2nd | 446,288 | 0.216644% |
| B0I_p16_q4_4th | 713,432 | 0.409930% |

全4軸で2%未満。control最大誤差はB0が1.484e-13、B0+Iが2.489e-13。
witness込みwallはB0 1163.533秒（約19分24秒）、B0+I 1935.699秒（約32分16秒）。
[実測・native終了・最終gate受入](../../artifacts/resource_applicability/track_a_h8_dba/2026-10-18/completion/)。
このPASSは登録4軸に限定する。全候補のモデル精度・母平均PR優位・厳密biasを認定しない。
既存結果・grant・ledger・STOPを保持し、総費用再会計・追加compile・fit・retryは0。
A/Bは別cloneでそれぞれ16 signalと4群32 trajectory/64軸を正常終了している。
次の担当：ユーザーが保存実測の費用統合を行うか判断し、Codexが別承認後にその契約で解析する。
