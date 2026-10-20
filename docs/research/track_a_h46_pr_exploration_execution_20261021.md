# H4/H6高次PF PR本探索：承認範囲と実行契約

2026-10-21の利用者の「その方針で進めてよいが、できるだけ並列化高速化簡単なものでいいがしてから行って」を、直前の推奨標本数案の採択とH4/H6それぞれの一回の取得承認として結合する。先行準備commitは`2fcdec15341c0b3371b37ab5d5eb5bec1009de54`。過去の未採択資料・無効grant・STOPは変更しない。

## 採択範囲

`MCSE_2PCT_F4_GROUP_MIN33`：保存pilotに基づく相対MCSE目標2%・分散倍率4・群別配分・下限33。名目設計であり、未観測S4、母平均、分散の保証ではない。既存pilot統計を再評価せず、保存済み案を採用する。

| 系 | 公式 | q1 | q2 | q4 | q8 |
| --- | --- | ---: | ---: | ---: | ---: |
| H4 | S2 | 255 | 143 | 62 | 48 |
| H4 | Yoshida4 | 54 | 33 | 33 | 33 |
| H4 | m5_best | 33 | 33 | 33 | 33 |
| H6 | S2 | 167 | 110 | 33 | 33 |
| H6 | Yoshida4 | 33 | 33 | 33 | 33 |
| H6 | m5_best | 33 | 33 | 33 | 33 |

H4は793 trajectory/1586 compile軸、H6は607/1214。合計1400/2800。各系を独立したEXPLORATION cohort・grant・ledger・marker・STOPへ結合する。新seedは明示された既知registryと照合し、pilot・独立確認・旧未使用案とも衝突を拒否する。再hash・retry・top-upは行わない。

## 固定科学条件

線形1 Å、STO-3G、tol-only 1e-8、DF10/19、保存Hamiltonian/state/preparation、T=.8、P=0、prefix3/5、K4、ABS_TIME_PER_OUTER=32、q1/2/4/8、固定S2/Yoshida4/m5_best係数・signed scheduleを維持する。full Hadamard wrapper、cosine/sine paired、symmetric-directional control、scalar once、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17を変更しない。compiled vectorはCT01の1e-7未満、他のnative/oracle/Hadamard検査は1e-9未満。phase alignmentなし。

生成law、S0/S2_single/S2_multi/S4、RTE event、両軸のnative検査とcompiled RZ/CX/depthを各trajectoryへ保存する。新科学callは契約内のsampling/native作用/compileのみで、新signal・SCF/DF/state生成・fit・shot・会計・rankingを含まない。

## 軽量な高速化と資源

- forkによるreadonly context共有と、1系1prefixの事前読込みを再利用する。
- SHA確認付き入力JSON cacheとnative fingerprint cacheを再利用する。
- 候補表の再生成を、入力全byteをkeyにしたcacheへ変更する。返却値はdeep copyとし、SOURCE/input全SHA検査は従来どおり毎回実行する。
- q×合成片数の大きい処理から割当て、cosine受入後に同trajectoryのsineを実行する。
- 外側processを並列化し、内側BLAS/Qiskit threadの重複を避ける。実空きCPU・RAMからH4/H6配分を決め、歴史的worker数を固定上限にしない。

容量は既存338708447996 byteのpeak growth計画を変更しない。新しい容量推定を作らず、同じ計画のscratch枠・RAM proxy・fresh空き容量/inode内でworkerを配分する。計画は実出力の上界ではない。新規QPY/NPZは既承認level1 byte可逆保存、原SHA/archive SHA、復元検証とmanifest永続化後の専用一時raw解放を継続する。旧rawやworktreeは整理・削除しない。

## 起動とSTOP

専用preparerが新SOURCE・入力・runtime/compiler・既知seed・既存容量計画・ユーザー承認をsealし、H4/H6別one-shot grantを作る。両方の静的gateを通過した後、durable controllerが専用supervisorを起動する。各launcherは起動前とcontext共有後にfresh gateを再確認する。

数値FAIL、容量不足、SHA不一致、seed衝突、異常exitは該当契約をSTOPする。共通SOURCE/input/storageの整合不良は影響する自分のstageだけを止める。一時的PSIや無関係なworktree更新をSTOP根拠にせず、他jobへのsignalは送らない。自動retry・条件変更なし。

取得後の監査は全登録・原byte SHA復元・native gate・正確なcall台帳・結果SHA・native exit/全worker回収と技術資源記録に限定する。public finalizerはprivate原本・seed・内部pathを除外した技術summaryをlocal commitする。公開pushは自動で行わない。

新しい統計/MCSE評価、正式shot政策、shot込み費用比較、順位、独立確認P、m5主比較採用AD03は今回の範囲外。取得・終了監査後はmandatory STOP。次の担当はユーザー：終了結果を確認し、後続の限定統計評価を承認するか判断する。

## 実装入口

- `scripts/prepare_h46_pr_exploration_execution.py`：新承認packとfresh共同資源配分。
- `scripts/launch_h46_pr_exploration_acquisition.py`：独立2取得・技術監査・終了処理。
- `scripts/finalize_h46_pr_exploration_acquisition.py`：公開可能な技術summaryとlocal commit。
- `scripts/h46_pr_exploration_contract.py`：独立scope、seed、SOURCE/input、proposal cache。
- `tests/test_h46_pr_exploration_execution.py`：cache・資源・新承認pack・終了/局所STOPの人工検査。既存native経路の人工・回帰検査も実行前に行う。
