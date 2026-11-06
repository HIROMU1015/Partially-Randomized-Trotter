# C6b：高速化SOURCEによる残task限定引継ぎPLAN

2026-11-07 JST。利用者は、具体的な[高速化差分](ACCELERATION.md)を確認し、**「新SOURCEへの残task引継ぎを許可」**と明示した。本PLANは旧[PLAN](PLAN.md)のSOURCE固定・中断/引継ぎ境界だけを今回の移管に限定して変更する。新sampling・追加標本・科学条件の変更は許可対象に加えない。承認証明/新grantは作らない。

## 元runの保全と調整停止

旧SOURCE `23725fc6f6d6d0c2bec93e31e32074711804e250`、旧namespace `artifacts/track_d_c6b_q8/2026-11-07/run01`。controller PIDだけへのSIGINTで新規投入を止め、既にsubmitされたtaskはProcessPoolExecutorのshutdown待機で完了させる。worker/他jobへsignalを送らず、workerの再起動・外部からのcode注入・固定SOURCE hot-editをしない。

旧runnerはKeyboardInterruptを記録してSTOPする。これは利用者承認済みの運用移管であり、測定演算子不一致やthreshold failureではない。旧SOURCE/result/failure/STOPを上書きせず、そのまま保全する。新workerを起動する前に旧taskの終了と全保存statusを確認する。

全32本のevent列・256 occurrence/8192 events・seed/domain・hashは旧runをそのまま使う。**新sampling0**、新ground/DFfit0。H4/1Å/STO-3G/Nα=Nβ=2、number-CDF8/元順LD2/S2/s1/r32/K0/h=.1/q8、元H厳密ground仮定・準備費除外、packet/endpoint/law、scalar/identity・signed係数・normalization/attenuation、global cut4、strong融合・whole Reference gadgetは旧PLANのまま。

## 引継ぎ集合と再実行禁止

旧run停止後に[inherited_evidence](../../../artifacts/track_d_c6b_q8/2026-11-07/accelerated_technical/inherited_evidence.json)を固定する。旧SOURCE/STOP/resultと、event32本・CAR cache・完了raw/QPY/NPZ/checkのbytes/hash、完了IDと残IDを記録する。これは科学来歴で、承認証明ではない。

- raw PASSが保存済みのpathはraw回路生成・検証を再実行しない。
- compile marker/QPY/postcheck/observableがすべてPASSのwrapperは、そのbytesと価格を再利用しcompile/postcheckを再実行しない。
- rawが始まった可能性があるのに完了証拠が無いpath、STARTED compile、compile済みだがpostcheck未完了の可能性が残るwrapper、科学failureが一つでもあればSTOP。都合よく再実行しない。
- 新規taskは、旧runで一度も開始していないraw、またはraw PASSだが一度もcompileしていないwrapperだけ。各新pathのraw PASS後だけ最大4 compileへ進む。
- 旧新合算compileは最大128、全32pathを必要とし、欠測のcomplete-case統計を作らない。追加標本・別q・cut/seed/モデル/幅/family変更なし。

原本を新namespaceへ別inodeでbyte一致コピーし、旧出力へ一切書き込まない。新namespaceは `artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion`。新SOURCE・input・引継ぎsnapshotを実行前/後に照合する。旧runのsource_input_preservedフラグが調整停止によりfalseで終わった履歴も消さず、新runで別に前後照合する。

## 高速化とpipeline再利用

新runnerは `run_track_d_c6b_accelerated.py`。C6b canonical event ingestion/CAR context/strong builder/reference gadget/raw harness、C2 P0 compile/check、C3最終pairedχ、C6b paired統計/凍結モデル診断を再利用する。新sampler/compiler/numerical simulatorは作らない。

新しい自分のworkerだけで、既存state-action moduleの `StateActions` 参照を明示的に `BatchedStateActions` へ選択する。旧workerや元SOURCEファイルは変更しない。公式 `simulation_circuit`、AerJob/native callback、double precisionを再利用し、回路copy削減・全入力列の分割処理・Aer実行thread内のOpenMP配分を適用する。全256入力/全出力振幅、global phase保持、component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなしを維持。

P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1.0/initial-zero false/topology-freeは変更しない。state-actionは検査用だけで、価格対象のraw/compiled回路を変更しない。

raw/Reference/directの完了済みprocess peakを工程別に使い、実空きCPU/RAMを観測してtask admission・CPU余り配分・path PASS後compileを行う。上記peakは見積りであって上界ではない。恣意的な低いworker capや共有OS/venv/scheduler変更は加えない。

## 統計・来歴・公開・STOP

凍結C6a係数/共分散、n32、一次12量、tau=1+.05×保存symmetric規模、名目的alpha=.05、一次frozen条件付き/別欄population-line診断を保持する。再fitなし、確認的PASS/95%同時coverage/全law/長q保証なし。X/Yは32whole-pathを共有する。32本の予定結果が揃ってから既存集計を一度だけ実施。

価格recordのexecution_sourceは新しい価格評価SOURCEを表し、別のprovenance欄にsampling SOURCE、raw生成SOURCE、compile/postcheck SOURCE、旧request fingerprintと再利用の有無を個別に保存する。旧新混合証拠をすべて新SOURCEで作り直したとは表示しない。

raw event/QPY/NPZ/CAR cache・内部資源/絶対pathはローカル保全のみ。公開はSOURCE/PLAN、scalar checks、全価格・統計/共分散・診断、hash索引、旧新STOPと来歴。B6介入、N再算定/費用乗算、全13-round総費用、q>8、追加校正/ground/DFfit/Born shots、詳細科学レビューは引き続き対象外。

科学failure、SOURCE/input/phase/profile/公平性/seed/hash不整合、完了証拠不足、再実行・上限超過が必要ならSTOPし原本保全。無断retry/閾値緩和はしない。新SOURCEを専用branchで公開しtree/bytes確認後だけ残taskを実行し、結果公開・remote照合後**mandatory STOP**。次の判断はユーザー／GPTへ返す。
