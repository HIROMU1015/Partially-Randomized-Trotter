# B実行報告

**11 attempted／11 completed／0 failed。各case1attempt。** 既存B-PRE runner／contract／engine／I exact代数・serializer／固定RTEと、B-ENV環境管理helperを再利用した。新しい科学pipeline、SOURCE修正、pilot、retryはない。

承認根拠は今回のユーザー添付指示そのもの。B11件・SOURCE・環境・scope・one-shot・STOPを明示した原文をbyte変換なしでuser_approval.txtへ保存し、新contract／authorizationへhash bindingした。直前のassistant提案は文脈資料として保存したが、ユーザー同意文として扱っていない。取得できない別会話の同意原文を創作していない。

科学SOURCEは`3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c`、active10file／canonical closure `4b4fbde323bce42a87e44b2aad5ac0504efbcfabac21ffb536da489466092e12`。CPython3.11.9／NumPy1.26.4の既存専用環境、public lockとprivate environment/full receiptは固定値に一致した。新しい専用SOURCE checkoutでHEADを維持し、結果文書は別branchへ分離した。

実行前は17固定GitHub資料と個別SOURCE/spec/recipe hashを照合した。既存environment helperの別process再照合を通過し、stdlib permission gateをconsumeなしで確認した。この時点でscientific import0、markerなし、output未作成。その後、既存runnerを1回だけ起動し、verify→consume→engine import→登録11case→STOPという固定順序を保った。

固定P0の4 mode・2粒子・LD1・残差density-square二fragmentの合成model。分子geometry／AO basis／分子DF rankは非該当。identityはfaithful_identity_in_tail。4armのF1用lawはT=1/5、q2、delta=1/10（この一点のみ）、r2、h=1/20、K2。catalogはT=delta=1/20、q=r1、K2。

成功出力は11science payload、4IO payload、11attempt、11completed、manifest、STOPの39JSONとrun.log1本。合計 **162,973bytes**、log24行。15payloadのcanonical bytes/hash/producer、依存DAG、全attempt/completed、manifest参照を保存済みfileだけから照合した。文書のために係数やlawを再生成していない。

GNU timeの実測はelapsed `0:00.51`、user 1.73s、system 2.92s、最大RSS 70,024kbytes、CPU 904%。gate/import/初期化も含む1回のprocess全体であり、科学関数別timingやspeedupではない。登録caseは小さいexact代数とfail-first DAGに従う逐次処理。native初期化のthreadも計測へ含まれる。資源cap、OS／scheduler／共有venv／cache／他jobへの操作はない。GPU driverとの通信は利用不可だったが、この固定CPU exact pathではGPUを使用しない。

runtime errorなし、stderr空、exit0。新grant1はmarker1で消費済み、retry/resume不可。ledger相当の記録は固定runnerのattempt/completed＋manifest＋logであり、別のledger fileを生成したとは表示しない。

F1/F2、Operator／mean／signal、回路build/compile、sampling/events/trajectory、price／interval資格は0件。config内のseed／PCG64はmetadataで、drawを行ったという意味ではない。
