# 保存q4の実測性能と長qへの限界

[固定科学条件と全結果](REPORT.md)。4 wrapperを4独立processで同時に処理した。投入前に実空きCPU／RAM／loadを確認し、scope内の4 taskを十分に並列化した。共有設定・他process・旧環境を変更せず、独自の小CPU/RAM上限は設けていない。

予測phase全体のwall timeは **24.957092564秒**。原本検査・worker起動／import・raw読込・lowering・bridge・count/transfer・private payload保全・結果保存を含む。一方、後続の保存P0比較・read-only診断・人工回帰・文書作業は含まない。比較phaseの独立wall timeは計測していない。診断は4.613秒だった。既に完了した処理を時間取得だけのために再実行しない。

## 各workerのwall timeとpeak RSS

単位は秒、RSSはMiB。四捨五入前の全値は[PREDICTIONS.json](../../../artifacts/track_d_h4_native_bridge/2026-11-10/run01/PREDICTIONS.json)に保存した。

| wrapper | 原本read/QPY load | Lowering | Bridge | Count/transfer | private保存 | Worker全体 | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong X | .128 | 10.235 | 9.816 | .812 | 1.131 | 22.508 | 278.555 |
| Strong Y | .130 | 10.628 | 9.888 | .792 | 1.145 | 22.986 | 276.137 |
| Reference X | .112 | 10.366 | 9.978 | 1.143 | 1.133 | 23.190 | 280.805 |
| Reference Y | .113 | 10.365 | 9.934 | 1.136 | 1.138 | 23.135 | 279.980 |

worker全体にはQiskit等のimportと小さいI/Oを含み、表のstage和とは一致しない。Bridge時間は既存optimizer自身の入力validation／count・出力countを含む。Count/transferは最後のnative列から独立に計数・transferを作って一致を検査する追加stageであり、bridge内の時間と混同しない。

RSSは各workerの `getrusage(RUSAGE_SELF).ru_maxrss`、Linux KiBをMiBへ換算したprocess全体のhigh-water mark。Qiskit／Python／保存raw QC／native列／bridge全てを含む。stage別記録もその時点までの累積peakで、instantaneous RSSではない。import後約89.5〜90 MiB、raw読込後92.5〜93.5 MiB、lowering後258〜263 MiB、bridge後276〜281 MiBだった。worker peakの和1115.477 MiBは同時peakの実測ではなくworker分の上側集計で、parentの記憶は含まない。

## 命令数・反復・frontier

| wrapper | 保存raw命令数 | Native命令数 | 最終命令数 | Bridge反復 | 最大live node | 最大retained node slot | 最大frontier参照数 | 最大1wire stack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strong X | 8386 | 200882 | 42839 | 10 | 60482 | 60848 | 70721 | 10651 |
| Strong Y | 8387 | 200887 | 42839 | 10 | 60482 | 60848 | 70721 | 10651 |
| Reference X | 7422 | 203178 | 42562 | 10 | 60243 | 60617 | 69794 | 5611 |
| Reference Y | 7423 | 203183 | 42562 | 10 | 60243 | 60617 | 69794 | 5611 |

live／frontier／retained slotは各CX pass内の最大値。retained slotは消したnodeのtombstoneも含む。frontier参照数は各wire stackの参照の和であり、同一CX nodeを両wireで数える。最大native word containerは入力native列の200882〜203183で、他containerと共存するためこれだけで総記憶を見積もらない。1Q runと全体の複数wordも保持し、process RSSが実際の合計コストを捉える。

10反復には最終のsize不変確認を含む。Strong Xは200882→60482→55389→50561→47135→45091→43729→43059→42893→42839→42839、Reference Xは203178→60243→55112→50284→46858→44814→43452→42782→42616→42562→42562。途中のCX／1Q連鎖を固定小回数で打ち切っていない。

## ボトルネックと省略範囲

lowering約10.2〜10.6秒とbridge約9.8〜10.0秒が主なコスト。Bridge内の1Q合成合計は約3.85〜3.98秒、CX pass合計は約4.38〜4.60秒。残りには入力validation/count、list生成、最終count等がある。時間計測と各feed後のfrontier観測のoverheadを含み、観測を外した速さは測定していない。価格を知ってからcacheやruleを変更する再実行はしなかった。

保存rawの全QCは読み込むが、native全体を新QuantumCircuit／full DAGへ組み立てず、全H4 wrapper構築・P0 pass manager・CAR target・全入力simulationを回避した。標準gate definition／小1Q QCの生成はnative loweringの一部として残る。既存P0を新実行して比較したわけではないため、compileに対する実測speedup倍率は示さない。

Lをnative命令数、Wをwire幅、Rを反復数とすると、T0直接計数O(L)時間/O(W)記憶、transfer capture O(LW)時間/O(W²)記憶、現bridge O(RL)時間/O(L+W)記憶・最悪O(L²)。今回の観測はq4でR=10という有限値だけ。W一定でも1wire stackは1万超で、固定幅frontierへの切詰めやchunk価格独立加算は正当化されない。

q1024〜4096では、native列の膨張・複数wordの同時保持・長い連鎖相殺・declared phaseのbinary64累積・raw QC自体の保持が障害になる。今回の25秒／281 MiBから長qの速度・RAM・精度を保証しない。q比例の線形外挿や新benchmark、全13-round費用会計も行っていない。
