# 冪漸化式：人工入力でのexact同値性

新moduleのendpointは`power=z`、`square=-z*z`から始め、項ごとにpower/(2j+1)を加算しpower*=squareとする。
各stepのpowerは元の`(-1)^j z^(2j+1)`で、J回後のpower/(2J+1)が元の次項。
同じFractionのpartial/min/max/remainderを返す。型は厳密Fraction、Jは非負整数で、floatを許さない。
これは既存有限和の通常の代数的書換えであり、新たな誤差保証・精度探索ではない。

standalone artificial enclosureは固定enclose_atanのbodyを基準にし、root_interval、exact型validation、context、Evidence codec、σ、radius≤τ受入、F3Failure detailsを維持。
二つのendpoint選択だけを明示的に変える。APIはARTIFICIAL_ONLY context必須であり、本番F3をmonkeypatchしない。
固定g6_f3_precision.py/F3 contract/ReturnKernelはbytes不変。

28件でr=0/微小/2/2近傍、完全平方root、1024bit比、endpoint4096bit分母、正負σ、J/K変化、τ=radiusとradius/2、invalid入力を検査。
元関数とのendpoint4値、全Evidence document/digest/read、精度失敗の全detailsがexact一致。
原precision/root/Evidence関数はこの**人工有理数検証**とbenchmarkでだけ実行した。G7/F3 sessionや実科学入力の評価ではない。

## 接続できた範囲と未接続範囲

4構成の人工workloadではangle callbackへこのprototypeを明示的に渡し比較できた。
現行Full F3 sessionのangle_for_eventは固定enclose_atanを参照するため、driverのprecision辞書を替えるだけではFull側へ接続できない。
固定F3をpatch/monkeypatchして無理に接続していない。canonical B/Full Bの本番は元angle/requeryのまま。

本番へ採用するなら、ユーザー承認後の別F3 SOURCE版でendpoint/backendを明示して接続し、Full sessionの_guard/context/event→packet再queryとbackend/source identityを保持する必要がある。
canonical Bのloaderも新しい固定precision APIへ明示接続し、F3/source/admission/observer pin、source_id由来の証拠、失敗STOPを再回帰する。
そのSOURCE差分・pin/契約更新は今回未実装・未採択。したがってkernelの人工同値性は確認済みだが、本番統合候補の総合分類は**B**。
