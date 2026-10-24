# 直接関連文献の差分比較

著者全員・正式タイトル・使用版・DOI・読取範囲は[参考文献](references.md)、旧分類・書誌は固定501b4232…を参照。
以下は今回確認した範囲の要約。[CSV](papers_comparison.csv)にPF方式、高次化箇所、q/次数、partition、化学/DF、精度、費用、shot、理論/実装、crossover、独立確認、保証の12項目を分けた。
「今回の範囲では未確認」は文献全体での不存在を意味しない。

|文献|確認した具体箇所|既にある内容|今回との比較境界|
|---|---|---|---|
|[P01 Güntherら](https://arxiv.org/pdf/2503.05647v2)|§V、VII B、App.A3 A32–A41、D1/Fig.16、E21–E23|一般次数PR、signed stageと絶対時間、DF適用、反復込み費用、S2/S4の効率比較|本研究の標準Y4・有限K4・固定outer32・native wrapper標本の定量例とは条件が異なる。PR高次化の初提案にはできない|
|[P03 Hagan–Wiebe](https://arxiv.org/html/2206.06409v3#S5)|§5.1 Definition7、Eqs.57–58、§5.2–5.3|高次outer decompositionと内側Trotter、QDrift partition、query数と誤差の理論|finite paired-Taylor RTEのHadamard signalと同じ法ではない。高次deterministic/random混合の一般着想は先行する|
|[P08 Kissら](https://arxiv.org/html/2212.05952v2#S3.SS3)|§3.3 Eqs.34–40、§4|termごとの実装単価を含む平均cost、sampling/partitionの選択|本研究はlaw固定。cost-aware samplingそのものやC×反復の初提案ではない|
|[P10 Mottaら](https://arxiv.org/pdf/1808.02625v2)|Eqs.3/8、Fig.1|低ランク二段分解・基底回転による電子構造回路|DF/Givensは既知の構成要素。今回のfull wrapper/control付きcompiled RZは具体的実装指標|
|[P21 Moralesら](https://arxiv.org/html/2210.15817v3#S4)|§II.1、IV.1–IV.2、V.2|PF長・次数・時間・許容誤差を考慮した効率比較と次数間threshold|本研究の精度別順位を普遍的な次数選択法の発見にしない|
|[P25 Paganelliら](https://arxiv.org/html/2604.07448v1#S5.SS2)|§IV、V.2/Figs.3–4、Eq.29、App.C|ランダム/決定論の精度依存crossoverを数値比較|random ensemble・qDRIFT/SparSto・T合成と本研究のDF signal/RZ費用は異なる。Eq.29の固有値proxyを一般非normal行列のspectral normと同一視しない|
|[P26 Kanasugiら](https://arxiv.org/html/2603.22778v2#S2.SS2.SSS4)|II B4 Eqs.17–18、V、App.B|PRのpartition cost最適化、各反復の単価×shots、UWC/early-FT資源|固定T/P=0の本研究を、より広いenergy/FT資源解析の初実施とは言えない|
|[P27 Casaresら](https://arxiv.org/html/2606.30741v1#S3.SS5)|Introduction/Fig.1、III.5、III.6概要|SPRINT/GRADE、群内orderingのrandomization、構造と合成費用の最適化|randomizationは本研究のtail RTEとは異なる。最新PF baselineとの未実施比較が残る|
|[P32 Kim–García-Pintos](https://arxiv.org/html/2608.07720v1#S4.SS2)|Table I、S4.1–S4.3、Figs.S3–S4|near-integrable系のrandomized PF、整数stepの最小化、次数/精度crossover、finite-sampling誤差|本研究の固定q構成・compiled-cost標本とは誤差・費用・access modelが違う。一般的method-selectionやsampling問題の初認識にしない|
|[P33 Abeら](https://arxiv.org/html/2605.30967v1#S3.SS4)|II B、III A/C/D、Fig.3|分子energy taskの決定論高次PF費用・新4次formula|別のdeterministic PF成果。標準Yoshida4のPR比較で新係数を発見したとはしない|

P01 Fig.16のS4は記載されたSuzuki構成（stage数の比較）であり、本研究の標準Y4の3 signed pieceと同じ具体列と認定しない。
一方、App.A3は一般のsigned stageを扱うので「Yoshidaという名前が違う」だけで高次PRの原理的新規性は生じない。

独立確認について：P32 S4.3等はobservable推定のsamplingを既に論じる。
今回の追加は、固定候補・固定shot政策で別cohortの**回路費用方向**を確認した具体的結果であり、新しい統計手法でも、全先行文献に同種の確認がないとの認定でもない。
詳しい証明や全実装を未確認の文献の欄は、その不足を保持する。
