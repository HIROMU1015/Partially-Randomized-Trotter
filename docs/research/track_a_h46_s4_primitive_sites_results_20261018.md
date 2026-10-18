# H4/H6 S4：実block別の不一致原因を確認

**COMPLETE_MANDATORY_STOP。** 承認された66 primitive評価を一回実施し、
元の8 metric行の差をblock・qubit別に完全に照合した。
新compile・trajectory・signal・reference・full-wrapper生成・理論費用表の再計算は0。
元のcounter/native生成器・科学結果・grant/ledger/marker/STOPを変更していない。

[全証拠・SOURCE・実行記録](../../artifacts/resource_applicability/track_a_h46_s4_primitive_sites/2026-10-18/README.md)。

## 固定条件・認可

調査起点は`b7385211e789ce9e9ab9515f3d5779fbd117bfb6`。
今回の**実行SOURCEは`4d60519e11ecf70a2d19f536ce69ed95b314724c`**で、公開・remote一致後に実行した。
元科学SOURCE `536a3671c0ec2939ff3886e40467db26a27ec65a`、
native provider `217c971dd537b1cc7fbc13742bc0f6c263ff7f45`の実装は不変。
branchは`track-a-h46-s4-primitive-site-validation-20261018`。
利用者の「次の作業に進んで大丈夫」を、直前に提示した最大66 primitive評価案への承認として記録した。
元の未承認planを書き換えず、新しいplan/auth/output/one-shot lockを使用した。

H4：linear 1 Å/STO-3G/legacy DF12/8 modes、p12/q1/S4/T=.8。
H6：linear 1 Å/STO-3G/tol-only DF19/12 modes、p19/q8/S4/T=.8。
両者のsymmetric-directional構成、scalar・wrapper、明示lowering規則は維持する。
元compiled profile（Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし）を変更していない。
今回はQiskit import0で、元compilerの再実行・出力完全同一性検証は行っていない。

## 実施内容と入力

H4 codec：1,450,047 bytes、SHA-256
`f9618a6fda0c28bccfc415f95079ad137e846de12fbe2acaf74456d8369a2c48`。
H6 archive：5,670,237 bytes、SHA-256
`ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010`。
元Stage4の固定入力と同じbytes/SHAを実行前後に照合した。

元typed DAG readerで全node hashと選択rootのpreparation hash・register・fragment順を照合し、
実回路/live preparationを復元せず、保存diagonalのfloat hexを読み取った。
Hamiltonian/DF/stateの再生成・NPZ科学arrayの読込はない。
H4の13 block、H6の20 block（各one-body 1を含む）について、
元producerにmodel/nativeの負時間を各1回だけ与え、**26+40=66回**を予約して実施した。
q反復・forward/reverse・cosine/sineで同じprimitive評価は再利用した。

sourceは前段と同じ関数本体を使用し、変更・数式の簡約・epsilon pruningは行わない。
NumPy1.26.4と実Python版、Python実体・NumPy初期化file・native extensionのbyte hashを前後照合した。
新primitive予約は別namespaceで保持し、旧compile/trajectory消費や予約を返却・resetしていない。

## 特定した発生箇所

modelの負係数は`-a*(1/(2-a))`、nativeは`-a/(2-a)`、`a=2**(1/3)`。
1 ULP異なる負stage時間が、共有`df_squared_diagonal_primitives`内の
逐次浮動小数点加算と厳密0判定へ作用していた。
block index・fragment index・qubit indexは**0始まり**。block0はone-bodyである。

|系|block index|DF fragment index|qubit|modelのRZ角度|nativeのRZ角度|
|---|---:|---:|---:|---|---|
|H4|2|1|0|`0x1.4000000000000p-55`|厳密0で省略|
|H4|2|1|1|厳密0で省略|`0x1.0000000000000p-54`|
|H4|2|1|7|厳密0で省略|`0x1.0000000000000p-56`|
|H6|6|5|11|`0x1.8000000000000p-62`|厳密0で省略|

H4負時間：model `-0x1.5ca78b1289ec2p-1`、native `-0x1.5ca78b1289ec1p-1`。
H6では同じ仮数で指数が−4。
H4 fragment1の保持RZ数は正味+1、H6 fragment5は正味−1。
全33 blockで非zero RZZのsupport maskとphaseの有無は一致した。
phaseの数値末尾bitやRZZ角度自体は異なり得るが、命令数差の原因ではない。
H4の3箇所を「1個のgateだけが変わった」と要約しない。

## 保存8行との照合

同じblockのforward uncontrolledはRZ 1命令、reverse directionalはCX–RZ–CXへ写る。
保持RZが1項増減すると、1組当たりrawが±4、RZ/CXが各±2変化する。
今回RZZ/phase有無の差は0だったため、局所差の合算で次を得た。

|系|軸|metric|保存理論|保存native|局所差からのnative−model|
|---|---|---|---:|---:|---:|
|H4|cosine|RZ|48,272|48,274|+2|
|H4|cosine|CX|21,444|21,446|+2|
|H4|sine|RZ|48,273|48,275|+2|
|H4|sine|CX|21,444|21,446|+2|
|H6|cosine|RZ|1,521,189|1,521,173|−16|
|H6|cosine|CX|667,264|667,248|−16|
|H6|sine|RZ|1,521,190|1,521,174|−16|
|H6|sine|CX|667,264|667,248|−16|

raw差もH4 +4、H6 −32、SX/X差0で全て保存差と一致した。
これは負stageの局所差と保存総数の照合であり、新しいfull counter総数を作ったものではない。
cosine/sineのwrapper差は両側で同じため、同一の局所差が両軸に対応する。
正stageの時間は元の静的検査でbit一致しており、今回は追加評価していない。

## 分類と修正の扱い

8行の原因は、**counter scheduleのnative binary64互換性の不整合**と、その結果の共有producerでのzero選別である。
数学的なYoshida式の違いではなく、有限精度の演算順の違いである。
既存のproducerの厳密0規則やnative生成器をバグと判定していない。
Qiskit lowering/opt1の前に差が生じており、今回の8行差をcompiler optimizerへ帰属しない。

最小修正案はcounter側の負係数・stage時間の演算順をnativeと揃えること。
**今回この修正は未実装**。元sourceへパッチを入れたり、過去結果を差し替えたりしていない。
仮に将来修正しても、新SOURCE/profile/closureを使い、元数値を歴史として保持する。
全wrapper wordのbit一致、修正counterの全表、修正後signal精度は今回未検証である。
これらを今回の局所66評価の成功からPASSへ格上げしない。

## 検証・資源・終了

今回追加の**12人工/mock unittest PASS**（0.179秒）。
入力破損、symlink、SOURCE/環境差、認可不足、非finite、block順、予約前STOP、
q/axis重複評価の抑制を検査した。前段16 testsはsource不変のため不要に反復していない。

科学実行は一回、66予約/66評価、失敗0、自動retry0。
native child PID3213264、native exit0、親がwait/reap完了、同一childの残存0。
実測本体wall **0.622166秒**、外側Popen/wait wall **0.730039秒**、
peak RSS **71,344,128 bytes（約68.04 MiB）**。
これはprimitive評価・照合の資源値で、実compile速度・メモリの測定ではない。
直前MemAvailable約468.85 GiB、affinity/disk/inodeをread-only記録した。
小規模1 process、内部thread1で、並列化の起動費用を避けた。
共有環境・既存venv・他jobの変更はなく、作業・一時fileはhome配下のみ。

## 科学的結論への影響・次の担当

先行調査で未観測だった実block別の因果関係を確認し、8行のUNATTRIBUTEDを今回の新しい証拠で説明できた。
元のUNATTRIBUTED監査自体は変更せず、追加証拠として公開する。
元compiled値・wrapper fingerprint・shot・signalは不変。
PR/PF費用比・順位・統計的信頼性の再計算はなく、並行担当のH4統計調査に干渉しない。
「モデルがcompiledより大きいからバグ」という判定ではない。
母平均保証、希少次数上界、旧compiler出力のbit完全同一性は引き続き未認定。
GPT/Claudeの詳細科学レビュー・別担当の独立reviewは実施していない。

次の担当：**ユーザー**が、counterの最小互換性修正と新SOURCEでの限定受入確認を進めるか判断する。
**Codex**はその指示がある場合のみ元counter修正の範囲を扱う。
今回の承認済み66評価は完了し、結果公開後に**mandatory STOP**する。
