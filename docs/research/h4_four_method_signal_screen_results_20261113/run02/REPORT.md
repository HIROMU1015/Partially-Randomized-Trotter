# H4・4方式signal screen run02：4セル保存、数値gateでSTOP

2026-11-13 JST。**再実行の明示承認を受け、import経路だけを補正して同じ20セルbatchを1回実行した。m6のFull PF4セルを保存し、6セルが固定norm/unitarity gateで失敗、9セルは未開始、1セルは空の例外文字列で未解決。部分結果を保全してSTOPし、追加retry・閾値変更は行っていない。**

科学SOURCE：`256caad3414a80de02fb68eca89c13c957ed6354`。結果公開の基点は`37e30194b0fca129a35f651ee260c66618bbb7e4`。旧[run01のSTOP](../REPORT.md)、raw、SOURCEを保持し、元の科学コードは変更していない。

[原結果](result.json)／[PLAN](PLAN.json)／[20セルの状態](SCREEN_STATUS.json)／[CSV](SCREEN_SUMMARY.csv)／[実行条件](EXECUTION.json)／[実行ログ](../RUN02_EXECUTION.log)／[技術確認](TECHNICAL_CHECKS.txt)／[B HANDOFF](HANDOFF.md)。CSVの欠測欄は空欄、JSONの欠測値はnullであり、科学値0を意味しない。

## 固定条件と実行

Linear H4/1.00 Å/STO-3G、保存tol-only DF10・weighted-Hermitian completion、sector(2,2)/36、保存H/state/snapshot。T=.1×2^m、M12、元prior/decoder/誤差budget・P0を維持。Full PFは全10fragment＋one-body、S2・標準Yoshida4・M5 best、Δ=.1/.4。B0Iはprefix7/Yoshida4/Δ.4。Full-H pure RTEはone-body込み全非scalar・決定論block0・K2/K4、κなしの元q/R layout。負時間・signed順・scalar一回会計も固定した。

`src/`と`scripts/`をimport経路に含めた。NumPy1.26.4/complex128、CPU affinity64・有限batch設定20 workers。2026-11-13 15:52:57〜15:53:00 JST、約3.60秒でSTOP結果を保存した。これは全20セルの完了時間ではない。共有環境・他担当の設定、process、worktreeは変更していない。

SOURCEと元PLANは一致し、保存H/stateからのcontext構築、共有spectra、macro-step準備を通過した。元のPF matrix-power経路を使用し、巨大なevent列、sampling、回路compile、新reference、N/RMSE/C/Gは起動していない。

## 保存できた4セル

すべてFull PF、prefix10、m6/T=6.4。保存full-H referenceは`0.2689351412041076 + 0.9631582890810391i`で、元独立pairを用いた数値gateを通過した。新referenceの取得は0。

| label | 公式・Δ | qPF | raw = corrected mean |
|---|---|---:|---|
| F2-FINE | S2/.1 | 64 | 0.2697592256363573 + 0.9629277487000245i |
| F2-COARSE | S2/.4 | 16 | 0.2821546061502322 + 0.959353764366455i |
| F4-FINE | Yoshida4/.1 | 64 | 0.2689304437424606 + 0.9631596007122267i |
| F4-COARSE | Yoshida4/.4 | 16 | 0.26775999266467543 + 0.9634856407233132i |

| label | 保存referenceからの絶対signal bias | 局所phase差 rad | raw半径 | state norm誤差 | power unitarity誤差 |
|---|---:|---:|---:|---:|---:|
| F2-FINE | .000855724266 | −.000855724314 | .9999999445162108 | 2.61390909e−12 | 9.29434587e−11 |
| F2-COARSE | .0137560409 | −.0137562419 | .999985433381847 | 6.55919763e−13 | 2.32564523e−11 |
| F4-FINE | .00000487714284 | .00000487714284 | 1.000000000007826 | 7.87259147e−12 | 2.78894456e−10 |
| F4-COARSE | .00121989065 | .00121989072 | .9999999967759002 | 1.98085992e−12 | 6.97009201e−11 |

値は新recordの既存診断fieldの転記・表示丸め。complex bias・全精度値・SOURCEは個別recordと`result.json`を参照。4セルともB=1、logB=0、attenuation=1、raw/corrected closure=0。finite-RTEを使用しないPFなのでraw=correctedであり、別々の伝播を行ったとは記録していない。

4セルの経験的uは`1.000005964543397e−9`。norm/unitarity閾値はいずれも1e−9で、保存4件は数値gateを通過した。F4-FINEの半径が1を約8e−12上回る値は数値としてそのまま残し、1への投影・再正規化はしていない。allowanceとpower境界normはrigorousな誤差保証や全eventの最大normではない。

## 全20セルの状態

| label | m6 | m12 | m3 |
|---|---|---|---|
| F2-FINE | 保存 | norm/unitarity gate失敗 | 対象外 |
| F2-COARSE | 保存 | norm/unitarity gate失敗 | 対象外 |
| F4-FINE | 保存 | norm/unitarity gate失敗 | 対象外 |
| F4-COARSE | 保存 | norm/unitarity gate失敗 | 対象外 |
| F5-FINE | norm/unitarity gate失敗 | norm/unitarity gate失敗 | 対象外 |
| F5-COARSE | 未開始 | 未開始 | 対象外 |
| T4-SCALAR/B0I | 未開始 | 未開始 | 保存値のみ |
| R2-ADAPT/K2 | 未開始 | 未開始 | 未開始 |
| R4-ADAPT/K4 | 未開始 | 空例外・未解決 | 未開始 |

各physical IDと正確な状態は`SCREEN_STATUS.json`に列挙した。6件の失敗文字列は`Deterministic power norm/unitarity gate`。このgateは`abs(norm(Uq psi)−1)>1e−9`又は`norm(Uq†Uq−I,F)>1e−9`で発生する。**固定実装は失敗時の両残差値を保存しないため、どちらがどれだけ超過したかは未取得。後から再計算して埋めていない。**

Hermitian blockの指数関数からなるPFは厳密算術ではunitaryであり、今回のgate失敗をPF近似biasの科学的FAILと同一視しない。macro-step構成やpowerでの数値誤差蓄積は調査候補だが、原因の断定や新しい数値診断は行っていない。m12の有資格signalは0件で、M5・B0I・Full-H RTEの局所比較も未完了。

Full-H normalizerは**保存record上0件**。R4-ADAPT/m12の空例外はpending futureの取消と整合するが、固定実装は例外型を保存していないので取消と断定しない。他のFull-H5件は明示的に未開始。したがって実normalizer総call数は0〜1の範囲、正確値はnullとした。6件という承認上限は超えない。最初の進捗で伝えた「normalizer0」は、この保存recordの件数として訂正・限定する。

## 保存成果との対応と資格

保存40セルはPLANの元locatorと資格を保全し、再計算・新しい比較診断は行わなかった。内訳はFull PF m3/m4の12、B0I m3/m4の2、PRとB0の各13。元SOURCEはFull PF/B0I長時間`155529e20a8fac39e00786c40b57830b71203c4f`、PF common `0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c`、B0 `89c91c692e1d35de94d2e7afbe890aabe6d80b7e`等、PR `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60`と物理契約`bba150834ca30bdb186fbfd58c55a8d51177b855`。m3/m6/m12 referenceも保存pairのまま。

新4セルは局所signalの観測としてBへ渡せる。局所phaseをTで割ってenergy biasとしたり、目標エネルギー精度の達成・全RPEのRMSE適格性へ昇格したりしない。保存stateのground資格、ideal/actual law、十分N/global RMSE、同RMSEでの資源順位は未資格のまま。欠測を0 biasや方式の敗北へ置換しない。

## 次の作業

**Codex A**：まず固定gate失敗の数値安定性と失敗値保存についてSOURCE確認・人工テストで技術調査することを提案する。閾値を維持したまま観測不足を補う。今回これ以上の科学計算はしない。

実H4の追加診断が必要なら、**ユーザー**に代表失敗セル1点、例えばF2-COARSE/m12（T409.6/q1024/Δ.4）について、既存macro-step/powerのnorm・unitarity・spectral orthogonalityを記録し独立経路と照合する有限scopeを提示する。新reference0、候補・閾値・物理条件は不変とし、全20セルの再走を先に始めない。失敗値の取得自体が新しい科学数値診断なので別途承認が必要。

**B**：今回の4局所recordと欠測状態を受け取り、全13roundへ進める候補の判断を保留する。4方式のscreenが揃わないため、winnerや正式な最適候補を選ばない。C/Dの価格作業へ新しい要求は出していない。

7件のSOURCE・metadata・保存gate整合性チェックを実施した。これは再計算・独立科学検証ではない。旧run01/raw/結果/grant/ledger/marker/KEEP/STOPを保全し、commit/push・remote SHA/tree/主要成果物確認後mandatory STOP。
