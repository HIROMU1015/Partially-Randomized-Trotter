# GPT向け引継ぎ：SOS-PF候補AのK0

2026-10-15 JST。対象：元の有限基底電子Hamiltonianのground-energy推定。**K0-1〜K0-5を完了、全体 `UNDETERMINED_DATA_REQUIRED`、K1未認可、正式採択なし、Mandatory STOP。**

## 読む資料

入口は[README](README.md)。証明と範囲は[技術監査](technical_audit.md)、先行性は[文献registry](literature_registry.json)、機械可読状態は[method status](method_status.json)、将来案は[未認可K1草案](k1_design_draft.md)。受領した[指示](received_instruction.txt)と[GPTレビュー](supplied_gpt_review.md)をbyte保存した。Claude `レビューouter.md` 原本は未取得。

研究状態の正本は `78b50c515d9df9556f261e74ca7c57af461b7199` の採用decision、M案K0原固定は `155336b07ad0d5e6e9c6cad35736e9412060a299`。root checkoutは別のdirty branchなので、これらのlocal Git objectを読んだ。今回の新資料は未commit・未公開であり、GitHub再取得／CI固定成果としない。

## 最重要結論

1. **数学の中心は成立するが、それだけでは独立新規性にならない。** PSD分散補題、S2のBCH、低energy評価、SOS Gram、平方指数化はいずれも既存数学。二分割の符号は `d=<A(H'−E_g)A>/12≥0` で確認した。
2. **多fragmentにも古典normだけの明示Kを与えた。** S2順序を固定し、raw BCH辞書から非負対称行列の最大行和を作れば `|d|≤K_SOS E_g`。groundのphiや固有値計算なしに定数を評価できる。今回その定数の分子数値は計算していない。
3. **有限hは無条件に閉じない。** Cauchyによる明示剰余とSchur補元から、認証済みgap下界があれば有限bias／overlapの十分式を紙上で得た。ただし半径・定数が粗く、化学で有用なstep削減は未確認。h²係数上界だけで実用保証とは言えない。
4. **PF-native化に情報・費用の損失がある。** 複素GramをHermitian一体平方へ実化すると、実平方数はrank(ReG)、最大2rank(G)で、さらに一体残余が出る。そのPSD化は下界を悪化させ得る。一般SOS gapの文献値はそのまま使えない。
5. **既存DFのPSD化が必須対照。** 定数shiftだけなら元Hへ復元したPF回路・実bias・overlapは厳密に同じ。保証の改善と分割の改善を混同しない。
6. **有利な独立生成則はまだない。** 構造化平方完成は既知の一体再配分と重なり、一般Gram SDPも既知。K0資料だけから新アルゴリズム採択やK1 prototypeの正当化はできない。

## 追加の近接先行研究

[Kang–Su, Quantum matrix arithmetics with Hamiltonian evolution, arXiv:2510.06316v3](https://arxiv.org/html/2510.06316v3) の§6.1–3は、SOS Hamiltonianの積公式シミュレーション、Hermitian一体平方、固定粒子数normを明示している。**「SOSをPFで使う」という広い構想だけの新規性は支持できない。** 同論文はHamiltonian block encoding/QSVTを使うため、直接軌道回転PFのaccessやSOS gap型固有値certificateと同値とは認定していない。既存のblock-encoding除外を解除したものでもない。

Hejazi二論文、Mizuta–Kuwahara、Yi–Crosson、Low、King、Rubin–Low–DePrince、Casaresの指定一次資料も関連本文を確認した。全付録の独立再証明・全PDFのbyte固定・網羅的先行性調査ではない。MizutaのHTMLの本文日付はPDFと異なったため、v1 PDFの主条件・補足S2定理S1を再確認し、registryに取得差を記録した。

## blocking issueと判断候補

| 問題 | GPTで判断すべきこと |
|---|---|
| 独立した構成原理 | 既知PSD-DF／SOS因子化を超える具体Rがない現在、算法候補として保留するか |
| 保証の意味 | 同回路の有限h認証を独立研究として扱うか。扱うなら主題の変更として明示する |
| 有用性 | 古典取得可能なG・gap・剰余とnative価格で利益が残る条件を先に絞れるか |
| K1 | 未固定Rの探索ではなく、固定仮説の反証に必要な限定検証を別途設計・承認するか |
| 停止 | 具体Rも有用な保証目標も定まらなければ、現Aを保存して追加投資しないか |

**K0の分類は正式な採否判断ではない。** 補題の既知性だけからA全体の一般no-goを宣言せず、逆に未取得データがあるというだけでK1へ進めない。費用レバーの一般的不能性は示していないが、正の総費用改善も示していない。2倍改善は感度分析の参考であり正式gateにしていない。

## K1草案の使い方

二軸比較、強対照、最大2保存入力、最大3表現×6kernel×8h、最大288PF spectrum cells、memory/output/log上限、attempt1/retry0、wall-time capなしを草案化した。**R、主指標・成功基準、入力hash、実行契約が未確定なのでlaunch-readyではない。** SDPは草案でも0で、必要なら別の明示範囲が要る。H4の保存入力候補を挙げただけでarray開封や新実験を始めていない。

## 実施境界

新規Hamiltonian生成、SDP、固有値・期待値・RDM、保存科学数値再集計、PFシミュレーション、optimizer／certificate探索、compile／T合成／benchmarkは**すべて0**。実行したのは読取り、文献調査、紙上導出、新規ローカル文書作成とJSON・リンク・ファイルhashのmetadata検査だけ。

既存source/results/認可/STOP、研究概要・規範文書・global manifestの既知FAIL、他Trackの状態を変更していない。第一候補CLOSED、H-E1終了、M-a/M-b STOP、小改変系列PAUSED、旧D HOLDを保持。候補B/Cへ自動移行なし。commit/pushなし。

**K0後Mandatory STOP。次の科学判断、正式採択、K1、新しい計算、公開はそれぞれ別の明示指示を待つ。**
