# 方法差と近接先行研究

2026-10-23 JST。調べた近接手順との比較であり、全世界のpriority監査ではない。外部一次資料の取得版・箇所・失敗は[SOURCE_EVIDENCE_INDEX.json](SOURCE_EVIDENCE_INDEX.json)に記録する。

## 1. 設計したものと、現在擁護できる主張

今回定義したものは、固定された残差fragmentそれぞれにS/Pの二つの辞書を用意し、全体collect後のcanonical finite-RTE normalizationと、既存run共有を含めたliteral期待価格で、二端点・固定一passの選択を行う手順である。決定論block、物理の分割、r/K、sampler familyを同時探索しない。

**混在辞書、残差Pauli化、cost-aware選択、basis共有の個別着想は新しくない。** PR原論文§VII.Bは残差をPauliへ書き戻す構成と、異なる軌道基底のinvolutionの利用を記載する。これだけで独立新規性は成立しない。[原PR v1](https://arxiv.org/pdf/2503.05647v1)

さらに固定SOURCE `C-CONSTRUCT` の `Primitive` はPauli labelとbasisをprimitiveごとに持つ。`PRConstruction.primitives`の型に同一basisの制約はなく、`trajectory`は各primitiveの`apply`を呼ぶ。現行の公開toyはuniformな比較armを構築しているが、異種辞書を受ける接口は既にある。mixed involution coreを今回発明したとは言えない。一方、このSOURCEに本案のfragment別二端点selector、canonical次数頻度を使うrun価格DP、production用mixed routerが実装済みとも読めない。これは対象箇所の静的確認であり、全branch不存在証明ではない。

現在の分類は `KNOWN_METHOD_APPLICATION_CANDIDATE` / `INDEPENDENT_METHOD_DELTA_UNESTABLISHED`。独立新法のclaimは `NO_DEFENSIBLE_NEW_METHOD_YET`。この否定的なclaim状態と、実用的なPR接続案の具体化は両立する。完全な手順同値性は未証明であり、「既知論文と一字一句同じ算法へ還元済み」とは書かない。

## 2. 近接する外部一次資料

| 一次資料・確認範囲 | 重なる構造 | 今回の具体手順との差・留保 |
|---|---|---|
| [Günther et al., PR v1](https://arxiv.org/pdf/2503.05647v1), §VII.A–B、Appendix E関連箇所 | 軌道選択、DF残差Pauli化、回転基底involution、回路費と係数の交換 | 本案は軌道／splitを固定しfragmentの辞書だけを選ぶ。原論文のどの記述にもないとの全面断言はしない |
| [Mukhopadhyay–Wiebe–Zhang 2023](https://www.nature.com/articles/s41534-023-00697-6), Abstract、Introduction、Algorithm 1前後 | 安く実装できるfragment辞書、係数質量と回路費を用いたgreedy割当 | 本案は可換群を新規探索せず、固定fragmentの二表現を選ぶ。qDRIFT channelの費用／誤差評価をcoherent paired finite-RTEへそのまま移さない。個別PR scoreを含む同一手順かは未確定 |
| [Casares et al., SPRINT/GRADE v1](https://arxiv.org/html/2606.30741v1), Abstract・導入・構成／selection guide関連節 | 分解、残差、階層PF、乱択、回路生成を結ぶ近接構成 | 全frameworkの再実装は要求しない。今回採る辞書と既知fusionに関係する部分だけを共通条件へ移す。別taskの倍率は使用しない |
| [Martínez-Martínez et al.](https://arxiv.org/html/2210.10189v2), Abstract・導入 | fermionic／Pauli partitioningと誤差・生成費の交換 | 一般的なrepresentation tradeoffを新claimにしない。本文全方式の独立再検証はしていない |
| [van den Berg–Temme 2020](https://arxiv.org/abs/2003.13599), Abstract | 可換Pauli clusterの同時対角化と回路最適化 | 今回は新cluster探索をしない。順序を保つ合法なcluster簡約は両armに許す。要約の確認であり詳細証明読了ではない |

RC-DFは近似fitやrankを変更する別の自由度であり、今回の固定input上の唯一の対照とはしない。適用可能な既存RC-DFが同じH精度へ接続されるなら将来比較候補に残すが、再fit実行や全方式の比較は今回行わない。IS、native配分、multi-round、hardware routingは保留のまま。

## 3. 同じtask・情報で比較する軸

| 軸 | 本案 | 近い既知構成へ与えるもの |
|---|---|---|
| 入力・アクセス | 保存fragment、factor／軌道行列、recipe、PR設定、共通allowance | 同じinputと同じ係数map。E0・exact state・gapの優遇なし |
| H・表現 | 同一Rのfragment単位S/P混在 | 全S、全P、既知cost-aware辞書選択に同じS/P候補を開放 |
| 分割 | `H_D`とR、`L_D`固定 | 同じ物理分割。既存H2 Pauli/DF比較の異splitを純表現対照としない |
| sampling | 変更後係数のcanonical paired finite-RTE | 同じlaw、q/r/K、scalar policy。全armでB・角度・確率を更新 |
| 回路 | typed中央dispatchと既知basis共有 | basis非制御、中央制御、実adjoint、phase集約、support/full共有、D境界融合、合法な相対Gaussian／Pauli簡約 |
| 古典work | 係数map生成、全体collect、有限state価格、最大2+2M評価 | 同じcacheとrecipe。cache初回構築費とhot lookupを混ぜない |
| 保証 | PO未充足。scoreはliteral plan用 | 同じ精度階層と保証の要求。actual compiler費のoracleはどちらにも与えない |
| 費用 | N×literal期待countを選択に利用 | 実測では同じwrapper・prep/readout・compiler。RZ／CX／T／depthを別々に報告 |

## 4. 比較対照を具体化する

F1の最小poolは二つのfragment、従って四つのbit列である。全S・全Pに加え、**同じpoolを既存の異種Primitive接口で明示的に組み立てる構成**を比較対象にする。このpoolの全bit列列挙は小系の既知の有限探索対照であり、候補だけにmixedを許す比較を避ける。全体価格の最良bit列に一致することは独立新原理の証拠ではない。選択に要したworkも比較する。

外部Algorithm 1の比率型greedyは、可換fragment lexiconを前提とする。これを同じ二表現poolへ移す場合、全fragment一括表現の排他条件やPR価格への変更箇所を明示し、原論文そのままの定理を移したとは呼ばない。本案と同一scoreを当然備える「万能baseline」を想像して、独立差なしと断定することもしない。

今回説明できる**設計上の差**は、固定fragmentを保つ二値更新、全体Pauli collect、canonical paired次数lawによるrun期待価格の再計算、有限回の探索規則を一体で指定した点である。それが既知構成の通常の組合せを越える研究上の方法差かは未確定。F1で単にmixed列が出るだけなら「既知法のPR接続」の証拠にとどめる。生成手順・取得費・適用域・保証に追加差がないと判定された場合、新算法路線は停止し、必要なら応用実装として別採択を求める。

## 5. 旧negativeの扱い

P-A interval DPは強い一区間対照と同planだった。これは今回の辞書選択のno-goではないが、旧区間分割を名前だけ替えて再開する根拠にもならない。Gaussian積生成closureは追加手順が未定義だった記録であり、性能負けの証拠ではない。BMのcompact BCH還元は同じ式・scoreの書換えを新法と呼ばない制約として使う。各固定資料のscopeは `EV-0032`、`EV-0128`、`EV-0084`。本案はそれらのSTOPを解除しない。
