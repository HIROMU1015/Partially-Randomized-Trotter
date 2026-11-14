# 次の有限検証案（未採択・未実行）

**最初の新規科学実行の推奨は、m9・全6候補・X/Yの最大12 compile。** m3の保存12軸は先にcompile 0件のmetadata bindingで引き渡す。ここにある候補選択、価格精度、実行予算は提案であり、今回の許可は計画公開までである。

## 推奨順と主張できる範囲

| 優先 | 対象route・公式・round・軸 | 必要な新規compile | 保存で代替できない理由・問い | 成功時の範囲 |
|---|---|---:|---|---|
| 0 | Full/p7 × S2/Y4/M5、m3、X/Y | 0 | 元trialは揃う。Cの現configuration/axisへSOURCE付きでbindingする | m3の12直接価格。別幅モデルの資格は増えない |
| **1** | **Full/p7 × S2/Y4/M5、m9、X/Y** | **最大12** | 同じactual幅でqをm6の8倍へ移す。保存m6は長qのcompiler contextを検証していない | 12モデルのm9点、実装/作用gate、実負荷の証拠。m7/m8やm10～m12の自動資格はない |
| 2 | Full/p7 × S2/Y4/M5、m0、X/Y | 最大12 | 全候補q1・幅.1という短時間端点。較正幅と異なり保存価格なし | m0の12直接価格のみ。m1/m2や別幅affine式の資格はない |
| 3 | Full/p7 × S2/Y4/M5、m1・m2、X/Y | 最大24（roundごと12） | m0/m3と異なるT・q・scalar phase。短い有限集合を直接埋める方が新しい別幅モデルより単純 | 取得した24軸。m0/m3と合わせ短roundを直接価格で解決できる |
| 条件付き | Full/p7 × S2/Y4/M5、m12、X/Y | 最大12、または必要group×2軸だけ | 長時間終端まで使う根拠。m9成功だけではm12のcompiler systematic不明 | 直接取得した終端点のみ。中間roundの誤差上界は別途必要 |

すべて現在の受入可能な保存結果が追加で見つからない場合の**不足分上限**。§10.5で重複不明なら、再compileで解決せず該当分保留。表全体を一括実行する計画ではない。特にm12直接compileは既定の次作業ではなく、資源負荷と科学的必要性を見て別途判断する。

m9を第一案にする理由は、較正時と同じactual幅の12モデルすべてに対して、途中roundの連続取得をせず一つの離れたq点で反例を探せること、m12一括取得より入力piece数が少ないこと、既存m6を再利用できることである。q8倍は設計上の選択であり、最適な検証配置の証明ではない。BのNがまだ揃わないため、どのモデルが総Gに最も効くかは今回決められない。N側の比較から対象groupが絞られた場合は、実行前に対象集合を明示して別途採択する。

## 第一案の物理点

正本alignmentの各行番号は0-based。X/Yを同じ観測値で埋めず、別wrapperをcompileする。全点m9、T=51.2、R/Kなし、deterministic、P0除外。

| route / 公式 | physical ID | q | actual幅 | alignment X/Y行 | compile上限 |
|---|---|---:|---|---|---:|
| Full S2 | a4ec53ff4e60d60a2930 | 224 | .2285714285714286 | 36/37 | 2 |
| p7 S2 | 1bd0a5dd4bbf876da1d6 | 224 | .2285714285714286 | 38/39 | 2 |
| Full Y4 | b2c479e7805c4d1f0b30 | 96 | .5333333333333333 | 88/89 | 2 |
| p7 Y4 | 9924c226ee246283ec54 | 96 | .5333333333333333 | 90/91 | 2 |
| Full M5 | a53547a5fb707b9550d5 | 32 | 1.6 | 140/141 | 2 |
| p7 M5 | 60111109e0024d1a1064 | 32 | 1.6 | 142/143 | 2 |

configuration IDはalignment行の値をそのまま使い、model_groupは固定CELL_PLANのroute/formula/幅/axisへ対応させる。m9は元CELL_PLANの取得対象外なので、既存stageのm6制限を単に回避して実行しない。将来の承認に合わせて既存callerへ有限cell selectorを最小拡張し、人工metadata test後、新実行SOURCEをcommitで固定する。現Stage 1/2のコード・出力・契約は保全する。

## 事前固定・判定基準の提案

1. Cの元凍結JSON、coefficients、m4/m5較正、m6判定を変更しない。**m9予測RZは今回算出しない。** 将来、予測計算を含めた有限範囲が承認された場合に、Cが元係数を使った12個の有理数予測と対応group/physical IDを観測前に固定する。新fit・m6/m9による係数再較正は含めない。
2. 実装判定は元SOURCE/input/scalar一回、fragment順、signed piece、full controlled測定wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を維持。既存low/high-control、独立occupation、Hadamard X/Y、compiled-vectorの元gate・許容差を保持し、失敗時はUNKNOWNとしてSTOPする。
3. モデル判定案は軸ごとに `|frozen_prediction-observed_RZ|/observed_RZ <= 1/50`、観測分母、有理数の丸め前判定、全12個別PASSをcampaign成功とする。**これはm6で採択済みの規則をm9という新範囲に使う提案であり、自動的に採択済みとはしない。** 取得/出所/実装UNKNOWNと2%超過FAILを分離する。誤差0も別に記録する。2%は統計的95%保証ではない。
4. m0～m2を直接取得する場合、比較対象の適格な凍結モデルがない。元実装/作用gateを通過したその物理点の直接価格として受け入れる案であり、異なる幅の凍結式に2%を課してPASS/FAILを付けない。
5. G比較の必要価格精度は、後でBのqualified Nと候補差・採択claimに合わせてC/Bが定める。モデルによる候補iの価格誤差区間があるなら `Σ N_{i,m,a} ε_{i,m,a}` を総価格不確かさへ伝播する。候補間の区間が分離するのに十分な精度を優先し、全156軸に一律高精度を要求しない。**未検証modelの2%を全roundの誤差上界に置かない。今回この和やGは計算しない。**

## 長qの終端と未観測中間点

m7～m12のqはS2=56,112,224,448,896,1792、Y4=24,48,96,192,384,768、M5=8,16,32,64,128,256。actual幅が同じでもcompilerの数値角度・scalar phase・1Q最適化・境界contextは無条件に移送できない。m6の完全一致は全q affine性の証明ではない。

m9を通過した後も、m7/m8やm10～m12のモデル誤差はUNKNOWNのまま。終端まで正式利用するには、(a)固定compilerと入力族に対して未観測qを含む誤差制御を示す、または(b)C/Bが採択する有限domain・不確かさ・感度条件に沿った独立検証を追加する必要がある。m12の直接点を一つ取得しても、すべての中間点の2%保証にはならない。検証点間を補間してよいとの数学的主張も今回は行わない。

追加直接compileが重すぎるなら、既存SOURCEを使った反復block/境界のcompiler閉包の技術調査を先に行い、成立条件が示せなければ価格をUNKNOWNとする。補正なし構造S、event数、旧他候補のqモデルで区間を保証したことにしない。全roundをcompileしない方針と、未資格領域を残すことは両立する。

## 実行資源・範囲・STOP

参考は保存済みStage 2性能だけ：Full m6のroute wall 663.902 s、最大worker RSS 1140.92 MiB、p7は440.316 s、955.46 MiB（process peak）。m9の時間/RSSは未測定で、この値から新しい性能予測や資源上限を採択しない。m9のS2/Y4/M5のsigned S2 piece数は224/288/352、m12はその8倍であり、入力規模を選ぶmetadata上の目安である。

将来の有限案は6 evolution cell/最大12 X/Y compile、stochastic sampling/event=0、fit=0、N/G=0。SOURCE、output、重複照合、scalar/binding、C予測固定、個別判定、科学call範囲の明示承認後にのみ開始する。保存H/state/snapshotを再生成せず既存復元・prepare・native/Accounting/fingerprint/cache/compilerを再利用する。既存runnerの性能経路・数値gateは保持する。資源観測に応じて独立cellを並列化し、CPU/RAM/workerの恣意的な恒常上限を設けず、共有環境や他processへ干渉しない。

- **今回**：計画のcommit/push・remote確認でmandatory STOP。
- **将来の実行前**：保存/進行中の同じwrapper、出所不明、SOURCE/input/axis/scalar/domain不一致、未固定の予測/判定policyは該当対象STOP。旧one-shot枠や旧run出力は転用しない。
- **将来の取得中**：実装gate失敗、取得不能、OOM等の実異常では部分結果を保全しUNKNOWN STOP。seed/軸/roundの代替、guard緩和、無断retryをしない。一時的CPU競合だけを理由に正常workerを強制終了しない。
- **将来の判定後**：FAILでも自動refit・追加点・閾値変更なし。全PASSでもその有限点の結論でSTOPし、m12、N/G、winnerへ進まない。

ロードマップ本体の反映は統合担当、価格受入/誤差domainはC、有限取得はD、N/riskはB。次の科学実行は別途明示承認待ち。
