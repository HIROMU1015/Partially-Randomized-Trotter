# 検証する価値のある仮説と反証

以下は正式採択前の候補。既存C/N/Gや方向は引用に限定し、新しい閾値・ratio・予測値は計算していない。

## H1：構成長だけでなく、精度境界に近い候補の扱いが選択を支配する

|項目|設計|
|---|---|
|予測対象|同一物理条件・候補域で、S2/Y4のqを変えたときの適格性と総費用方向|
|入力情報|Tier0の公式係数列・q・prefix・epsilon。Tier1候補は構成長、既存cost proxy。Tier2 bias照会を使う拡張は有料として区別|
|構造的根拠|SOURCEではh_axis=epsilon/sqrt(2)-abs(bias_axis)-u_axis、log Nの主要項は2log B_upper-2log h_axis。小さいheadroomでは単価減少だけでGを判断できない。高次化はq低減とpiece長増加を伴い得る|
|既存結果との対応|異なるqの方向差と同一q対照、H4 Full Y4/q2の保存適格性。固定pairを最適化問題へ拡大しない|
|先行研究との差分候補|次数/step threshold自体は既知。候補固有signalを取得しない規則がどこまで判断でき、有料照会で何が改善するかという測定課題へ限定|
|反証・縮小条件|構成proxyが単価の方向すら保持しない、または単価を説明しても適格性誤り・G損失を防げない。安価なbias情報がない限り安全な事前判定を主張しない|
|必要な新規計算|最小はP1の保存16 endpoint選択・評価。新signal/compile不要。ただし新regret/選択計算には別承認|
|利益の条件|同程度以下の選択Gで、取得するsignal/compileが減る。既存全labelを無料参照した成功は利益に数えない|

Gは軸別N*Cの和である。有効平均単価をNで重み付けして作り、その値を事前の単価予測と呼ぶことは循環になる。上の関係はSOURCEから読める依存関係であり、新しい定理や数値thresholdではない。

## H2：Full/PRの構造応答差は、signed piece・RTE配分・tailとnormalizationを分ければ検査できる

|項目|設計|
|---|---|
|予測対象|次数/q/partition変更時のCとshot負担の応答、共通成分とPR固有成分|
|入力情報|Tier0のq・piece係数・prefix/K/allocation law。Tier1候補はlambda_R、finite distribution由来log B、support/基底回転の構造proxy|
|構造的根拠|FullにRTE tailはない。PRは各pieceでlambda_R*t/rと有限cutoffの法、q回の積、整数r配分に依存する。signed時間を絶対時間総量だけへ潰すと、law・順序・誤差情報が失われる|
|既存結果との対応|同一qではFull/PR双方の低費用次数が一致した保存対照がある。ただし費用変化の大きさが同じとは言えず、q単独原因を示さない|
|先行研究との差分候補|高次PR・絶対時間・cost-aware partitionは既知。有限K・整数allocation・full wrapperのどの診断が有用かを具体化できる場合のみ狭い余地|
|反証・縮小条件|normalization構造proxyを加えても選択誤りを減らさない、またはlambda取得自体が候補全signalと同程度に重い。新しい経験則を一般法則へ拡大しない|
|必要な新規計算|固定prefixの保存比較は開発解析として可能。PR partition効果には複数prefix/K/lawの同条件証拠が不足。必要性を判断するまで新gridを作らない|
|利益の条件|追加特徴取得の負担を含めて候補評価を削減できること。既存R3/SPRINTとの差分が残ること|

現時点では **H2の固定law・固定partition内の記述は可能、partition最適化則・普遍則は困難** と整理する。

## H3：少数の有料適格性照会を含む選択が、固定構成より役立つか

|項目|設計|
|---|---|
|予測対象|決められた候補域からの選択ID、適格性、選択G、取得量、保留|
|入力情報|最初はTier0/1のみ。Tier2は照会したcandidate/field/取得費用を記録した場合だけ利用。全Gと順位はOracle評価側へ隔離|
|構造的根拠|全候補のGを知る必要がない場面はあり得るが、候補誤差を縛れなければ安全な枝刈りはできない。heuristicな選択とcertificateを区別|
|既存結果との対応|旧PF qモデルの価格置換は選択が価格近傍に敏感な例。PR-2のSELECTION_LIMITEDとR3停止は選択可能性を事前に保証しない反例資産|
|先行研究との差分候補|MF best-arm、cost-aware selection、abstainは既知。C0では既知型規則をbaselineとして設計し、新アルゴリズムとは呼ばない|
|反証・縮小条件|ほぼ全candidateのTier2情報が要る、固定baselineより悪いGになる、保留を除外したときだけ成功、一般baselineと同一になる|
|必要な新規計算|P1の固定保存値だけによる情報照会replay。実データfit・新科学取得は0の案|
|利益の条件|取得費用を明示してregretと評価量のtrade-offが有用。少数既参照系での成功だけでは移送性・論文化を採択しない|

現在の優先候補はH1/H3の組合せであり、その新規性は未成立。H2を証明済みの説明変数にしてH3へ入力しない。

## 採用しない近道

- G=C*Nの恒等式そのものを予測理論とする。
- qだけ、次数だけ、またはPRという名称だけで一般的な低費用方向を決める。
- 保存bias、headroom、Nや最終順位をTier0へ移す。
- qが大きければbiasやGが必ず単調改善すると仮定する。
- PF588のB0/B0IをPRのrandom partition系列と誤認する。
- 多数epsilon、複数cohort、独立費用標本を独立した未知分子のtest件数へ数える。
