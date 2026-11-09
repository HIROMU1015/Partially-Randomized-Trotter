# 次段階の最小実装・検証案（未開始）

2026-11-09 JST。**最初はnative stream＋count/max-plus backendの技術実装とmock試験を推奨する。** qを増やす科学実行を先行させる必要はない。後続段階はユーザー/GPTが範囲を採択し、Codexへ指示する。承認証明書・新grant・承認原文保存は不要。既存の個別実行契約・STOPは別に維持する。

## T0：小さな技術実装

担当：採択後のCodex。対象は純粋な価格評価用受け口・mockであり、H4入力・P0呼出しは含めない。

- `track_d_c7_2_toy.weighted_depth`の既存recurrenceを共通化／再利用し、native tuple iteratorからのcount・arrival更新を受けられるようにする。新しいsimulatorを作らない。
- 最初は確定native tupleを入力するbackendまでとする。CRZ／Gaussian／CSWAP等の非native gateを一操作のnative価格として数えず、既存loweringへの接続が暗黙にP0を起動しないようにする。その展開とP0互換の資格はT1で扱う。
- 全wire transferとembeddingを最小追加する。単一exit vectorではなく、非一様arrivalへ作用する表現を持つ。metricごとに重みを分け、countとpeak widthも別に扱う。
- C7-4 `event_features`の定義を維持してstream受け口を追加する案を検討する。`basis_runs`がstepごとに始まること、lane境界、support work、zero/signを変えない。生eventを再生成しない。
- C3 `head_recipe/mixed_head/physical_segment`、StrongBuilder、tail emitter、Reference gadgetの**既存命令生成順**を使うための接続点を設計する。`prepare_steps`のCAR target／dense reference構築を価格用iteratorへ持ち込まない。既存builderを別の物理意味へ書き換えない。
- actual A†は逆順命令の適格templateを使う。保存descriptorの逆読取やstep単位の再訪が必要なら設計する。長q全回路を逆転のためだけに保持しない一方、定数memoryを無証明に約束しない。

出力には対象IR（pre-P0／確定native／P0対応）、SOURCE・compiler・optimization policy、入力/phase規約、適用domain、未閉包残差の有無を簡潔に付す。`P0_UNQUALIFIED`な設計出力を既存C5/C6a〜C8-1の適格price recordへ流し込まない。新しい管理・承認frameworkは不要。

## T1：P0閉包を切り分ける最小人工検証

担当：ユーザー/GPTが対象と新P0実行範囲を採択し、その後Codexが実施。**今回は新しい人工P0も禁止されており、以下は未実行。** まずC7-2の既存traceをテスト仕様のoracleとして再利用できるか確認する。

| 固定するケースfamily | 切り分ける疑問 | 比較するもの |
|---|---|---|
| RZ/RZZ：ordinary/directional/physical、＋/−、zero/special-angle | 中心premiumとpruneのdomain、parityの跨界面相殺 | 同じ全wordのP0とstream結果。局所価格和だけをoracleにしない |
| Gaussian：同rotation／異frame／gauge／actual inverse | relative橋・頭尾・位相がcountとdepthへ与える影響 | 既存emitterのphase付き意味と全P0資源 |
| step連結と一global cut | Strong q−1／Reference q−2の会計が閉じるか | 分割位置を事前固定した連結と全word |
| 二CSWAP＋χ＋lane端部 | 孤立CSWAP8CX・scalar shell28CX等の非加法性 | 固定shell＋laneを含むfull人工word、既存反例を保持 |
| 連鎖相殺／長い1q runを跨ぐchunk境界 | 固定frontier幅が十分か、可変frontierが必要か | 同じwordのchunk分割を変えても全P0 oracleが変わらないこと |

一つの小さい固定familyを先に扱い、1q再合成・inverse cancellation・translation・停止条件をSOURCEで確認する。任意量のP0 compileを包括承認されたものとは扱わない。追加ケースは切り分け目的と本数を実行前に固定する。有限一致だけで全support証明にはしない。閉包しない場合は未資格範囲を残し、都合のよい固定窓や補正定数でexact扱いしない。

## H0：既存q≤64による限定照合

担当：ユーザー/GPTが**新しい保存科学データ再解析**として対象を明示し、許可後Codexが実施。今の設計許可には含まれない。

既存88path／q64 8pathの公開scalar・構造descriptorと、必要な場合だけ許可されたprivate event/保存P0を利用する。q1はcutと構成が違うのでq≥2の閉包へ混ぜない。固定SOURCE、same P0、C1 OFFを維持する。新sampling・新compile・state simulationなしでも、候補評価器のCX/RZ/size・各depthが保存値と一致するかを照合できる可能性がある。

ただし公開特徴だけでは角度・native界面が不足する場合がある。private原本がなければ欠測を明記してSTOPし、再生成で原本の代わりにしない。全path再処理を初手の必須条件にはせず、T1で解く界面を持つ保存ケースを結果非依存に事前固定する。有限照合は実装移送の証拠で、全q・全law証明の代用ではない。

## 長qと全RPEへ進める条件

1. どの資源・domainがexact、どこがproxyかを先に確定する。完全なP0互換を主張するなら、実pass列の連結閉包を示す。
2. 既存価格の意味資格と新しいstream emitterの意味資格を接続する。B6bのideal theoremは利用できても、新しい浮動小数点IR・最適化・compilerの誤差保証まで含まない。
3. 新しい長q測定が本当に必要なら、検証したい未解決項からq・本数・cut・入力・数値精度を選ぶ。q128やq4096を順番だけで自動生成しない。別の明示実行許可が必要。
4. 全law期待値には、選んだdescriptorの分布／joint-law／有限bit・標本の資格、E[max]、biasと共分散が必要。絶対価格とpaired差を別々に評価する。新fitや統計推定は別scope。
5. 全RPEでは価格だけでなくN・測定同値性・誤差配賦を合わせる。主指標や新しい価格許容誤差・energy/failure目標はユーザー/GPTが採択する。本提案ではN・全13-roundを更新しない。

## Full compile回避の見込みと限界

固定native streamの資源計数にはfull QuantumCircuitも全256入力検証も不要である。P0の特定domainが局所／圧縮interfaceで閉じれば、以後そのdomainでfull compileを省ける可能性がある。閉包が示せなければ、native構造価格または部分exact＋未保証proxyに留まる。少数のfull compileを限定oracleとして残す選択肢も排除しない。

価格計数のたびに全入力simulationを繰り返さない設計にできるが、新builderや最適化の測定同値性を無検査で認定してよいという意味ではない。新しいcheckerや精度条件を使う場合は、その範囲を別に固定する。

## STOP条件と担当

- SOURCE/input/order/scalar/identity/χ、actual inverse、fair baseline、P0 policyが不整合なら依存評価を停止する。
- chunkingにより価格が変わる／P0 oracleと不一致なら、そのdomainのexact主張を停止し原因を保存する。失敗閾値を緩めず、結果依存でfixture/cutを変えない。
- 既存mandatory STOPを再開せず、承認された新scopeだけを実施する。rawやprivate payloadをGitHubへ出さない。
- **今回のCodexは設計成果の公開照合後mandatory STOP。次の担当ユーザー/GPTは、まずT0の技術実装、続いて必要なT1/H0の限定実行を判断する。** 詳細科学レビュー、q追加、N、全RPEへ自動進行しない。
