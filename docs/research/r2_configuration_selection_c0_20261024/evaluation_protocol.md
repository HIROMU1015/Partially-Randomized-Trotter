# 予測・選択法の評価protocol案

状態：DRAFT、未実行・未採択。scopeごとに候補域と取得可能情報を先に固定する。

## 問題と入出力

入力は物理条件・task・精度・固定候補集合Dと、Tier0 metadata、取得済みTier1情報である。各特徴は入力由来、取得SOURCE、cost、利用可能時点、祖先fieldを持つ。hashは識別用途に限り、性能ラベルをhashに符号化しない。

選択器はcandidate IDまたはABSTAINを返す。Tier2はquery(candidate, field)の明示的な照会だけで渡し、1候補1回のsignal取得で複数epsilonを得る場合は共有取得と下流会計を区別する。label table全体は選択器へ渡さない。

評価側のみ、保存適格性・C・N・G・cohort不確かさへアクセスする。schema上は次を分離する。

|channel|許容内容|禁止/注意|
|---|---|---|
|metadata|system/task/epsilon、formula ID、q、prefix、指定K/law、科学条件ID|reference/ground-state vector、bias、N、G、選択履歴由来のwinner flagは不可|
|features|Tier0/1の由来付き特徴|候補全signalから作ったproxyや事後有効単価は不可。別candidate/別systemの取得も校正費用に計上|
|acquisitions|照会ID、candidate/field、SOURCE、実測または未計測cost|保存値のreplayを実測した新取得費用と呼ばない|
|labels|適格性、二軸価格/shot/G、paired uncertainty、reference政策|評価側専用。Tier2の有料照会へ値を渡す場合も、取得履歴が必要|
|decision|選択ID/ABSTAIN、rule ID、query trace、proxy情報|事後のbest IDへの差替えは不可|

候補のlabelは、取得時の役割ではTier2、予測の評価時にはOracleになり得る。この役割差を宣言する。派生fieldも祖先のTierを引き継ぎ、名前変更や別ファイル保存で独立な情報にはならない。

## 費用と適格性

二軸の総量は G(d)=N_cos(d)*C_cos(d)+N_sin(d)*C_sin(d)。状態準備P=0、full Hadamard wrapper、対称方向controlという既存scopeを維持する。PRのCは有限cohortのpaired推定であり、母平均・希少tail保証ではない。

Fullのnormalization B=1という構造上の性質と、数値allowanceのlog_B_marginを区別する。SOURCEの適格性は単純なbias閾値ではなく、三状態headroom、整数shot可否、u/headroom guardを含む。P1では保存statusをそのまま使い、新しいu/N/Gや適格性を計算しない。

科学条件、Hamiltonian/状態、rank-policy、prefixの意味、q/T、law、cost scope、compiler、reference/allowance、cohortが一致した単位で評価する。厳密byte一致を科学目的にしない一方、条件が違う行を合算しない。

## baselineと公平性

|baseline|候補域・使用情報|本C0の扱い|
|---|---|---|
|固定S2|同一物理条件の登録S2/q8|P1でそのIDだけ選択。適格性は評価側で確認し、不適格を隠さない|
|固定標準Y4|同じ候補域のY4/q4|同上。既存主比較の再ランキングではなく、P1の固定参照|
|既存PF次数/q選択|Morales等のerror constant/公式長による比較、既存内部PF探索|前者のerror constantを候補全signalから無料推定しない。後者の保存最小値はOracle扱い。誤差定義が違う方法の未実装を比較勝利に数えない|
|cost-aware selection|Kiss/Hagan/SPRINT等の入力・費用/誤差モデルに対応する既知規則|law・taskが異なるため本文の保証を流用しない。実装比較が必要なら別契約で入力と取得費用を固定|
|構成長のみM0|全候補の指定q・piece数・prefixから相対長を使う|P1の低情報baseline。費用予測器・保証付き法とは呼ばない|
|有料適格性照会M1|各公式でq昇順、最初の保存ELIGIBLE候補を得て構成長で選択|既知の逐次探索型baseline。signalを見た候補はTier2取得として計上。新手法ではない|
|有限集合Oracle|全Dの評価labelから適格G最小とexact ties|評価専用。全域最適とは言わない。取得数削減法のbaselineには全Dの必要な評価量も併記|
|将来の提案法|未固定|誤差proxyの独立取得・比較可能な費用・明確な差分がないうちは作らない|

FullとPRは候補域の厚さが異なる。方法別Oracleと共通域Oracleは区別し、PRが固定prefixしかないときはpartition最適化後の比較にしない。m5_bestを標準Y4に混ぜない。

## 評価指標

|指標|定義/記録規則|
|---|---|
|選択G|適格で価格完備の選択にのみ保存Gを提示。不適格・欠測・ABSTAINはnull、0費用扱い不可|
|regret|同一Dの適格Oracle Gとの差、およびG_selected/G_oracle-1。将来別承認で計算。相対量は正のOracleがある場合だけ|
|方向・順位|正解最良tie集合への選択一致、事前固定したpairの一致。全順位を返さない選択器へ架空の順位精度を付けない|
|適格性の誤り|ELIGIBLEとの主張と保存statusの照合。誤受理、誤棄却、未確定、保留率を別々に提示。未確定を陰性に潰さない|
|取得量|signal/prepare/reference/compile軸/PR trajectoryとsampling event数。既存情報の取得費用、校正費用、最終選択候補の費用も記録|
|古典費用|選択CPU/wall/RSSとfeature/校正の実測。保存値replayのwallを元signal/compileの費用にしない|
|失敗損失|有効選択のexcess G、無効選択の取得浪費、ABSTAIN後のfallback費用。成功したsubsetだけを残さない|
|移送性|新system/geometry/DF構造/candidate域を事前固定。epsilon再会計だけなら精度移送であり新物理sampleではない|
|PR不確かさ|paired cos/sin、cohort別保存mean/SE/nを保持。再標本化・新CI・poolingは別承認。母平均のOracle最良は未確定とする|

選択Gと古典取得費用を勝手に合算しない。必要なら将来、利用回数Rに対し「校正/取得古典費用」「R回の量子G」「失敗/fallback費用」を別成分で提示する。交換レートとRを固定するまで合成スコアは作らない。

## 検証分離

H2/H4/H6の既存結果は全てdevelopment。新標準Y4の独立cohortは第1研究の固定費用方向に対する独立取得であって、本研究の仮説形成後のblind testではない。旧PR-2のheld-out条件も本研究へ自動移管しない。

epsilonを跨ぐ同じcandidate・trajectoryは同じcluster、geometryを跨ぐ同じHamiltonian生成履歴も漏洩確認対象とする。model fitを行う将来段階ではsystem/geometry/Hamiltonian単位で分割し、分子数が少ない場合の不確かさを隠さない。真の未使用性が証明できなければretrospectiveと表示する。

独立testへ進むには、開発上の有用性と先行手法との差分を先に示す必要がある。現C0では詳細科学レビュー・held-out計算・科学実行を開始しない。
