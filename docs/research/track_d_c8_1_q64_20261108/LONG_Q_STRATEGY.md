# 長q価格の検証戦略（C8-1数値preflight不成立後）

2026-11-08 JST。q64 sample00はraw PASS・4 P0 compile完了後、両方式X/Yの全4 postcheckがCAR-observable Frobenius許容値2e−10で不成立となった。残7 pathは未開始、n8価格/モデル診断は未取得。[最終報告](REPORT.md)。8a6bbdacに保存した実行中draftを履歴として保持し、本版ではこの観測を追記する。次の科学実行を採択・開始するものではない。q≥128の新sampling/回路/compile、fit、全RPE総費用更新は行わない。既存[B/C固定結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md)とC7-2/C7-4の構造・数値証拠に基づく技術計画であり、詳細科学レビューではない。

## 今回の結果から先に必要となる最小診断

今は下記の長q H4実行案へ進む前に、失敗したFrobenius値と誤差の位置を取得できる数値診断が必要である。4 postcheckとも同じ例外だが、途中check listとcompiled observableは例外時に保存されておらず、超過量や個別誤差の配賦は分からない。既存原本の高精度再計算・再postcheckを今回追加していない。

|次の候補|新H4 sampling/compile|追加科学評価の範囲案|得られるもの・限界|
|---|---:|---|---|
|失敗診断の保存処理を新revisionへ追加し人工テスト|0/0|H4評価0。既存checkerの数値・ラベル・閾値・失敗値/途中records保存を人工fail例で検証|失敗値欠落を防ぐ。H4精度改善を示すものではない|
|保存sample00の限定診断を別途承認|0/0|同じraw reference/compiled QPY hashの4 wrapperを各一回、全256入力・同閾値で再評価し、例外前にobservable/残差を保存|新SOURCE・新namespaceの診断実行として最大4 post評価。旧FAILを置換せず、成功を求めて再試行しない|
|原因候補ごとの人工安定性比較|0/0|局所gate/declared phase/actual inverse/CAR積/observable集約の既存部品を用い、同じ参照と許容値で比較|独立な数値参照と誤差の意味が必要。C7-4のnative/generic同値だけを改善済みとしない|

上記のコード・評価は未実装・未実行で、ユーザー/GPTが新しい目的・SOURCE・call範囲を採択する。旧科学runのretry/resumeとして自動実行しない。計算precision/実装を変更する場合は別revisionと科学条件の変更範囲を明示し、phase fit/renormalization/列削減/閾値緩和でPASSを作らない。

数値gateが資格付けできるまで、q128以上のfull-wrapper compileを次の直接実行候補として採択しない。一方、科学call0のCX構造条件やdepthの事前feature/compiled-DAG bridgeの静的整理は、数値verifierの改善と独立に進める選択肢がある。下記A/B/Cはその後の判断材料であり、C8-1不成立によって採択済みになったものではない。

## 判断対象と検証地点

今回の中心指標はCX countとtotal depth。既存主Aでの削減率は11.17%/54.65%だが、総資源の約99.5%は長q proxy。依頼に示された保存主A total-depth差へのq1024/2048/4096の寄与は約9.62%/22.18%/59.46%。これを新たに再集計したり、q64観測で旧費用を更新していない。

したがってq64は新domainへの最初の移送・数値gateを調べる点であり、最終的に重要なのはq4096付近の価格biasと差の符号である。q64 PASSだけでq128を次の地点に自動採択せず、**高寄与のq1024またはq4096付近で何を直接確かめ、残りをどの構造・誤差証拠でつなぐか**を決める。

## 三つの方法と必要作業

以下の本数は追加実行を見積もるための探索的案であり、保証付き必要標本数ではない。qや本数の採択・新SOURCE固定・実行許可が必要。

|方法|q4096までの役割|最小追加H4案とfull-wrapper数|負担・不足|
|---|---|---|---|
|A：少数の独立長q直接compile|大きいqの全体P0/符号・予測の反例を直接得る|候補q1024の固定2 path→8 wrappers。まず同じcohortのsample00/4 wrapper数値gate。q4096も必要なら別scopeの固定2 path→8 wrappers|n2は数値/構造pilotのみ。母平均精度保証には不足。q1024はq64の16倍、q4096は64倍のstep数で、gate処理量・保存量が増える。wall timeやcompile次数は未測定|
|B：構造的価格モデル|全qを覆うCXの会計/残差包絡、depthのmax-plus伝播候補|人工/静的閉包なら新H4 sampling/compile0。H4 bridgeが不可欠なら独立q64と上記の少数長qを再利用|局所価格和/P0文脈非加法性・Gaussian/zero-angle閉包が不足。depthのshell+max(lane)は既存反例がある|
|C：event/run特徴と限定calibration|安価な古典特徴からpaired価格・lane揺らぎを推定し、長qの実compileを絞る|新calibrationが必要なら別に固定例q256/2 path→8 wrappers＋未使用q1024/2 path→8 holdout wrappers。既存係数を凍結維持する案なら新fit0|新fitは今回未許可。feature lawと価格biasの区別、calibration/holdoutの分離、E[max]が必要。q4096までの移送保証には追加根拠|

どの方法でもnew lawが同じこと、固定cut/P0/strong融合/位相、全256入力correctnessを維持する。n2は最小の技術pilot候補であり、平均価格を確定する本数ではない。必要な母平均精度が決まれば、独立性・variance/tail/biasの根拠に応じて別に標本数を設計する。観測SD0、観測min/max、PRNG seed分離を母分散0・全support・IIDの証明にしない。

## 推奨する分岐

**q64が数値PASSし、価格も移送と整合する場合**：次候補の第一比較は、q128を順番に増やす案と、寄与の大きいq1024へ限定pilotを置く案。後者が結論への情報量を大きくし得る。SOURCE/数値負担が実行可能と事前に確認できる場合に、q1024の固定少数pathを候補とする。q256/512を使うなら「順番だから」ではなく、q1024の正確な検証が現実的でない理由と中間点で切り分けられるbias/数値機構を明示する。

**q64で価格予測や整数候補に反例が出てcorrectnessはPASSする場合**：反例とbasis/sign/support/cut/DAGを保全する。今回のA/F/S係数を改変しない。次に反例の構造的原因を説明する限定calibrationか、高寄与qで反例の移送を調べるかを選び、同じq64を修正モデルの独立holdoutと呼ばない。

**q64が固定数値精度でSTOPする場合**：同じverifierを長qへそのまま拡張する案を先に進めない。局所gate/位相・CAR積・A/B/actual inverse・native propagation・256列observable集約のどこが寄与したか、保存証拠で候補を絞る。新しい人工高精度参照/安定集約/実装経路の比較を別scopeで検証し、固定閾値での同等性と独立な参照を確かめる。保存配列のextended subtractionだけでは前段gate適用誤差を回復できない。C7-4 native/generic比較は精度改善を示さなかったため、改善済みverifierとして自動採択しない。新SOURCEでのH4 retryはユーザー/GPTの明示判断後のみ。

## CXとdepthで必要な構造証拠は異なる

CX候補208−224qの全q移送には、Gaussian basis/relative transition、directional diagonalのnonzero/signed/zero-angle、scalar集約、CSWAP/global-cut、head/tail/step境界の各テンプレートが全pathで同じP0会計へ閉じる条件が必要。C7-2の局所反例を無視して節約値を単純加算しない。候補が有限全pathで一致しても、未検証compiler文脈の残差を0と認定しない。可能なら固定費＋step費＋境界費＋P0残差という明示会計にし、残差を全q/全lawへどう制御するかを別に示す。

depthはwire依存を持つ。compiled DAG/max-plus wire finishは保存された特定回路の価格を正しく再記述できるが、未来の全体compileを省略する事前モデルにはなっていない。global P0が境界をまたぐと、局所compiled blockを連結したdepthとfull-wrapper depthは異なり得る。未来へ使うには実際のcompiler境界作用、wire-timeの伝播、左右lane/cut/CSWAPの依存と、対応するbiasを制御する必要がある。

event/run特徴はO(qr)の古典処理で比較的安価だが、左右laneのmaxはpathごとに取る。max(E[A],E[B])をE[max(A,B)]の代わりにしない。q64の8本のfeature分布は全lawのfeature分布ではない。feature母平均やjoint lawを推定する追加samplingも別の科学callとして定める。

## 符号判定と削減率定量化の精度

既存点推定をΔĜ<0、将来のpaired価格bias/推定誤差を各round・axisでεΔ,m,bと置き、一様または適切な期待値の意味を資格付けできる場合、符号を支える十分条件の候補は

\[
\sum_{m,b}N_{m,b}\,\epsilon_{\Delta,m,b}<-\Delta\widehat G.
\]

これは設計用の条件式で、今回新しい数値boundや総費用を算定したものではない。方式別絶対誤差しかなければ、εΔ≤εRef+εSymという保守的な候補になる。共通biasがpairedで相殺することは、別途のjoint証拠なしに仮定しない。

削減率R=1−GRef/GSymの定量化には、差だけでなく分母の絶対価格も必要。|GSym−ĜSym|≤εS、|GRef−ĜRef|≤εR、ĜSym>εSなら、候補として

\[
|R-\widehat R|\le\frac{\epsilon_R+(\widehat G_R/\widehat G_S)\epsilon_S}{\widehat G_S-\epsilon_S}.
\]

paired差の符号が安定でも、削減率が同じ精度で定量化できるとは限らない。RZ約1.1%やsize約2.6%の小さな差と、CX約11%/depth約55%の差で必要精度は異なる。今回の優先CX/total depthをまず扱い、他資源の符号や保証まで一律に推定しない。classical confidence/familyと標本数はこの必要精度・law/tail/biasの証拠を踏まえユーザー/GPTが採択する。既存5%幅を新しい保証基準へ格上げしない。

## 共通Nと全13-roundへの再接続

新しい長q価格を得てもB4-2の保存ideal十分Nとactual Nは別。方式別実装誤差/law条件が違えば同じNが使えるとは限らず、actualへ進む際はB5/B6bの資格を別に扱う。長q価格が整えば、今回凍結したB/C全13-round結果とは別versionとして、価格入力・共分散・bias/law資格を明示した再会計を別科学実行で承認する。

新joint共分散が得られない項を0にせず、同じモデル係数のround間依存とdirect/calibration/holdoutの共有を保持する。長q検証の結果だけで既存A/F/S総費用を自動上書きしない。

**次の担当：ユーザー/GPT**。C8-1の数値/価格結果を踏まえ、安定verifierの技術検証、q1024等の独立pilot、構造bridge、feature calibrationのどれが最小で有効かと、具体的な新SOURCE/標本数/compile数/精度・STOPを判断する。**Codex**は今回の公開照合後mandatory STOPし、提案scopeを自動開始しない。
