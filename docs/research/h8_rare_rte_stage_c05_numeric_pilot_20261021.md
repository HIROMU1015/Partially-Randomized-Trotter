# H8希少RTE C-0.5：数値規約候補とvariance/load pilot契約

2026-10-21。**PROPOSED / 未承認**。元のH8確率・費用・SE・逆転境界を新規評価していない。C-0の[標本設計](h8_rare_rte_stage_c0_sampling_inference_20261021.md)を維持し、未確定な数値処理とpilotの選択肢を具体化した。

## 一意に定義した数値候補

候補ID：`EXACT_INTEGER_CONDITIONING_OF_STORED_BINARY64_V1_PROPOSED`。

1. 元SOURCEの`make_rte_config`が返すK4の保存binary64次数確率p0,p2,p4を、各floatの厳密な整数比として解釈する。三つの共通分母を消して非負整数w0,w2,w4へ変換し、共通GCDだけを除く。これは原τからcanonical pを新しく再導出する操作ではない。
2. 全count cellに`C(R,n4) C(R−n4,n2) w0^n0 w2^n2 w4^n4`という整数weightを割り当てる。層支持の全正cellを保持する。binary64への再変換、log/exp、少数countへの固定、support打切りはしない。
3. PCG64のuint64 raw wordから下位bitを順番に取り出す。必要bit数ごとに新しいwordを消費し、余った上位bitは次回へ持ち越さない。`getrandbits((total−1).bit_length())`の値がtotal以上なら整数抽選をやり直す。これはカテゴリ内部の整数変換であり、trajectoryの科学call失敗を再試行する処理ではない。
4. 整数CDFをbinary searchで抽選する。count後の位置はpartial Fisher–Yatesで全R位置へ一様配置する。S2_MULTIは全N2≥2、S4は全N4≥1・N2任意を保持する。
5. componentは元`_validate_components`と`math.fsum`相当のtotalを使い、元SOURCEと同じ`component.probability/total`というbinary64除算を先に行う。その結果を厳密整数比として正規化し、rotation index→k productsを独立抽選する。元`_make_event`を呼び、event係数・normalization・phase・identity/scalar・適用順・angleを変更しない。

K4の正の次数確率やcomponentが入力表ですでに0なら拒否する。有限の正floatを整数へ変換した**後**の巨大整数累乗では浮動小数underflowを起こさない。これはcanonical法則をfloat化する前の失われた支持まで回復する主張ではない。元nativeの命令数guard超過も費用0ではなく失敗STOPとする。

count/CDF表とcomponent表を承認後の親processで一度作り、workerへread-only COW共有する。整数表の生成も実H8では科学処理に含め、C-1予算へ予約する。C-0.5で作ったのはtoy表だけである。

## 段階B・旧samplerとの相違と採択gate

段階Bは保存τにおけるcanonical有限lawをDecimal100桁で評価し、保存categorical pの和を使ったsimplex丸め診断を別保存している。本候補の厳密な抽出対象は、**保存binary64 tableを有理数として正規化した有限カテゴリモデル**であり、canonical pそのものの厳密実数モデルではない。componentの浮動小数除算後の表も同様である。

PCG64は維持するが、元NumPy `Generator.choice`とbitの消費・カテゴリ変換が異なるので旧event列のbit再現は主張しない。新namespaceの新cohortとして扱う。量子アルゴリズムの無条件法則・係数を差し替える案ではなく、古典的な層別費用取得に使う数値実装候補である。しかし、この候補とcanonical期待費用の対応を無断で正式採択しない。

将来、別途承認した`law_compatibility` callで、Bの保存canonical p文字列と、有理数正規化した保存pとの差を照合する実装を用意した。候補のengineering gateは既存Bに合わせた絶対`1e-12`。これは数値整合の受入候補で、費用biasの上界、丸めinterval certificate、実PRNG bit法則の一致ではない。未知費用が大きければ小さい確率差の費用への影響を無条件には制限できない。

C-1でこの候補を採用するか、canonical高精度法則を別方式で直接抽選する案を求めるかはユーザー判断。本実装で候補ID・正規化・bit変換は一つに固定したが、`numeric_model_approved=false`を保持した。**実H8のcount/差分計算が必要になる地点からは別科学承認が必要**であり、今回は行わない。

## pilotの具体的選択肢

|選択|取得scope|得られる情報と制限|
|---|---|---|
|A：最小情報候補|各条件S2_SINGLE|主要な未観測層の分散・loadを取得。multi/S4の平均・分散は未知のまま|
|B：残余も探索|各条件single/multi/S4|全3層の条件付き標本分散・loadを取得。有限pilotだけで厳密全support保証にはならない|
|静的証拠を先に補う|未取得層の妥当boundを別途調査|適用できるboundが成立すればcompile削減候補。現Tier Rの全閉包未確認を解消済みとはしない|

元q4/q8 pilotは実装・履歴の参考で、p8q2・p7q16の分散として代入しない。原P n48のS0は別anchorとして維持する。

分散未取得なのでpilot nは最終固定していない。**分散推定のためn>1は必要だが、それだけで十分な精度とはしない。** 選択可能な計画は、仮に正規モデルを置いた場合のsample varianceのchi-square区間を使い、希望する相対分散精度η（候補1/2または1/4）と計画信頼水準（候補90%または95%）から、区間比/幅の事前基準を満たす最小整数nを決めるもの。式は`[(n−1)s²/χ²_(1−α/2,n−1), (n−1)s²/χ²_(α/2,n−1)]`である。このモデル仮定・η・基準を承認せずnを決めない。重い尾・非正規性では精度保証にならず、pilotは診断情報として終える。

productionの候補は信頼水準95%または99%、G半幅は`|保存PF費用−保存P点費用|/4`または`/2`。これは目標の具体案で、実際のH8値を計算したり、順位に都合のよい最終値を採択したりしていない。厳しい半幅/高信頼水準ほど必要情報が増え、S0誤差で目標達成が制約され得る。

pilotで得た分散・compile負荷を用い、S0項と残余の不確かさを含むvariance予算からproduction nを設計する。cost-aware allocationはC-0の`n_s∝π_s σ_s/√c_s`案を維持。pilotを見た後でNを固定するため、primary production推定には新cohortを使いpilotをpoolしない。pilotが完了したらSTOPし、productionは別契約・新grant・明示承認へ分ける。信号の符号が分かるまで逐次増量する規則は採らない。

## 条件別call予算を凍結する式

Nは承認した層ごとの本数の合計、Lは取得層数、Vは事前登録したvalidation代表task数（候補は取得層ごと1本）。科学予算は資源上限と区別する。

|call|p8/q2/r4|p7/q16/r4|
|---|---|---|
|saved input/preparation decode|1|1|
|純粋prefix復元|p8を1|p7を1|
|原law形成 / law差分診断|各1|各1|
|count表 / component表|L / 1|L / 1|
|trajectory|N|N|
|occurrence / RTE event|2N / 8N|16N / 64N|
|native evolution|N|N|
|測定full-wrapper build|2N|2N|
|実compile|cache miss最大2N|cache miss最大2N|
|control probe|4V（同じ4V statevector simulationを含む）|4V（同左）|
|event oracle / 非測定validation wrapper|V / 2V|V / 2V|
|pilot数値会計 / 独立moment検査|各1|各1|
|新signal / SCF / DF / state solver / Gaussian生成|0|0|

control/oracleは元関数を再利用するため実simulationを含む。compile承認だけから自動許可せず、V・validation task IDと共に明示承認する。予算はcall前に共有lock下で非返還予約し、cache hitと実compileを分離する。cacheはSessionEquivalenceLibrary等のidentity封印までデフォルト無効。

pilot会計は同event両軸の`G_i=Ncos*RZcos+Nsin*RZsin`の層別mean・variance・SE、別実装raw moments照合まで。未知層はnull、n1のvariance/SEはnull、CI・母平均順位保証・全support上界は付与しない。S0を含む期待費用/区間の新評価を行う場合はその数値解析範囲も別途承認・予約する。
