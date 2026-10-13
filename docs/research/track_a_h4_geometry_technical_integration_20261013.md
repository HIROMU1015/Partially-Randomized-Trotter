# H4 geometry：回路調査・全保存値解析・原稿統合

2026-10-13。資料入口。元結果 `7c4483378047b57f514c47dad98905e3f2a9a756`、
B0+I診断 `c81f7326dde3f40686b3caefe05e2f544d8d1826` を保持した専用branchでの技術的事後解析。
科学SOURCEは `d0726e946f89f8c30a08d383fb8ff48b363629ec`、seed/input identity SOURCEは
`5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。新しい科学計算・compiler実行はない。

**登録B0/B1/B2/B3の点最小は全1,812点でB2のまま。B0+Iは登録外の未測定対照として分離する。**
今回の利用者指示に従って前回保留したA〜Eを完成し、科学的採否はGPTへ戻す。
前回の「主要claimに影響し得るため統合を保留」は当時の履歴として保存している。

- [公開データ・再生成・検証の入口](../../artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/README.md)
- [更新通し原稿v0.3](../manuscripts/track_a_resource_study_geometry_native_v0_3.md)
- [比較条件対応表](track_a_resource_comparison_conditions_20261013.md)
- [GPTへ返す判断事項と追加実測案](../../artifacts/resource_applicability/track_a_h4_geometry_technical_integration/2026-10-13/SCIENTIFIC_DECISIONS.md)
- [元結果](track_a_h4_geometry_results_20261013.md)／[先行B0+I診断](track_a_h4_geometry_scalar_diagnostic_20261013.md)

## A. B0+Iの技術的差分

### 定数項・位相・wrapper

固定sourceの `h4_geometry/signal.py::_prepare` はone-bodyと保持DFのI/Z/ZZを保持する。
B0はtail全体をdiscardする定義であり、tailのIもdiscardされる。B2/B3はtail Iをconstantへ移し、
tailの非Iだけをrandomizeする。これはB0の定義の違いであり、B0の実装ミスと認定しない。

DF fragment `lambda*(sum eta_i n_i)^2` のidentity係数は

`c_L = lambda/2 * sum(eta_i^2) + lambda/2 * sum_{i<j}(eta_i eta_j)`

`= lambda/4 * [(Tr G_L)^2 + Tr(G_L^2)]`。

`c_tail(p)=sum_{L>=p} c_L` とする。先行診断の下三角Hermitian化（`eigh`既定UPLO=Lと整合）と
6入力byte SHA／保存120 signalの照合を再利用した。新しいeighや入力array読込は0。
微小非対称な保存行列に対して上三角や平均化を代用しない。

B0+Iを別対照として作る最小差分は、B0準備のコピーに対して
`constant_I = constant_B0 + c_tail` とすること。元B0のmethod・結果・FPは変更しない。
`circuits.py::build_evolution` は各q stepの冒頭で既に `p(-delta*constant, ancilla)` を置く。
この角度へ加算すれば、**この段階のQiskit instruction数は増えない**。
各stepへ別の位相命令を挿入する実装ならq命令増え、数学的に一つへまとめる実装なら1命令増える。
いずれも最終compiled RZ差を意味しない。

evolution中のancilla scalar phaseはsystem Gaussian basis変更、ancilla-Zを保存するcontrolled
diagonal gatesと可換で、q stepの加算は `-q*delta*c_tail=-T*c_tail`。
wrapperの端点Hadamardを越えて無条件に移動できない。control branch間の相対位相であるため、
全wrapperのglobal phaseとして捨てられない。
cosine wrapperはH→E→H、sine wrapperはH→E→Sdg→Hで、両軸とも位相を保持する。
保存値の複素signalは `z_I=exp(-i*T*c_tail)*z_B0` となり、B=1のまま。
これを元の `exp(-i*T*E)` targetと比較したaxis biasだけが補助対照で変わる。
元targetのstate/reference契約や数値保証をこの操作で強化したわけではない。

### compiled RZ差をsourceだけで決められるか

固定basis rz/sx/x/cxでは位相のbasis変換、1-qubit runの再合成、角度打消し、identity削除が関係する。
局所installed Qiskit `Optimize1qGatesDecomposition.run` の静的読取は再合成があることを支持するが、
modified wrapperのpass実行履歴・最終RZ数を取得した証拠ではない。
同じ既存gateへの角度吸収によってRZ数が不変になる可能性、特別な角度の打消しによる減少、
異なるrunの再合成による増加を排除できない。**Δ=0、Δ=1、Δ>=0の実測保証はない**。
共通非負Δを用いる以下の表は感度仮定である。

sourceから証明できるのは追加pre-transpile命令数と可換性まで。
最終RZの自明な制約は `C_I,a>=0`、したがって `Δ_a>=-C_B0,a`。
全compiler pipelineに関する有用な小さい差の上下界は今回証明していない。
数学的に同じunitaryでもinstruction配置・丸め・wrapper軸によってcompiler出力が同じとは限らない。

### fingerprintと別対照

numeric canonical bytesは角度、global phase、control state、instruction順、axisを含む。
constantの変更で数値FPが通常変わり、たとえpre-transpile命令数が同じでも旧B0のcompiled metricを
新実測として流用できない。角度が特殊値で最終unitaryが等価でも、canonical表現やsource/candidate
identityの一致を個別に証明する必要がある。
正式に追加する場合は新しい対照ID・template/semantics/source/plan bindingが必要で、旧B0を
上書きしたり旧compile cacheを無条件に再利用したりしない。今回、実対照のsource実装やgrant発行は0。

## B. 397点と追加費用の感度

詳細は `scalar_classification_1812.csv`、geometry・補助prefix/q・B2構成ごとの集約は
`scalar_reversal_groups.csv`、grid上の切替は `scalar_comparison_switches.csv`。
これは同じ保存結果から作った302精度点であり、397独立実験ではない。

|距離 Å|Δ=0で補助対照が低い点数|該当epsilonのgrid範囲|
|---|---:|---|
|0.70|178|0.017247616〜0.100000000|
|0.80|103|0.023505464〜0.064443967|
|0.90|29|0.026763740〜0.035397800|
|1.10|0|なし|
|1.40|87|0.042793207〜0.100000000|
|1.60|0|なし|

補助点最小はprefix5/q1が264点、prefix4/q1が46点、prefix3/q1が87点。
固定zero-Δ候補の逆転閾値は10.60068〜3228.61286 RZ/axis-shot。
最小は0.90 Å・epsilon=0.02676374、p5/q1、参考gap=888241.5625 RZ work。
最大は0.70 Å・epsilon=0.06316969、p5/q1、参考gap=34536471.8125。
極値24行を `scalar_threshold_extremes.csv` に収録した。

候補j、axis別の未知差に対して

`G_I,j = N_I,j,c*(C_B0,j,c+Δ_j,c) + N_I,j,s*(C_B0,j,s+Δ_j,s)`。

B2点最小より低い条件は `N_c*Δ_j,c + N_s*Δ_j,s < G_B2 - G_I,j(0)`。
同じ追加費用Δを全候補・全geometry・両軸に仮定すると、固定候補の閾値は
`(G_B2-G_I,j(0))/(N_c+N_s)`。候補を選び直す場合は、この閾値の**全適格補助候補に対する最大**が
存在条件を与える。zero-Δ点最小の候補が最大閾値の候補とは限らない。

|仮定共通Δ|補助候補を選び直す点数|zero-Δ候補を固定する点数|
|---:|---:|---:|
|0 / 1 / 10|397|397|
|100|377|377|
|200|358|358|
|500|294|294|
|1000|210|210|
|2000|96|94|
|3000|23|21|
|5000 / 10000|0|0|

実測Δは候補・axis・geometryで異なり得る。この表をcompiled費用の予測や採用済み比較に読み替えない。

## C. 全1,308候補・6距離×302精度点

H4 linear/STO-3G、固定DF12、8 system+ancilla、sector36、T=.8、delta=.8/q。
S2、prefix0/3/4/5/6/9/12、q1/2/4/8、r1/2/4/8、K2/4の登録集合とPM-1補完を保持。
各距離218候補、stochastic194×32 paired trajectories＋deterministic24×1。
登録全1308×302=395016回の保存スカラー評価。
列ごとにmethod適格件数・最小ID/費用、全点最小のprefix/q/r/K、bias/B、headroom、shots、
axis平均RZ、paired covariance/SE、deterministic比を `precision_grid_1812.csv` に保存する。
空欄はそのmethodが不適格であり、費用0ではない。

shot会計は `h_a=epsilon/sqrt(2)-bias_a`、`N_a=ceil(2*B^2/h_a^2*log(80))`、h<=0なら不適格。
H4元契約にはH6のような明示uを追加していない。binary64保存値を元の演算順で用い、厳密certificateを主張しない。
P=0、`G=N_c*C_c+N_s*C_s=N_total*C_effective`。
`C_effective=(N_c*C_c+N_s*C_s)/N_total` なのでaxis配分が変われば同じ回路候補でも値が変わる。

全1,812点の点最小はB2/prefix3。q1は937点、q2は670点、q4は205点。
geometryごとの連続grid領域は計15、精度境界は隣接grid点のbracketであり連続epsilonの厳密根ではない。
各領域の最初/最後/次epsilonと件数は `configuration_switches.csv` に収録した。

epsilon=.05での元登録集合の点最小（B0+Iは入れない）：

|距離 Å|B2 q/r/K|N total|effective RZ/shot|G RZ work|best deterministic比|
|---|---|---:|---:|---:|---:|
|0.70|2/4/2|17187|12015.50|206510351.81|0.721428|
|0.80|1/4/4|28102|6096.92|171335623.31|0.658836|
|0.90|1/4/4|22921|5848.92|134062992.53|0.577896|
|1.10|1/4/2|19252|5864.70|112907239.50|0.573120|
|1.40|1/8/4|16474|6248.29|102934380.50|0.643758|
|1.60|1/8/4|15981|6087.06|97277315.91|0.679176|

deterministic比もshot比×effective-RZ比へ分解した。
geometry別に選び直した点最小の変化を、同じ構成の変化とは扱わない。
`fixed_configuration_factors.csv` は明示した6構成×3精度×6geometryの108行で、各軸と隣距離の比を追跡する。
代表的にp3/q1/r4/K4、epsilon=.05の0.80→0.90 Åでは
`G比=.78245837 = N比 .81563590 * C_effective比 .95932311`。
B²比は1.00147761で増える一方、cos biasが.01312847→.00947307へ減りheadroomが広がる。
この保存値上のshot減少が総費用の低下の主な算術要因である。分子相関・個別DF項の物理的原因を
新たに確証したわけではなく、geometry→bias→headroomの因果機構は別科学的検討が必要。

## D. Taylor次数・選択・不確かさ

### 保存記録で何が分かるか

固定execution sourceはeventsをtrajectory構築時に生成するが、wire/record/ledgerにはtrajectory seed、
numeric FP、paired metricsなどを保存し、次数列・component列を保存しない。
元全record照合を再利用し、今回はファイル名をstreaming集計、最大24 record×4KiBのschema読取だけで補強した。
`event_metadata_audit_v1.json` は74784 record、1308 signal、119077 ledgerのfile-family数とsample SHAを記録する。
source payloadの調査とsampleを根拠に、次数観測件数・非zeroを含んだtrajectory数・component観測は
**NOT_SAVED**。K2でorder4は **OUTSIDE_CUTOFF**。NOT_SAVEDを0回観測に置き換えない。
seedからの再sampling、numeric FPからの逆算、新trajectory生成は行わない。

### 法則から計算できる確率

R=q*r、w_n=tau^n/n!*sqrt(1+(tau/(n+1))^2)、n=0,2,...,K、b=sum w_n、B=b^R。
保存Bから単調なb(tau)を数値逆算し、`p_n=w_n/b` を算出する。丸めたBによる近似的な法則診断であり
新たな観測・原tauのbit完全復元・costの母平均上界ではない。
trajectoryが非zeroを含む確率は `1-p0^R`、32本で全zeroの確率は `p0^(32R)`、
次数nが未観測の確率は `(1-p_n)^(32R)`、期待draw数は `32R*p_n`。
1164 stochastic候補について `rare_order_law_diagnostics.csv` に保存cost平均/SEを対応づけた。
そのうち172候補は保存Bが1へ丸められており、tau/次数確率を一意に復元できないため
ROUNDED_TO_ONE_UNRESOLVEDとし、確率を空欄にする。残り992候補に法則逆算を適用した。
Bが1に近い場合のb−1は安定式とexpm1を使い、桁落ちを避ける。
これは無条件法則の確率であり、winner選択後の条件付き確率ではない。

epsilon=.05の選択構成で、32本全zeroの無条件確率は距離順に
65.79%,43.55%,43.04%,41.06%,60.24%,57.24%。
K4を選んだ.80/.90/1.40/1.60 Åでorder4の期待総draw数は
.0009027/.0009286/.0001674/.0002029、order4未観測確率は99.9098%〜99.9833%。
**今回実際にorder4が観測されたかどうかは未保存なので不明**。
希少componentの確率にはcomponent係数・抽選記録が必要で、Bだけでは復元できない。

### SEとwinner

同じtrajectoryのcos/sinをpairedとし、標本数は32。
`SE(G)^2=(N_c^2*s_c^2+2*N_c*N_s*s_cs+N_s^2*s_s^2)/32`。
paired covarianceを落とさず、`nearby_candidates.csv` に±2SE区間が重なる全近接候補の費用・SEを保存する。
全1812点に重なる候補があり、B2内のK/rの正式winnerは認定しない。
一方、登録deterministic点最小に対しB2の `G+2SE` が低いという記述的条件は全1812点で成立する。
この二つは異なる比較である。ただし±2SEは希少事象未観測・多重比較・選択後の楽観性を補正した
confidence intervalではない。同じ32本で候補選抜と費用評価をしており、母平均優位の保証にはならない。
未観測次数の母平均寄与はcomponent別costも未保存なのでcertifyできない。
H6条件付きpilotの0.00734%は別geometry/source/lawの局所診断であり、H4へ流用していない。

## E. 原稿・比較条件と返す判断

v0.3は固定H6 v0.2を基礎にH4 geometry/anchorを統合した通し草案。
図1は元登録method、図2は元点最小と構成、図3はshot/回路要因、図4は未測定B0+I感度。
後者を元最良値表へ混ぜない。H4/H6/H8でDF、PF次数、control、Gaussian lowering、compiler、u、標本数を
対応表へ分けた。H8 feasibility20 compileの固定資料と、後日のone-shot core承認済み資料は区別し、
H8最終shot-map結果をこの作業へ取り込んでいない。H8 worktree・grant・実行への操作は0。

GPTへ返す論点はB0+Iの正式比較採否、元登録集合に限定したclaimの範囲、
compiled費用の最小測定を6/10/240 wrapperのどの範囲で実施するか、H4の独立確認・希少次数診断の必要性、
比較条件が異なる系列の提示方法。新実測を判断前提として要求して本作業を止めたわけではない。

**技術的事実**：元集合点最小、全grid算術、回路のphase位置、FP変更条件、event schema、保存cost SE。
**条件付き感度**：共通Δ表、Bから逆算した次数確率。
**未測定**：B0+Iのcompiled RZ、未観測成分の母平均寄与、旧compilerとの完全同一性。
**未採択**：新baseline・比較集合変更・主要claim変更・系サイズscaling。

14限定人工算術検査と公開bundleの別算術照合を実施した。結果とSHAはbundle内のverificationを参照。
local解析の合格でありGPT詳細科学review・CI・外部再現ではない。
軽量資料だけをcommit、認証可能ならnon-force pushと公開byte照合。
認証できない場合は設定を変えず手動pushを報告し、公開確認は未完了と記す。最終報告後mandatory STOP。
