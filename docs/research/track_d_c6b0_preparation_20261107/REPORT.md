# Track D C6b-0：paired価格モデルの検証条件と追加校正の準備

2026-11-07 JST。**最小案は、6資源すべてのpaired差を一次評価family（X/Yを含む12量）の候補とし、方式別絶対価格24量の精度確認を別目的に分けること。追加q2/q4 calibrationを一律に要求せず、目的・幅・信頼度・統計資格を採択した後、独立q8 whole-pathだけを次の科学domainにする。** ただし、現在の資料は確認的な必要標本数を正当化していない。32本は探索枠の候補であり、保証数ではない。確認的なモデルPASSを必要とする場合は、統計条件と必要数を正当化する作業を先に許可する必要がある。

本書と[実行前契約案](C6B_EXECUTION_DRAFT.md)は静的な技術資料であり、研究条件の採択・新しい科学実行許可ではない。新sampling、保存数値の再解析・fit・共分散再計算・新予測、回路生成・compile・postcheck、科学アルゴリズム変更、N算定、全RPE会計、B6介入、詳細科学レビューはすべて未実施。数値は指定commitに保存された値の転記・判定欄の分類だけである。

## 1. 固定条件・取得した資料・保全

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、各h=δ=.1。元Hamiltonian Hの厳密groundを仮定し、状態準備費を除外する。endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、元H packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`、signed atoms・scalar/identity・normalizationを維持する。

P0はQiskit1.3.0/optimization_level1/seed17/rz,sx,x,cx/approximation_degree1.0/qubits_initially_zero=false/topology-free/位相保持。費用境界は `single_hadamard_interrogation_without_state_preparation`。Referenceの追加register・CSWAP・χ・actual inverse・cut損失を含むfull wrapperで、kernel＋shellの旧serial価格へ差し替えない。RZ/CX count、各depth、total depth、sizeは個別の資源、qubitsは幅制約である。

共通・repository AGENTS.md、PROJECT_MAP.md、研究概要・現状と研究目的、正式共通指示2026-10-25（承認・保全・STOP・再利用規約）を確認した。専用branch `codex/track-d-c6b0-paired-model-preparation-20261107` はC6a公開commitから分離した。正式共通指示は共通資料で読み、未収録のGitHub URLを作らない。

| 固定commit・資料 | 取得・確認内容と資格 |
|---|---|
| C6a `e9d4e2f602d5f9e014507ca7e391719bab017b88`：[RESULT][C6A]／[提案MD][C6AMD]／[提案JSON][PLANJSON]／[paired統計][STATS]／[affine][FIT]／[SOURCE][SOURCE]／[STOP][STOP] | q2/q4各16 fresh path、raw32・postcheck128・paired X/Y/χ PASS。実行SOURCE `6b6667026ec8b670f610e41141d195883a9eb513`、planning後処理は同公開commitの[script][PLANCODE]（保存SHA256 `ea57809f2375dfaee385847a4d8c5a5557c996b1d7e161d932e09af0cd20476e`）。既存JSONを実行・再解析していない |
| C5 `9a6920b8355643701ab7436a4fd3b34862e9e3bd`：[RESULT][C5]／PLAN／SOURCE／STOP／共通入口 | 保存00→01/q2/cut1、実行SOURCE `976c898fbdb79477c1db3178b5d32df1fe5aeb5c`。強い融合条件を共通化し、7価格は旧C2と一致。有限fixture |
| C4 `a5784c4bd3e3fb9779849a1e1fc1759559e2caea`：[REPORT][C4] | full-wrapperとfenced kernelの境界、affine/boundary/lane候補、shared calibration、必要domainだけのholdout、費用・幅の扱い |
| C3完了 `cc554cf6daea08df6c20ea8e1e92405259ce49b5`：[COMPLETION_RESULT][C3]／補完PLAN／SOURCE／STOP | 保存q4/q8固定fixture、構築SOURCE `52b3cab50d5c5edcbd7e3249637168e9978be535`、補完SOURCE `88f0496349c056d0736f08487889f942c1aa4d37`。旧run01失敗／run02中断と完了証拠を区別する。既知q8は独立holdoutではない |
| B4-2 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8`：[RESULT][B42]／SOURCE／STOP | 元H・ideal finite-RTE IID・exact nativeの条件付き13-round十分shot数14738。actual Nを与えない。再算定なし |
| B5 `0a8b953aea1be20b110784960763d7f36ebccac8`：[REPORT][B5] | finite-bit law、shot/occurrence joint-law、全path/全13-roundの実装誤差不足と、共通Nに必要な条件。固定Git objectだけを読んだ。B5/B6の作業領域・実行には介入しない |
| 再利用SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`：[StrongBuilder][STRONG]／[event emitter][EMITTER]／[P0][GATES] | C6a SOURCEが指す元C側の実装。現在checkoutの同名fileを無条件に実行元としない。adapter/方向制御/phase compilerはC6aが固定した版を読む |

C6aの上記MDとSOURCE/STOP/3統計JSON/result/公開範囲はe9固定Git bytesと照合した。C3/C4/C5の関連MD・STOPも指定commitと照合した。B側資料は固定Git objectから取得した。C6aの143-file closure再監査・分子NPZ展開・QPY decode、private raw/archive読取り・追加公開はしていない。[公開範囲][PUB]とhash索引を優先する。

C6a時点でq8 holdout domainは未使用、本作業でも使用0。依頼文の「C6aの保存q8固定fixture」は、資料上はC3由来の既知fixtureに対応する。C6aは新しいq8標本を作っていない。このfixtureを新holdoutとして流用しない。

## 2. Paired差・絶対価格・予測精度の目的を分ける

同じq・law・full-wrapper条件で、資源gについて

\[
\Delta C_g(\omega)=C_{R,g}(\omega)-C_{S,g}(\omega),\quad
\mu_{a,g}=\mathbb E_{p_q}C_{a,g},\quad d_g=\mathbb E_{p_q}\Delta C_g
\]

とする。次の三つは別の評価である。

| 評価 | 推定・検証対象 | 必要な情報／この評価だけでは得られないもの |
|---|---|---|
| 同qのdirect paired比較 | 新q8標本のd_g、差の符号・大きさ | 同一pathの差とその有効な標本不確かさ。直接holdoutを推定量にするなら、affine係数のcalibration分散を足す必要はない。モデル予測を使っていないためである。ただしlaw/IID/tail資格は別に必要 |
| pairedモデル精度 | d_g(8) と保存モデル差 f_Δ,g(8) の隔たり | population校正線を評価する旧提案ではholdout誤差に校正不確かさも関わる。凍結済みモデルそのものを条件付き評価する案とは下記で分ける。差が負というだけでは精度PASSにならない |
| 各方式の絶対価格精度 | μ_R,g(8)、μ_S,g(8) と各モデル予測 | 方式別の平均・分散・校正誤差。両方式それぞれの全RPE費用を出すには将来こちらも必要。pairedモデルだけでは絶対価格は資格付けされない |

同pathの方式間covarianceが大きければ差で共通変動が相殺される。絶対価格の大きな不確かさを無条件にpaired差へ加算せず、同pathの差を保存する。一方、paired差が精密でも各方式の絶対価格が同じ精度で分かったとはしない。

6資源の価格・不確かさをすべて保存する。価格モデルの一次検定対象をpaired差に限定することは、RZ増加・幅増加・絶対価格の未資格を隠すことではない。単一の総合スコア、T count、実機実行時間へ勝手に変換しない。

## 3. C6a保存統計と共分散の確認

以下は[paired_statistics.json][STATS]の保存値の転記。X軸を代表表示し、X/Yの共有価格はC6aの有限32 pathでの観測として扱う。各qの標本単位はwhole path16本、SDはddof1。±は標本SDでありCIやSEではない。

|q|資源|symmetric 平均±SD|Reference 平均±SD|paired差 平均±SD|
|---:|---|---|---|---|
|2|rz_count|11428.1875 ± 728.716883|11766.3125 ± 729.292417|338.125 ± 1.82117178|
|2|cx_count|4296.125 ± 222.605743|4056.125 ± 222.605743|-240 ± 0|
|2|rz_depth|2777.4375 ± 158.556391|1467.0625 ± 124.109344|-1310.375 ± 72.3591736|
|2|cx_depth|2083.0625 ± 85.9336711|922.125 ± 66.5110267|-1160.9375 ± 39.6492434|
|2|total_depth|6431.4375 ± 348.060142|3254.1875 ± 271.143804|-3177.25 ± 159.53286|
|2|circuit_size|23279.0625 ± 1459.61952|23569.1875 ± 1460.20566|290.125 ± 1.82117178|
|2|peak_qubits|9 ± 0|17 ± 0|8 ± 0|
|4|rz_count|21399.125 ± 883.163698|21618.8125 ± 882.798748|219.6875 ± 2.98258836|
|4|cx_count|8152.875 ± 272.971763|7464.875 ± 272.971763|-688 ± 0|
|4|rz_depth|5176.8125 ± 225.98384|2621.25 ± 125.117278|-2555.5625 ± 157.632894|
|4|cx_depth|3974.4375 ± 123.434176|1632 ± 68.066634|-2342.4375 ± 85.158651|
|4|total_depth|12039.3125 ± 495.544108|5809.25 ± 272.291143|-6230.0625 ± 345.360385|
|4|circuit_size|43666 ± 1782.34333|43389.6875 ± 1781.98719|-276.3125 ± 2.98258836|
|4|peak_qubits|9 ± 0|17 ± 0|8 ± 0|

表示は丸めのみ。全pathでCX/depth減、RZ増、幅9→17。sizeはq2で増、q4で減という有限観測である。CX差の標本SD=0は全law分散0を示さず、観測min/maxもsupportにはならない。

保存共分散の代表欄をそのまま示す。方式間だけでなく、資源間・axis間の依存も保持する。

|保存欄（X、特記時Y）|q2|q4|
|---|---:|---:|
|Cov(S CX, R CX)|49553.316666666666|74513.58333333333|
|Cov(S RZ, S CX)|162008.50833333333|240432.28333333333|
|Cov(R total depth X, R total depth Y)|73518.9625|74142.46666666666|

`by_q/q/sample_covariance_labels` が28列を定義し、`sample_covariance_ddof=1`、`covariance_rank_limit=15`、`XY_are_not_independent_samples=true`が保存されている。行列の独立性・可逆性を仮定せず、新しい相関係数・rank・分散を計算していない。同じsample番号をq2/q4間の対応標本として扱わない。

### 探索的係数とq8不確かさ

以下は[exploratory_affine.json][FIT]のcosine=X欄の転記。式は f(q)=b q+a。sine=Yの係数も同JSONに保存されている。新しいfitや方式差係数の再算定はしない。

|資源|S: b|S: a|R: b|R: a|
|---|---:|---:|---:|---:|
|rz_count|4985.46875|1457.25|4926.25|1913.8125|
|cx_count|1928.375|439.375|1704.375|647.375|
|rz_depth|1199.6875|378.0625|577.09375|312.875|
|cx_depth|945.6875|191.6875|354.9375|212.25|
|total_depth|2803.9375|823.5625|1277.53125|699.125|
|circuit_size|10193.46875|2892.125|9910.25|3748.6875|

2つのq平均に2係数なので校正残差自由度0。残差0はモデルbiasの推定や長q成立の証拠ではない。係数の48変量共分散は[planning JSON][PLANJSON]の `conditional_parameter_covariance.labels/matrix` に保存済み。例えば Cov(b_S,CX,b_R,CX)=1938.5453125、Cov(b_S,RZ,b_S,CX)=6288.137369791666、Cov(b_S,CX,a_S,CX)=−5425.631770833334 である（全てX）。これらも保存欄の転記である。

C6aの既存planningは、q-domainの擬似独立を条件として Cov(mean_q)=S_q/16、f(8)=3 mean4−2 mean2、Cov(f(8))=(9S4+4S2)/16 を用いた。pairedには方式間covarianceを残す。式の再計算・新しいqの予測はしていない。共通係数を将来のround間で独立扱いすると不確かさを誤って縮めるため、係数共分散と校正IDを引き継ぐ。

保存された**仮想q8 planning値**の例は、paired CXがf_Δ(8)=−1584、校正分散0、名目的校正halfwidth0、tau=794.31875。paired total depthはf_Δ(8)=−12335.6875、校正分散73454.19348958333、名目的halfwidth866.4513295030238、tau=1163.753125。実q8平均や新しい予測値ではない。前者の0は有限標本と仮定したモデルから得たもので、未知q8分散0や必要数0を認定しない。

未取得項は、affine biasとq-domain適用性、q8実分散、tail/support、理想law対finite-bit実装law、PRNG/joint-IID、shared input/codeの系統的誤差、P0全supportの誤差資格である。これらを係数共分散の一部として0埋めしない。

### 検証するモデル誤差の対象も先に指定する

C6aの旧合成分散を使う案は、population平均で校正する線の残差、例えば μ8−(3μ4−2μ2) を、独立な標本平均から評価する設計に対応する。この対象にはμ2/μ4推定の不確かさが残る。

これとは別に、保存済みのfrozenな予測hat f8を固定し、**この一つの予測値がμ8にどれだけ近いか**をcalibrationに条件付けて問う設計候補がある。その場合、正当化された独立holdoutのμ8区間をhat f8だけ平行移動して誤差を判定する形を検討でき、旧calibration分散を機械的にもう一度足す対象ではない。これは新しいCIの構成・有効性証明ではなく、検証するestimand/conditioningを明記する候補である。population校正線や将来の未観測qの係数不確かさを消したことにはならない。

したがって旧20量のfloorは**旧合成不確かさ規則での計画上の制約**であり、どんなq8検証でも追加校正が必須という不可能性証明ではない。paired優先か絶対価格優先かと、このconditioningの選択は別の設計軸。どちらの命題を採るか、intervalの資格・多重性も含めてholdout前に決める。既存C6aの計算・判定を変更せず、新しい数値比較もしていない。

## 4. Calibration floorの所在：旧36量の判定を分類する

次表は[planning JSON][PLANJSON]の `per_quantity` の保存status・保存数だけを分類したもの。**alpha=.05、K=36、二側normal z=3.1969502291312533、sigma8_plan=2×SD4、幅1＋5%というC6aの未採択案を固定した診断**であり、新しいK=12/24の再判定ではない。X/Yは同じ保存判定だった。

|資源|S絶対価格|R絶対価格|paired差|
|---|---|---|---|
|rz_count|floor不成立|floor不成立|floor不成立なし／条件付き案n=32|
|cx_count|floor不成立なし／条件付き案n=41|floor不成立|floor不成立なし／条件付き案n=32|
|rz_depth|floor不成立|floor不成立|floor不成立なし／条件付き案n=32|
|cx_depth|floor不成立なし／条件付き案n=32|floor不成立|floor不成立なし／条件付き案n=32|
|total_depth|floor不成立|floor不成立|floor不成立なし／条件付き案n=32|
|circuit_size|floor不成立|floor不成立|floor不成立なし／条件付き案n=32|

20量の不成立は全て絶対価格family内で、SのRZ/RZ depth/total depth/size、Rの6資源すべてに属する（各X/Y）。絶対価格で残る非block欄はSのCX（保存案n41）とSのCX depth（保存案n32）の各X/Yである。paired12量は全て非blockで、保存案n32。この32/41は十分数ではない。保証された必要数は保存JSONでもnullである。

したがって「20量のfloorがあるからpaired比較にも必ず追加q2/q4が要る」とはならない。一方、各方式の絶対価格を同時に同じ幅で資格付けする目的なら、この制約を消せない。holdout nを増やすだけではcalibration由来の不確かさは減らず、model biasやlaw誤差も解消しない。幅やfamilyを変更した後の数値診断は別の許可対象として残す。

## 5. 評価family・価格幅・古典信頼度の候補

### 5.1 Familyを先に固定する

family failureは保存JSON/scriptの0.05、旧各量配分は0.05/36として区別する。旧RESULT本文の略記を別のfamily failure値の採択としない。

|候補|対象|提案・成立条件|
|---|---|---|
|pairedモデル優先（推奨候補）|6資源×X/Yのモデル誤差12量|alpha_C=.05を候補に、Bonferroni alpha_C/12。絶対価格・direct優劣の正式な同時資格は別に扱う。新しいz・floor・nは今回算定しない|
|絶対価格のみ|2方式×6資源×X/Y=24量|absoluteを主要な成果に必要とするとき。旧36設定の20 blockを起点に、必要な追加calibrationを検討する。K24での結果を計算済みとしない|
|paired＋絶対価格|上記12＋24=36量|C6aの旧候補そのもの。20量のfloorを解消せず全量PASSにしない|
|direct paired優劣を一次目的にする|同q8の差12量|モデルから独立したholdout差平均の区間を使う。モデル精度は診断に留めるか、別の有効な同時coverageを設定する|

Bonferroniはfamily内の独立性を必要としないが、各周辺区間の有効性は必要。X/Y同価格と資源間covarianceがあっても、そのままnを倍増したり、holdoutを見てKを半減したりしない。全対象law・P0でのX/Y価格同一性を**事前に**資格付けした場合だけ、重複する価格命題の統合を検討できる。有限32本で同価格だっただけでは足りない。multivariate手法で共分散を使う場合も、事前に手法・coverage・特異行列の扱いを固定し、今回は新しい数値判定をしない。

pairedモデル誤差とdirect優劣を両方「同時に確認的」と呼ぶなら、それぞれのfamilyへ別々に.05を付けて全体.05とはしない。候補は alpha_model+alpha_adv≤.05（例.025ずつ）、または両命題を覆う正当化済みjoint手順。絶対価格の正式判定も加える場合は同じ全体予算内で再設計する。これは将来の設計案で、新たな有効CIを構成した結果ではない。

### 5.2 各資源の価格幅

候補式はC6aを継承して、tau_g=epsilon_abs,g+epsilon_rel,g S_g。旧候補は絶対1 compiled単位＋5%。pairedのS_gはholdout前に固定したsymmetric予測規模、絶対価格のS_gは各方式の予測規模とする。|paired差|を分母にしない。5%はpilotを受けた粗い分解能の提案であり、研究上必要な精度・十分性の採択値ではない。

|資源|幅の候補／単位|採択の根拠・留意点|
|---|---|---|
|RZ count|1 RZ＋5%×事前S_RZ、またはユーザー指定の絶対RZ予算|RZ増加の負担を残す。T数へ換算しない。差が0付近なら絶対幅・別の増加許容条件が必要|
|CX count|1 CX＋5%×事前S_CX|保存差−240/−688を比較する粗い解像度候補。標本SD0をsupport定数の証明やnの根拠にしない|
|RZ depth|1 DAG layer＋5%×事前S_RZd|RZ countの幅と交換しない。lane依存・融合で非加法的|
|CX depth|1 DAG layer＋5%×事前S_CXd|CX countが一定差でもdepthの差が一定とはしない|
|Total depth|1 DAG layer＋5%×事前S_D|保存されたdepth削減を比較する。実時間・T-depthでなく、q/2則も仮定しない|
|Circuit size|1 instruction＋5%×事前S_size|compilerのsize定義・measurement包含を固定する。q2とq4で差の符号が異なる保存事実を保持|
|Qubits|幅制約 w≤W_max と実幅の保存|W_maxはユーザーが利用条件から決める。観測S9/R17は別々に保持し、価格やNへ加算しない|

別の根拠ある幅を採る場合は、比較に必要な許容誤差・実装用途からepsilonをholdout前に決める。floorを消すためだけの事後拡幅や、資源を都合よく選び直す変更はしない。RZ/CX/depthを6資源のvectorとして報告し、単一の主資源を無断採択しない。

### 5.3 予測精度と優劣の判定を分離する

- **モデル精度（旧population校正線の不確かさを含める案）**：有効な同時不確かさ幅u_pred,gを得た後の候補は、|hat d_g(8)−f_Δ,g(8)|+u_pred,g≤tau_Δ,g。旧normal案のuはz sqrt(v_cal+s_H²/n_H)であり、calibration/holdout独立とnormal近似等を置いた名目的式にすぎない。
- **frozenモデルの条件付き精度**：有効なholdout差区間[L_H,U_H]を得て、[L_H−hat f_Δ, U_H−hat f_Δ]が[−tau,+tau]内に入るかを問う案。校正dataに条件付けた命題・scopeを明示し、旧population線の資格と混同しない。
- **direct優劣**：qualifiedな差の区間[L_Δ,g,U_Δ,g]について、U<0ならその資源の削減、L>0なら増加、0を含むなら未決。0は「費用差なし」という定義上の閾値。実用上必要な最小削減epsilon_adv,gを使うならユーザーが先に決め、U<−epsilon_adv,gとする。
- **tradeoff**：幅17やRZ負担が許容されるかは別条件。CX/depthだけの優位から全資源・全RPEの優位を宣言しない。

予測誤差が幅内でも優劣の符号は確定しない。保存された仮想q8 ΔRZ=−17.1875に対し旧tau=2068.05という例は、その違いを示すための保存値であり、新しいRZ削減結果ではない。validなCIが無い場合は点差・有限標本範囲の観測までとし、上のqualified判定を起動しない。

### 5.4 統計資格と標本数

C6aのnominal normal approximation、alpha_C=.05、K36、n_H≥32、sigma8_plan=2 SD4は全て未採択の条件付きplanning。normal近似のcoverage、tail/support、PCG64列のjoint-IID・ideal lawへのbridgeは未確認。観測min/maxや標本SDから全law support/母分散上界を作らない。CXのSD0により保存された条件式n0も、必要なholdoutが0という意味ではない。

旧planningの条件式は

\[
n_H\ge\frac{z^2\sigma_{8,plan}^2}{\tau^2-z^2v_{cal}},\qquad \tau^2>z^2v_{cal}.
\]

これは候補選択を説明する既存式の引用で、新しいnを計算していない。分母が非正ならその条件ではholdout増だけで解消しない。分母が正でも未知分散の移送・model bias・有効なcoverageは別。固定nの選択か、有効な逐次手順を先に採択し、通常の区間を見てPASSまで任意延長しない。

厳密な確認的結果が必要なら、宣言した確率モデルの下で正当化された全law範囲・tail/variance根拠・同時推論法等が先に必要。C4の既存bounded-cost手順を使う候補も、適用supportが得られた後に限る。**現段階の保証された必要nは未確定**。探索的n32は事前に採択する計算予算の候補としてのみ残す。

## 6. CX差の構造：静的に確認できることと証明義務

次は[whole-path builder][BUILDER]、[Reference gadget][GADGET]、[StrongBuilder][STRONG]、[tail emitter][EMITTER]、[方向制御][DIRECTIONAL]、[compiler][PHASECOMP]・[P0呼出し][GATES]を読んだ結果であり、新しい回路・gate価格・証明を生成していない。

|要素|SOURCEから確認できる範囲|全lawの一定CX差へ移す前に残ること|
|---|---|---|
|強いstep境界|`head_recipe('boundary')`はdirectional half＋uncontrolled halfの同block0をordinary controlled fullに融合。symmetricはq−1境界|native表現の規則と一体P0後のgate数を区別。跨境界の削除・同basis/符号相殺を含めた価格の一様性が必要|
|Reference global cut|`physical_segment`はA/B内部を融合し、合計q−2境界。cutの1境界を失う|cut前後の実際のgate列とhead/tail interfaceで、両方式の節約が全ordered列に同じか未証明|
|A/Bの分配|前半B・後半Aを構成、gadgetはsystemへA.inverse、referenceへB各一回。両laneは別register|AとA†を両方実行した二重価格にしない。逆順・逆角度やcompiler相殺の影響を含む必要がある。depthはmax型lane依存も持つ|
|CSWAP|n=8の前後cascade、論理CSWAPは保存条件で16、共通ancillaと追加8 qubits|`16×単体compile価格`は一体wrapperの価格証明でない。CSWAP展開と周囲のCX・phaseの相殺を対象P0全域で覆う必要|
|directional control|Gaussian基底はsystem上uncontrolled、中心だけordinary/directional。方向制御はparity-CXとRZ、定義上のzero-angle分岐を持つ|全制御Gaussianへ弱体化しない。control-oneへの変換や対角位相集約を含む全caseの対応が必要|
|tail共有|同basis runをまとめ、signed eventを同じordered列でemit。Reference側は同じtailのcontrol-one制限を利用|同basis/異basis・support/符号・隣接相殺がcompiled差へどう伝わるか、有限標本からの外挿は未資格|
|P0全体|公式定義のunroll後にwhole-wrapper transpile opt1/seed17、basis指定、couplingなし。MPFR160はdeclared phaseの蓄積規約|固定seedはcost不変性の証明でない。角度・gate消去・逆回路・shellとの融合を含む全rewriteのcost差の正当化が必要。MPFR160はT合成精度・丸め包絡を与えない|

これらはCX差が小さな構造項に依存するという**候補説明**を支える。C6a q2/q4で全16差がそれぞれ−240/−688だった観測は保持するが、SOURCE読解だけで全law/q8の定数差を証明していない。旧C3 q8一fixtureやC5固定q2も構造回帰の既知証拠で、独立holdoutへ混ぜない。

将来の証明・検証候補は、(i) 同じ入力・P0・cut下でnative差の全caseを定義し、(ii) 局所dictionary/境界/符号/特別角度を含むcompiler変換のcoverageを示し、(iii) 必要な限定caseだけを人工検証・独立whole-path postcheckで確認すること。単に未観測のrandom pathを増やして全supportの証明としない。追加証明・回路生成・compileは今回は行っていない。CX規則が得られてもdepth/RZ/sizeへ自動移送しない。

## 7. 追加calibration・holdoutの3案

|項目|案1：現在calibration＋paired優先q8|案2：追加q2/q4 calibration後q8|案3：paired構造モデル優先|
|---|---|---|---|
|前進する事項|新q8 domainでpaired比較とモデル残差の独立観測|方式別/pairedの校正標本誤差を、妥当な統計前提の下で小さくする|CX差の低分散・定数候補の理由と適用domainを特定|
|残る事項|絶対価格の旧20 block、normal/IID/tail、model bias、q>8|law/IID、bias、compiler全域、q>8は標本増だけで解決しない|全P0 rewriteと任意列coverage。depth/RZ/size/絶対価格は別|
|最小実装差分|fresh q8 request/seed/価格資格をopt-in追加。既存C3/C5/C6aの組立/P0/checker再利用|新calibration domain・追加ID・固定したpooling/係数更新。旧標本を上書きしない|既存dictionary/境界/phase/sourceの記述・将来許可された限定検証。新compiler/samplerは不要|
|sampling単位|q8のfresh whole-path n_H。各path8 occurrenceを一回生成し方式/axis共有|q2 k2本・q4 k4本を追加、holdoutとはdisjoint。旧各16本を校正用に全て保持|定理が成立するdomainの構造資料。経験検査が要る場合は別に固定whole-pathを定める|
|compile対象|raw PASS後、最大4 n_H full wrappers|追加校正は最大4(k2+k4)、q8は4 n_H。旧128を不要に再compileしない|証明だけで省略可能と認定しない。6資源の経験価格が必要ならfull wrapper比較も必要|
|統計条件|paired family/幅/alpha/estimandと資格を先に採択。旧n32を十分数にしない|必要なfamily/幅と分散根拠、固定n2/n4/停止規則、更新後モデルの独立holdout|全support・compiler領域の定理か、有限検証と未資格の区別|
|推定負荷|新しい長さはq8だけ、raw n_H・compile最大4 n_H。探索枠n_H=32なら最大128 wrapperというcall予算案|案1にraw k2+k4・compile4(k2+k4)を追加。k2/k4は未決|古典的な構造検証が中心だが全P0証明の難度は未確定。最初から全支持列列挙を要求しない|
|実行前固定|契約案の全項目、claimを探索的/確認的に分ける|calibration追加数・seed/ID・pooling・fit法と、その後に凍結する係数/共分散|使う構造・compiler版・局所集合・適用domain、証明が残る場合の有限検証scope|
|STOP|correctness/identity/fairness不一致、判定不能・混入・未許可domain|同左＋追加数/選択/更新規則違反、floor解消不能|一般化根拠がない場合に全law claimを停止。弱い対照へ戻らない|

**推奨は案1を入口とし、案3の未解決な構造説明を保持する。** q8の保存モデルそのものの評価が目的なら、上記frozen条件付き検証も採択候補とする。population校正線の残差・長q係数の資格が目的なら旧合成不確かさを省略しない。 paired差の旧12欄にfloor blockが無いことと、最終目的が方式間の資源比較であることが理由。絶対価格の全24量を同時に5%で資格付けする要求を先に課して追加samplingを一律増やすことは避ける。ただし絶対価格の保存・標本SD/covariance・将来の資格確認は省略しない。案2は、採択された目的が絶対価格も必要とする場合、またはpairedの新しい採択条件でも校正不足が残る場合に選ぶ。

追加calibrationでは、条件付きに v_cal=4 S2/n2+9 S4/n4 というC6aと同じ線形誤差伝播の形を用いる設計が候補になる。pairedではS_Δ,qを使う。ここでn2=16+k2、n4=16+k4。新しい数値・最適配分を計算していない。必要kは採択幅/alpha/有効な分散やtail前提なしに決められない。新しいkや係数更新を許可するときは、16本の都合のよい部分選択でなく、新IDの追加・規則固定・全校正情報の適正な合成を条件にする。旧pilotで追加数を選んだ後のpoolを、無条件に固定n IIDとして扱わない。pilotを設計用・新calibrationを確認用と事前分離するか、poolingとsample数選択を扱える推論手順を先に定める。

q8に関しても、**確認的な必要nは未確定**。ユーザー／GPTが探索枠としてn32を採択する場合は、correctnessと有限paired観測・名目的モデル診断までをscopeとし、確認的PASSを主張しない。確認的C6bが必要なら、有効な統計条件と必要数の評価を先に含める明示許可を求める。細かな関数ごとの承認を増やさず、採択scopeの実装・人工回帰・必要な科学callを次の指示でまとめる。

## 8. 将来C6bの実行前条件

[別紙契約案](C6B_EXECUTION_DRAFT.md)で、q8 law・cut4・未使用holdout domain・scope・P0・correctness・call数・統計手順・判定とSTOPを一覧化した。**草案の採択と科学実行許可は別**である。未確定のnや統計資格をboolで通さず、ユーザーの明示許可を元に具体化する。新grantや承認証明の仕組みは作らない。

成功・不成立・未資格を分ける。技術correctness PASSは有限pathの成功、model幅を通らないことはその価格モデル条件の不成立、interval/IID等が足りないことは統計未資格。数値誤差不足や不公平なbaselineを幅の変更で救わない。q8で仮に価格資格が得られても、認定domainは検証したq8に限定し、q>8/全13-roundへの外挿を止める。

## 9. Stage Bとの接続条件

B4-2の14738は元H厳密ground、ideal finite-RTE IID・exact native、保存B1数値operator誤差等を前提とした13-round Hoeffding十分shot数。actual必要Nではない。[B5][B5]はfinite-bit law誤差、shot/occurrence joint-law、全path/全13-roundの実装誤差を未資格としている。C6aでfresh sampler接続・有限path検証は追加されたが、この不足を解決した証拠ではない。B5が当時C6a未取得と書いた履歴を、現在もfresh sampler自体が無いという意味で引用しない。

将来、同じ対象law・測定分布・実装誤差資格で同じround/axis別Nを使える場合に限り

\[
G_{R,g}-G_{S,g}=\sum_{m,b}N_{m,b}\,\mathbb E_{p_m}[\Delta C_{g,m,b}]
\]

と結べる。方式別Nなら

\[
G_{R,g}-G_{S,g}=\sum_{m,b}\{N_{R,m,b}\mu_{R,g,m,b}-N_{S,m,b}\mu_{S,g,m,b}\}
\]

であり、paired価格差だけでは不足する。これは記号上の会計整理で、数値和・N再算定・14738との乗算を行っていない。G_R/G_Sを各々算定するための絶対期待価格、全使用m×axisのdomain、共有校正共分散も必要である。

価格の古典的不確かさalpha_Cと、RPE energy/failureの条件を分ける。B4-2のtargetやalphaを変更しない。価格標本数を量子Nに掛けない。B5のbounded信号biasのlaw誤差式を、価格のrange無しにcost期待値の誤差式へ流用しない。全path誤差・同じNの資格が未確認なら、q8での資源差が負でも全RPE費用優位は未認定のままにする。

## 10. 未解決事項・次の判断・今回のSTOP

|次の判断|担当と必要な許可範囲|
|---|---|
|pairedモデル12量を一次familyにするか、direct比較を一次にするか、絶対価格も確認的に求めるか|ユーザー／GPTがclaim・価格幅・alpha予算・資源/幅条件・探索的/確認的資格を採択する|
|確認的なnを必要とする場合の統計設計|Codexは別の明示許可後、指定保存値の限定再解析、family変更後のplanning、必要なsupport/tail/IID根拠・sample数評価を行う。今回未実施|
|独立q8を実行する場合|ユーザーが対象SOURCE・input・law・cut・nまたは事前規則・max call・精度・判定を具体化して許可。Codexが最小adapter・人工/回帰・sampling・raw/P0/postcheck・paired統計を同scopeで行う|
|追加q2/q4校正を必要とする場合|追加k2/k4、seed/ID、全標本合成・fit/共分散更新、凍結後holdoutを別途scopeへ明記。今回の静的案は追加samplingを許可しない|
|B側保証・長q移送・全RPE総費用・詳細科学レビュー|ユーザー／GPTが必要性と別scopeを判断。B6は今回調査/変更/実行しない|

未取得：q8の実価格・実分散、全support CX一定差、qualifiedな価格区間/必要n、長q適用域、implemented joint-law/全path誤差、共通actual N、routing/T合成/実機時間。未知を0とせず、追加sampling・別q・再解析が必要ならそこで判断へ戻す。

今回の変更は本書・契約草案と入口索引のMarkdownだけ。科学code/input/raw/result/SOURCE、既存契約・grant/ledger/marker/STOP、科学manifest/VALIDATION_STATUS/研究概要は変更しない。C6a raw archiveはローカル保全のまま、追加公開しない。重い処理・研究module import・test起動はなく、他jobや共有環境への変更もない。

**C6b-0の静的調査を終了し、専用branchへcommit/push、remote SHA/tree/公開Markdown bytes照合後mandatory STOP。** 次担当はユーザー／GPT。本書の推奨を採択・科学実行許可と扱わず、追加calibration/独立q8/長q/全RPE/詳細レビューへ自動進行しない。

[C6A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/docs/research/track_d_c6a_paired_price_20261106/RESULT.md
[C6AMD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/docs/research/track_d_c6a_paired_price_20261106/C6B_PROPOSAL.md
[PLANJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/artifacts/track_d_c6a_paired_price/2026-11-06/run01/C6b_planning_proposal.json
[STATS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/artifacts/track_d_c6a_paired_price/2026-11-06/run01/paired_statistics.json
[FIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/artifacts/track_d_c6a_paired_price/2026-11-06/run01/exploratory_affine.json
[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/artifacts/track_d_c6a_paired_price/2026-11-06/run01/SOURCE.json
[STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/artifacts/track_d_c6a_paired_price/2026-11-06/run01/STOP.json
[PUB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/artifacts/track_d_c6a_paired_price/2026-11-06/run01/payload_publication.json
[PLANCODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/scripts/summarize_track_d_c6a.py
[C5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a6920b8355643701ab7436a4fd3b34862e9e3bd/docs/research/track_d_c5_q2_20261106/RESULT.md
[C4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a5784c4bd3e3fb9779849a1e1fc1759559e2caea/docs/research/track_d_c4_price_model_preparation_20261106/REPORT.md
[C3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md
[B42]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md
[B5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0a8b953aea1be20b110784960763d7f36ebccac8/docs/research/track_d_b5_actual_rpe_static_survey_20261106/REPORT.md
[STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/c1_h4_strong_baseline.py#L130
[EMITTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/pr_dictionary_validation/circuits.py#L87
[GATES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/pr_dictionary_validation/gates.py#L85
[BUILDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b6667026ec8b670f610e41141d195883a9eb513/src/trottertracks/track_d_c3_whole_path.py#L19
[GADGET]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b6667026ec8b670f610e41141d195883a9eb513/src/trottertracks/track_d_h4_reference.py#L129
[DIRECTIONAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b6667026ec8b670f610e41141d195883a9eb513/src/trottertracks/resource_applicability/ax2b_native_df_v5.py#L94
[PHASECOMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/c1_h6_compiler_precision.py
