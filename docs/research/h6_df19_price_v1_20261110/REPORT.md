# H6 DF19 compiled価格モデルV1：保存データでの近似精度と構造誤差

2026-11-10 JST。**許可された保存PF/PRの全対象を評価した。M1は全174軸行で過小予測し、整数完全一致は0/174。PR独立確認の平均絶対相対誤差はA族16.98%、B族15.94%、PF留保点は17.29%だった。** 非0残差だけで処理を中断せず、全cohortとbinding確認済みPF補助点まで完了した。

これは構造モデルの近似精度検証であり、全compiler再実装や残差0の達成を研究目的にしていない。一律の%採用閾値・補正係数fitは設けない。新回路build/compile、trajectory、fit/bootstrap、signal/N、N×C/全Gは0。[実行前PLAN](PLAN.md)、[開発](DEVELOPMENT.json)、[留保](HOLDOUT.json)、[補助](AUXILIARY.json)、[個別174軸CSV](PATH_RESIDUALS.csv)、[構造誤差集計](STRUCTURE_ANALYSIS.json)、[完了scope](COMPLETION.json)。

## 1. 科学条件・SOURCE・入力

Linear H6/隣接1.00Å/STO-3G、tol-only1e−8の保存DF19、weighted-Hermitian completion、12 system mode、Nα=Nβ=3のsector400、保存DF-H同sector groundがtarget。fragment順・signed weights/scalar・保存数値state条件付き/P=0を保持した。RPE条件はh=.1/全13round m0…12、εE=.00159360010199040 Ha、各方式αtotal=.05、βtotal/stat=.4/.2で、今回Nや成功条件を再計算していない。

| SOURCE / 役割 | 固定版・用途 |
|---|---|
| [H6設計][DESIGN] | 4c965ff69。旧REPORT/SOURCE_INDEX/COVERAGEは無変更 |
| [H6 D1][D1] / [D2][D2] | 45607a50 / 95d01b6a。candidate/H/state/snapshot、保存weights/allocationを引き継ぐ。m6担当の科学結果は変更・採択しない |
| [元主価格][MAIN] | carrier155529e2、artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/の探索/確認/再利用価格と元provenance |
| 実provider | c31ef424175c02379eafebe8fc6c7548c1d9aabe。元control-plan、DF primitiveとwrapperの本文を確認 |
| [H4 V1][H4] | eec5bdf9。局所ゲート展開・literal消去核と人工testを再利用し、H6実SOURCEに適合することを確認。H4係数/8〜11%誤差率は移植しない |
| **モデル/実行SOURCE** | **960e8f91140cce37a0eb9091a65a41e18f6e54a3**。実データ評価前にcommitし、開発→留保→補助まで変更なし |
| 記述的誤差分析SOURCE | d726dc49a63244c6d815c729b72473d07b716744。完了予測を読み、役割・event層別だけを集計。予測再実行/fitなし |

元Qiskit1.3.0/NumPy1.26.4、rz/sx/x/cx、optimization_level1、seed17、symmetric_directional、測定付きfull Hadamard wrapperを維持。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。別P0/opt2/他系列の価格へ移さない。既知の公開集計値とH4不一致結果は設計時に既知なので、完全blindとは呼ばない。

保存H6 snapshot/df receipt/snapshot receiptのbytes/hashとH/state/completion bindingを確認した。state配列はロードせず、保存12×12のone-bodyと19 DF係数から元providerのdiag_hermitian/元Gaussian関数で局所価格記述子を取得した。20局所行列の分解は価格descriptor処理であり、新SCF/DF/Hamiltonian/state生成やground-state固有値問題ではない。

PR84 pathはcandidate/cohort/index/preparation hash/PASS/conditional=false、application順、保存basis ID/hash/ordered operationsを照合した。元trial_filesの完了順をpath番号順に対応付け、cohort集合を変えていない。各cohort・軸の実価格平均は元aggregateと一致した。入力hashは各JSONのinput_sha256/private_trial_bindingsへ保存し、raw event/seed/NPZ/QPY/内部pathは公開していない。

## 2. 対象と計数rule

Full19/m5/qPF1/T.8（δ.8）を開発、qPF1/T1.6（δ1.6）を留保。PR Aは `2606efdab20e3b8522e5`、p5/R32/K2/q1/T.8、PR Bは `2d2af6ca830dff14a695`、p7/R16/K2/q1/T1.6。それぞれ探索9を開発、独立確認33を留保した。PF旧 `4ecf37965885c656e554` / qPF2/T.8/δ.4は元Full19/preparation/compiler/axis receiptが一致したため補助評価した。

モデル名は`H6_DF19_M5_SOURCE_NATIVE_BASIS_CANCEL_ZFOLD_V1`。[実装](../../../src/trottertracks/h6_df19_price_v1.py)は実価格、平均単価、cohort、seedを予測入力に取らず、fitした係数を持たない。

1. 11 signed pieceを保存順に展開し、各時間w_j T/qPFの半stepでforward uncontrolled/head→PR tail→reverse directional/headを会計する。Full19はtailなし。中央負時間を含む符号を保持する。
2. R32/R16を保存ABS_TIME_PER_OUTER配分として一outerに一回適用する。32や16を11pieceへ重複乗算しない。PR A/Bのheadはone-body＋prefix5/7、Fullはone-body＋19 fragmentで、H4 S2のhead/step数をコピーしない。
3. 保存tailのapplication/product/rotation/identity/phaseとbasis共有を保持する。basis逆変換→diagonal→復帰を、同じproviderに対するGaussian P/XX+YY等の局所記述子で扱う。scalar/抽出identityは元の最後のP(−scalar T)一回のみ。
4. 解析的rz/sx/x/cx/measure展開、直に隣接する同basis inverse対の消去、wire上で他gateに遮られないRZの加算を行う。primitive生成は元SOURCEのzero policyを維持し、追加Z-foldではbinary64の厳密0のみ消去する。近い角のsnapはしない。H/Sdg/measurement wrapperは各軸一回、control wireは12（全13 quantum wire）。
5. 既存NativeCounterへstreamで渡す。全QuantumCircuit/DAGや実回路の新build/transpile、一般1Q再合成、CX固定点最適化は実装しない。選択した等価分解が元basis translatorと同じRZ数とは仮定しない。

`structural_rz`が候補予測で、native展開後countと限定Z-fold後countを別に保持する。compiler correction/全law期待価格/CIはnull。展開のglobal phase欄はモデル内の局所会計であり、新しい分子signal/RPE位相ではない。

## 3. 価格予測・残差と近似精度

以下は**各軸の値**。X/Yを別に計数・照合し、今回のC/Ĉ/残差は両軸で等しかった。相対誤差はpathごとに|C−Ĉ|/Cを求め、そのcohort平均を表示する。平均価格同士の比とは区別し、探索/確認や軸をpoolしない。ΔC=C−Ĉ、全行で正だった。

### 開発（19 path単位、38軸）

| 族 | n | 平均C | 平均Ĉ | 平均ΔC=平均絶対誤差 | 平均絶対相対誤差 | 最大path絶対誤差 | 最大path相対誤差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full19 PF | 1 | 427996 | 354760 | +73236 | 17.1114% | 73236 | 17.1114% |
| PR A p5/R32/K2 | 9 | 141775.111111111 | 117807.444444444 | +23967.666666667 | 16.9095% | 24860 | 17.7158% |
| PR B p7/R16/K2 | 9 | 179583.333333333 | 150787.888888889 | +28795.444444444 | 16.0336% | 30168 | 16.6475% |

### rule固定後の留保（67 path単位、134軸）

| 族 | n | 平均C | 平均Ĉ | 平均ΔC=平均絶対誤差 | 平均絶対相対誤差 | 最大path絶対誤差 | 最大path相対誤差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full19 PF | 1 | 428918 | 354760 | +74158 | 17.2896% | 74158 | 17.2896% |
| PR A p5/R32/K2 | 33 | 141173.212121212 | 117206.454545455 | +23966.757575758 | 16.9793% | 25340 | 17.8974% |
| PR B p7/R16/K2 | 33 | 179579.363636364 | 150948.181818182 | +28631.181818182 | 15.9429% | 29756 | 16.5271% |

### 保存補助点（1回路、2軸）

| 族 | n | C | Ĉ | ΔC=絶対誤差 | 絶対相対誤差 | 最大絶対誤差 | 最大相対誤差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full19 PF 補助q2 | 1 | 831662 | 708732 | +122930 | 14.7812% | 122930 | 14.7812% |

留保の整数残差0は**0/134**、開発/補助を合わせ**0/174**。これは174独立標本を意味しない。PRは同pathのX/Y paired、同じ回路族内の有限cohortである。PF q2補助点はδ変更を伴うので、固定δのq外挿精度へ昇格しない。

## 4. 誤差構造：説明できたことと未説明のこと

[STRUCTURE_ANALYSIS](STRUCTURE_ANALYSIS.json)にcohort・軸別のfeature/role範囲、nativeとfolded RZ、order-2有無による残差を記録した。これは記述集計であり、残差を説明する補正係数のfitや因果分解ではない。

| 観測 | 解釈できる範囲・限界 |
|---|---|
| Full19 q1では展開697257→限定消去354760、実CはT.8で427996/T1.6で428918 | native countをcompiled価格と同一視できない。同じM1 countでも実価格は時間で変わる。どの角度/1Q/CX規則が差を生んだかは未同定 |
| PF q1の明示basis逆対消去21、補助q2は43 | 11piece内中央とpiece/outer境界を構造として扱えた。これらはM1内の消去数で、元compilerの全融合数ではない |
| PR A/Bの明示basis逆対消去は各path10 | head/tailとpiece境界の一部を分離した。異なるprefix/Rで一般補正を共有できたという証拠ではない |
| PR Aのtail basis group：開発16…26、確認14…24。Bは開発14…16、確認13…16 | 保存event列の差をモデルへ反映した。同じqでもpath価格は異なる。各特徴へ残差を一意配賦する証拠はない |
| Aのorder-2ありpathは探索1、確認1。Bは両cohortで0 | order-0だけのpathにも全て正の残差がある。不一致をrare eventだけに帰せない。Bのゼロ観測を真のevent確率/費用寄与0にはしない |
| 一般1Q run再合成・basis translation選択・CXと1Qの相互作用・head/tail/piece/wrapper端部 | 現在の解析展開＋literal消去で閉じていない候補機構。今回はcompiled DAGの分解や個別ablationをしておらず、主因を断定しない |

全targetで過小予測した有限観測はM1を価格の厳密下界にする根拠ではない。原始展開はCより大きく、限定消去後は小さい。H4の残差率や一定倍率を掛けて補正することもしていない。

## 5. 境界補正研究の再利用判断

`rte_boundary_cost_validation.py`は固定short-step分布で1/2/3-eventのcompiled校正を行い、`L μ1+(L−1) μ2+(L−2) μ3+…`を検証する。`hierarchical_cost_validation.py`は同分布clusterとq1/q2校正・q4留保を分ける。`rte_connected_cluster_cost_validation.py`もcompiled calibration/holdout・pattern係数を必要とする。

今回のH6は11 signed pieceでτ・配分が異なり、A/Bでprefix/R/δも異なる。同じH6/preparation/K2/law/compilerの対応する保存cluster係数は本scopeに含まれていない。これらの実行入口はsampling/build/compileや新校正へ進み得るため呼ばず、**V1の数値補正へは採用しなかった**。局所境界項を分け、校正と留保を分離する方法は再利用した。再利用必須として無理に係数を移植しない。

## 6. 三つの結論と移送資格

1. **保存価格を再現できたか：** 完全一致は0/174。正確なcompiled RZ再現器ではない。一方、全対象の近似誤差と構造内訳を取得でき、非0残差だけでV1全体を研究上失敗として打ち切っていない。
2. **近似モデルとして有望か：** 費用の規模と構造内訳を与える探索用の基線として有用性はあるが、確認cohortでも約16〜17%の系統過小が残る。精密な絶対価格providerとしての採用は推奨しない。用途に必要な精度や不確かさとの接続は別判断で、一律%閾値は置かない。追加compileより先に、少数の局所分解/1Q界面を改善して説明力が上がるかを調べる余地がある。
3. **全13round providerに使えるか：** 未資格。q方向の独立証拠、未観測event/全law期待費用、long-q systematic、round間/校正との相関が不足し、M1を正式providerへ採用しない。

| 移送する対象 | 今回得た証拠・未資格事項 |
|---|---|
| 別trajectory | A/B各族で探索9の後に独立確認33を同じruleで評価し、同程度の過小が継続。有限同q/cohortでの経験的支持に限る |
| 別δ/prefix/R | PF T1.6留保、PR別族Bも評価したが、両PR族の探索データは開発対象。未使用の任意prefix/Rへの一般化ではない |
| qPF増加 | 補助q2はδ.4。目標δ1.6のq1→q2→q4の価格検証は未実施 |
| PR全law | finite-bit/ideal law差、未観測order/supportの条件付き価格とtail boundが不足。標本平均やorder-2観測数だけで保証しない |
| 全G | N/C誤差の相関、価格bias、欠測roundが未解決。今回N×C/費用比/全RPE優位は評価していない |

## 7. 最小の次案（未許可・未実行）

まず新compileなしの局所会計改善を候補とする。同じ保存開発点で、元providerのGaussian/XX+YY・basis translatorのnative分解選択と、1Q run/CX界面のRZ価格を切り分ける。全Qiskit compilerの再実装や一律倍率fitを目的にせず、支配的な局所列だけの説明力を調べる。局所再合成・新予測/残差が必要なら範囲を別途固定して許可を受ける。

今回の留保結果を使って次モデルを設計するなら、同じ33本を未使用独立holdoutとして再認定しない。既知データ上の改善確認と、別domainの独立検証を分ける。その後も必要な場合に限り、固定δ1.6のqPF1保存点、qPF2/T3.2校正、qPF4/T6.4留保の小さい価格pilotを検討する。新価格取得/compileは今回0、m6の科学適格性は別担当の結果を引き継ぎ、勝手に採択しない。

## 8. 実行・検査・STOP

人工8件PASSを[記録](SYNTHETIC_CHECKS.md)に保存した。再利用した局所ゲートの位相付き作用、basis/RZ消去、H6 control12対system8、11piece/R配分、価格を予測入力に取らないscopeを確認した。人工PASSを実H6の完全作用検証や独立科学レビューと呼ばない。

科学実行は一回で全対象を完了し、技術的再試行/モデル変更は0。descriptorはparentで共有し、最大64 processのpure countingで約216.11秒。worker数は利用可能affinityと独立job数から選び、CPU/RAMの恣意的capや共有設定変更をしていない。実QuantumCircuit constructor/transpileを禁止する検査下で、build/transpile/sampling/fit/bootstrap/signal/N/G/新q価格取得0を完了記録に残した。[execution.txt](execution.txt)は開発/留保summaryの出力記録。

同じ`codex/resource-eval-h6-df19-price-design`を継続した。H6 m6/H4 V2担当のworktree/source/processへ操作・制約を加えていない。旧H6設計/H4 V1/SOURCE/input/raw/価格/cache/grant/ledger/marker/KEEP/STOPを保全し、新しいapproval証明書やgrantは作っていない。

GitHub公開後、remote SHA/tree・主要成果物を照合して**mandatory STOP**。次は**ユーザー**が本結果で区切るか、局所会計改善/独立q検証の具体的scopeを許可するか判断する。**Codex**は追加fit/compile/長q/全G/詳細科学レビューへ自動進行しない。

[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c965ff69e3619a7b2f8313a2093cde679f542b8/docs/research/h6_df19_price_design_20261110/REPORT.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/45607a5086721ea1d21b1ed2d7edfe870549f0c8/docs/research/h6_df19_rpe_d1_results_20261110/REPORT.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/95d01b6ab1d820921fcd2521324e41e82389713a/docs/research/h6_df19_rpe_d2_results_20261110/REPORT.md
[MAIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[H4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eec5bdf9cc391fe49b55b5673f3e385deeb20d42/docs/research/h4_df10_price_v1_20261110/REPORT.md
