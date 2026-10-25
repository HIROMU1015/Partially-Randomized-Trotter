# H4 Pauli-PRパラメータ探索：実装接続・一括実行範囲案

2026-10-25。**実装準備と人工検証まで完了。実H4の新しい科学実行は未承認・未実行。**
添付指示§9は受領だけを実行承認と扱わず、範囲固定後に一度だけ明示承認を得る指定である。
以下の一括範囲を提示し、その承認後は通常の段階ごとの承認を求めず進める。
研究の主要claim採択・詳細科学レビューを今回の技術検証に含めない。

基点 `fbfbfd21a252a4a32f52e3b40e50acd91b3c626a`、専用branch
`codex/non-df-h4-pauli-pr-search-preparation-20261025`。
[正確な非実行範囲JSON](../artifacts/non_df_h4_pauli_pr_search_preparation/2026-10-25/execution_scope_DRAFT_DISABLED.json)、
[初期候補100行](../artifacts/non_df_h4_pauli_pr_search_preparation/2026-10-25/initial_candidates.csv)。
`execution_enabled=false` のまま保存し、新grantや承認証明は作成しない。

## 1. 固定条件と比較対象

Linear H4、隣接1.00 Å、STO-3G、同じ保存Pauli-JW184非identity項＋constant、
abs(c)降順/label tie-break、保存raw rayとfull-H reference全区間、S2/T=.8。
D2 central-rotation制御、system8/ancilla8、cosine/+Im sine。
full Hadamard wrapperのcompiled RZ×shot、state preparation費用除外。
Qiskit1.3.0/NumPy1.26.4、basis rz/sx/x/cx、opt1/seed17、backend/coupling/layout/routing指定なし。
ε_sig=.05/.01/.005/.001/.0005、固定ε/√2配分・α_axis=.025・u/h≤.01。

決定論的側は[106構成の保存結果](non_df_h4_pauli_b0_comparison_20261025.md)を引用する。
新しいB0/full PFのbuild・compile・状態作用は行わない。

| ε_sig | 保存最良B0 prefix/q | 保存G（仮想ρ=0） | 保存最良full PF prefix/q：G |
|---|---|---:|---|
| .05 | 132/1 | 23,237,760 | 184/1：32,680,930 |
| .01 | 144/2 | 1,440,845,160 | 184/2：1,723,689,280 |
| .005 | 176/2 | 9,734,669,448 | 184/2：10,167,897,280 |
| .001 | 180/8 | 579,283,719,324 | 184/8：615,615,107,120 |
| .0005 | 180/8 | 2,663,993,522,484 | 184/8：2,982,596,955,988 |

上表は保存[best_by_precision.csv](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/best_by_precision.csv)／結果JSONからの引用。
今回N/Gの再計算・順位付けをしていない。
旧Pauli-PR prefix27/q8/r4/K4の256 paired価格、bias/B/u/N/Gも歴史的baselineとして保持する。
その標本を新探索や独立確認へpoolしない。

## 2. 初期候補・追加規則

巨大な全直積を避け、保持数/qの骨格とK/rの対照を組み合わせる。

| block | prefix L_D | q | r | K | 候補数 |
|---|---|---|---|---|---:|
| 保持数/q骨格 | 0,8,16,27,36,37,64,96,128,176 | 1,2,4,8 | 4 | 4 | 40 |
| Taylor次数対照 | 同上 | 1,4,8 | 4 | 2 | 30 |
| event数対照 | 8,27,36,96,176 | 1,4,8 | 2,8 | 4 | 30 |

計100。うち旧固定PR1候補は保存証拠を再利用し、新規平均信号候補は最大99。
prefix0/8/16は低保持、27は旧baseline、36/37は最初の非対角項の境界、64/96/128は中間、176はfull近傍。
full184は決定論的比較側に既存値があり、空tailのPR候補として重複登録しない。
DFのfragment数・Hamiltonian・価格・旧科学契約は移さない。

適応候補は最大2round、各最大32、合計64追加まで（全登録≤164）。
前roundの各εの有限G上位2候補を親として、prefix±1/4/8、隣接q/r、K切替を一座標ずつ提案する。
許可domainはprefix0–183、q1/2/4/8、r2/4/8、K2/4。
ε=.05を先頭に5精度を巡回し、親の保存G/決定論的G、prefix距離、candidate IDで順序を固定して重複を除く。
第2roundは>1%改善、最小候補が探索境界、または決定論的との差≤5探索SEが残る場合だけ。
各batchの実対象・理由・直前結果SHA・SOURCE/input/runtime/call範囲を**実行前**に固定する。
これは有限登録探索で、大域最適性の証明ではない。

## 3. 探索標本と独立確認

精度適格な候補だけ、最初は**8 paired trajectory**をcompileする。
追加探索終了後、各εの上位2候補のunion（≤10）と、僅差の場合の最大2候補について、
新しい探索index8–63を追加し、最大12候補を64本にする。
旧固定PRは既存256本を別baselineとして引用し、新しい探索標本と合算しない。

その後、探索G上位2/εのunion（≤10）を**確認前に固定**する。
独立seedで48本から始め、必要な候補だけ累積96→192→384本へ追加する。
追加は、決定論的または凍結finalistとの差が2 combined SE以内で、nominal t95相対半幅>1%の場合。
384本でも重要な符号が未確定ならINCONCLUSIVEを残す。
探索nomineeの確認値を主表にし、確認標本で再選抜した最小値は別表で事後選択を明記する。
探索winner、確認値、確認集合内winnerを同一の推定量として扱わない。

SeedSequence entropyは `[20261025,3101,L_D,q,r,K,cohort_tag,index]`、
exploration=101、confirmation=202、出力uint64を既存PCG64 samplerへ渡す。
同じtrajectoryの両軸はpaired。SEは `G_i=N_cos*C_i,cos+N_sin*C_i,sin` から求める。
PCG64の数学的独立性証明、逐次観察に有効なCI、全候補同時保証を主張しない。
未観測order4や標本分散0から母平均tail=0とはしない。

以前のDF研究は、探索n8/32と独立確認n48/104を分けた
[H6確認資料](research/track_a_h6_pr_confirmation_adoption_20261019.md)、
[H8独立確認](research/track_a_h8_new_pr_confirmation_results_20261019.md)、
[H4 B0I候補地図](research/track_a_h4_b0i_map_results_20261014.md)を確認した。
候補登録・独立確認・不確かさの整理方法を参考にし、分子条件や数値結果は移していない。

## 4. 再利用コードと数値接続

[search接続](../src/trotterlib/h4_pauli_pr_search.py)は、`PauliSignalAdapter`のprefix/S2/constant、
`rte.sample_rte_events`、D2 central制御、既存Hadamard wrapperを再利用する。
新seedのevent記録を価格へjoinし、異なるaxis/config/compilerのcacheを混合しない。
両軸の価格が等しいと仮定して片軸compileを省略しない。

[Arb mean接続](../src/trotterlib/h4_pauli_pr_mean.py)はCのbinary64→exact変換、CDF/grid mass、
Pauli map、全reference区間、L2 recenterを再利用する。
旧`EnclosureEngine`はq8/r4/K4の固定契約なので変更・迂回しない。
既存Cのdense意味論に適合するstate-vector接続だけ新設した。
準備版の公開入口は2/3qubit人工系のみで、実H4 production dispatcherはまだ設けていない。
承認後に既存B0 batch runnerのprocess/SOURCE/native保存方式を最小拡張し、実行前に新SOURCEをcommitする。

符号付きPauliのeffective平均を `M=Σ_j p_j P_j` とすると、各event平均は
`F=Σ_(k=0,2,...,K) π_k (-1)^(k/2) [cos(θ_k) M^k − i sin(θ_k) M^(k+1)]`。
productの発生順と最後のrotationを独立component選択で平均した同じfinite lawであり、アルゴリズム変更ではない。
`p_j,π_k`はNumPy1.26.4の53bit uniform-word/CDFモデルのexact質量、θはruntimeのbinary64角。
IID uniform-wordは明示的統計モデルの前提である。

各eventは位相・符号を含めunitaryで、Fはその凸平均なので `||F||≤1`。
Fをunitaryと仮定せず、このcontractionから `e_new≤e_old+rounding_radius` を使用する。
retainedはCと同じexact binary64 operandの積、scalarはnative step operand。
compiled回路のangle/aggregate位相/π/control/最適化との差は未保証ρとして別に残す。
共通raw normをexactに求め、stateファイルを書き換えず、B_savedを最後に一度だけ適用する。

meanは最初64bit、適格性や僅差のG判断を幅が妨げる場合だけ96→128bit。
各新候補最大3回、full-H reference作用は0。bias区間の半幅がu_encで、経験的な任意allowanceを置かない。
N/Gは既存`h4_pauli_b0_comparison.account`/`h4_conditional_cng`を呼び、
候補自身のB_saved・そのlogの外向き上界・bias/uからguardを作る。
旧PRのguard/N/Bを別候補へ転用しない。nullは0/∞へ変換しない。

公平な比較のため、新探索では全候補に同じphysical-ray/whole-mean区間の会計を適用する。
旧p27の保存C interval/νと実B_savedを使う共通policyの**別の保存値再会計行**も、承認後に作る。
旧log marginやuを含む歴史的N/Gは無変更で引用し、会計の締め直しによる差とパラメータ改善を分ける。
決定論的106構成には同じwhole-target/νの保存会計があるので再計算しない。

## 5. 人工検証と実行前に残す技術作業

[compiler側テスト](../tests/test_h4_pauli_pr_search.py)6件と
[Arb側テスト](../tests/test_h4_pauli_pr_mean.py)4件、計10件PASS。
2/3qubitで4パラメータ伝播、CDF graph、cohort分離、order2/4非可換product・負符号・order phase、
constantを含むcentral relative phase、compiled cosine/+Im sine、候補自身のB/guard/nullを検査した。
4つの2qubit lawで64bit vector包含が192bitの既存C dense演算包含を含むこと、
raw rayのexact倍によるphysical信号不変、reference区間全体の保持を確認した。
これは技術検証であり、H4精度・資源改善や独立科学レビューの証拠ではない。

人工oracle初回は、旧C helperがK4固定のstrict zipを使うためK2で例外になった。
K2をorder4質量0として同じ旧helperへ埋め込むよう**人工oracleだけ**修正し、旧C SOURCEを変更せずPASS。
実H4科学callの失敗・retryではない。

実行前に、production接続、batch対象・入力SHA・code SOURCE・runtimeを固定する。
固定p27のsaved descriptor/angles/scalar/CDF/state/referenceとの対応を最初に確認する。
必要な数値照合も承認範囲内のcallとして数え、旧成功計算を不要に繰り返さない。
既存STOPを再開しない。通常技術修正は新しい来歴で、旧失敗・成功証拠を保全する。

## 6. 提示する一括科学範囲・終了

| 上限 | 科学的対象数 |
|---|---:|
| 登録PR候補 | 初期100＋適応64＝164 |
| 新しいmean候補 | 最大163（固定p27は保存C再利用） |
| Arb mean状態作用 | 最大489（各64/96/128bit） |
| 新探索trajectory | 最大1,984 |
| 独立確認trajectory | 最大3,840（10候補×384） |
| 合計新paired trajectory | 最大5,824 |
| 両軸top-level compile | 最大11,648 |

上記は実験の最大範囲であり、CPU/RAM/worker数の制限ではない。
全候補・精度を機械的に最大まで実行せず、適格性・候補選抜・早期終了を適用する。
実際の空き資源に応じ候補/trajectoryをprocess並列化し、共有設定や他jobは変更しない。
新SCF/FCI/DF/Hamiltonian/state/full-H reference生成は全て0。
最初は5精度だけを完成させ、必要なら既存303点gridを保存値会計で整理する。

compiled実ρ、PR母平均費用tail、全law数値保証、正式certificateは未解決のまま。
[固定PR母平均価格の限定scalar結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d4f6ff7f1df085f292d36c36e460354df8a740/docs/non_df_h4_pr_mean_cost_scalar_execution_20261025.md)
は読取確認したが、別パラメータのUや母平均順位へ転用しない。
R2-C証明継続、新PRアルゴリズム、別geometry、論文草案、主要claim採択は対象外。

**次の担当：ユーザー**がこの一括科学範囲を一度承認する。
**Codex**はその後、合意範囲のproduction固定・探索・独立確認・比較・保存公開を連続して実施する。
科学条件変更、重大な不整合/公平性、上限超過、共有環境への不可避な影響では安全に判断を戻す。
承認前は実H4科学実行を開始しない。準備commit/remoteは最終チャットで報告する。

[入力参照](../artifacts/non_df_h4_pauli_pr_search_preparation/2026-10-25/input_references.json)、
[人工検証記録](../artifacts/non_df_h4_pauli_pr_search_preparation/2026-10-25/verification.json)。
旧SOURCE/raw/結果/grant/ledger/marker/STOPは保全。科学計算完了後はmandatory STOPし、研究方針の採択へ自動移行しない。
