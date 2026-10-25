# H2・H4・H6：Hybrid PR対全ランダム化RTEの条件付き資源比較

2026-10-25。専用branch `codex/pauli-hybrid-rte-execution-20261025`。
準備commit `25d248a1f2ed88de62b0e621c2d1cea8532804c4`。
科学driver SOURCE `b19ed83c6b4b64821965fa0b756cd2691e34eda8`。

## 結果の読み方と今回得られた比較

この比較は固定linear H2/H4/H6・隣接1.00 Å・STO-3G、Pauli-JW、S2/T=.8、central-controlの有限登録候補に限る。以下は保存Nと経験的価格を組み合わせた仮想ρ=0のGであり、実精度まで保証した母平均Gではない。

- **H2**：保存独立確認n96では5精度すべてHybridの低G（Hybrid/RTE=.855463–.862817）。探索n8ではRTE側の点Gが低く、確認で逆転するため、小標本だけで優劣を固定しない実例となる。Hybridの単回価格24.2188 RZはRTEの13.8021 RZより高い一方、Bが1.35747対1.93575で必要Nが小さく、Gで逆転する。新H2実行は0。
- **H4**：確認集合ではε=.05のみHybrid（L36/q2/r8/K4）が低G。ε=.01/.005/.001/.0005ではRTEが低Gで、Hybrid/RTE=1.04035/1.04144/1.05097/1.06317。新Hybrid L1/q8/r8/K4の48本を加えても、この確認結果は変わらなかった。ただし全登録候補の母平均最適性を確認したものではない。
- **H6**：確認済みの有限G部分集合では5精度すべてHybridが低く、比は.000697795–.00211103。例えばε=.05ではHybridは約7969.46 RZ、RTEは約239.229 RZと単回価格の関係が逆であるが、Bは約1.027対271.3、Nは約1万対5.17億/軸で、総費用はHybridが低い。ε=.0005のRTE Bも約283.089である。r候補域の非対称性と4精度の未解決RTE会計を残すため、一般的なHybrid優位や全登録集合の確定winnerとはしない。

新確認は3候補×48本で完了し、7つの事前指定候補/精度セルはすべて固定延長規則を満たさなかった。96/192/384への拡張は0。[元の延長判定](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/data_archives.json)はstages/n048配下の可逆保存に含む。

## 確認結果と探索結果

以下は仮想ρ=0、有限価格標本、保存会計可能な候補内の結果である。
R=G_Hybrid/G_RTE。R<1はHybrid、R>1はRTEの点費用が低い。表示値は丸め、exact値はCSV/JSONに保持する。
「低G」は名目2 combined-SE基準で近接とされなかった経験的分類であり、母平均winnerや選抜後の厳密な信頼保証ではない。

| 分子 | ε | Hybrid L/q/r/K | RTE L/q/r/K | 確認n H/R | G Hybrid | G RTE | G SE H/R | R | 観測分類 |
|---|---|---|---|---|---|---|---|---|---|
| H2 | 0.05 | 1/1/4/4 | 0/1/2/4 | 96/96 | 625812 | 725313 | 20809/36861.5 | 0.862817 | Hybrid低G |
| H2 | 0.01 | 1/1/4/4 | 0/1/2/4 | 96/96 | 1.56457e+07 | 1.81392e+07 | 520239/921861 | 0.862538 | Hybrid低G |
| H2 | 0.005 | 1/1/4/4 | 0/1/2/4 | 96/96 | 6.2586e+07 | 7.25916e+07 | 2.08106e+06/3.68922e+06 | 0.862166 | Hybrid低G |
| H2 | 0.001 | 1/1/4/4 | 0/1/2/4 | 96/96 | 1.56525e+09 | 1.82179e+09 | 5.20463e+07/9.25859e+07 | 0.859184 | Hybrid低G |
| H2 | 0.0005 | 1/1/4/4 | 0/1/2/4 | 96/96 | 6.264e+09 | 7.32235e+09 | 2.08285e+08/3.72133e+08 | 0.855463 | Hybrid低G |
| H4 | 0.05 | 36/2/8/4 | 0/8/8/2 | 48/384 | 1.19864e+07 | 1.2602e+07 | 37995.7/44580.4 | 0.951156 | Hybrid低G |
| H4 | 0.01 | 1/8/8/4 | 0/8/8/2 | 48/384 | 3.27791e+08 | 3.15077e+08 | 2.845e+06/1.11461e+06 | 1.04035 | RTE低G |
| H4 | 0.005 | 1/8/8/4 | 0/8/8/4 | 48/384 | 1.31265e+09 | 1.26042e+09 | 1.13929e+07/4.86679e+06 | 1.04144 | RTE低G |
| H4 | 0.001 | 1/8/8/4 | 0/8/8/4 | 48/384 | 3.31167e+10 | 3.15105e+10 | 2.87429e+08/1.2167e+08 | 1.05097 | RTE低G |
| H4 | 0.0005 | 1/8/8/4 | 0/8/8/4 | 48/384 | 1.34005e+11 | 1.26042e+11 | 1.16307e+09/4.86679e+08 | 1.06317 | RTE低G |
| H6 | 0.05 | 684/1/8/4 | 0/8/4/2 | 48/48 | 1.72547e+08 | 2.47274e+11 | 29251.4/4.40065e+09 | 0.000697795 | Hybrid低G |
| H6 | 0.01 | 78/8/8/2 | 0/8/4/2 | 48/48 | 7.77669e+09 | 6.2185e+12 | 8.78226e+06/1.10668e+11 | 0.00125057 | Hybrid低G † |
| H6 | 0.005 | 78/8/8/2 | 0/8/4/2 | 48/48 | 3.19453e+10 | 2.50591e+13 | 3.6076e+07/4.45968e+11 | 0.0012748 | Hybrid低G † |
| H6 | 0.001 | 78/8/8/2 | 0/8/4/2 | 48/48 | 1.00454e+12 | 6.65499e+14 | 1.13443e+09/1.18436e+13 | 0.00150946 | Hybrid低G † |
| H6 | 0.0005 | 78/8/8/2 | 0/8/4/4 | 48/48 | 5.64832e+12 | 2.67562e+15 | 6.37868e+09/5.24433e+13 | 0.00211103 | Hybrid低G † |

†：適格性/会計domain等が未解決の登録候補が残る。表示された有限G部分集合の点順位を、登録全体の確定順位とはしない。
[精度別の未解決ID・比較scope](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/comparison_scope_status.csv)を参照。

| 分子 | ε | 探索Hybrid L/q/r/K | 探索RTE L/q/r/K | 探索G Hybrid | 探索G RTE | 探索R |
|---|---|---|---|---|---|---|
| H2 | 0.05 | 1/1/4/4 | 0/1/2/4 | 555560 | 512372 | 1.08429 |
| H2 | 0.01 | 1/1/4/4 | 0/1/2/4 | 1.38894e+07 | 1.28138e+07 | 1.08394 |
| H2 | 0.005 | 1/1/4/4 | 0/1/2/4 | 5.55602e+07 | 5.12798e+07 | 1.08347 |
| H2 | 0.001 | 1/1/4/4 | 0/1/2/4 | 1.38954e+09 | 1.28694e+09 | 1.07972 |
| H2 | 0.0005 | 1/1/4/4 | 0/1/2/4 | 5.56082e+09 | 5.17262e+09 | 1.07505 |
| H4 | 0.05 | 36/2/8/2 | 0/8/8/4 | 1.19893e+07 | 1.24666e+07 | 0.961713 |
| H4 | 0.01 | 1/8/8/4 | 0/8/8/4 | 3.20148e+08 | 3.11656e+08 | 1.02725 |
| H4 | 0.005 | 1/8/8/4 | 0/8/8/4 | 1.28204e+09 | 1.24662e+09 | 1.02841 |
| H4 | 0.001 | 1/8/8/4 | 0/8/8/4 | 3.23445e+10 | 3.11656e+10 | 1.03783 |
| H4 | 0.0005 | 1/8/8/4 | 0/8/8/4 | 1.3088e+11 | 1.24662e+11 | 1.04987 |
| H6 | 0.05 | 684/1/8/4 | 0/8/4/2 | 1.72561e+08 | 2.42774e+11 | 0.000710791 |
| H6 | 0.01 | 78/8/8/2 | 0/8/4/2 | 7.79049e+09 | 6.10531e+12 | 0.00127602 |
| H6 | 0.005 | 78/8/8/2 | 0/8/4/2 | 3.2002e+10 | 2.4603e+13 | 0.00130074 |
| H6 | 0.001 | 78/8/8/2 | 0/8/4/2 | 1.00632e+12 | 6.53386e+14 | 0.00154017 |
| H6 | 0.0005 | 78/8/8/2 | 0/8/4/4 | 5.65835e+12 | 2.80533e+15 | 0.002017 |

探索と確認をpoolしていない。探索標本の小ささ、確認後選抜、候補ごとのn差を保持する。

## 代表条件の費用内訳

Cは軸ごとの平均compiled RZ。biasはsigned cosine/sine。BとNは保存値を維持し、Nやnormalizationを再計算していない。
数値allowanceのexact値はcost_breakdown.csvのu_cos_exact/u_sin_exact列に保持する。

| 分子 | ε | 群 | B | bias cos/sin | N cos/sin | C cos/sin |
|---|---|---|---|---|---|---|
| H2 | 0.05 | Hybrid | 1.35747 | 1.80585e-07/1.58488e-07 | 12920/12920 | 24.2188/24.2188 |
| H2 | 0.05 | RTE | 1.93575 | 1.98853e-06/1.40674e-06 | 26276/26275 | 13.8021/13.8021 |
| H2 | 0.0005 | Hybrid | 1.35747 | 1.80585e-07/1.58488e-07 | 129329412/129313239 | 24.2188/24.2188 |
| H2 | 0.0005 | RTE | 1.93575 | 1.98853e-06/1.40674e-06 | 265701264/264824048 | 13.8021/13.8021 |
| H4 | 0.05 | Hybrid | 1.38375 | 0.00119806/0.000188581 | 14384/13570 | 428.792/428.792 |
| H4 | 0.05 | RTE | 1.66209 | 1.05881e-07/-7.30379e-07 | 19369/19370 | 325.305/325.305 |
| H4 | 0.0005 | Hybrid | 1.58715 | 6.86777e-06/1.11787e-06 | 183683129/177738535 | 370.771/370.771 |
| H4 | 0.0005 | RTE | 1.66237 | -1.83762e-12/1.28059e-11 | 193753833/193753845 | 325.263/325.263 |
| H6 | 0.05 | Hybrid | 1.02701 | 0.0048229/0.00728858 | 9916/11735 | 7969.46/7969.46 |
| H6 | 0.05 | RTE | 271.3 | 3.06406e-05/-2.14593e-05 | 516948681/516680062 | 239.229/239.229 |
| H6 | 0.0005 | Hybrid | 2.63064 | 7.01691e-05/0.000113624 | 755225928/1053563646 | 3122.71/3122.71 |
| H6 | 0.0005 | RTE | 283.089 | -5.39459e-09/3.81166e-09 | 5619116778073/5619066460994 | 238.083/238.083 |

## 実行・照合

追加trajectoryは144本、両軸compileは288回。新状態作用・Hamiltonian/state/reference生成は0。
人工10件PASS。保存N/B/bias/allowance、seed、compiled QPYのSHA/RZ/CX/wire、native exit0/reapを照合した。
新rawは旧形式の可逆archiveへ保存し、既存restorerで3 archive・1020ファイルの元SHA一致を確認した。元rawは変更していない。
元engine sessionからnative exit0も直接回収した。[復元結果](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/archive_restore_verification.json)と[元engine終了観測](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/engine_native_exit_observation.json)を保存した。


## 固定条件と比較の意味

Linear H2/H4/H6、隣接1.00 Å、STO-3G、中性singlet。Pauli-JWの保存Hamiltonian、各分子のraw state/ray・full-H referenceを維持した。
非identity項数Mは14/184/918。L_D=0を全ランダム化RTE、0<L_D<MをHybrid PRとする。Constant phaseは両方式で保持する。
L_D=Mは非自明なランダム化tailがない決定論的側で、RTE/Hybridの2群には混入させない。

S2、T=.8、Pauli central-rotation control、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、basis rz/sx/x/cx、opt1、seed17。
Backend/Target/coupling/layout/routingの追加指定なし。状態準備を除くfull Hadamard wrapperのcompiled RZを価格とする。
Cosineと+Im sineは同じtrajectoryから測定回路を作り、`G=N_cos*C_cos+N_sin*C_sin` で会計する。
確認nは回路価格を調べる古典Monte Carlo標本数で、保存Nはshot十分条件からの会計値である。新しい量子測定shotは実行していない。
主表示は仮想ρ=0。保存候補固有のB、bias、数値allowance、Nを転用せず、その候補・精度の保存値を使用する。
新しい状態作用、Hamiltonian/state/reference生成、Arb状態計算、shot法則・閾値の変更は0。

目的は、登録有限候補内で「一部を決定論的PFに保持する構成」がL_D=0より低い条件付きGを得たかを見ることである。
q/r/Kも候補ごとに異なる最小費用比較であり、L_Dだけを動かした因果効果や普遍的なアルゴリズム優位ではない。
特にH6の登録RTE7候補はr4だけで、Hybrid側にはr2/4/8がある。未登録RTE r8等まで含めた最適化優位は主張しない。

## 保存探索と独立確認を分離する

| 分子 | RTE登録 | Hybrid登録 | 元の独立確認 |
|---|---|---|---|
| H2 | 13 | 51 | RTE1・Hybrid1候補、各96本 |
| H4 | 13 | 128 | RTE2候補、各384本；Hybrid2候補、各48本 |
| H6 | 7 | 94 | Hybrid4候補、各48本；RTE0候補 |

H4 Hybrid128のうち1件は歴史的p027/q8/r4/K4・256本。実exploration140候補のminとこのhistorical baselineを分離した。
新表は `exploration` と `confirmation` を別々に最小化し、探索標本と確認標本はpoolしない。
同じ候補の価格標本を5精度で再利用する。15精度セルは15個の独立な価格実験ではなく、表の行数から新しい標本数を足し上げない。
旧H4 primaryは探索nomineeを確認前に固定している。今回の確認群minは事後選抜として表示し、旧primaryを別列に保持する。
H2/H6の旧主表も確認集合内の事後最小を含む。これらを一意な母平均winnerや厳密な同時信頼保証に読み替えない。

Gがnullの候補は0/∞として順位付けしない。比較は適格・保存会計可能な部分集合の観測minで、未解決IDを残す。
H6の一部RTEでは保存N/Gが `SAFE_INTEGER_LIMIT`、最厳精度の一例では `FIXED_GUARD_FAILED`/数値未確定である。
それらは全パラメータでの不可能性や無限費用の証明ではなく、今回の会計domain・数値証拠の限界である。

## 追加確認の設計と再利用

保存探索Gだけから不足セルのnomineeを選び、ID・各精度の保存target/N selector・SOURCE・入力・seed/indexを新sampling前に固定した。
追加はH4 `p001_q8_r8_K4`（ε=.001/.0005）と、H6 `PR_p0000_q8_r4_K2`（ε=.05/.01/.005/.001）、`PR_p0000_q8_r4_K4`（ε=.0005）。
新H4候補の確認価格を他の精度へ使う場合も元の候補固有Nを維持し、その精度の確認集合内minは事後選抜として扱う。
H2の新samplingは0。新規候補探索や別geometryへの拡張は行っていない。

各追加候補を48 paired trajectoryから開始し、固定規則のときだけ96→192→384へ追加する。
規則は、方式間の観測G差がcombined SEの2倍以内、かつ新候補の名目t95半幅がGの1%を超えること。
旧比較候補のSEが支配して追加では解けない場合は、成功済み旧cohortを再計算せず未解決で停止する。
この名目規則はtail・選抜・逐次観察の厳密なcoverageを保証しない。

Pauli adapter、RTE sampler、D2 control、wrapper、compiler、H2/H6 price_job、既存paired/cost_summary/_statisticsを再利用した。
H6は旧price_jobを独立child内で新OUTへ接続し、固定model/referenceの同一bytesを参照する。
H4は準備版のcentral_trajectoryにaudit引数がない版差を、既存AuditedPauliSignalAdapter＋D2＋wrapperの小さい接続で吸収した。
人工2qubitで元のbody/wrapperと一致し、旧科学SOURCEを変更していない。

SeedSequenceの元規約はH4が `[20261025,3101,L_D,q,r,K,202,index]`、H6が `[20261025,6,L_D,q,r,K,202,index]`。
同候補の旧探索seedとの非重複、既存確認の不在、計画した新seed同士の一意性を確認した。
両軸価格をtrajectory単位で重み付けしてGのSEを求め、cosine/sineを独立と仮定してSEを合算しない。

資源観測では64コアがほぼ使用中だったため、新jobを自分の低優先度workerへ投入した。
worker数は実測の空きCPUから選び、CPU/RAMの固定人工上限や共有環境の変更は設けていない。
旧workerの停止・再起動、他ユーザーのprocess/file/設定変更は行っていない。

## 新確認の高次event観測

| 対象（各48 trajectory） | order0 event | order2 event | order4 event |
|---|---|---|---|
| H4 L1/q8/r8/K4 | 3060 | 12 | 0 |
| H6 L0/q8/r4/K2 | 1398 | 138 | 0 |
| H6 L0/q8/r4/K4 | 1400 | 132 | 4 |

これはeventの個数でありtrajectory数ではない。K2のorder4はlawのsupport外であり、K4の未観測order4とは意味が異なる。
H6 K4ではorder4を4 event観測したが、母平均tailや全eventの費用上界を保証する証拠にはしない。H4の未観測order4の寄与も0としない。

## 証拠と保存先

- [15精度セルの探索・確認比較CSV](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/comparison_by_precision.csv)
- [全選択行・tie・未解決IDを含むJSON](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/comparison_by_precision.json)
- [選択候補の費用内訳CSV](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/cost_breakdown.csv)
- [新候補の会計CSV](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/new_candidate_accounting.csv)、[新価格統計](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/new_price_statistics.json)
- [SOURCE](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/SOURCE.json)、[scope](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/active_scope.json)、[事前固定nominee](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/nominations.json)
- [完了照合](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/completion_verification.json)、[保存データ入口](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/README.md)

固定元資料は統合 `e2b9215fd7432e7406ef8c7b8b5f55fba563a696`、H4探索 `d3fdfad880902b7ff35b710d22e1c9759f7087c2`、H2/H6探索 `8fd1ceb283309a7e515ff4a533a253260bdd0f78`。
取得済み固定ref・各SHA・候補別cohortの来歴は[準備報告](pauli_hybrid_rte_preparation_20261025.md)と同保存manifestに保持した。
新旧SOURCE、raw、grant、ledger、marker、STOP、旧15行統合CSVを混合・上書きしていない。

## 次に判断が必要な範囲

今回許可された有限候補・有限確認標本の比較は完了した。現在の条件付き観測を報告するだけなら、追加科学計算は必須ではない。
より強い主張を選ぶ場合に限り、ユーザー/GPTが次の必要性と実行範囲を判断する。

1. **H6の比較条件**：RTE側に未登録r8等を含める価値、SAFE_INTEGER_LIMIT/数値未確定行を扱う必要性。パラメータ候補追加、target/N評価や数値domain変更は今回範囲外であり、Codexによる実行は別の明示承認を要する。
2. **母平均費用**：高次event費用上界または別設計の統計保証。確認n48/96/384だけでtail保証にしない。新samplingや保証証明は今回完了後に自動実行しない。
3. **実精度**：実compiledρ/全law転送の必要性。既存H4の6局所Arb結果はL27/q8/r4/K4のselected trajectoryに限られ、今回の新候補やL0 law全体の保証へ転用しない。新状態作用・Arb評価・R2-C等は別承認。

優先度は主張scope次第であり、今回は新しい研究方針や主要claimを採択しない。

## 未保証事項と停止

実compiledρ、PR全law残差、母平均価格tail、大域最適性は未保証のまま。
有限確認標本の平均・名目SEに基づく費用比較を、無条件のend-to-end資源優位や母平均winnerへ格上げしない。
希少eventの未観測を0価格とせず、観測があっても全eventの費用上界とは呼ばない。
H2/H4/H6ではstate/reference・候補集合・確認n・選抜履歴が異なる。一般的な分子サイズ依存則や全RPE費用へ一般化しない。
今回の技術照合は著者Codexによるもので、独立科学レビューではない。

結果保存・commit/公開・local/remote SHA照合後にmandatory STOPする。
次の担当はユーザー/GPT：条件付き有限候補の比較を研究上どう位置付けるか判断する。Codexは新計算、主要claim採択、論文執筆、詳細科学レビューへ自動移行しない。
