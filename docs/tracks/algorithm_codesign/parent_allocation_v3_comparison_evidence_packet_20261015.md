# Parent allocation v3：GPT科学レビュー用比較Evidence Packet

2026-10-15 JST。保存結果の抽出・索引化のみ。科学的採否／新規性／一般化を判断せず、GPTレビューは開始していない。

## 正本と今回の取得確認

- 固定公開commit：`cd055232f9612982a60953341b9cbf6168dd0bd8`。
- Science source S3：`53a36adf9bdd84690e9cdf3e1f97f38a178441dc`。
- 実行authorization-only A3：`b1dfe53ca98a80d60831dbf9d28c5c9fc6a08f31`。
- 原classification：`PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW`。既存918監査PASSを参照し、再実行しなかった。
- GitHubから必要15資料を取得し、固定Git blob／local原本のbytes一致、manifestとterminalのSHA256、S3／A3参照を照合した。
- Resource map：3,647,912 bytes、SHA256 `a781242e3136c51253862d9569388c0ac7fc9093d4c1f28c2fbe378523ba94fe`。
- 固定contract SHA256：`adf2a75ecaadbd012f1ec20bb6b741979e5542614f151da6988d26938dfd82ad`。
- 新科学計算、N／forecast／交点／包絡線生成、再ランキング、合成、operator guard、science source import：すべて0。

正本：[v3結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/docs/tracks/algorithm_codesign/parent_allocation_v3_results_and_gpt_handoff_20261015.md)、[GPT採択済み科学設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/docs/research/track_b_parent_allocation_scientific_design_review_20261014.md)、[資源表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json)、[evidence manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/evidence_manifest.json)、[918保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/saved_output_audit/saved_output_audit_v3.json)、[contract v3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/execution_contract_v3.json)。

大きな資源表のlossless取得：[part0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json.part0000.txt)、[part1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json.part0001.txt)、[part2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json.part0002.txt)、[part3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json.part0003.txt)、[part4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json.part0004.txt)、[part5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json.part0005.txt)、[part6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd055232f9612982a60953341b9cbf6168dd0bd8/artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v3/resource_map.json.part0006.txt)。順序どおり結合した原identityは既存manifest／監査を参照する。全70 saved rowは各10,684 bytes以下でCSV参照先から取得できる。

対象は既知development用3-qubit distinct-basis synthetic、p=(1/5,3/10,1/2)、x=tau=5/7、状態000、sigma+1、finite P5、r1。
Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4)R_ZZ01(pi/4)、Qi=Vi†ZiVi。
分子geometry／basis／DF rank／split L_Dは対象外。primitive1e-6、root bits256、proposal bits160、rho=eta=1e-12、strict phase100dps、axis精度1/200、alpha_axis1/40。
費用は十分試行数Nを使う加法的nativeモデルの`2 N E[C_j] + 2 N h_j`。実測quantum shots／必要最小shots／whole-circuit最適費用ではない。
Tはnative1Qに含まれるため合算しない。hは指標ごとの共通準備・読出しgate単価で、古典秒との換算ではない。

## A：15構成・70 lawを追跡する索引

[全law索引CSV](../../../artifacts/track_b_parent_allocation_review_packet/2026-10-15/v1/law_comparison_index.csv)は、
config／law ID、arm／候補／proposal、適用性、N、moment／range、mass／acceptance、各bias／統計幅、
一試行単価、T／CX／native1Q切片、primary、準備費用slope、workspace、online費用、原row／law／eventのpath・SHAを対応付ける。
**70実law＋5正常適用外slot＝75索引行**。5適用外にNや費用の0を補っていない。

- `_exact`は保存string／integerをそのままコピーした値。
- `_display`は表示専用の小数丸め（scalar `.9g`、費用 `.2f`）。moment／range／biasの長いexact有理数は複製せず、原rowの同名ledger fieldへ戻る。
- 原rowへの`resource_map_pointer`、row SHA、fixed commitは[抽出JSON](../../../artifacts/track_b_parent_allocation_review_packet/2026-10-15/v1/comparison_extract_and_identity.json)とCSVで特定する。
- 下表の費用・差の単位は**百万T**、小数6桁への表示丸め。差・順位の判定は表示丸めから作らず、既存exact relationへ戻る。
- 欠測でなく正常適用外なのは`config_02`／`03`／`08`／`09`／`14`のanalytic IS。positive supportのzero Tが原因で、失敗・性能negativeではない。

今回コピーした比較はB1の46組×3指標、B2の55組×3指標、B3の2組×3指標＝309既存affine records。
値・left/right・classification・crossoverを変更せず、原`/comparisons/affine_relations/<index>`への参照を付けた。
原row70件と全コピーrecordの参照・コピー一致を確認した。momentsやNの元計算を再監査した意味ではない。

## B1：同じproposalでcommonと配分候補を対応付ける

下表は**保存された差 `G_common,T(0) − G_candidate,T(0)`**の表示。
正ならこの組合せでは配分候補側が低T、負ならcommon側が低T。
両armでcommonからvertex_0／1／2、KKT、halfwayへの全適用可能組を掲載する。
proposal名・正則化scaleは同一でも、係数／unitary／価格が変わるため実qは同一とは限らない。

| 候補 | canonical | analytic IS | regularized C*/16 | regularized C* | regularized 16C* |
|---|---:|---:|---:|---:|---:|
| config_01 / ordinary / vertex_0 | -62.506668 | 10.265501 | 4.261141 | -32.981371 | -59.298628 |
| config_02 / ordinary / vertex_1 | -15.689226 | 適用外 | 35.276289 | 5.297082 | -13.528920 |
| config_03 / ordinary / vertex_2 | -3.221357 | 適用外 | 36.256680 | 12.405346 | -1.629134 |
| config_04 / ordinary / kkt | 2.386513 | 2.427117 | 2.427336 | 2.417104 | 2.390867 |
| config_05 / ordinary / halfway | 1.221330 | 1.245582 | 1.245670 | 1.239316 | 1.223661 |
| config_07 / closed_P5_full / vertex_0 | -12.546395 | 36.006195 | 31.633231 | 5.779921 | -10.665373 |
| config_08 / closed_P5_full / vertex_1 | -9.694072 | 適用外 | 30.810986 | 6.976310 | -7.978965 |
| config_09 / closed_P5_full / vertex_2 | 4.297915 | 適用外 | 34.641906 | 16.168617 | 5.496650 |
| config_10 / closed_P5_full / kkt | -4.761353 | -4.766839 | -4.767483 | -4.766934 | -4.762347 |
| config_11 / closed_P5_full / halfway | -6.198518 | -6.200006 | -6.200634 | -6.201534 | -6.199245 |

保存値の読取りとして、ordinaryのKKT／halfwayは5 proposalすべてで正の差、closed P5のKKT／halfwayは5 proposalすべてで負の差。
ordinary vertex_0やP5 vertex_0などは、canonicalとIS側で差の符号が異なる。
したがって「同一proposal条件で配分側の低T点がある」と「全proposal・全配分で利益がある」は別の内容。
頂点と内部候補を一つの成功／失敗へ統合しない。KKTは旧common価格の固定surrogate候補で、native最適内部点ではない。

## B2：配分を固定してproposalを並べる

各値は保存済みprimary切片の表示で、並び順は元のconfig／proposal順。
各configの新しい勝者を計算・選定していない。canonicalから各ISへのT／CX／1Q保存relationは抽出JSONのB2にある。

| 構成 | canonical | analytic IS | regularized C*/16 | regularized C* | regularized 16C* |
|---|---:|---:|---:|---:|---:|
| config_00 / ordinary / common | 258.668184 | 258.691510 | 258.687337 | 258.666469 | 258.666316 |
| config_01 / ordinary / vertex_0 | 321.174852 | 248.426009 | 254.426196 | 291.647840 | 317.964944 |
| config_02 / ordinary / vertex_1 | 274.357410 | 適用外 | 223.411048 | 253.369387 | 272.195237 |
| config_03 / ordinary / vertex_2 | 261.889542 | 適用外 | 222.430657 | 246.261123 | 260.295451 |
| config_04 / ordinary / kkt | 256.281672 | 256.264392 | 256.260001 | 256.249365 | 256.275449 |
| config_05 / ordinary / halfway | 257.446854 | 257.445928 | 257.441667 | 257.427153 | 257.442655 |
| config_06 / closed_P5_full / common | 189.773346 | 189.782118 | 189.780105 | 189.771852 | 189.772560 |
| config_07 / closed_P5_full / vertex_0 | 202.319741 | 153.775923 | 158.146874 | 183.991931 | 200.437933 |
| config_08 / closed_P5_full / vertex_1 | 199.467418 | 適用外 | 158.969119 | 182.795542 | 197.751524 |
| config_09 / closed_P5_full / vertex_2 | 185.475432 | 適用外 | 155.138200 | 173.603234 | 184.275910 |
| config_10 / closed_P5_full / kkt | 194.534700 | 194.548957 | 194.547588 | 194.538786 | 194.534907 |
| config_11 / closed_P5_full / halfway | 195.971865 | 195.982124 | 195.980740 | 195.973386 | 195.971805 |
| config_12 / partial_return_tail / common | 191.990315 | 191.992037 | 191.987891 | 191.973637 | 191.986451 |
| config_13 / closed_P3_tail / common | 198.483503 | 198.517980 | 198.514638 | 198.494280 | 198.483971 |
| config_14 / matched_CTS / common | 270.222826 | 適用外 | 237.608885 | 257.908188 | 268.989877 |

common内のproposal変更と配分を変えた後のproposal変更を別々に参照できる。
一例としてordinary vertex_0の保存T切片はcanonical約321.174852百万、analytic IS約248.426009百万。
P5 vertex_0は約202.319741百万と約153.775923百万。
この並置は同一配分内の既存IS効果を示す証拠であり、best-to-best差をすべて配分単独の寄与へ帰属させない。
参照partial／P3+tail／matched CTSにも同じproposal機会があるが、任意parent配分を今回追加した全面比較ではない。

## B3／C：保存済みbest-to-bestと複数資源の交換

`/comparisons/mechanisms`に元々記録されたcommon／noncommonの組だけを抜粋する。
ordinaryはcommon regularized_16対vertex_2 regularized_1/16、P5はcommon regularized_1対vertex_0 analytic IS。
**B1の同一proposal比較と混同しない。**

| 保存law ID | N/axis（exact） | E[T]/trial | G_T(0) | G_CX(0) | G_native1Q(0) | second moment | range |
|---|---:|---:|---:|---:|---:|---:|---:|
| config_00:regularized_16 | 924186 | 139.9428 | 258666316.46 | 15263828.18 | 694789433.91 | 2.256285 | 1.515636 |
| config_03:regularized_1/16 | 1587522 | 70.0559 | 222430656.80 | 21883969.29 | 590555983.32 | 3.874503 | 3.076045 |
| config_06:regularized_1 | 680318 | 139.4729 | 189771851.79 | 11574441.76 | 479835618.53 | 1.660139 | 1.334415 |
| config_07:analytic_IS | 2270017 | 33.8711 | 153775923.47 | 29657567.70 | 414245982.26 | 5.538145 | 5.114787 |

両組でnoncommon側はT／native1Q切片が低く、N／CX切片は高い。
second moment／rangeも上がっている。**複数資源座標のdominanceと呼ばない。**
費用差は元mechanism／affine recordにexactで保存されており、新しい百分率や寄与率を計算していない。

係数・strict native誤差・bias・統計残余幅の対応は次のとおり（表示9桁）。

| 保存law ID | coefficient mass | coefficient error上界 | weighted native error上界 | axis bias上界 | 統計残余幅 |
|---|---:|---:|---:|---:|---:|
| config_00:regularized_16 | 1.50209305 | 1.29542528e-77 | 3.35221582e-07 | 0.000369587967 | 0.00463041203 |
| config_03:regularized_1/16 | 1.69219435 | 1.29542528e-77 | 2.67728851e-07 | 0.000369452982 | 0.00463054702 |
| config_06:regularized_1 | 1.28844456 | 4.31808428e-77 | 4.02440459e-07 | 0.000369722405 | 0.0046302776 |
| config_07:analytic_IS | 1.61654314 | 4.31808428e-77 | 2.35537098e-07 | 0.000369388598 | 0.0046306114 |

Taylor上界は共通保存値`3125/16941456`。
係数L1誤差と`weighted_native_error_upper`を別々に保存し、後者にはexact ideal angleへのstrict guardが含まれる。
angle誤差を二重に加算しない。dyadic確率の丸めは実qでweightを補償し、保存`probability_rounding_mean_bias=0`を維持する。
配分を固定したproposal間では同じ実装meanのbiasを使うが、配分間ではbiasも変わり得る。
T価格差と統計予算差を独立に加法分解した新しい寄与率は取得していない。

### 合成価格・取得費用・古典費用

- inventoryは旧20 unique keysの再利用＋新16 positive root keys、positive／inverseは保存sequenceのactual adjointを共有。
  新16合成は各一回・strict phase1e-6で、既存918監査がsequence／T・T†／1Q／errorの保存を確認済み。
- rootの端点はexact処理でもprovider、controlled phase、pure-Qの費用が残る。zero Tをzero全費用としない。
- 候補／C*／inventoryは新価格前freeze。KKT・halfwayの価格を見て配分を再選択していない。
- 新角度のratio／tool／seed／取得区分は固定inventory、実sequence価格とworker CPU／RSSは旧監査JSONの`new_worker_receipts`へ戻る。
  合成取得は一回の実験setupであり、N倍のquantum費用や選択law固有のonline生成費用と混ぜない。
- `classical_setup`の候補生成CPUとconfig別law／ledger CPUは取得済み。
  共有acquisition・ledger setupは選択されたアルゴリズムのsetup費用と同一ではない。
  **online sampling／generation費用は全lawで`NOT_MEASURED`**。未知費用を0に置換しない。
- 実価格に条件付けた費用関係であることは追跡可能。
  ただし新角度key別の期待寄与分解、counterfactual price、別tool／precision／seedでの順位不変性は今回抽出できない／未取得。
  本作業で新しいincidence集計・価格感度・合成を行っていない。

## D：既に保存された準備・読出し費用の関係

h=0の全70 lawの最良IDは、元の`finite_registered_best`をコピーした**`config_07:analytic_IS`**。
下表はB3の同じ固定law同士の保存affine関係。ΔG=common−noncommon、Δslopeも原値であり、交点を再計算していない。

| arm | 指標 | 保存Δ切片（丸め） | 保存Δslope（exact） | 保存crossover（丸め） | 元classification |
|---|---|---:|---:|---:|---|
| ordinary | T | 36235659.66 | -1326672 | 27.3132015 | RESOURCE_TRADEOFF |
| ordinary | CX | -6620141.11 | -1326672 | 保存値null | LEFT_NONWORSE_ALL_NONNEGATIVE_H |
| ordinary | 1Q | 104233450.59 | -1326672 | 78.5676117 | RESOURCE_TRADEOFF |
| closed_P5_full | T | 35995928.32 | -3179398 | 11.3216176 | RESOURCE_TRADEOFF |
| closed_P5_full | CX | -18083125.94 | -3179398 | 保存値null | LEFT_NONWORSE_ALL_NONNEGATIVE_H |
| closed_P5_full | 1Q | 65589636.26 | -3179398 | 20.6295771 | RESOURCE_TRADEOFF |

- ordinaryのT：保存交点h≈27.3132015より下ではnoncommon側が低く、上ではcommon側が低い。交点では同額。
- P5のT：同じ意味でh≈11.3216176。native1Qの交点はordinary≈78.5676117、P5≈20.6295771。
- CXは両組で`LEFT_NONWORSE_ALL_NONNEGATIVE_H`、ここでLEFTはcommon。全h≥0でcommonがnon-worseという元の分類を保持する。
- T／CX／1Qのhは別の単位・別の座標。上の三つの条件を足したり、単一scalar条件へ変換したりしない。
- B1同一proposalとB2内のproposal比較にも、既存T／CX／1Q relationとcrossoverを索引化してある。
- これらは**指定二lawの関係**であり、交点後の全70 lawの最良方式を確定しない。
  全hの最適包絡線／h別の最良配分・proposalは正本に保存されていないため**未取得**。本作業で生成しない。

## E：GPTレビューの7問に対応する証拠と境界

| 問い | 保存済み事実／参照 | 未取得情報 | GPTが判断すべき論点 |
|---|---|---|---|
| 1 同一proposalでも配分側に低T点があるか | B1の46組。両armに正の差がある組合せがあり、ordinaryの内部点とP5の内部点は方向が異なる | 連続配分family、未登録proposal・入力 | 固定proposal規則下の差をどの改良機構として評価するか |
| 2 best-to-bestと既知ISを区別できるか | B1とB2とB3を分離。commonにも同じIS機会。B3のproposalは左右で異なる | 配分×IS×bias×単価の一意な寄与分解、最適SPS等との全面比較 | 条件付き改良と独立新規性の境界 |
| 3 ordinary／P5の共通点と相違点 | 両B3にT低下・N/CX増加。内部候補の同proposal差はordinaryで正、P5で負。元の全体最良はP5 vertex_0 analytic IS | 他provider、他rでの再現、return集約固有の因果分解 | 配分改善とreturn集約の追加価値をどう区別するか |
| 4 頂点と内部点はどう違うか | B1/B2、保存freeze配分。頂点はproposalで符号が変わり、KKT／halfwayはarm間で異なる | native最適内部点、頂点／内部family全体の優劣 | endpointの回転除去とsurrogate内部移動の意味・限界 |
| 5 T利益にN/CX/hが与える制約 | B3の全資源表とDの保存交点。T低下は全資源dominanceではない | 全h包絡線、実際のprep/readout単価、online classical費用、最小shots | どのresource座標／実装条件で有用性を認めるか |
| 6 実際の回転価格への依存をどこまで確認できるか | 固定1e-6／tool／seedの旧20＋新16価格、保存IRとguard、T単価、候補before-price freeze | key別価格感度、別精度／合成器での頑健性、whole-circuit最適化 | 特定価格の機構証拠を一般的選択則へ発展できるか |
| 7 今の結果で判断できないもの | science adoption／priorityは元からfalse。finite synthetic比較の完遂のみ | 独立新規性、一般化、全既知法への優越、分子PR/QPE総資源、研究継続・論文化の十分性 | 継続／縮小／終了と、追加取得の必要性・範囲 |

未取得項目の存在を根拠に、新実験や研究方針を採択しない。

## 成果物identityと停止

| 資料 | bytes | SHA256 |
|---|---:|---|
| law_comparison_index.csv | 84,807 | `25370d9e59a41734f24750eea3280fa899cd1d0593d13b6dbd8429fc82486945` |
| comparison_extract_and_identity.json | 253,219 | `878c6e651ae7a35ee5f2852177513bd791cfab511133b82ebb95bf1984456062` |

本Markdownと上の二つの索引だけを新資料として公開する。いずれも512 KiB以下。
詳細監査・原出力・source／contract／authorization／marker／STOPは複製・変更しない。
親worktreeの既存dirty変更とindexを保護する。
公開前の索引・原値コピー・参照リンク・保全確認は450件PASS。元の918項目の科学値監査とは別の資料整形確認である。
親HEAD／index／70 status entries／131既存ファイルは不変。

**mandatory STOP。** GPT重要科学レビューは利用者の明示的な開始承認後。
科学計算も別途の明示承認まで開始せず、v1/v2/v3再実行、native合成／guard、追加provider／precision／rを行わない。
