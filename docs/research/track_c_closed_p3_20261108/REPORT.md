# Track B Closed P3 × Track C改良PR：固定H4統合検証

2026-11-08 JST。専用branch `codex/track-c-closed-p3-integration-20261108`。
科学基準は `94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9`、作業基点は文書総括 `100377dac8635fc49074147bfd8b42f52f2b7311`。科学SOURCEは変更前と同じである。

**Closed P3をCの218成分へ接続し、同じ有限P3・D構成・h/resolutionで全18roundの条件付き評価まで完了した。** 独立paired holdoutによる総RZは **1.422901381e12 → 1.394938574e12、1.9652%減**。CXは1.9722%、modular serial depth上界は1.9628%、sizeは1.9655%減。量子資源の利益は主に高価なroundの十分trial数減少による。**強Full PFはなお低コスト**で、PR全体の優越や実行時間の短縮は成立していない。

結果の正本は [FAIR_FINAL.json](../../../artifacts/track_c_closed_p3/2026-11-08/FAIR_FINAL.json) と [FAIR_ROUND_RESOURCES.csv](../../../artifacts/track_c_closed_p3/2026-11-08/FAIR_ROUND_RESOURCES.csv)。以下の数値は今回の固定条件の新しい計算結果であり、過去報告の転載と区別する。詳細GPT科学レビュー・別AIによる独立レビュー・実機測定は行っていない。公開・照合後 **mandatory STOP**。

## 1. 固定条件と両サーバー調査の照合

Linear H4、1.00 Å、STO-3G、保存DF12・generation order、Partial LD3（D4 block・R218 signed Z/ZZ成分、7 basis）、Full LD12。8 system qubits、4電子sector70、保存元Hのexact groundを利用可能と仮定し、状態準備費を除外した。18round、δ0=.02、ε_E=1.5936e-4 Ha、α総和≤.05、位相予算.4 rad。

D内部BCH係数、各roundの既存case/h/resolution、共同isometry、対角融合、Qiskit1.3.0/reference A/rz-sx-x-cxを保持した。m17はD調整0321、h=.32、resolution64、8192 D/R step、8388608 RTE microstepのまま。Rの次数はK2/P3から変更しない。P5、新分割、新分子、新PF設計、General Full/Partialの追加実験は行っていない。

- SERVER 2 [接続調査4d48a6f2][S2]とSERVER 1 [既存成果調査a8299889][S1]は、K2=P3、G7 closed P3のm3適用、signed-Q・語順・相対位相・weighted全trial会計が必要という点で一致した。SERVER 1の報告とevidence inventoryを実際に取得し、旧「未取得」を解消した。
- SERVER 1の旧local inventory、C1の324 label/別partition、m5のH4試行を今回のC入力へ移していない。旧G9/G10/C1の負結果・技術STOP・研究凍結は保持した。本件はユーザーが別途許可した新しい固定H4/P3研究である。
- 収録前のC総括 [100377da][CSUM]は上書きしていない。旧private common cacheのSHA256は `68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。新SCF・DF fit・ground solveは0。

## 2. 再利用SOURCEと新しいadapter

| 既存資産 | 再利用と今回の差分 |
|---|---|
| G7 `CanonicalGenerator` / `g7_generator.py` | `m=3, arm='closed_P3_tail'`で係数、group、conditional law、実proposal、補正weightを生成。conditional/eventのcacheだけを加え、samplerの係数式を再実装しない |
| G6 `ReturnKernel` | 人工非可換例でparent係数・angle比の参照に使用。一般Fullの研究を再開していない |
| `dyadic_distribution`, `dyadic_index`, F3 `enclose_atan` | 256-bit root/確率、η=ρ=10^-24、64項angle enclosureを再利用。Full用F3 preflightをcanonicalの証拠へ転用していない |
| C1 `whole_path.AllTrials`（af75af03） | 元classを固定Git objectから読み、人工全trial・zero・failure・未完走拒否を検査。元のsynthetic/m5 runnerを改造して再開しない |
| C `track_c_h4_finite_bits.weight_interval` | 丸めたevent weightの頻度で圧縮したpath積を100 dps区間評価。巨大な有理数の逐次積を強制しない |
| C `schedule`, `split`, D価格、`assemble`, `wrapper` | D/R順、D/D junction、scalar、X/Y、決定論価格を保持 |
| C partial QR/共同isometry/`diagonal_word`/`emit` | 元の列projector、same-basis共有、未登録full fallback、exact角度和、parity合成を使用。新eventの明示angle/applicationだけを接続 |
| C `compile_reference`、旧source/native/basis区間証拠 | controlを付けたjoint回路をlower。新しい中央回路はnative RZ/CXと既存CZ/P。旧角度表だけで新RA角度を認定しない |
| C `allocate`, `rigorous_shots`, RPE decoder | 認証rangeでスケールしたweighted変数へ接続。training-only α配分とE0非依存decoderを再利用 |

新SOURCEは `src/trottertracks/track_c_closed_p3*.py`、入口は `scripts/run_track_c_closed_p3.py`。既存SOURCEファイルは変更していない。新しいapproval証明・grant・receipt管理・認証基盤は作成していない。

## 3. 同じP3平均への数学的接続

Cの保存binary64確率をexact rationalとして正規化し、p_i>0、Σp_i=1、符号をQ_iへ吸収する。Q_i=B_i Z_S B_i†、Q_i²=I。identityはCの既存scalar側に残す。x=F(λ_R)F(t_R)/n、σは時間方向として分離し、全使用条件で0<x≤1を確認した。

\[
A=\sum_i p_iQ_i,\qquad
M_3=I-i\sigma xA-\frac{x^2}{2}A^2+\frac{i\sigma x^3}{6}A^3.
\]

CのK2は次数3まで残すpaired Taylorである。Bのm3 closed P3は同じM3の形式returnを集約する。χ=Σp_i²、μ3=Σp_i³として空parentは

\[
a_0=1-\chi x^2/2,\quad s_0=x-(2\chi-\mu_3)x^3/6,
\quad a_i=p_i[x-(2\chi-p_i^2)x^3/6].
\]

相異なるj,kのparentではa_jk=(x²/2)p_jp_k、s_jk=(x³/6)p_jp_k(1−p_j)、tanφ_j=x(1−p_j)/3。単一Q_childのrotationとordered parentをpairする。operator語(j,k)の回路時間順は(k,j)、even parentの位相−1、負Qのproduct位相を保持する。異なるbasisのreorderや、一般の奇数語をinvolution扱いする操作はない。

C入力のχは約 **0.029775729744**。全12使用cellで、groupを解析的にfactorしたH4 sector70の平均をP3行列と照合し、最大作用素差 **1.66534e-15**。218成分の正規化generatorと保存tail行列/λ_Rとの差は **7.44714e-16**。これは数値実装検査であり、数値一致を一般恒等式の代用にはしない。理想同値性は既存G6/G7の構成とその適用条件を使う。[STAGE1.json](../../../artifacts/track_c_closed_p3/2026-11-08/STAGE1.json)

人工非可換例はσ±、負Q、identity labelを含み、exactなπW=α̃、proposal和1、range/M2上界を全eventで確認した。別々の非可換microstepのweighted平均が**正しい作用順の積**になり、逆順とは異なることも検査した。focused testsは3件PASS。

## 4. Finite-bit、weight、統計会計

### 4.1 実proposalは理想normalizationと別

理想closed P3はB_RA=Σgroup rootをnormalizerとし、weightが定数になる。実dyadic proposal π_eではroot midpoint係数α̃_eを使い、w_e=α̃_e/π_eはevent依存である。全218 conditional law、collected-root law、group interval、positive supportを検査した。実mapはG7のbit mapと照合して一致した。

最大4つのcategorical factorから、全eventについて

\[
|w_e|\le W_*=(1+\rho)B_{RA}/(1-\eta)^4,
\qquad E[Y^2]\le W_*^2.
\]

M2はこの認証range²の保守上界を採用し、観測weight最大・経験的二乗平均を使わない。root係数・angle enclosure・binary64 angle丸めのmean biasを別に加算した。πの丸めはweightで補正するため、同じproposal差を追加mean biasとして二重計上しない。

### 4.2 Ordered pathと有限算術

\[
\widehat z_b=\frac1{N_{trials}}\sum_a
 A_a\Bigl(\prod_j w_{a,j}\Bigr)Z_{a,b},\qquad Z_{a,b}\in\{-1,+1\}.
\]

分母は全trial。今回はclosed P3なので正当なrejection/zero質量は0、acceptance=1、期待quantum callsと十分trial数は一致する。技術failureはzeroにせず、未完走prefixから正式meanを作らない。この区別は既存C1 accumulatorを使った人工検査でも確認した。fresh独立bits・trial間独立性は明示したモデル条件で、固定PRNG traceの一致を物理RNGの認証と呼ばない。

実行可能なweight算術として、eventのexact wを1回binary64へ丸め、同値weightの頻度を圧縮し、既存区間積を使用する。event丸めu=2^-53、最終float丸めと区間幅≤10^-24を**rangeとmean biasの両方へ**計上した。新しい全path rangeは概念的にΠW_*×(1+u)^N×(1+u+10^-24)。実際の登録path長・正のweight範囲で積の区間精度が保たれる。精度不足やoverflowを正当なzeroにしない。

有限非unitary meanの積にはTaylor remainderと実装差を含むtelescopingを用いる。旧source/native/basisの一様diskは保守的に保持し、新しいroot/angle/weight算術分だけ追加する。新しいnative角度は明示的に囲い、native RZのparameter pass-throughとbinary半角、既存の角度和丸め上界を使う。

### 4.3 Weighted測定の十分trial数

full-pathの認証上界L_mでYを割れば、Y/L_m∈[−1,1]。その**スケール済みweighted平均**の半径下限をrho_mと定義し、

\[
N_{m,b}\ge\left\lceil\frac{4\log(2/\alpha_{m,b})}
 {\rho_m^2\sin^2\beta_{stat,m}}\right\rceil
\]

を適用した。無重み物理Hadamard平均にそのまま旧shot式を当てたものではない。36軸α和をexact rationalで.05以下にし、192-bit区間算術で整数Nを取得した。

m17のcanonical log Bは約.27869543、Closed P3のlog path rangeは約.27039707。rho下限は約.756674→.762979。全round最小rhoはcanonical **.572700145**、closed **.582284385**。最小βstatは約.366215976/.366215973で、追加weight算術等を0扱いしていない。[FINITE_ARITHMETIC_QUALIFICATION.json](../../../artifacts/track_c_closed_p3/2026-11-08/FINITE_ARITHMETIC_QUALIFICATION.json)

## 5. 回路・native correctness

明示RA eventをproduct/rotation applicationへ写し、最大same-source-basis run、個別列projectorを保持する共同isometry、full/singleton fallback、効率的単一Pauli control、対角融合へ接続した。sampling lawの異なるeventを旧order-only samplerに偽装していない。

7 basisと符号/σ±/非空parent/共有runを含む9組×3合成方式、計27件の全512次元joint operatorを検査した。

| 検査 | 最大差 |
|---|---:|
| 元のsigned Fock演算子からのsource joint Frobenius差 | 1.43533e-14 |
| native joint、1個の共通global phase下のFrobenius差 | 1.51841e-13 |
| X/Y native readout差 | 4.21885e-15 |
| D/R・scalarを含む1/2step、4 X/Y条件の独立sector参照との差 | 2.69784e-13 |

inactive branchにも同じ共通phaseを用い、branch別phase fittingをしない。basisのactual逆演算を使い、negative angleをwhole-wordの随伴と取り違えない。625採用isometryとlegacy singleton/full fallbackの既存一様列/native証拠を再利用した。新中央演算はnative RZ/CX・既存CZ/Pであり、未知のCRZZ角度を旧certificateへ付け替えていない。[CORRECTNESS.json](../../../artifacts/track_c_closed_p3/2026-11-08/CORRECTNESS.json)

## 6. 価格pilot・独立holdout・公平性

最初は最短cellとm17のt=.32,n=1024で各2層×8、計32 paired occurrenceを評価した。m17の旧α固定では各軸258→254 trial、総RZ1.5504%減、paired差MC SE1.8981e8に対する差1.4084e10となり、全roundへ進む基準を通過した。このpilotを最終holdoutの独立証拠に数えない。

全roundは使用する12 cell×2層×16、training384と別seed holdout384。canonicalとclosedそれぞれのrare層確率を戻し、whole occurrenceを直接Compiler Aでlowerした。D/D junctionは保存された同じ実装価格、wrapperは同じscalar/X/Y規約。RZ/CX/sizeは具体的modular回路の加算価格で、長q fitの無検証移送はない。

候補選択とαはtrainingだけで固定した。さらにRA adapterが持つfull/single/joint融合fallbackをcanonicalにも与え、同じrawを再利用する公平性確認を実施した。新samplingは追加せず、元raw/policy/暫定結果を保全した。trainingからαを固定してから強化canonicalのholdoutを評価した。holdout384件で追加のRZ/CX削減は0件で、最終数値は変わらなかった。全指標は1本の実回路から取得し、指標ごとの別回路minを足していない。

paired共分散はcell・stratumごとに独立なseedを持つreplicateから計算した。共通cellが複数roundに使われる重みを合算してから共分散を戻し、各armの独立SEを差のSEとして足していない。価格用conditional-first-rare mapは既存float64実装を使用し、量子shot用の無条件dyadic map・数学的confidenceとは区別する。MC SEは経験的価格の不確かさで、rigorousな価格上界ではない。

## 7. 全18roundの結果

主比較はRZ優先。状態準備費は除外。depth欄は**modular serial上界**であり、m17巨大回路全体の実DAG depthではない。

| 方法 | 十分trial数 / 期待quantum calls | 総RZ | 総CX | serial depth上界 | size |
|---|---:|---:|---:|---:|---:|
| 改良C canonical P3、今回paired再評価 | 17186 | 1.422901381e12 | 1.972696355e11 | 5.464353695e11 | 2.077113378e12 |
| 同条件Closed P3 | 17028 | 1.394938574e12 | 1.933790045e11 | 5.357097303e11 | 2.036287194e12 |
| 同等control/対角合成済み強Full PF（既存固定結果） | 14232 | 5.660906974e11 | 1.014634368e11 | 2.093319366e11 | 8.390911294e11 |

| paired削減 | 値 | 差のMC SE（1 SE） | 削減率 |
|---|---:|---:|---:|
| RZ | 2.796280709e10 | 2.871016745e8 | **1.965196%** |
| CX | 3.890631060e9 | 3.328893922e7 | **1.972240%** |
| serial depth上界 | 1.072563912e10 | 1.191170253e8 | 1.962838% |
| size | 4.082618442e10 | 4.148133479e8 | 1.965525% |

総trial数は158減。最終αでm16は各軸304→299、m17は258→253となり、高価な後半roundのtrial減少が総ゲート削減へ効く。旧公開報告のcanonical 17164 trial/RZ1.42343e12と今回17186/1.42290e12の違いは、新しい価格標本とtraining α配分による。これを追加アルゴリズム改善と数えず、主比較は今回のpaired2方式同士で行った。

同じcanonical trial数を固定した回路単価ablationでは、RZ差はcanonical−closedで約 **−6.8629e4**に留まる。総RZ1e12規模に比べて極小であり、native単価が改善の主因という証拠はない。小角rotationが少し変わってもRZ数はほぼ同じで、return集約後のnormalization/rangeと十分Nの改善が主要機構だった。normalized logだけで勝敗を先取りせず、実whole occurrence価格まで確認した結果である。

元のcanonical P3平均位相をE0なしのdecoderへ入力し、branch復元とenergy精度が通過した。平均energy誤差 **1.284014863e-5 Ha**。新しいweighted/native平均は追加diskに囲まれる。既存alias条件、36軸の残余予算、3×.4<πと最終幅 **1.52587890625e-4 Ha < ε_E**から、保存Hermitian-H・そのexact ground・ideal native・fresh独立finite bitsのモデルで成功率≥.95の十分条件が閉じる。実機shotや有限shot反復実験を実施した結果ではない。

## 8. 古典費用・負結果・限界

古典費用は [CLASSICAL.json](../../../artifacts/track_c_closed_p3/2026-11-08/CLASSICAL.json) に分離した。代表2 cellのG7 group setupは各約.052 CPU秒、全conditional/angle資格は各約43.7 CPU秒。4096 microstepのclosed生成はcold約1.24 CPU秒、固定trace/cacheを再利用したwarm約.0041秒、圧縮weight積約.0033秒だった。canonicalの記録境界（毎回event構築、約.14秒）とはcache境界が異なるので、**CPU速度優越は認定しない**。実運用の全RPE古典wall time・setup償却の最適性も未評価で、CPU秒をRZへ無換算で足していない。

同じbinary64 weightが1種類だった固定traceでは、m17の登録8388608因子も頻度圧縮により短い区間累乗で処理できた。これは全eventのexact weightが等しいという主張ではない。実πWの補正、float weight丸めbias、全path rangeは引き続き必要である。

保存した技術失敗は、人工x=1/3に16項angle enclosureが不足したもの、mpmath区間asin API/辞書更新の接続不備、classical診断で短時間cellへ不適切な長さを与えてoverflowしたもの。64項の既存enclosure、既存atan2式、登録path長への修正で解消した。元logを保全し、通過済みjoint結果を不要に再生成していない。これらをP3平均の科学的不一致や旧研究のSTOP解除に読み替えない。

残る制約：

- 強Full PFのRZは約5.66e11で、closed PRはなお約1.395e12。**Fullの逆転には至らない**。
- h/resolution・D係数は固定。sampling変更後に再最適化したglobal最適解ではない。有限menuや全13!順序探索も実施していない。
- 旧Cの保存H包含条件を継承し、元の連続積分Hamiltonian、近似ground情報、別分子/状態、ノイズ/routing、T合成、状態準備の費用へ一般化しない。
- 価格MCの差が明瞭でも、実機成功率・実時間優越・一般的な量子優位・新規性の認定ではない。
- G9/G10/C1の過去の負結果やpauseは元条件のまま維持。今回のP3接続成功でP5や別sampling familyを再開しない。

## 9. 再現・SOURCE・raw

[RAW_INDEX.json](../../../artifacts/track_c_closed_p3/2026-11-08/RAW_INDEX.json) と [RAW_DATA.tar.gz](../../../artifacts/track_c_closed_p3/2026-11-08/RAW_DATA.tar.gz) が全新規cell/回路検証/価格/技術logへの入口。元byteとarchive memberを照合する。最終の統計・資源値はFAIR_FINAL/FAIR_ROUND_RESOURCESを使用し、FINALは公平性補助確認前の保存結果として保持する。

private pickleを公開せず、必要な既存科学配列・preparation・basis演算列を **MODEL.json.gz / MODEL_ARRAYS.npz** に構造化した。公開replayは同一科学model identityを持ち、private pathを含まない。[INPUT_REPLAY_FINAL.json](../../../artifacts/track_c_closed_p3/2026-11-08/INPUT_REPLAY_FINAL.json)。新しいfit/diagonalization/ground solveをせず既存classとgateを復元する。625共同isometry catalogueは固定C rawから取得する。

Qiskit1.3.0 / NumPy1.26.4 / SciPy1.14.1 / mpmath1.3.0 / Python3.12を使用。新しいworktree/専用出力で実行し、利用可能affinity64・開始時MemAvailable約445GiBを確認した。独立価格worker64、job-local BLAS1。training約75.7秒、holdout約82.0秒、同じrawの公平性追加compile各約12.7秒。これらは記録したpipeline wallで、研究全体の所要時間ではない。共有設定・他ユーザーのjob/ファイルを変更していない。

再実行は保存成果を上書きせず、別の出力先を指定する：

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/prepare_track_c_closed_p3_inputs.py
export TRACK_C_CLOSED_P3_OUTPUT="$PWD/artifacts/track_c_closed_p3/reproduction"
python scripts/run_track_c_closed_p3.py stage1
python scripts/run_track_c_closed_p3.py correctness
python scripts/run_track_c_closed_p3.py pilot
python scripts/run_track_c_closed_p3.py holdout
python scripts/run_track_c_closed_p3.py fair
```

上の科学再実行は本実行の公開・STOP後に自動起動しない。科学SOURCE完全SHA、入力identity、生成段階は各JSONとSOURCE.jsonに記録する。科学実行時のlocal SOURCE履歴は **SOURCE_HISTORY.bundle** に原commitとして保持し、公開用閲覧commitとの差を明示する。旧SOURCE/grant/ledger/marker/STOPや旧rawを付け替えない。

## 10. 結論と次の判断

この固定H4では、Bのclosed P3をCの改良PRへ接続でき、weighted/finite-bit/native/RPE会計を通した後にも約2%のPR内部量子資源改善が残った。主要機構はreturn集約によるrange/normalization低下と高価格roundのtrial減少であり、basis/native単価の大幅改善ではない。

次に検討する価値があるのは、同じP3でsampling変更後のh/resolutionを共同選択したときにも利益が残るか、及び公平な古典cache/前処理境界を揃えた実運用費用である。今回その追加探索はしていない。

次の担当：**ユーザー**が本結果の採否と、追加範囲を検討するかを判断する。**Codex**の現在の後続実行はない。公開commit・remote SHA/tree・成果内容・raw索引を照合したら **mandatory STOP**。P5、別分子、別sampling family、Track B単独研究、別PF研究、詳細GPT科学レビューへ進まない。

[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4d48a6f2097ea81ff3b7c64114e30bef35b1b685/docs/research/track_c_b_connection_20261108/REPORT.md
[S1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a8299889de93c201bf21e727415d629f5e4b9f48/docs/research/track_b_existing_results_survey_20261108/REPORT.md
[CSUM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/100377dac8635fc49074147bfd8b42f52f2b7311/docs/research/track_c_o3_research_summary_20261108/TRACK_C_O3_RESEARCH_SUMMARY.md
