# Track B：既存成果の発展RQ・技術具体化と最小比較計画 v1

2026-10-13 JST。**設計準備完了／科学実行・新source実装は未認可／mandatory STOP。**

研究目的は、**同じ有限Taylor平均を実現する表現について、集約・生成・正規化情報・有限精度実装が資源順位を変える条件を、取得可能な情報と同じconfidenceで説明すること**。

[最新GPTレビュー原本](../../research/track_b_existing_results_development_broad_scientific_review_20261013.md)の§12を具体化した。原本46,337 bytes、SHA256 `9ce0cf6662b0f99b8cb35ebf5817f3cb98a9249122154579c510b3d80587870e`を保持する。既存成果を発展させる設計の優先順位を更新し、旧科学結果・個別closure・消費済みmarkerを変更しない。

branch `track-b-existing-results-development-preparation-20261013`、base `792af43ef237c0c5cd9b42ec102a15dfc3c60be4`。今回実施したのは文書受領、固定git blobの静的読取り、手計算による式展開、限定した一次文献照合と公開確認。sampler、matrix、scientific tests、native IR生成、compile、合成、LP、分子/NPZ、GPU処理は実行していない。

## 1. 同じもの／違うものを先に固定する

| 系列 | target・入力 | ensemble/proposal/normalizer | 実装・bias・confidence | 次計画への使用 |
|---|---|---|---|---|
| G10 | 同一次数の有限 `P_m(-i x R)`。3 system qubits、3 involutions、`p=(1/5,3/10,1/2), x=5/7, sigma=+1`。m=3/5/7を各々比較 | ordinary、partial、closed P3+tail、local full、closed P5(/tail)、matched CTS。fullはzero-fill、closedは少数群直接生成 | literal direct controlled lowering、共通adjacent inverse cancellation、strict phase、primitive error 1e-6。finite target外のTaylor remainderは旧taskに含まない。axis 1/200、旧34 axes配分49/34000 | 発展計画の同target基点。known/development。既存17行を新実験やheld-outにしない |
| R1/R1.5 | 2-qubit synthetic `P3`、`p=(3/4,1/4), x=1/8,1/4`、符号±。distinct-basisとPauli controls | ordinary/PTSC-K0/A、情報accessによる対照を分離 | native precision 1e-3/1e-4/1e-6、保存primitive価格・strict error。axis .005、旧264 axesのfamilywise .05。basisとrotation費用を別記 | representation/precision相互作用の機構資料。G10の価格・bias・proposalへ移さない |
| SP-1 | 2-qubit synthetic coherent wrapper。A/B/C、n=8/16/32/64。有限Taylor/RTE入力ではない | outer toy pathとinner PAI。joint lowering→mask非依存fusion→placement→PAI | 既存23 literal keys、exact pi/4 catalogue、additive primitive T、complex .05、旧96 axes配分1/1920 | conditional momentと累積の機構資料。G10と改善率を掛けない。実RTE接続は別の未認証問題 |

固定根拠：[G完成ノート](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md) §§6–8、[R1.5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md) §§1–5、[SP契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)、[SP結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d2bb1fa439748b02084bd9fbc9b10a705328f8a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)。分子geometry、chemistry basis、DF rank、Hamiltonian splitはこれらsynthetic条件に適用外。

## 2. G系列内の表現差と情報差

| 同じfinite P5への方式 | 理想ensemble・生成入口 | 予算へ使える情報 | 統計・取得・実装上の差 |
|---|---|---|---|
| ordinary | `CanonicalGenerator(...,'ordinary')` | paired order群のnormalizer | IID word。既存baseline。通常RTEにも同じnative改善・precision機会を許す |
| partial-return | `CanonicalGenerator(...,'partial_return_tail')` | root吸収＋distinct pair群＋ordinary高次tail | 強い近接baseline。P5の遠いordinary対照だけを主比較にしない |
| closed P3+tail | `CanonicalGenerator(...,'closed_P3_tail')` | 低次数のcollected root/first-label群 | 低次数集約の対照。m5 targetにordinary degree4/5 tailを残す |
| local full | `FullReturnGenerator`／`ReturnKernel` | ordinary envelope `B_env`、local query、`U_+`。reference `B_full`を入力しない | reduced判定とaccept/reject、zero-fill。`M2_+`はenvelopeとUに依存。全trial会計。precomputeとonline queryを別座標 |
| closed P5 | `P5Closed` | root、ordered pair、first-label length4、completion DP、群root intervals | 同じ理想full P5 ensembleを少数群で直接生成。finite-bit群midpoint targetはlocal parent-midpoint版と完全同一とは限らない |
| matched CTS | `g9_native.cts_events`のfinite specialization | 同じproviderの明示Pauli情報・collection費用 | real identity correction、zero-T event、odd pairingを残す。literal対照でありCTS family全体の最適性は主張しない |

source：科学source `b9ed01455351628c9073748f5ba5751aa794b789` の `g7_generator.py`、`g9_p5.py`、`g9_native.py`、`g10_generator.py`、`g10_comparison.py`。全blob identityと静的確認箇所は[参照manifest](../../../artifacts/track_b_existing_results_development_preparation/2026-10-13/reference_identity_v1.json)。既存 `make_generator` にclosed P5 armを直接渡せるとは仮定せず、P5は別constructorへ接続する。`g10_comparison.row()`はp/xを旧domainへ固定してmatrixを生成するため、将来のmultistep runnerへそのまま流用しない。

**m5のlocal fullとclosed P5は理想係数表現が同じでも、proposal・受理率・weight・finite-bit mean・使用可能上界は同じではない。** 費用差を「新しい演算子」「normalizer上界だけ」の効果へ一意帰属しない。m7 fullとP5+tailは別ensembleであり、m5の情報効果の対照へ混ぜない。

Pauli情報が取得可能なexplicit providerでは、全方式へ同じ情報と実装機会を公開する。方法ごとに実際に取得した係数・normalizer・DP・Pauli collectionのsetup/online時間とmemoryを記録する。I0-only実問題での優位、DF scaleのclassical優位、無料のcontrolled oracleは認定しない。

旧G10の巨大JSONを毎回読む必要はない。[版付きreview-access manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e65e3c0680fff4cfc4243ed5d6c87429cd5ce753/artifacts/track_b_g10_v3_review_access/2026-10-10/manifest.json)から、m5の`rows/row_00`–`row_05`と対応event indexを取得できる。今回はこれらの保存identityだけを参照manifestへ記録し、科学値の再集計・再分類はしていない。

## 3. 最新レビューの式(1)–(6)の独立確認

本節は手計算。登録係数・event・matrixを生成しない。一般の有限bit定理とG6/P5の既存証明は再実行しない。

`chi=sum p_i^2`、正のp、fixed degree5、x→0+。ordinaryの3群を展開すると、root `sqrt(1+x^2)` が `1+x^2/2+O(x^4)`、degree2群が `x^2/2+O(x^4)`、degree4群が `O(x^4)`。従って

\[
B_{ord}=1+x^2+O(x^4).
\]

partial rootでは `a0=1-chi*x^2/2, b0=x-chi*x^3/6`。`a0^2+b0^2=1+(1-chi)x^2+O(x^4)`、distinct pairが `(1-chi)x^2/2+O(x^4)`、tailが `O(x^4)`なので

\[
B_{partial}=1+(1-\chi)x^2+O(x^4).
\]

P5 rootは同じleading項、ordered pairのleading総質量は `(1-chi)x^2/2`、length4は `O(x^4)`。従って

\[
B_{P5}=1+(1-\chi)x^2+O(x^4),\quad
B_{ord}-B_{partial}=\chi x^2+O(x^4),\quad
\log(B_{P5}/B_{partial})=O(x^4).
\]

固定総dimensionless時間tau、x=tau/r、独立stepのideal canonical samplingに限れば、weightは `B(x)^r`、実測M2は `B(x)^(2r)`。従って

\[
\log\frac{B_{P5}(\tau/r)^{2r}}{B_{partial}(\tau/r)^{2r}}
=O(\tau^4/r^3).
\]

これはideal normalization部分だけの漸近式であり、符号、有限rの総費用、native価格、拒否生成、biasの順位を決めない。G10の局所改善率をr乗しない。

既存 `g7_generator.quartic_coefficients(p)` はm引数を持たず、ordinary/partialの返値にはdegree4/5 tailの質量を加えていない。**このhelperのx4係数をfull P5のx4係数へ直接流用しない。** 今回は上記leading-order確認で十分であり、P5のquartic値や新しい符号定理は取得していない。

外側eventの正係数をa_e、全trial proposalをq_e、inner joint-circuitのsigned係数をg_(e,k)、`Gamma_e=sum|g_(e,k)|`とする。inner drawは `|g|/Gamma_e`、outer weightは `a_e/q_e`。±1 outcomeの二乗を使うと

\[
M_2=\sum_e\frac{a_e^2}{q_e}\Gamma_e^2,\quad
R_{max}=\max_e\frac{a_e\Gamma_e}{q_e},\quad
\bar C=\sum_e q_e\,\mathbb E[C\mid e].
\]

これは最新レビュー式(6)と一致。零trialはweight/cost0、classical生成費用は0にしない。一般に `M2_outer * unweighted average Gamma^2` へ分離しない。history-dependent inner lawならGammaの代わりにconditional weightの二乗期待値とsupremumへ戻す。target一致はinner **joint channel**の誤差を認証してから使う。

## 4. 同じ物理時間・confidenceへ戻す契約骨格

最初の物理task案は、共通Rについて `U=exp(-i sigma*tau*R)` のcomplex Hadamard signal。初期状態は全方式共通とし、truth-free operator boundを使う場合でも結果前にstate identityを記録する。D blockを含むfull PR/QPE、FRのphase taskへ拡張しない。

`x=tau/r`、`M_j=P5(-i sigma*x*R)`、`||R||<=1`。Taylor remainderの共通解析上界として

\[
\epsilon_j\le e^{|x|}|x|^6/6!,\quad
\left\|\prod_j M_j-\prod_j U_j\right\|
\le\prod_j(1+\epsilon_j)-1
\]

を候補にする。`e^x`は将来outward認証する必要があり、現在数値化していない。各M_jをunitary/CPTP contractionと扱わない。

一stepの係数・probability丸めとstrict native実装のmean errorをe_jとすると、finite meanへの実装errorは

\[
b_{impl}\le\prod_j(1+\epsilon_j+e_j)-\prod_j(1+\epsilon_j)
\]

で保守的に分離できる。これはtelescopingをscalar norm envelopeへ戻したもの。係数丸めはe_jに一度だけ計上する。複素signalの各axisでは、旧G10の保守的factor2を再利用する案として `b_axis<=2(b_Taylor+b_impl)`、PAIを使うならそのjoint-channel/numerical項を追加する。PAIをunitary first-moment errorへ無検証で置換しない。

誤差欄はTaylor/PF、finite-bit、native/control、inner signed approximation、finite-shotに分離する。候補は同じ最大予算を受け、証明済み上界が異なる部分は同一規則で残余幅へ戻す。exact signalや開封済みreference normalizerでshotを減らさない。

独立whole-trial shotsで

\[
s_a=\epsilon_a-b_{a,+}>0,\qquad
n_a=\left\lceil\log(2/\alpha_a)
\left(2M_{2,a,+}/s_a^2+4R_{a,+}/(3s_a)\right)\right\rceil,
\quad G_j=C_{init,j}+\sum_a n_a\mathbb E[C_j].
\]

十分予算であり最小必要shot数ではない。全rows/axes、resource failure、semantic checksの失敗確率を新契約へまとめる。旧G10の17 rows/34 axes配分を新row数の認可として流用しない。

## 5. multi-step零trialと費用の会計

各stepを独立に準備し、**全stepのclassical drawが成功してから量子準備を始める**方式を比較案とする。どれかが零trialならwhole trialの測定値を0とし、全trial数に数える。accepted-only平均は使わない。

stepごとのnonzero proposal massをZ_j、per-whole-step-trialのnative加法期待値をc_jとすると、cross-step fusionなしのモデルで

\[
Z_{path}=\prod_j Z_j,\qquad
\mathbb E[C_{path}^{add}]=\sum_j c_j\prod_{k\ne j}Z_k.
\]

同質stepなら `Z_path=Z^r, E C_path=r Z^(r-1)c`。共通prepare/readout費用hは `Z_path*h`、二axesの期待量子呼出しは `sum_a n_a Z_path`。M2とrangeのstep積には、そのstepの**実際のzero-fill law**に対する認証値を使う。local fullを拒否なしcanonical B^2へ置き換えない。

上記は加法的nativeモデル。whole-pathの隣接簡約・fusionを許す場合は費用とinner momentをpath別に再定義し、同じpassを全方式へ適用する。今回はそのIRや費用を取得していない。生成途中のrejection、CPU、random bits、cacheの費用は量子費用と分けて残す。

T、CX、native1Q、workspace、classical setup/per-trialは別座標。1Qに含まれるTを二重加算しない。expected、accepted-tail、hard upperは別量。hは非負の記号変数として範囲を示し、後付けの換算でwinnerを作らない。

## 6. 最小の前向き比較案と段階境界

**主比較はpartial-return対closed P5**。ordinary、closed P3+tail、local full、matched CTSを同targetの機構対照に置く。P5が最良であることを成功条件にしない。全parameter grid、再配分探索、proposal/solver最適化は行わない。

| 段階 | 固定する内容の案 | 識別するもの | 現在の状態 |
|---|---|---|---|
| D1a：physical-taskへの会計接続 | m=5、既存p/provider、tau=5/7をdevelopment anchorとして提案。六方式、共通direct native model、共通finite-confidence task。r=1の保存価格はmetadata同一性を確認して参照可能 | 旧finite-only taskへTaylor remainderを戻したときの予算と、表現／normalizer情報／有限bit差 | source未実装・再集計未実行。再会計は新しいdevelopment結果として別契約が必要 |
| D1b：短い反復への前向き接続 | tau固定、m固定、r>1を一条件だけ選ぶ案。r自体は未固定。全方式に同じrとgate error allocation、同じclassical zero-fill rule | small-step normalization差とnative価格・bias・call数の相互作用。G10単一step比の反復では説明できない差 | 新x、angle keys、費用取得が必要。別認可前は一切生成しない |
| D1c：synthesisとのinteraction | 必要性が示された場合だけ、同targetでpartial/P5×ordinary synthesis/一種類のjoint PAIの四条件。D/R populationを同時探索しない | 表現順位がconditional moment・価格・biasで変わるか | 条件付き設計のみ。catalogue、branch law、joint phase/error認証、native inventoryは未固定 |

D1aは旧G10の再開ではなく、新しいphysical-task会計の基点案。D1bを初回に含めるか、まずD1aのsource検証までとするかは次のscope固定事項。実際のquantum shotsは初期比較に不要であり、認証十分予算と実取得native価格の比較を想定する。

既存p/x/providerは全てknown/development。D1bも同providerなら独立検証・blindではない。新しい確認入力やseedを今は採択しない。将来の独立条件は新結果を見る前に別途定める。

**今回新しいr、seed、precision、改善率閾値、数値資源上限を採択していない。** これらを埋める前に計算を開始しない。部分価格を0、未知normalizerをexact、uncertified biasを小さい値として補完しない。

## 7. 数値仕様の継承候補と未固定事項

機械可読案：[比較設計draft](../../../artifacts/track_b_existing_results_development_preparation/2026-10-13/comparison_design_draft_v1.json)。**実行contract/authorizationではない。**

- 継承候補：既存3-system-qubit provider、p、m=5、tau=5/7、sigma=+1、H_prob=160/H_root=256、eta=rho=1e-12、native primitive 1e-6、strict phase、literal direct lowering、exact inverse cancellation、same controlled access。
- 精度案：旧axis 1/200を共通physical taskへ継承するとcomplex誤差はsqrt(2)/200以内。Taylor・有限bit・nativeを含む残余幅が正かは未評価。global alpha=1/20を継承する案でも、rows/axes/resource failure配分は新しくfreezeが必要。
- 未固定：D1の採択scope、D1bの単一r、共通初期state identity、primitive precisionの反復時配分、native/synthesis取得key数、reuseの許可範囲、runtime/tool source manifest、CPU/RSS/output/log/call上限、結果前の予測と必要な解釈基準。
- SPを追加する場合の未固定：具体的joint decomposition、mask非依存fusion、catalogue、signed branch/support、finite-bit認証、合成precision、inventory。R1/SP保存価格をこのfixtureへ流用しない。

物理taskへのtruth-free biasが不足した場合、結果に合わせてr・precision・error budgetを救済しない。原因と選択肢を先に報告する。旧のwall cap等は凍結契約でそのまま保持し、新規契約では利用者のperformance指示に従いphase/total wall capをdefaultで設けない。新CPU/memory値は現時点で未採択。

## 8. 実行sourceと費用取得を準備する場合の仕様

次の明示scopeで認可された場合に限り、B固有の新namespaceから既存数学/APIを参照する。共有trotterlibをtrack別にcopyせず、旧generator・contract・budget helperを書き換えない。

1. fixed provider/target/rounding仕様を入力manifestへ束縛し、同target meanとfinite-bit target差を別recordにする。
2. `P5Closed`、canonical、local fullのproduction入口へreference event tableを渡さない。小providerの全列挙は評価専用oracleと明記し、取得費用・必要byte数を別会計する。
3. CTSのPauli情報を全方式に同じ条件で公開し、CTSの実際のcollection取得費用を記録。I0とI1を片側にだけ無料付与しない。
4. native/T inventoryを結果前にfreeze。keyはrational tangent、signed convention、controlled lowering、precision、synthesizer version/options、sequence identityを含む。同一key cacheは全armで共有し、旧cacheとの一致と未取得keyを分ける。
5. 全positive-support eventのcost/error coverageを要求する。T部分coverageなら下界とunknown上界を記録し、全T勝敗を確定しない。RZ数からのT推計をactual synthesis Tと呼ばない。
6. 全resultを512 KiB以下のレビュー用断片へ索引化し、原bytes/hashの復元を確認。stream serialization、bounded cache、partial/terminal/outer statusを保存。以前のRSS/JSON failureを反復しない。
7. 新sourceとcontractをcommit固定し、必要な独立source確認条項の適用を明記。既存markerを保護した新namespace、新authorization-only直接子、remote SHA/clean HEADを確認して、別の明示one-shot指示を受ける。
8. off-domain semantic/phase/rounding/zero-fill/cap/tamper focused testsの範囲を先に認可する。今回source実装・testsは行っていない。

性能準備は取得前に実施する。保存timingと静的sourceから、synthesis/interval処理、reference binding、serializationを分ける。共通key batch/cache、branch DP再利用、読取り専用共有入力、independent keyの並列化を検討し、process×numeric threadsとaggregate RSSを予算化する。小providerへGPU使用を自動追加しない。actual CPU/memory preflightとworker数は新実行scopeで定め、speedup倍率は未測定。

## 9. 今回目指すclaimと近接文献

| 既知内容・一次資料 | この計画で狙う限定した追加知見 | 今回主張しないもの |
|---|---|---|
| [IS §II Theorem 1、III Theorem 2](https://arxiv.org/html/2603.13495v1)：固定ensembleのcost/moment trade-offと同path実装のbias不変 | finite-RTEのnormalizer access、proposal、rounded meanと価格が同taskで順位を変える条件 | 新しいIS最適law、全confidence目的の普遍最適性 |
| [SCU/CTS II.3、IV.2、Supplementary Note 5/6](https://arxiv.org/html/2407.21095v2)：Taylor/Pauli収集、normalizationと反復、collectionの取得負担 | 同じfinite polynomialへ特殊化した強いCTS対照とのnative/resource成立範囲 | 全CTSが必ず全語列挙、CTS family全体への勝利 |
| [Sparse PS II.1–II.2、B.2](https://arxiv.org/html/2402.15550v2)：process/channel decompositionと積L1 overhead、近似誤差との交換 | 必要なら、RTE event条件付きの価格・weightと表現順位のinteraction | 一般probabilistic synthesis/crossoverの再発見、無検証のcontrol位相保証 |
| [TE-PAI II.3、IV、Appendix A](https://arxiv.org/html/2410.16850v3)：time-evolved observableと離散回転・shot burden | coherent signalの同taskへ対応可能な範囲と必要な追加costを明示 | system-only observable taskをcoherent Hadamard taskと同一視した優劣 |

今回の式・条件付き会計は既知の代数と確率恒等式を使う。新しい数学primitiveを得ていないことだけでは計画を停止しない。一方、係数構造、アクセス、有限精度、同taskの適用条件について新しい説明が得られなければ、その限界を報告する。世界初・独立論文成立・普遍的資源改善は未確定。

## 10. 結果前の評価とSTOP案

future primaryはpartial対P5の同task `G_T(h)` interval/rational relation。CX/native1Q/classical/workspaceは別座標。local full対closedは理想ensemble共通の情報・proposal・rounding差として診断し、causal percentageへ一意分割しない。primary comparatorは全armで同じ実装機会・confidence・残余幅規則を受ける。

結果分類案は `COST_ORDER_EXPLAINED`、`RESOURCE_TRADEOFF`、`NO_SEPARATION_UNDER_FIXED_POLICY`、`BOUND_OR_COST_INCONCLUSIVE`、`TECHNICAL_INCONCLUSIVE`。これはdraftであり、新5%閾値やalgorithm GOを登録したものではない。保存比の違いだけを一般条件とせず、予測・対象domain・不確かさを結果前に示す。将来SPのinteraction比も単独で新規性・実用性の証拠にしない。

technical incomplete runのprefixで勝敗を作らない。制御位相／support／mean／certificateが不明ならINCONCLUSIVE。cost未知は0にせず、cap hitは自動grid拡張せず、全outcomeでmandatory STOPして重要な科学的解釈をGPTへ戻す。

## 11. 完了状態と次の担当

**`DEVELOPMENT_PLAN_PREPARED_NOT_EXECUTION_READY`**。最新レビュー§12の技術具体化、式(1)–(6)のleading/conditional展開、source入口の対応、最小比較案、未固定事項と必要取得を一束にした。source実装・semantic tests・新resource値は取得していない。

次は新しい比較のscopeと未固定数値仕様を確定し、その範囲のsource準備・登録外focused validationへ進める指示が必要。単にdocs commitが増えた理由で研究全体の詳細レビューを要求しない。coherent semantics、比較対象、誤差保証、主要claimを変える必要が出た場合は、その論点だけGPT判断へ戻す。

FRは独立第二候補、DF basis改善は独立応用候補。FR-R2、BF/BM、generic whole、Gaussian、相関sampler、G11/RA-D0、Track A/N3を自動再開しない。旧判断は保持し、今回は既存成果の発展RQの**設計のみ**を進めた。科学実行には別途の明示認可が必要。**mandatory STOP。**
