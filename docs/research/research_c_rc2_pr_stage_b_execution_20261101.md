# 研究C：保存RC2／PR-tailのStage B・143-call限定科学実行

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。

**固定SOURCE・契約を変更せず143/143 logical units COMPLETE、correctness FAIL0、retry/resume0で完了した。両候補の13/13 roundがPOTENTIALLY_FEASIBLE。PR-tailは全13roundでattenuation Aが大きかったが、PF誤差も全13roundで大きく、補正RTE誤差の改善はm6..12に限られた。** 同じ固定予算政策の未収束保存候補の結果であり、CA達成・最適RC-DFへの優位・総資源削減は認定しない。

[全結果JSON](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01/result.json)／[26セルCSV](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/rounds26.csv)／[call journal](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01/science_calls.jsonl)／[STOP](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01/STOP.json)。CSVは保存結果の値を転記したもので、PF/RTE等を再計算していない。

## 1. 固定SOURCE・条件・許可範囲

| 項目 | 固定条件・identity |
|---|---|
| 科学実行SOURCE | d6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6、無変更で一度実行 |
| Adapter実装commit | 50a490b20a12a5387a84b0bccc42b7e2ca45d70b |
| [143-call契約](research_c_rc2_pr_stage_b_contract_unapproved_20261101.json) SHA256 | 851f43acc65be42dfcc21cef30f8c019bab7085566a93f9d42453917c4db048b |
| [SOURCE74](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/technical/SOURCE.json) SHA256 | d55fb55b1a6c9db609f2fd9667ba937a9fbdadeb1d9ec6aa14f602de6760b9ce |
| 保存候補の元fit科学SOURCE | e2f7dfd44357510d54ab9c6c9c5032e0072ab376 |
| RC2 J/U export SHA256 | d6cb5d56dfc6a83090a6ad1bc1207fdaee62a14cf120542fcb5a7d116644a798 |
| PR-tail J/U export SHA256 | 3fdb0cb77afc770294cbe7aebf47d8a4529e3567ef9e886895495f4490b7ee21 |
| 共通h/g/constant snapshot SHA256 | 1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db |
| 元P-H4入力の来歴 | 8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50 |
| 分子・表現 | 保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、sector36、通常number CDF rank8 |
| 分割と時間 | 元frame順LD2、partial ordered symmetric S2、δ=h=.1、s1／r128／K0、m0..12、n=2^m、t=.1×2^m |
| 物理・位相条件 | 元g補正κ、核scalar一回、CA=.00159 Ha／DF reserve=.00019875 Ha、βtotal=π/6／βstat=π/12、alias window[-10,10] Ha |
| State／law | 各DF固有のsector ground-state oracle、状態準備費用除外、ideal finite-RTE IID reference |
| Fresh output | artifacts/research_c_rc2_pr_stage_b/2026-11-01/run01 |

[技術準備](research_c_rc2_pr_stage_b_preparation_20261101.md)の既存コード・人工検証を再利用した。ユーザーのチャット上の明示許可を受けて今回の固定範囲を実行し、歴史的なunapprovedというファイル名・metadataは変更しない。承認証明・承認原文保存・新grantは作成していない。共通運用指示2026-10-25版の最新版をworkspace原本で確認した。重要GPT科学レビューは実施していない。

今回のRC2／PR-tailは、同じ元x0、初期penalty寄与一致α=γ=.1、共通L-BFGS-B・関数評価上限政策で得た**未収束の有限終点**。旧Stage Aのγ0 Baseline／旧RC2／旧PR終点と混同しない。DF予算PASSとoptimizer未収束は独立に保持する。新fit・optimizer・DF norm再評価・入力再生成・rank探索は0。

## 2. 正しさgateと独立reference

[実行preflight](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/execution_preflight.json)でSOURCE74／保存入力13／契約・依存lock・NPZ header・fresh outputを照合した。実行前に[新規人工24件](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/execution_synthetic_tests.txt)を再確認してPASS。準備時の88件を今回すべて再実行したとは扱わない。

| Gate | 結果 |
|---|---|
| SOURCE／input／元fitとrole・export来歴 | PASS |
| 保存DF予算／既存guard | 両PASS、再計算0 |
| Base＋8 frameのraw CAR／一般K | 18/18 PASS、atol1e-10・rtol0 |
| 両armのsum | PASS |
| Signed tail、identity phase、保存exact λとの整合 | 両PASS |
| 各arm固有のevent確率・K0 normalization | 両PASS、旧armのlaw流用0 |
| Physical／RC2／PRのground・normalization・alias | 3/3 PASS |
| PF／RTE／A・phase分類 | 26/26 COMPLETE、全セルPOTENTIALLY_FEASIBLE |

General-Kと独立CARのsum max-entryはRC2=1.887379141862766e-15、PR=2.8033131371785203e-15。Tail sumのmax-entryは4.884981308350689e-15／7.216449660063518e-16。いずれも登録allowance内。Factor／sum／tailの原値はrun01のrole別JSONに保持した。

| 各armの固有量 | RC2 | PR-tail |
|---|---:|---:|
| Law λ_R [Ha] | 8.545687128230103 | .9545432846581517 |
| Tail identity coefficient [Ha] | 3.4764356801835334 | .2557642181923188 |
| Nonidentity signed atoms | 216 | 216 |
| 保存DF norm＋guard [Ha] | 3.880714511121459e-9 | 7.216837988604903e-9 |
| Own-DF ground energy [Ha] | −2.166387448492221 | −2.1663874485603927 |
| Ground eigenpair residual | 2.3548410028413465e-15 | 2.5375025148413406e-15 |
| Alias margin [rad] | 1.6179938779914944 | 1.6179938779914944 |

Physical groundは−2.1663874486347625 Ha、residual1.7642254846760187e-15。これらは各Hamiltonianのground/alias確認値であり、RPEのenergy推定誤差ではない。κのHermitian化、U polar置換、frame再順序付けや新たなDF norm計算は行っていない。

原子・係数・確率・normalization・groundは候補ごとに取得した。Referenceは粒子数を保存する**完全な因子**のsector36 operatorで、Gaussian gate単体の射影や実controlled回路の検証ではない。Native circuit build／compileは0。

## 3. 全13roundの比較

PF誤差は||PF(h)^n−exp(−iH_DF t)||₂、補正RTE誤差は||corrected(h)^n−PF(h)^n||₂。両方とも無次元で、物理DF reserve [Ha]と単純加算しない。AはRTE normalizationの減衰係数であり、raw complex signalの絶対値と同一視しない。

| m | RC2 PF | PR-tail PF | RC2 corrected RTE | PR-tail corrected RTE | RC2 A | PR-tail A |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.95305162e-05 | 6.19687016e-05 | 1.69468313e-05 | 2.36198321e-05 | 0.997151442 | 0.999964409 |
| 1 | 0.000118737531 | 0.000123598525 | 3.38851733e-05 | 4.71984415e-05 | 0.994310999 | 0.999928819 |
| 2 | 0.00023491177 | 0.000244513482 | 6.77014491e-05 | 9.40668118e-05 | 0.988654362 | 0.999857643 |
| 3 | 0.000450087563 | 0.000468412877 | 0.000134852294 | 0.00018554522 | 0.977437447 | 0.999715305 |
| 4 | 0.000764410099 | 0.000796247265 | 0.000265426993 | 0.000352140399 | 0.955383964 | 0.999430692 |
| 5 | 0.000898684454 | 0.00095104884 | 0.000499684272 | 0.00059394407 | 0.912758518 | 0.998861708 |
| 6 | 0.00112508632 | 0.0012659963 | 0.000840187411 | 0.000814743757 | 0.833128112 | 0.997724711 |
| 7 | 0.0020800758 | 0.00237730842 | 0.00159886544 | 0.00107425838 | 0.694102451 | 0.995454599 |
| 8 | 0.00371829995 | 0.00422320989 | 0.00319050959 | 0.0019797868 | 0.481778213 | 0.990929858 |
| 9 | 0.00733433856 | 0.00842637216 | 0.00638180688 | 0.00376211719 | 0.232110247 | 0.981941983 |
| 10 | 0.0146480624 | 0.0168350191 | 0.0128006799 | 0.00749068271 | 0.0538751666 | 0.964210058 |
| 11 | 0.0292421531 | 0.0336057364 | 0.0257560584 | 0.014917479 | 0.00290253357 | 0.929701037 |
| 12 | 0.058425543 | 0.0670810994 | 0.0521750245 | 0.0300545902 | 8.42470113e-06 | 0.864344018 |

**両候補とも全13roundでA>0、εPF＋εRTE≤sin(π/12)=.25881904510252074を満たした。** INFEASIBLE／UNRESOLVEDセルは0で、両候補の全体分類もPOTENTIALLY_FEASIBLE。これは登録十分条件への適格性であり、未知の合成／implemented-law誤差を加えた最終CAの資格ではない。

Raw complex signalはA〈ψ_DF|corrected(h)^n|ψ_DF〉を同じreferenceで保存した。m12ではRC2=1.6902216412271795e-6＋8.701613767232747e-6 i、PR=.17389748207284125＋.8504056791423663 i。[26セルCSV](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/rounds26.csv)に物理時刻・coherent反復数・全signal実虚部・分類を含む。実量子測定・trajectory samplingではない。

## 4. 観測と未資格事項

固定候補・固定政策で得た観測は次の通り。

1. PR-tailのAは全13roundで大きい。特にm12でRC2は8.424701129970533e-6、PRは.8643440178248293となり、長roundのnormalizationによる減衰に明瞭な差があった。λの値からAを推測して済ませず、各arm固有のlawで評価した。
2. PF operator誤差は全13roundでPRの方が大きい。λ低下がPF改善を含意するとは扱わない。
3. 補正RTE operator誤差はm0..5ではPRが大きく、m6..12ではPRが小さい。小さいλだけから全roundの補正RTE改善を主張できない。
4. 今回のPF／RTE tradeoffの下では、両候補とも全13roundの登録十分条件内だった。片候補だけが位相適格性を失う結果ではなかった。

この観測は、**未収束の固定保存候補における減衰改善とPF/RTE tradeoff**を示す。最適なRC-DF全体に対する優位、大域最適性、最適強度での比較、λだけの因果効果、別分子への汎化や独立新規性は認定しない。両optimizerのsuccess=false／一次基準未達も変更しない。

Synthesis bound、implemented-law bound、cost support、shots、native価格、CA_achieved、総費用、保証付き優位はすべて未取得／nullを維持した。Binary64のsector operator数値pilotで、厳密な丸めenclosureではない。**Aが大きいという結果だけからshotsや総RZの優位を計算・主張していない。**

## 5. Call・資源・保全・公開

| 科学処理 | Units | 結果 |
|---|---:|---|
| Input／保存資格／sector／physical | 4 | COMPLETE |
| 両fit export load／model metadata | 4 | COMPLETE |
| 18 raw factors／18表現gate | 36 | COMPLETE |
| 両sum／law／tail／mass／distribution | 10 | COMPLETE |
| 3 ground／3 alias | 6 | COMPLETE |
| 両kernel reference | 2 | COMPLETE |
| 26 exact／26 reference／26分類 | 78 | COMPLETE |
| 両summary／matched比較 | 3 | COMPLETE |
| 合計 | **143** | **143 COMPLETE、FAIL0、inflight0** |

新fit／optimizer／DF norm再計算／native build／compile／trajectory sampling／shots／総費用は0。Retry／resumeも0。結果を見た後の科学条件・数値allowance・SOURCE変更や追加候補実行はない。

Preflightでaffinity32 CPU、available RAM56,244,256,768 bytes、load約5.88を確認。固定実装の動的resource planを使い、26独立round task／26 threads、BLAS各1でoversubscriptionを避けた。実行記録はwall2.9690023362636566秒、process CPU40.52977573秒、peak RSS366456 KiB。Wallはrunner内部、CPU/RSSはprocess計測で区間が異なる。GPU backend追加や速度向上率の主張はない。CPU/RAM/worker/wallの人工上限、共有環境・venv・他ユーザーのjobへの変更もない。

[Postflight](../../artifacts/research_c_rc2_pr_stage_b/2026-11-01/execution_postflight.json)に、143 attempts/COMPLETE、SOURCE74／入力13／契約とrun copy、元common SOURCE44／JSON修復SOURCE47、旧fit raw37／修復raw7、今回raw69件のhashを保存した。旧partial result.json／failed STOPも不変。CSVは保存26セルとの値一致を確認し、新しいoperator評価や分類を追加していない。

結果・SOURCE・journal・STOPをcommit/pushし、remote SHA/tree・公開本文・rawとの一致を確認する。以降の古い準備表示は履歴として保持する。

## 6. 次の担当とmandatory STOP

**ユーザー／研究C**が、今回の減衰差・PF/RTE tradeoffを踏まえ、保存26セルからのStage C shots評価を先に準備する情報価値があるか判断する。Native価格・合成/law資格の取得範囲も、その次の比較目的に応じて別途決める。**Codex**は具体的な依頼後に必要最小限の技術準備を行い、新しい数値再解析やbuild/compileはその固定範囲の明示許可後だけ実施する。

**公開照合後mandatory STOP。** 今回からshots、native校正、合成誤差上界、最終CA／総資源、追加fit、重要科学レビューへ自動進行しない。
