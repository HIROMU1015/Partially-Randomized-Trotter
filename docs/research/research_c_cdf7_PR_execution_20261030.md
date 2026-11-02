# 保存CDF7：PR reference・RPE・損益分岐119-call実行結果

2026-10-30 JST。**119/119 COMPLETE、correctness FAIL0、retry/resume0。8因子の一般K／raw CAR、own-ground／alias、全13 roundの登録PF/RTE十分条件がPASSし、`POTENTIALLY_FEASIBLE`となった。** Ideal仮定のshotsは21,768。新CDF7のnative価格は未測定で、総資源優位・実際のCA達成は未資格のままである。

保存直鎖H4・1.00 Å・STO-3G、Nα=Nβ=2／sector36、通常number CDF7、元順LD2／tail F2..6、partial ordered symmetric S2、δ=h=.1／s1／r128／K0／m0..12。各DFのground-state oracleとideal finite-RTE IID参照モデルを使い、状態準備費用を除外する。結果はこの条件の数値pilotであり、別rank・分子・lawへの一般化ではない。

## 固定SOURCE・契約・実行

ユーザーの「実行してよい」に基づき、[技術準備](research_c_cdf7_PR_preparation_20261030.md)で公開したSOURCEと[ready契約](research_c_cdf7_PR_contract_ready_unapproved_20261030.json)を**無変更**で一度実行した。歴史的な`unapproved`ファイル名・metadataはそのまま保存し、承認証明書・承認原文保存・新grantは要求していない。

| 項目 | 値 |
|---|---|
| Execution branch | `codex/research-c-cdf7-pr-execution-20261030` |
| 科学SOURCE commit | `3e75b16901f0f4a9966d2bd5b9ff6bde8e75f472` |
| Scope | `C_H4_CDF7_PR_REFERENCE_RPE_PRICE_V1` |
| 契約 SHA256 | `55032e96e745095611b52aebe300056f0ce12231742dce2f6007ee5d0e427c81` |
| SOURCE102 manifest SHA256 | `e79781e8d63acb60f993aeb1d13ee8d9132d38d24f8905f4c8ac3ca086c85d16` |
| 保存fit SOURCE | `c13d6f82a22d497b3334eaaa40d8430a5010f70f` |
| 保存fit export SHA256 | `4e4939d1380386ebb98c3b39a13954cbfc3148e35fb8e2cfd3cf536bd0b57ff4` |
| 元P-H4 input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| Fresh output | [run01](../../artifacts/research_c_cdf7_PR/2026-10-30/run01/) |

独立worktree・task専用venv/cacheを使用した。[Preflight](../../artifacts/research_c_cdf7_PR/2026-10-30/execution_preflight.json)でSOURCE102件、保存input20件、旧保護2214件、依存版、fresh不在、空き資源を照合した。準備時の人工39 PASSを再利用し、同一SOURCEの新人工11件を実行環境でも再確認して**11 PASS／3.98秒**。[今回のテストログ](../../artifacts/research_c_cdf7_PR/2026-10-30/execution_artificial_tests.log)。これはCI／外部独立再現ではない。

実行したのは既存の[専用CLI](../../scripts/run_research_c_cdf7_PR.py)一回だけである。旧rank8 runner、one-fit runner、旧callの残枠やgrantは再利用していない。

## 表現・law・groundの正しさ

| Gate | 結果 |
|---|---|
| Input／export／構造／来歴 | PASS。J/U float64、actual rank7、frame0..6／LD2を維持 |
| Base＋7 frameの独立表現 | **8/8 PASS**。一般K対raw CARのmax-entry最大3.9968028886505635e-15、登録絶対1e-10／rtol0内 |
| Hermiticity・own-ground／alias | PASS。E₀=-2.1663874400468766 Ha、residual2.876538565024285e-15、normalization error5.551115123125783e-16、alias margin1.6179938779914944 |
| Tail law／identity／mass | PASS。180 atom、λ_R=2.200160822452493 Ha、probability sum1。CAR対law＋identityのmax-entry9.992007221626409e-16 |
| 保存DF reserve | PASSを再利用。誤差＋engineering guard3.022641747295844e-7 Ha、CA/8=.00019875 Ha内。DF norm再計算0 |

核scalar一回・元g補正κ・frame順・spin conventionを維持した。Own-groundは新CDF7 Hamiltonianに対するoracleであり、状態準備アルゴリズムの検証ではない。Physical-ground overlapはnull。個別factorの表現gateはsector36のもので、full512 native circuitの同値性を検証したという意味ではない。

## PF／補正RTE／減衰：13 round

下表は登録済みJSONの転記である。全13行が`POTENTIALLY_FEASIBLE`。未知の合成／implemented-law誤差を除いた登録十分条件を満たしたという意味で、CA達成の認定ではない。[全精度・complex signalのCSV](../../artifacts/research_c_cdf7_PR/2026-10-30/presentation/reference13.csv)と[raw各round](../../artifacts/research_c_cdf7_PR/2026-10-30/run01/reference/)を保存した。

| m | PF operator誤差 | 補正RTE operator誤差 | Attenuation |
|---:|---:|---:|---:|
| 0 | 6.27594e-5 | 6.05116e-5 | .999810928 |
| 1 | .000125181 | .000120924 | .999621892 |
| 2 | .000247687 | .000241046 | .999243927 |
| 3 | .000474835 | .000475754 | .998488425 |
| 4 | .000809752 | .000904783 | .996979134 |
| 5 | .000980747 | .001532927 | .993967394 |
| 6 | .001325943 | .002106217 | .987971180 |
| 7 | .002521860 | .002784321 | .976087053 |
| 8 | .004709936 | .005209898 | .952745936 |
| 9 | .009402707 | .010033846 | .907724818 |
| 10 | .018795465 | .020064683 | .823964346 |
| 11 | .037569399 | .040270005 | .678917243 |
| 12 | .075094505 | .082154333 | .460928623 |

これらは時間発展operatorの無次元誤差であり、Ha単位のDF誤差やenergy統計誤差と根拠なく加算しない。既存AccuracyTarget／raw-signal・attenuationへの対応をそのまま使用した。Binary64数値残差は厳密rounding enclosureではない。

## RPE・shots：証拠経路と仮定経路

共通CA=.00159 Ha、β_total=π/6、β_stat=π/12、α_axis=.01/26、α_total=.01、13 roundを維持した。[39 lane-round行](../../artifacts/research_c_cdf7_PR/2026-10-30/presentation/accuracy39.csv)／[lane summary](../../artifacts/research_c_cdf7_PR/2026-10-30/presentation/lanes.csv)。

| Lane | 資格・13 round | Energy bound pilot [Ha] | X/Y合計shots | Kernel反復重みW |
|---|---|---:|---:|---:|
| evidence | UNQUALIFIED／13 UNRESOLVED_SIGNAL_ERROR | null | null | null |
| ideal_exact_native | ASSUMED_SCENARIO／13 CONDITIONAL_ROUND_PASS | .0012786195874127638 | **21,768** | **35,983,272** |
| assumed_raw_signal_budget | ASSUMED_SCENARIO／13 CONDITIONAL_ROUND_PASS | .0012786195874127638 | **21,862** | **36,152,228** |

Idealはexact nativeとideal finite-RTE IID lawを仮定する。もう一方は合成・lawのraw-signal誤差がそれぞれ`.001*A_m`以下と仮定する。いずれも仮定を測定済み上界として認定していない。表示energy値は登録条件下の上界pilotであり、実際に推定したenergyの誤差ではない。量子shotsやtrajectory samplingの実行は0。

Idealの各軸shotsはm0..12で **512,512,513,514,516,520,528,543,575,645,815,1304,3387**。仮定budget側は **514,514,515,516,518,522,530,545,577,648,819,1310,3403**。Evidence laneのshots／W／CAは未知のままnullを維持した。

既存helperの`old_M3_*`という歴史的field名が残るが、この契約での参照は固定したM4-r128結果である。これらの差分やhigh-m accounting substitutionは登録helperの会計診断であり、新候補・別PF条件を実行したものではない。

## 条件付きRZ損益分岐と次のnative提案

新CDF7の一kernel平均価格をμ_RZとすると、登録会計式は次の通り。

- Ideal：`C_RZ = 35,983,272 * μ_RZ + 76,188`。
- Assumed budget：`C_RZ = 36,152,228 * μ_RZ + 76,517`。

保存head/X/Y shellはinterrogationごとに一回だけ加算し、coherent occurrenceに再加算しない。μ_RZは未測定の変数であり、frame数、CDF8平均、tail massから埋めていない。

| 同じ仮定laneの比較先 | Idealのμ_RZ損益分岐 | Assumed budgetのμ_RZ損益分岐 |
|---|---:|---:|
| 保存通常CDF8 γ0・s1/r128/K0 | **17,426.37199236634** | **17,424.524450298057** |
| 保存PR-aware CDF8 γ.1 M4・s1/r128/K0 | **6,506.293033801929** | **6,504.591725861266** |

非負のμ_RZがこの値より**小さい**ことが、それぞれの保存条件付き点費用を下回る条件であり、等号はtie。比較先もfinite32 cohortの記述的な費用点推定で、全law supportや母平均の保証付き優位ではない。[全6 metrics×4比較のCSV](../../artifacts/research_c_cdf7_PR/2026-10-30/presentation/break_even24.csv)。RZ以外も同じ登録式で保存したが、今回は新CDF7のCX／depthも未測定である。

事前固定したall13／ground-alias／条件付きenergy／正のshots・W／両比較先への正のideal RZ thresholdを満たし、判定は**`ELIGIBLE_FOR_SEPARATE_NATIVE_PROPOSAL`**。これはnativeを準備・実行する権限や資源優位の結論ではない。Rank7を得たことだけでは価格の改善は分からず、特にM4を下回るには上表のkernel価格条件の検証が必要である。

## Call・資源・保全・STOP

Reference62＋RPE48＋price8＋summary1＝**119 attempts／119 COMPLETE／FAIL0／inflight0**。[Kind別calls](../../artifacts/research_c_cdf7_PR/2026-10-30/presentation/calls.csv)、[journalとSTOP](../../artifacts/research_c_cdf7_PR/2026-10-30/run01/science_calls.jsonl)、[STOP.json](../../artifacts/research_c_cdf7_PR/2026-10-30/run01/STOP.json)。新fit／optimizer／rank探索／DF norm再計算／sampling／回路build-compileは0、retry/resume0。

開始時はaffinity32 CPU、空きRAM56,512,032,768 bytes、load約1.12、CPU/memory pressure avg10=0。既存resource plannerにより13独立sector36 roundを13 thread、各BLAS1で実行し、自job内のoversubscriptionを調整した。人工CPU/RAM/worker/wall上限は追加していない。小さい36次元sectorを既存CPU経路で処理し、共有venv・cache・全体設定・他jobは変更していない。

[GNU time](../../artifacts/research_c_cdf7_PR/2026-10-30/process_time.txt)：wall2.25秒／user15.11秒／system11.18秒／max RSS379204 KiB／swap0／exit0。Runnerのwall1.9297998678秒／CPU23.817583493秒／peak RSS378292 KiB。実測比較なしのspeedup倍率は主張しない。

[Summary](../../artifacts/research_c_cdf7_PR/2026-10-30/execution_summary.json)／[postflight](../../artifacts/research_c_cdf7_PR/2026-10-30/execution_postflight.json)へSOURCE102・保存input20・旧保護2214件の不変、raw32件／842974 bytes、call・null資格を保存した。CSVは登録JSONの転記で、operator／shots／価格式の追加科学評価を行っていない。旧H2/H4/H6、fit、grant／ledger／marker／STOP、非実行draftは保全した。

結果・raw・SOURCE・契約・STOPをcommit/pushし、remote SHA／SOURCE／input／成果物を照合して**mandatory STOP**する。次の担当は**ユーザー／研究C**が本条件付き結果を確認し、別のrank7 fixed32 native校正の技術準備を進めるか判断すること。指示を受けた場合に**Codex**が323-call案の専用実装・人工検証・新cohort/seed・SOURCE/契約を準備し、実科学実行はその固定範囲への別途明示許可後だけ行う。今回の結果から自動的にnative、追加fit、保証計算、総費用評価、詳細科学レビューへ進まない。
