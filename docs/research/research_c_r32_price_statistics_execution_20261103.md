# 研究C：update005/r32 独立native価格校正の限定科学実行

2026-11-03 JST。科学SOURCE `62fa4c0144b9e36f9d96b7dff67628051946632d`、[固定契約](research_c_r32_price_statistics_contract_unapproved_20261103.json)、fresh run01。**5845/5845 attempts、COMPLETE 5845・FAIL 0、retry/resume0。** 登録判定は **UNRESOLVED_PRECISION_TARGET**、精度目標判定は **False**。 本書は保存runner出力の報告で、新しい重要科学レビュー・追加計算ではない。

**独立640件によるPR/D点比は0.999580138349、95%近似区間は[0.993748669885, 1.00541160681]で1を含んだ。** 費用差の符号はこのモデル区間で未確定であり、同等性・非劣性も未判定。半幅615411095.204 RZは登録したD費用の0.5%目標を満たさず、`UNRESOLVED_PRECISION_TARGET`として終了した。追加sampling・閾値変更は行わない。

[Runner結果](../../artifacts/research_c_r32_price_statistics/2026-11-03/run01/result.json)／[主統計](../../artifacts/research_c_r32_price_statistics/2026-11-03/run01/primary.json)／[投影summary](../../artifacts/research_c_r32_price_statistics/2026-11-03/execution_summary.json)／[journal](../../artifacts/research_c_r32_price_statistics/2026-11-03/run01/science_calls.jsonl)／[STOP](../../artifacts/research_c_r32_price_statistics/2026-11-03/run01/STOP.json)／[preflight](../../artifacts/research_c_r32_price_statistics/2026-11-03/execution_preflight.json)／[postflight・raw inventory](../../artifacts/research_c_r32_price_statistics/2026-11-03/execution_postflight.json)。

## 1. 固定条件・実行範囲

直鎖H4・隣接1.00 Å・STO-3G、number CDF8、signed対称J・実直交U、Nα=Nβ=2/sector36、full256/controlled512。保存未収束endpoint update005、candidate SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。元g補正κ、核scalar一回、元frame/spin順を維持。PRはL_D=2・partial ordered symmetric S2、s1、δ=h=.1、r32/K0、m0..12。Own-DF ground oracle・状態準備費用除外。

共通CA=.00159 Ha・DF reserve=CA/8・既存RPE βtotalπ/6/βstatπ/12・failure=.01/axis=.01/26。P0はQiskit1.3.0/opt1/seed17・rz/sx/x/cx・approximation degree1・qubits_initially_zero=false・MPFR160 declared phase、元head相対Gaussian/tail exact basis_key共有とfenced serial会計。

科学契約SHA256 `4a78a3756243aa310df60a51d32a4c43bd318056c2fcf536da924f72b3e904a4`、SOURCE manifest SHA256 `f6f0be47a453349209237694826637cd289b57a00f7184d2cee2d6241172314b`。SOURCE70/input31と依存10packages、fresh outputを再確認した。人工・回帰42件PASSの[準備証拠](../../artifacts/research_c_r32_price_statistics/2026-11-03/technical/verification.json)を再利用し、SOURCEを編集せず実行した。契約のunapproved/proposalという歴史的表示は保持し、固定640件・confidence95%・半幅目標Dの0.5%の今回のチャット許可に従った。承認証明・承認原文保存・新grantは作成しない。

新fit/optimizer、rank変更、PF/RTE誤差・DF誤差・ground eigensolve、shots式、旧raw32の再解析は0。保存DF予算、ground、PF/RTE、RPE shots/W、Dを元の資格のまま再利用し、一般K/CAR9・ground residual・law/normalizationとnative正しさを確認した。

## 2. 独立cohortとcorrectness

既存32件はr選択と技術的標本数計画だけに使い、新推定へ混ぜない。事前固定20組×32件、master2026110302、cohort update005_r32_prospective_v1_batch00..19、axis R32_PRICE_HOLDOUT_ONLY、global ID0..639。SOURCEを含むhash seed key/PCG64を維持し、新640seedの相互重複と旧4条件128seedとの一致を検査した。異なるseedが同じevent列を出すことは棄却理由にしていない。Seed key分離を厳密な独立性・ideal IID law距離の保証と呼ばない。

各sampleは32 K0 eventsを持つcomplete controlled kernel。共通full CAR/Gaussian/head相対/ancilla shellは一回準備して共有し、raw/compiled full512 reference、actual raw/compiled adjoint、complex X/Y、metricsを全標本で検査した。Allowanceはcomponent1e-10/signal2e-10、rtol0、位相fitなし。個々のGaussianをsector射影してfull回路検証を代用しない。QPYは2560件。

組途中では価格区間・符号・精度を見て停止しない。完了数とcorrectness進捗だけを監視した。組別の既存記述的calibrationは保存するが、最終640件を一つの新cohortとして解析した。欠測・失敗標本を置換／削除／winsorizeせず、retry/resumeしない。

## 3. 主統計：固定r32対同一J/U強D

Estimandは実装seed-law下のkernel価格平均μ_implと、固定ideal会計の`C_PR=W μ_impl+S`、`Δ=C_PR−C_D`、`C_PR/C_D`。Wは保存13979232、Dは保存105532782860 RZ。Shellは保存RPE plansとancilla metricsを既存serial会計へ接続して固定した（[fixed_accounting](../../artifacts/research_c_r32_price_statistics/2026-11-03/run01/fixed_accounting.json)）。rや640をWへ再乗算せず、共通価格SEをW倍し、13round独立校正として縮小しない。

| 指標 | 新holdoutのrunner値 |
|---|---:|
| 新規独立cohort N | 640 |
| Kernel平均RZ | 7546.08125 |
| Kernel平均SE | 22.4187061536 |
| 条件付き総RZ | 105488473691.6 |
| 条件付き総RZ SE | 313396294.461 |
| 総RZ 95%近似区間 | [104873062596, 106103884787] |
| Dとの差 Δ | -44309168.4 |
| Δの95%近似区間 | [-659720263.604, 571101926.804] |
| PR/D点比 | 0.999580138349 |
| PR/Dの95%近似区間 | [0.993748669885, 1.00541160681] |
| 半幅 | 615411095.204 |
| 半幅目標PASS | False |
| 登録判定 | UNRESOLVED_PRECISION_TARGET |

区間は`x̄±t(.975,N−1)·s/√N`の**近似IIDモデル区間**を同じW/S/Dで変換したもの。価格の正規性・被覆率95%・非漸近保証を認証したものではない。Δ区間に0が含まれる場合は同等性を認定しない。Precision未達、観測分散0、登録skew/delete-one診断に該当すればUNRESOLVEDとし、追加samplingはしない。詳細diagnosticsはprimary.jsonに保存する。観測min/maxをsupportとせず、真のsupport/母平均のvalidated enclosureはnull。

登録された分布診断はmoment skewness −0.268077423938、最大delete-one平均変化2.59167644757 kernel RZ、skewness warning=false・delete-one変化が半幅を超えるflag=false・除外数0。工程screenへの該当はなく、precision目標が未達だった。これらのscreenは母分布・希少tail・区間被覆率の資格証明ではない。

旧32件の点推定106752458375 RZ、PR/D1.01155731406、SE1192089996.44は歴史的比較として保持する。新holdoutとpoolした推定や旧価格の新CIは作らない。今回の問いは選定済みr32対Dであり、r16/32/64/128の母平均最小性を検証していない。

## 4. 副指標・条件付き会計の限界

| lane | 総RZ | 総RZ depth | 総CX | 総CX depth | 総total depth | 総size |
|---|---:|---:|---:|---:|---:|---:|
| ideal_exact_native | 105488473691.6 | 36333256672.7 | 47168025652.8 | 32763060510.8 | 79524607750.9 | 207325302273 |
| assumed_raw_signal_budget | 105990831644 | 36506283148.7 | 47392649566.6 | 32919084995.1 | 79903320373 | 208312628290 |

副指標とassumed laneは記述的平均/SEだけで、別の有意差・同時区間・全指標支配を主張しない。Depthはfenced serial scheduled sumで、全長再compile最適化やhardware時間ではない。Assumedは各未知誤差≤.001Aという保存感度仮定で、実誤差の認定ではない。

今回の価格母平均はfixed finite-bit/PCG64実装lawのモデルに関するもの。Ideal-law母平均への移送、全law synthesis、implemented-vs-ideal signal距離、actual CA、状態準備、厳密量子総資源優位は未資格のまま。Evidence laneの未知誤差を0にせず、同等性／非劣性marginも未採択/null。SelectedのJ/U同時一次収束・他分子への一般化・DF目的だけの因果効果・独立新規性を追加認定しない。

## 5. Calls・資源・保全・次の担当

科学call実績は5845、COMPLETE 5845、FAIL 0、上限5845。契約通りprepare20・law3・共通native35・20組×289・7終端/metadataを使用し、内側library操作を新fitや量子shotsと混同しない。実量子測定0、旧call残枠の流用0。STOPとjournalに実内訳を残した。

事前資源はCPU affinity32・MemAvailable約54GB・disk約230GB・低pressure。固定32件の独立native仕事を実資源に応じて並列化し、BLAS協調・既存Qiskit排他・同一arm cache共有を維持した。組は既存builder/cacheのため逐次で、結果による打ち切りではない。実績wall=9422.50501859秒、CPU=152350.489202秒、peakRSS=2525384KiB。Speedup比較はしていない。CPU/RAM/worker/wallの人工上限や共有環境・他ユーザーjobへの変更はない。

旧SOURCE/input/raw/grant/ledger/marker/STOPと共有索引を保全し、Codex-Bのupdate064・最適化コード・契約・成果物を変更しない。今回scopeは固定update005/r32だけである。

次の担当は**ユーザー／研究C**。今回の近似価格区間・未資格事項と、別系列update064の結果を区別して確認し、後続の科学的問いと必要性を判断する。Codex-Aは追加r、追加cohort、CI再設計、U更新、全law保証、重要科学レビューへ自動進行しない。共有索引の更新が必要なら統合担当へ引き渡す。

Branch `codex/research-c-r32-price-statistics-execution-20261103`。結果・SOURCE・契約・journal/QPY/STOPを公開し、remote SHA/tree・本文・raw・元入力との一致を照合後、**mandatory STOP**。
