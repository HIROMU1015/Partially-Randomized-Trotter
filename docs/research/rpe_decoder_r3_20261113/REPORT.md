# R3：RPE decoder高速化とB S3 batch接続

2026-11-13 JST。**独立した高速版、Bへの任意batch hook差分、人工同値性・benchmarkを完了した。** 元scalar/vectorized decoder、RMSE wrapper、Bのbinomial/count/loss/CIは変更していない。実H4 decoder Monte Carlo、保存mean/Nの再解析、十分N/global RMSEの認定、価格/G、実S3は0。

成果物：[HANDOFF](HANDOFF.md)、[B最小差分](B_S3_PATCH.diff)、[benchmark原記録](BENCHMARK.json)、[新12テスト](TESTS.txt)、[旧decoder3件](REGRESSION_DECODER.txt)、[旧RMSE17件](REGRESSION_RMSE.txt)。コードは`src/trottertracks/rpe_decoder_fast.py`、人工benchmarkは`scripts/bench_rpe_decoder_fast_artificial.py`。

## SOURCEと作業の独立性

| SOURCE | 確認・再利用 |
|---|---|
| 監査 `a8f32abbd925276e126588b3648e3fb0deb014ea` | RECOMMENDED_PLAN/IMPROVEMENT_CANDIDATES/bench_decoder_artificial。閉形式の4候補窓を起点に次点・tie互換性を追加 |
| B `170047ef6e7a72aef5f4c6ff9c73f942097a694e` | rpe_decoder_statisticsのcount_mean / simulate / simulate_prepared / summarizeを固定参照・人工検証 |
| 最新B `bf515a976f1987ad6a19c61d14689b27b0894542` | 対象統計fileは170047eと同じblob `e4d5aba93a7a91c0061babaccc86f69e81dfa697`。最新branchにもhook差分を適用できる |
| A基点 `bba150834ca30bdb186fbfd58c55a8d51177b855` | 物理input契約・全13roundの科学的未資格状態は不変。今回その数値を読み込んでdecodeしていない |
| 元decoder | pr_rpe_rmse.decode_supplied_coordinatesとrpe_four_round_phase_validationのscalar/vectorized。既存A branchの同じfileを参照実装として保持 |

関連A branch `codex/pr-r-dependent-n-model-20261111`を再利用。B/C/Dのbranch/worktree/processを変更していない。過去SOURCE/input/raw/grant/ledger/marker/KEEP/STOPは保持した。サブエージェントは使用していない。

## 定数候補でのbranch選択

principal位相p、直前推定e、qについて、候補は`wrap((p+2πk)/q)`、k=0,…,q−1の等間隔格子。`k0=floor((q*e−p)/(2π))`を求め、`k0−1,k0,k0+1,k0+2`をmod qで比較する。q2は重複を除いた2候補。円上の最近傍と次点は隣接格子にあり、この窓は整数化の丸め境界も含む。候補自体と距離は**元と同じbinary64式・演算順**で評価し、簡易的な距離式へ置換しない。

scalar互換modeは候補ID順に並べた上でargminし、最小branch IDのtie ruleを保つ。最良・次点距離から`isclose(rtol=0, atol=1e−12)`でambiguousを累積する。scalar/B modeのbranch処理は、trial・roundごとにqと無関係な定数個、O(1)。全roundの診断を保持するため、batch全体の計算量はO(trials×rounds)、メモリもO(trials×rounds)となる。

**旧vectorizedの完全tieは別規則。** 元はデフォルトargsortで、scalarのargminとは順序が異なり得る。人工q32/p=0/e=2.650718801466388で、scalar13・vector14を実測した。これを無断で統一せず、`vectorized_reconstruction_fast`は最良・次点距離がbinary64で完全一致する行だけ元の全q argsortを再現する。したがって、この互換APIは通常O(1)/roundだが、exact-tie行ではO(q)。元vector規則を保持したまま、全入力で最悪O(1)とする主張はしない。B向けscalar modeにはこの例外がない。

1e−12近傍のtie、wrap・alias境界、zero X/Y、−phase/h、未解決Noneを保持。座標からの位相は元のmath.atan2を使用する。NumPy arctan2は丸めが1 ULP異なる可能性があるため、branch閾値に影響する置換を避けた。比較対象の有効入力で、位相・距離・energyに浮動小数点差は観測されていない。

M0〜12を検証domainとする。実装のbranch整数演算はq≤2^52までだが、より長いgrid全体の同値性や科学的成立性は未認定。非有限・不正grid/shape等は明示拒否し、有効入力の互換性と区別する。

## Bの実S3経路への接続

固定Bの`simulate`は、round→X/Yの順で全binomial countを生成し、その後trial別にcount_mean→A scalar wrapper→全branch scalar decoderを呼ぶ。vectorized関数を置き換えるだけではこの経路は速くならない。

今回の最小差分は、Bのsimulate/simulate_preparedへの`batch_decoder=None`引数のみ。Noneなら元trial loop、指定時は同じ生成済みcount arraysを新hookへ渡す。PCG64、binomial呼出し順・seed・確率、shot整数検査はBの元処理を使用する。Nやmeanを生成し直さない。

hookは各round/axisで同じcountをまとめ、B自身の`count_mean`でexact整数差とFraction→binary64変換を実施し、その結果を全trialへ戻す。`2*k/N−1`への置換や大きなNの丸めを行わない。2^53超・2^62超Nの人工countsも元経路と完全一致。unique count処理にはソートがあるため、この前処理を含む全APIがtrial数について厳密O(n)という主張はしない。

その後、元math.atan2と高速branch kernelでbatch復元し、Bの元trial dictを返す。provenanceの`decoder_acceleration_source`だけを追加。priorは所属診断のみ、alias修正・clipping・shiftは追加しない。元summarize・未解決loss・Clopper–Pearson/既存CI・trial_budgetはそのまま使用する。監査提案の別CI方式への置換は今回行っていない。

欠測・物理未資格のsimulate_prepared gateはhookより前に維持する。明示された条件付き人工inputでは元経路と同じ全trial・metadata・null qualificationを返す。既存S0/S1＋S2の実行許可をS3へ拡張していない。

## 人工同値性と回帰

新12件は次を対象とした。

- M0〜12の乱数位相、選択branch/全round距離・位相のscalar完全一致、vector tuple/ambiguous完全一致。
- 小qで5値位相gridの全組合せ、各段の全branchを列挙する元実装との照合。
- wrap境界、branch中点、1e−12 ambiguity閾値の近傍・ULP境界、±0、極端な有限位相±1e308。
- scalar/vectorの異なる完全tieをそれぞれ保全。0次元を含むbatch shape。
- zero座標・branch-tie・alias・energy変換・未解決statusとnull、不正入力の拒否。
- scalar modeで全branch生成を禁止するsentinel、q2^24を含む定数候補確認（人工のみ）。
- 固定B patchをtemporary treeへ適用し、同seedのbinomial trial全項目・loss・CIが一致、元fallbackも一致。
- Bのexact count conversion・大整数N、prepare gateとsimulate_prepared実接続、最新B fileの同一blob。

旧decoder3件＋既存RMSE17件もPASS、**計32件の技術テストPASS**。pytestの5つの既存invalid-escape DeprecationWarningは旧file由来で変更していない。これらは実H4の性能・全RMSE達成や十分Nの証拠ではない。

## 人工benchmark（実測）

全13round/M12、人工位相・人工raw座標のみ。各caseをfresh subprocessで実行し、同じimport条件、3回の処理時間中央値とLinux ru_maxrssを記録。Python 3.11.0rc1、NumPy 1.26.4、Linux x86_64、共有サーバー。CPU/RAM/thread/worker capを設定していない。caseは時間測定の相互干渉を避けて逐次実行した。NはMC trial数であり、量子shot Nではない。

| trial数 | 旧scalar kernel 秒 | 旧vector 秒 | 高速vector互換 秒 | B旧経路 秒 | B batch 秒 |
|---:|---:|---:|---:|---:|---:|
| 32 | 1.0210 | .02451 | .002119 | 1.0470 | .005363 |
| 128 | 4.1626 | .09306 | .002468 | 4.1447 | .009403 |
| 512 | 16.7007 | .38024 | .003947 | 16.7995 | .025279 |

512trialではvector kernelが**約96.3倍**、B経路が**約664.6倍**。Bの値はbinomial生成・exact count変換・math.atan2・branch処理・全trial dict作成を含む。kernelの値にはcount変換・dict作成を含めない。process起動・import・SOURCE fixtureの読取・出力digestは時間から除く。単発1trialの高速scalar APIの速度は別途測定しておらず、scalarをtrial loopで呼ぶ採用を推奨していない。

| trial数 | 旧vector peak RSS MiB | 高速vector peak RSS MiB | B旧 peak RSS MiB | B batch peak RSS MiB |
|---:|---:|---:|---:|---:|
| 32 | 349.64 | 347.55 | 347.55 | 345.85 |
| 128 | 374.80 | 347.55 | 349.04 | 349.05 |
| 512 | 487.15 | 347.56 | 352.86 | 353.95 |

RSSはinterpreter・共有import・入力・出力を含むprocess high-waterで、kernel専用allocationではない。baselineだけで約345〜348 MiBある。反復間のdigest/JSON作成のhigh-waterが後の測定へ残る場合もあるので、Bで一方のRSSがわずかに大きいことを数値kernelのmemory退行と断定しない。旧vectorのtrial×qの一時配列は除去できたが、Bは全trialの全round診断を保持し、出力memoryが残る。

高速scalar-rule batch kernelは4096/16384trialで**.01616/.05947秒**、peak RSS **348.06/364.56 MiB**。B batch全経路は同じtrial数で**.17959/.82102秒**、peak RSS **416.53/590.25 MiB**。旧scalar/vectorの同規模実測は行っていない。これらの数値へ旧速度を線形外挿して比較しない。

共有caseのbranch・位相・ambiguousのdigestは4 kernel間で完全一致、Bの全trial結果digestもprovenance追加を除いて一致。原記録の式`trials×2^M×16 bytes`は旧complex一時配列1個の**構造上の推定**であり、実測peak RSSとは別。512trialならその1個は32 MiB。新kernelは高々4候補だが、legacy-vector exact-tie fallback例外を除く。

## 次の担当とSTOP

**B**：独立moduleと任意hook差分を最新統計実装へ取り込み、固定SOURCEを新commitへ更新する。既定scalar fallback・loss/CI・RNG・資格を保持する。[HANDOFF](HANDOFF.md)にAPIと適用条件を記載した。

**ユーザー**：実S3のtarget proxy、未解決loss、数値不確かさ、統計精度/信頼度/有限試行scopeと固定SOURCEを別途承認する。今回の人工高速化をRMSE科学保証や実S3実行許可として扱わない。

**A**：今回のcommit/push、remote SHA/tree/主要成果物照合後mandatory STOP。実科学評価・価格/Gへ進まない。
