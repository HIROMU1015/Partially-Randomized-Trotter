# Track B：finite-RTE既存成果の発展比較 D1 source preparation

2026-10-13 JST。**READY_FOR_SEPARATE_D1_ONE_SHOT_AUTHORIZATION**。
契約・source・登録外focused validationを準備した。登録比較、native費用評価、T合成は未実行。
authorizationはpending、実result namespaceとone-shot markerは未作成。**mandatory STOP**。

## 入力、主比較、実行境界

[GPTレビュー原本](../../research/track_b_existing_results_development_broad_scientific_review_20261013.md)
（46,337 bytes、SHA256 `9ce0cf6662b0f99b8cb35ebf5817f3cb98a9249122154579c510b3d80587870e`）と
[前段の技術計画](existing_results_development_technical_plan_v1_20261013.md)を基点とする。
利用者の「次の作業に進んで大丈夫」「続きを行って」に従い、未固定の技術仕様を埋めた。
branch `track-b-rte-development-source-preparation-20261013`、base
`8c9b5dbdf96dcc7b4af0605fc24921f47e0971bd`。親worktree・他Trackを編集しない。

[機械可読contract](../../../artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json)が実行仕様。
既存の3-system-qubit明示provider、`p=(1/5,3/10,1/2)`、総時間`tau=5/7`、符号+、初期state `000`、P5を維持。
最初は`r=1`と最小の非自明反復`r=2`だけ。各stepは`x=tau/r`、finite targetは`[P5(-i x R)]^r`。
共通の物理target `exp(-i tau R)`へ、stateに依存しないTaylor上界を戻す。
これは既知providerのdevelopment比較であり、独立held-out、blind、DFスケールの実証ではない。

六方式はordinary、partial-return+tail、closed P3+tail、local full、closed P5、matched CTS。
**主比較はpartial-returnとclosed P5**。CTSの明示I1情報は全方式に同じ条件で利用可能とする。
CTS collection費用とreference tableの列挙はsetupとして別記し、production samplerへ評価表を渡さない。
旧G10のgenerator/native数学source等11ファイルは凍結source `b9ed01455351628c9073748f5ba5751aa794b789` とbyte一致。
新codeはB固有の`rte_development/`から参照する。共有trotterlibの実装は変更していない。

SPとの組合せ、proposal探索、precision探索、fixture追加、actual quantum shots、sampling、LP、DF、分子/NPZ、GPUは含めない。
BF/BM、generic whole、Gaussian、FR、旧G系列、RA-D0、Track A/N3のclosure・実行状態を変更しない。

## 共通accuracy、有限bit、confidence

各axisの許容誤差は`1/200`、複素誤差は高々`sqrt(2)/200 < 0.01`。
12 rows / 24 axesに`alpha_axis=49/24000`、resource failureはrowあたり`1/12000`。
合計`24*alpha_axis+12/12000=1/20`。accepted-call tailは`t=10`を共通に用いる。
H_prob=160、H_root=256、eta=rho=1e-12、primitive strict Rz error=1e-6、interval dps=100。
正負primitiveは同じ実際のsequenceとそのadjointを使い、相対control位相を消さない。

短時間`0<=x<=1`では`exp(x)<3`からstep Taylor remainderを
`epsilon=3*x^6/720`で包む。rounded native step errorは共通に
`e=8*rho+16*(1+rho)*primitive_error`。
finite meanはunitaryではないため、積に対して

```
Taylor_product <= (1+epsilon)^r - 1
implementation_product <= (1+epsilon+e)^r - (1+epsilon)^r
axis_bias <= 2*(Taylor_product+implementation_product)
statistical_width = 1/200 - axis_bias
```

を用いる。幅が尽きればSTOPし、r・precision・予算を変更して救済しない。
probability roundingは実際のdyadic proposalと`weight=coefficient/proposal`に戻す。
proposal丸め自体を平均biasへ再加算せず、係数丸めを一回だけ計上する。
matrix referenceは順序・位相・平均を確認するdiagnosticで、数値残差やexact signalでshot数を減らさない。

productionのM2/range上界を各stepで計算し、そのr乗へ共通Bernstein sufficient budgetを適用する。
local fullはproduction B/U envelopeを使うzero-fill lawであり、referenceのglobal normalizerをbudgetへ戻さない。
これは十分shot数の比較で、必要shot数の下界・最適shot policyの証明ではない。

## 全trialと費用取得

各whole trialのstep抽選は独立。全stepがnonzeroになってから量子準備を開始する。
どれかzeroならwhole値は0で、全trial数に含める。accepted-only平均は使わない。
step acceptanceをZ、全step-trialあたりのnative期待費用をcとすると

```
whole acceptance = Z^r
whole expected additive native cost = r*Z^(r-1)*c
two-axis expected cost G_k(h_k) = 2*N*r*Z^(r-1)*c_k + 2*N*Z^r*h_k
```

T/CX/native1Qについて、各`h_k>=0`での正確な有理数intercept・slope・交点を保存する。
同じstep内のliteral controlled lowering、CZ→H CX H、隣接inverse cancellationを共通適用し、cross-step fusionをしない。
これはadditive native costモデルで、whole-circuit global compilerの実測ではない。
Tは1Qに含まれ、T+1Qの二重加算をしない。workspace、classical setup、per-trialは別座標。
online classical samplingは未取得であり、未知費用を0として総費用優位を作らない。
expected、accepted-tail、hard upperを分離する。新しい5%閾値やalgorithm GOは追加していない。

`r=1`は固定G9 v2結果
[`c95736fd2990f5ef6dd1cb5866421fd78bb28687`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/artifacts/track_b_g9_p5_native_result/2026-10-10/v2/result_v1.json)
の同angle/precision/tool/strict-phase primitiveだけをreuseする。
SHA256 `3eb8430014a7881b5189ca8779a58ccd7ab2f50c37a58795bd0458b174abda20`。
旧budgets・wrapper価格・科学分類を移さない。r=1でmissing keyなら新合成せず失敗扱い。
r=2について全必要keyを価格取得前にfreezeし、inventory hashをjob/worker/result/集計へ束縛する。
新key上限53、保存key上限19、全cache上限72。
これはhard capであり、登録点の実際のkey数を今回計算したものではない。capを超えれば拡張せずtechnical STOP。
全positive eventにnative/strict-error coverageを要求し、不完全T取得から全方式の勝敗を作らない。

## 実行資源と性能準備

既存isolated runtimeはPython3.10.12、numpy2.2.6、mpmath1.3.0、pygridsynth2.0.0。
旧tool identityのpackage lockとPython source treeを照合済み。
合成設定は旧固定dps/seed/optionsを継承し、`up_to_phase=false`。
実行executable hashはcontractへ固定した。wheelを再インストールしていない。

| 制限・構成 | 固定値 |
|---|---|
| independent synthesis worker / numeric threads | 2 processes / 各1 thread |
| parent / worker / aggregate RSS | 512 / 512 / 1536 MiB |
| virtual address | 各process 1536 MiB |
| aggregate CPU / per-key CPU | 1800 s / 20 s |
| phase・total・per-key wall cap | なし |
| result output / per-key log | 128 MiB / 64 KiB |
| cache / sequence | 4 MiB / 20,000 characters |
| reference event bindings / matrix cache | 2000 / 1024 |
| sufficient shots | axisあたり1e8、clipしない |
| retries | 0 |

CPU affinity32、MemTotal65,522,480 KiBをmetadataで確認。GPU query/useは0。
旧G10 v3外側receiptの22.226 s wall / 20.029 s CPU / 240.016 MiB peakを実装参考にした。
dominant acquisitionを同一key cache、2 independent workers、thread=1で整理し、shared immutable provider/matrix cacheを再利用する。
whole pathの全列挙はせず、積の平均・moment・費用をfactorizationで戻す。speedup倍率は未測定。
periodic POSIX guard、worker CPU hard limit、aggregate memory/accounting、bounded cache/log/outputを適用する。
wallなしは科学契約の仕様であり、外側polling・失敗workerを終了するcleanup timeoutと区別する。

保存はstream JSONとUTF-8境界を保つ512 KiB以下のlossless chunks。
行・event・sequence・error・費用を索引化し、再結合bytes/SHA256を照合する。
partial保存は残すが、incomplete prefixでpositive/negative判定をしない。
外側observerはexit/signal、terminal identity、CPU、RSS、logsを保存し、runner内完了と外側正常終了を分ける。

## 登録外validationと来歴

[focused結果](../../../artifacts/track_b_rte_development_source/2026-10-13/v1/focused_tests_v1.json)：
**57 tests / 57 PASS**、local runtime lock一致。full test suiteは実行していない。
off-domain fixtureは2-label `p=(1/3,2/3), x=1/7`、generic X/Zの2x2行列、r=3積確認。
control loweringは別の2-system-qubit fixtureと人工exact sequenceを使った。
登録provider生成、登録Q/T/U、登録event/native IR、native compile、synthesis、sampling、LP、DF、molecule/NPZ/GPUは0。
mock parallel workerを実際の合成として数えない。

検証範囲：各ensembleの同有限平均、丸めlawのM2/range、controlled相対位相、signed adjoint、zero-fill積と費用、signal-free budget、bias/shot cap、source/contract/marker/inventory tamper、pending rejection、exclusive保存、UTF-8復元、512 KiB上限、資源超過、alarm conflict/restoration、worker failure no retry。
49 PASSと54 PASSの中間実行を最終57 PASSと区別して[test履歴](../../../artifacts/track_b_rte_development_source/2026-10-13/v1/test_execution_history_v1.json)へ記録した。
不成功testから数値設定を調整したことはない。

[技術監査](../../../artifacts/track_b_rte_development_source/2026-10-13/v1/technical_audit_v1.json)と
[source manifest](../../../artifacts/track_b_rte_development_source/2026-10-13/v1/source_manifest_v1.json)を参照。
旧source・authorizations・STOP・markersと固定G9/G10原結果等181 filesをprotected ledgerへ記録し、byte/hash不変を確認した。
共有API、旧science契約、旧markerを修正していない。今回のtestsはimmutable CI、外部再現、登録科学結果ではない。

## 次の承認とSTOP

本技術準備だけを理由に追加のGPT詳細sourceレビューを必須にしない。
登録比較に進むには、**今回の固定source Sを指定した別の明示D1 one-shot実行指示**が必要。
Sから独立の実行branch/worktreeを作り、直接子Aで変更できるのは新authorization JSONと任意receiptだけ。
authorizationはsource S・contract hash・一回のみ・retry=0・mandatory STOP・明示指示の正確な本文を束縛する。
remote=A、clean HEAD、source/protected identity、fresh namespaceを確認してexclusive markerを作る。
公開確認用の後続commitがあっても、実行source Sと混同しない。

将来の入口は`observe_rte_development_d1.py --source <S>`、内部runnerは`run_rte_development_d1.py`。
現在のpending authorizationでは、入力・backend・job開始前に拒否することをtestsで確認した。
成功なら`D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW`、不完全なら`D1_TECHNICAL_INCONCLUSIVE`。
外側exit/terminal不整合もtechnical扱い。どの場合もmandatory STOP、retryなし、次stage自動認可なし。

未検証なのは、登録条件の余裕・actual key coverage・新合成の費用・登録native operator diagnostics・資源上限内の完走。
これは別承認後のone-shotが取得する情報であり、source準備で先回りして計算しない。
新しいscience semantics/保証の変更が必要ならGPTへ戻す。結果後の研究価値・新規性・着地点はGPTが判断する。
