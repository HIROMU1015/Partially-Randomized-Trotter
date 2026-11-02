# 研究C・C1：H6 native限定実行のcorrectness FAIL・部分証拠

2026-10-25 JST。科学SOURCE `bf431765b1a83ca79d1c2fe15d30284e0b1f28c6` を無変更で使用。
Branch `codex/research-c-c1-h6-native-execution-20261025`。

**162/298 logical unitsでcorrectness FAILとなり停止した。30局所因子と3 ancilla shellはPASS。最初のFAILはCDF s=1/shared kernelのraw対referenceで、最大振幅成分差 `1.0400731857157292e-10` が固定閾値 `1e-10` を超えた。開始済みの8 kernel検証を回収し、154 COMPLETE／8 FAIL、retry/resume 0。kernel費用・adjoint・X/Y信号・13 round合成誤差profileへは進んでいない。**

[実行要約](../../artifacts/research_c_c1_h6_native/2026-10-25/execution_summary.json)／[局所30因子](../../artifacts/research_c_c1_h6_native/2026-10-25/local_summary.csv)／[shell](../../artifacts/research_c_c1_h6_native/2026-10-25/shell_summary.csv)／[8経路の到達状況](../../artifacts/research_c_c1_h6_native/2026-10-25/candidate_status.csv)／[gate](../../artifacts/research_c_c1_h6_native/2026-10-25/gate_summary.csv)／[call集計](../../artifacts/research_c_c1_h6_native/2026-10-25/calls_summary.csv)／[原本・52 QPY・STOP](../../artifacts/research_c_c1_h6_native/2026-10-25/run01/)／[postflight](../../artifacts/research_c_c1_h6_native/2026-10-25/postflight.json)。

## 固定条件・SOURCE・入力

ユーザーのチャット上の開始許可に基づき、独立worktree `/tmp/prt-c1-h6-native-science-20261025`、fresh `artifacts/research_c_c1_h6_native/2026-10-25/run01` で一度だけ実行した。契約の歴史的なunapprovedファイル名・metadataは保持し、承認原文保存・承認証明書・新grantは作成していない。

直鎖H6・1.00 Å・STO-3G、Nα=Nβ=3、sector400／full controlled8192。修正codecによる保存最終CDF14と保存selected XDF14、通常number・元g補正κ・scalar・元frame順を保持した。全frameのordered symmetric S2、s=1／4、literal／sharedの8経路。時間h=`1/(10*s)`のcontrolled kernelを対象とし、δ=0.1、m=0..12の登録条件を維持した。PR splitは非適用、own-DF ground-state oracle、状態準備費用除外。共通CA=0.00159 Ha、DF reserve=CA/8=0.00019875 Ha。

P0はQiskit1.3.0／opt1／seed17／basis rz,sx,x,cx／approximation_degree=1.0／qubits_initially_zero=false。既存NativeBuilderの同一basis共有規則を両armに適用し、一般相対Gaussian融合等の追加最適化は導入していない。

| 固定資料 | identity |
|---|---|
| 科学SOURCE commit | `bf431765b1a83ca79d1c2fe15d30284e0b1f28c6` |
| SOURCE tree | `91e82bc58971019f5f111a68ef42a05d6432ee0c` |
| [契約](research_c_c1_h6_native_contract_unapproved_20261025.json) SHA256 | `516bd17b89dfe74d25bdbd59b14d128cdb5ada493504586190389774d6df2040` |
| [SOURCE manifest](../../artifacts/research_c_c1_h6_native/2026-10-25/technical/SOURCE.json) SHA256 | `783a280916b7febbcd4c71628305cee94a7dafa7ec460393ab30dbd4f9a7fd27` |
| 元input SHA256 | `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447` |
| 保存CDF14 export SHA256 | `cd4de687f99d32bd6f644ac33100fb52d4f7543e710e6c273723a0908b87dc45` |
| 保存XDF14 export SHA256 | `559dc43bf923a421ccc62d05c4cefa0bdf39a48ba68824abf3c8f770c04642b1` |

元input／XDF SOURCEは `33c380d8aa792e5119bb3adec5dd07d1e1377214`、CDF fit／保存DF誤差SOURCEは `33432e78659b8b3e654e39f1392328cb0ba5c487`。前段PF SOURCEは `87ea924ae28353707bd2eb3adf94231042b67361`、結果公開は `554f5fa93231c561aaffce1b50335cbb3ce35b92`。それぞれの来歴を今回のnative SOURCEと分けて保持した。

[preflight](../../artifacts/research_c_c1_h6_native/2026-10-25/preflight.json)はSOURCE42 files・入力25 hash・固定依存を照合してPASS。Python3.11.1、NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Aer0.15.1等の既存task環境を変更せず再利用した。直前人工テストは **30 passed**（11.61秒、[log](../../artifacts/research_c_c1_h6_native/2026-10-25/pre_execution_artificial_tests.txt)／[XML](../../artifacts/research_c_c1_h6_native/2026-10-25/pre_execution_artificial_tests.xml)）。準備時の117 tests＋29 subtestsも保全している。人工PASSを今回の科学PASSへ転用しない。

保存DF norm＋guardはCDF `1.0803058513925193e-4`／XDF `1.0361230763094318e-4` Haで双方予算内。これは対応する保存値の再利用で、DF誤差の新規評価ではない。前段の一般K・ground/alias・52 PF roundのPASSは保存結果として再利用し、再計算していない。CDF／XDFの物理近似誤差は異なり、今回のnative FAILが過去のsector/PF結果を変更することはない。

## 局所・shell・kernelの到達範囲

| 検証 | 到達状況 | 確認できた範囲 |
|---|---|---|
| 30局所因子 | 30/30 PASS | 各arm15因子、τ=1/20、独立CAR sector400 referenceに対するfull8192出力 |
| head／X／Y shell | 3/3 PASS | ancilla-only構造、全2×2 raw/reference・compiled/raw・unitarity |
| raw回路build | 41 COMPLETE | 局所30＋kernel8＋shell3 |
| P0 compile | 11 COMPLETE | kernel8＋shell3。compile完了はkernel同値性PASSを意味しない |
| kernel state-action bundle | 8 FAIL | 各5 probe、最初の詳細FAILはCDF_s1_sharedのraw/reference |
| kernel compiled/raw・compiled/reference | 未資格 | bundleが例外で終了し、チェック完了reportは保存されなかった |
| actual adjoint／full X/Y信号 | NOT_RUN | 後続callを開始せず停止 |
| kernel費用／CDF-XDF・literal-shared比較 | NOT_RUN | kernel metrics0／比較0 |
| 13 round合成誤差profile | NOT_RUN | 予定104行は生成していない |

局所因子はGaussian gate個別のsector射影ではなく、complete controlled因子をfull8192状態に作用させた。各因子1個の固定complex-sector control+ probeで、sector外出力も含めて比較した。最大振幅成分差はCDF_block1の `1.558265651669714e-11`、最大norm driftは `1.2212453270876722e-15`。局所の最大per-probe L2残差は `1.1034253052368121e-10` だが、登録判定はmax-component／norm drift各1e-10であり、L2を同じ閾値で判定する契約ではない。これら有限probeから全operator同値性や厳密上界は主張しない。

shellのraw/reference最大差は0、compiled/raw最大差はhead/X `1.6653345369377348e-16`、Y `5.551115123125783e-17`。unitarity最大成分差は `2.220446049250313e-16`。shell全2×2の数値pilotであり、full controlled kernelやX/Y信号のPASSではない。厳密rounding enclosureも未資格。

## 最初のFAILと停止

[failure.json](../../artifacts/research_c_c1_h6_native/2026-10-25/run01/failure.json)の `first_failure` は **CDF_s1_shared**。raw/referenceのmax-entryは `1.0400731857157292e-10`、batch Frobeniusは `6.202629046913387e-10`、固定component allowanceは `1e-10`／rtol0。位相fitは行っていない。閾値超過が小さくても、固定契約ではFAILであり、事後的な閾値変更・PASS扱いはしない。

親の例外 `error` は、登録順にfutureを回収する既存 `pool.map` により **CDF_s1_literal** を伝播しており、同じ数値を記録している。時間順の最初のFAILとは区別する。残る6 bundleについてはFAIL eventだけが保存され、個別の例外詳細・残差値は保存されていない。8件すべてが同じ値・同じsubcheckでFAILしたとは断定しない。

最初のFAIL時には8 kernel_actionが開始済みで、以後の新規科学callは0。実行中作業を回収してSTOP・資源・部分証拠を保存した。全162 attemptにCOMPLETEまたはFAILの終端eventがあり、inflightは0。回収中の完了をretry/resumeとは数えない。

SOURCEの処理順では、raw／compiled両方のstate-action後にraw/reference gateを検査する。最初のFAILではそのgateで例外となり、compiled/raw・compiled/referenceの判定reportを保存する前に終了した。したがってcompiled correctnessをPASSとはしない。数値reference／state-actionの累積誤差、位相・Gaussian・順序等のどれが原因かは、今回保存された証拠だけでは特定していない。追加の数値診断は実行していない。

## 4候補×literal/sharedの費用表

「—」はNOT_RUN／未取得を表し、0ではない。QPYは存在するが、STOP後にdecodeしてmetricsを追加計算していない。

| arm | s | route | kernel bundle | RZ数 | RZ depth | CX数 | CX depth | total depth |
|---|---:|---|---|---:|---:|---:|---:|---:|
| CDF14 | 1 | literal | FAIL | — | — | — | — | — |
| CDF14 | 1 | shared | FAIL（最初） | — | — | — | — | — |
| CDF14 | 4 | literal | FAIL | — | — | — | — | — |
| CDF14 | 4 | shared | FAIL | — | — | — | — | — |
| XDF14 | 1 | literal | FAIL | — | — | — | — | — |
| XDF14 | 1 | shared | FAIL | — | — | — | — | — |
| XDF14 | 4 | literal | FAIL | — | — | — | — | — |
| XDF14 | 4 | shared | FAIL | — | — | — | — | — |

取得済みshell metricsのみを別表に示す。1 h-step kernel費用の比較や、全round・shots加重費用への外挿には使わない。

| shell | RZ数 | RZ depth | CX数 | CX depth | total depth | size |
|---|---:|---:|---:|---:|---:|---:|
| head | 2 | 2 | 0 | 0 | 3 | 3 |
| X | 2 | 2 | 0 | 0 | 4 | 4 |
| Y | 1 | 1 | 0 | 0 | 3 | 3 |

## 科学call・資源・保全

| 科学call | 実績／上限 | COMPLETE | FAIL |
|---|---:|---:|---:|
| saved bundle load | 1/1 | 1 | 0 |
| native model | 2/2 | 2 | 0 |
| sector context | 1/1 | 1 | 0 |
| sector generators | 2/2 | 2 | 0 |
| Gaussian build | 30/30 | 30 | 0 |
| local reference | 30/30 | 30 | 0 |
| circuit build | 41/41 | 41 | 0 |
| local action | 30/30 | 30 | 0 |
| compile | 11/11 | 11 | 0 |
| shell assessment | 3/3 | 3 | 0 |
| kernel action | 8/8 | 0 | 8 |
| adjoint action | 0/8 | 0 | 0 |
| Hadamard action | 0/8 | 0 | 0 |
| metrics | 3/11 | 3 | 0 |
| round synthesis | 0/104 | 0 | 0 |
| cost comparison | 0/8 | 0 | 0 |
| **合計** | **162/298** | **154** | **8** |

未開始136 unitsは失効した今回の残枠として再利用しない。fit／optimizer／rank探索／DF誤差・PF再計算／retry／resume／shots算定／最終CA・総費用算定はすべて0。

実行前にCPU affinity32、idle約99.56%、MemAvailable約52.57 GiB、CPU/memory pressure0を確認した。GPU driver利用不可のため固定CPU経路を使用。既存resource planにより8独立候補worker、各Aer4 threads、BLAS1で過剰並列を避け、P0 compileだけ既存mutexで逐次化した。CPU／RAM／wallの恣意的な上限や共有環境の変更はない。

runner wall `271.4197031077929 s`、process CPU `279.982797945 s`、peak RSS `3,137,324 KiB`。累積compile待ち `241.19206223729998 s` は複数workerの待ちの合計で、wall時間と混同しない。比較実行はなく、高速化倍率は主張しない。

raw89 files（56,263,062 bytes）、QPY52件を保持した。SOURCE42／入力25／旧原本469 filesのSHA256が実行前後で一致し、契約・SOURCE manifestも不変。旧H2/H4/H6のraw結果・grant・ledger・marker・STOPは変更していない。公開commitは結果・報告・状態索引の追加で、科学SOURCEを変更しない。

## 資格の限界と次の最小作業

局所有限probeとancilla shellの数値pilot PASSは得たが、full controlled kernelの正しさは資格付けできなかった。104行のprofileは未到達で、予定された `UNRESOLVED_FINITE_PROBES` を解消する証拠はない。global kernel operator norm bound／厳密enclosure／CA／shots／総費用はnull・未資格のまま。未知誤差を0へ置き換えず、CDF/XDF間・literal/shared間の費用削減も今回の結果からは取得できていない。

次は**ユーザー／研究C**が、今回のraw/reference差の原因調査を依頼するか判断する。**Codex**が行う最小の技術作業案は、既存SOURCEのreferenceとstate-actionの時間・位相・順序・数値精度を人工再現で切り分け、必要な修正と検証範囲を具体化すること。保存H6回路の新しい数値診断・再評価には、そのSOURCE・条件と別途明示許可が必要で、旧run01のretry/resumeや残callの流用はしない。

kernel correctnessを解決した後も、共通CA・shots・総費用へ進むには合成誤差の全operator／roundに対する資格等が別途必要。有限probe最大値をglobal boundへ格上げせず、重要な科学的判断・詳細レビューも今回の実行報告から自動開始しない。

**今回の限定実行はcorrectness FAILで完了。証拠のcommit・push・remote照合後にmandatory STOPし、追加科学計算へ進まない。**
