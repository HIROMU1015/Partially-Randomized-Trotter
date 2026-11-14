# Pauli LD1/m0 B0：run02復旧結果

2026-11-14、Codex D。ユーザーの「復旧再実行してよい」に基づく一回の手動復旧を完了した。
**normal 1 path・測定付きX/Yの2 compileが成功し、固定の実H4数値gateはすべてPASS。**
価格は各軸RZ **1380**、CX **578**。これは同一trajectoryの直接compiled価格で、Pauli lawの平均・CIではない。
旧run01の失敗とSTOPは変更せず、別のSOURCE・run02へ結果を保存した。B0完了、mandatory STOP。

## SOURCEと固定条件

| 項目 | 固定値 |
|---|---|
| 実行SOURCE | `9abbcc563aaa8a2650a5f799bc394f67d9c42534` |
| raw結果保存commit | `8502dd33ad16dc012fb3558668ef3e4cd8a4405c` |
| 元D SOURCE | `62ae1bd56718287de5d958cf0ed6411f5b7395c9` |
| C planner/guard | `c628e9e8447c11c984064b922cbce6c52dc72d48`の既存実装を保持 |
| A signal / P0辞書・演算子 | `714247619afb4714fd663eb15c0470ef32bdc7af` / `4220f3006d0d8d413e195ed948e250578513b557` |
| 系 | linear H4、1.00 Å、STO-3G、保存tol-only DF10、8 system qubit |
| PR構成 | Pauli LD1、S2、ΔPF=0.1、m0、T=0.1、qPF=1、R_outer=3、K=2、P=0 |
| 元canonical configuration | `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27` |
| C canonical physical ID | `RMSE:62a9a6b68c2c88951f1055d1da7f9b6a89da94a91092c9488cf5fce503288e54` |
| cohort / index / seed | `D_PAULI_LD1_M0_WIRING_V1` / 0 / `14788159701244282917` |
| 辞書 | 保存184項、SHA-256 `dbdde5a12791cd7b31b00d8ec6ae5074d56964b467389688163ece8bf186945c` |
| route / lowering | `PR_PAULI_TAIL` / `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF` |
| compiler | Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17 |
| compiler profile hash | `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` |

Cの`price_context`を正本とした元IDを保持し、Bの新Z1＋T1/Q3構成へ置換していない。
native/provider、D2 emitter、wrapperの固定SOURCEは[SIDECARS_PUBLISHED](run02/SIDECARS_PUBLISHED.json)に記録した。
state preparationは費用から除外し、controlled evolution・scalar phase・basis change・最終測定を含むfull wrapperを価格対象とした。

## 復旧内容と事前確認

旧run01は、worktreeのsparse checkoutでtracked reference JSONが未展開だったため、
入力復元moduleのimportで停止した。旧SOURCEの科学処理やscalar不一致による停止ではない。
[旧REPORT](REPORT.md)、旧raw/private job、元STOPは保全した。

このworktreeだけに`references/`を展開し、必要な2 JSONを元62aeのblobとbyte照合した。
保存H/state/snapshotのfile SHAも元input configに一致した。callerの変更はrun02出力先、
fresh processで実restore moduleまでimportするpreflight、元run01への来歴追加のみ。
`pilot_core/child/records/canonical_request/preflight`は元62aeとAST同一で、
connection、sampler、数値gate、tolerance、位相処理を変更していない。

[人工・metadata test](recovery_precheck/TESTS.txt)は親9件PASS（内部で子の人工回路5件もPASS）。
科学実行前にSOURCEをcommit/pushで固定した。import preflightではrestore/prepare/sample/build/compileを呼んでいない。
実行は既存callerの`--execute-approved-pauli-b0`をrun02へ一度だけ起動した。
これは追加承認後の手動復旧一回であり、旧枠の自動転用や自動retryではない。

## 実回路と固定gate

実H4 contextを一度取得し、normal trajectory 1 path・3 eventを生成した。
観測eventはorder0が2、order2が1。forced rare、seed変更、追加drawはない。
native headはone-bodyに続く保存fragment 0の順序を検査し、保存head blocksとの一致も確認した。
Pauli辞書の順序・係数・hashを固定し、mode-0-MSBからQiskit little-endianへの変換を適用した。

native tail identityと保存`s_R=0.4177175120626939`が完全一致し、
保存Hの`c=2.29310124732`と既存規則で構成したtotal scalarは`2.7108187593826942`。
各native stepのscalarは0、total Tのancilla phaseは一回であることを実回路上で確認した。
expected scalarの置換、guard緩和、tolerance追加は行っていない。

| 固定check | 実誤差 | 許容値 |
|---|---:|---:|
| low-control恒等作用 | 2.4599345360769775e-15 | 1e-9 |
| high-control leakage | 0 | 1e-9 |
| 独立full-space oracle vector | 3.8883493843496886e-15 | 1e-9 |
| X/cosine符号・信号 | 8.187894806610529e-16 | 1e-9 |
| Y/sine符号・信号 | 1.6132928326584306e-16 | 1e-9 |
| compiled vector X | 2.457857917960719e-13 | 1e-7 |
| compiled vector Y | 2.3964756048123724e-13 | 1e-7 |

bit-order checkはPASS。head/tailと保存block列の差は0、tailと保存`s_R`付きPauli演算子の差は
1.8424035611499477e-15だった。詳細は[RESULT](run02/RESULT.json)。
独立oracleは保存状態に作用させた全2^8次元vectorの照合であり、全入力に対するoperator同値性の認定ではない。

## compiled価格・資源

| axis | RZ | RZ depth | CX | CX depth | total depth | size |
|---|---:|---:|---:|---:|---:|---:|
| X/cosine | 1380 | 364 | 578 | 309 | 834 | 2695 |
| Y/sine | 1380 | 364 | 578 | 309 | 834 | 2695 |

9 qubit・1 clbit、最終測定あり。X/Yはそれぞれcompileし、cache hitは0、新compileは2件。
同価格でもwrapper fingerprintは異なる。

- X：`799be56b69ab6edf14195f25b11ba29c230c5eacbf674d2d397a4e7c43099ea8`
- Y：`ade235c59769dc0d74d5717b9e7445829721e3d44c9cd77b2617e9cbb7216586`

fingerprintとaxisの対応は生成済みwrapperの末尾ゲート・測定metadataから確認した。
再compileや追加statevector計算は行っていない。
[receipt trace](run02/RECEIPT_TRACE.json)と[CHECKS](run02/CHECKS.json)にprivate QPY/cacheのpath・bytes・SHAを保存した。

GNU timeの全command wallは**13.48秒**、peak RSS **626888 KiB**、swap 0、exit 0。
caller内elapsedは6.9676895760931075秒で、起動/import preflightを含む全wallとは測定範囲が異なる。
個別compile時間は未計測。短いm0の観測であり、長時間roundの性能や高速化倍率へ外挿しない。
[実行log](recovery_precheck/EXECUTION_RUN02.txt)／[資源log](recovery_precheck/RESOURCE_TIME_RUN02.txt)。

## Cへの受渡しと資格の限界

[PRICES](run02/PRICES.json)は`DIRECT_COMPILED_FINITE_SAMPLE` / `FINITE_COHORT_ONLY`、
`n_paths=1`、`law_expectation=false`を保持する。既存schemaの`mean.rz_count`欄には
1 pathの値が入るが、law平均としては受け入れない。S、C−S、mean/CI/fit/N/Gは今回取得していない。

raw sidecarのpublish前placeholderは改変せず、別の`SIDECARS_PUBLISHED.json`だけに
raw結果commitの証拠locatorを設定した。Cの既存guardはX/Yの2セルを
`IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`としてPASSした。
[guard記録](run02/C_GUARD_CHECK.json)はSOURCE・physical ID・law binding・compiler・lowering・scalar・axisの
metadata整合確認で、科学的平均価格の資格や比較policyの採択ではない。
詳細な組合せと注意点は[RECOVERY_HANDOFF](RECOVERY_HANDOFF.md)を参照する。
[公開・保全確認](PUBLICATION_CHECKS.json)では旧8ファイルとrun02の17ファイルのhash、
実行SOURCE不変、raw commit一致、既存manifest項目保全を確認した。

未取得はPauli law全体の平均/分散/CI、sampler law差、rare寄与、別path/LD2/長時間round、
価格modelの予測誤差、NおよびGである。この1 pathやorder2の一回の観測で未知差を0とはしない。

次はユーザーがB1や平均価格検証の要否・有限scopeを判断する。
Codex C/Dは本B0証拠を有限sampleの実接続証拠として参照できるが、追加科学実行には進まない。
commit/push・remote照合後、**mandatory STOP**。
