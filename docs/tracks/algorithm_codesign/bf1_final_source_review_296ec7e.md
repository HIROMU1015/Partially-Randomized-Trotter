# BF-1 final source review — 296ec7e

日付: 2026-10-05 JST。
対象source: `296ec7e4c025f09e4bfda96e56e08d32388b81c5`。
利用者から受領した判定: `PROCEED_TO_FINAL_BF1_EXECUTION_REVIEW`。
今回のlocal source review判定: **`REVISE_NUMERICAL_ASSEMBLY_GUARD_BEFORE_AUTHORIZATION`**。

研究方針は維持する。BF-1前にfamily、split、geometry、accuracy、budgetを追加しない。
以下の数値guardだけを閉じてから一回限りの実行をreviewする。
本書はCodexによるローカルsource reviewであり、独立研究者のsignoff、BF-1実行承認ではない。
source、準備packet、domain、正式authorization、one-shot registryを変更していない。

## 1. 問題のない部分

| review項目 | 確認結果 |
|---|---|
| source identity | inventoryの70 fileについて、working bytes・source commitのblob・既存SHA-256が一致 |
| plan identity | canonical fingerprint `c338e656fdb8ada30b8731dff23afe9570df4f356537ac1a4fc37b1a9ef6fd88` |
| domain | 既存v1 manifestを保持。raw SHA-256 `caece92bd2d9f827b0f1f17281865869420bf923273ff294fd30cabd24b51efe` |
| preparation test report | raw SHA-256 `ee0813aff72efc6ce4e3272757f36fa71f585ab61a9cb83da70e0d92ad28d54b`。旧artifactは保持 |
| environment | 固定interpreter・package・thread設定に一致 |
| O/L/F | 同じ32係数評価、Oは診断、Lはnovelty対照、共通finite再採点、5%のprimary routeのみ |
| cross-score | 全探索とprimary再採点後、bridge前。cache-only、不足cellは拒否、primary分類を変更しない |
| attribution | reachability/budget説明を排除できなければfinite固有の設計原理を主張しない |
| authorization方式 | source Sの唯一のauthorization-only child A。許可追加path限定、source/JSON/環境をinput操作前に検査 |
| STOP | 全caseで停止。retry、BF-2、grid拡張を自動認可しない |

source-bound focused再検査は`33 passed in 1.42s`。full suiteではない。
直接pytestを起動した最初の呼出しは`PYTHONPATH=src`の指定漏れでcollection段階に停止した。
指定を補った上記の限定再検査はpassした。prepare runnerや既存packetは再生成していない。

## 2. P1: DF assemblyの誤差をfinite積へ戻すguardが不足

対象は[`pilot.py`](../../../src/trottertracks/algorithm_codesign/pilot.py)の`Evaluator._signal()`。
現在は各factorのbinary64 spectral評価誤差を逐次伝播する一方、DF assemblyの摂動は積の最後に
`0.8 * matrix_assembly_bound * max(1, norm(vector))`として加える。
`Evaluator.__init__()`のtargetにも別途`0.8 * matrix_assembly_bound`を加える。

`Spectrum`のresidualは**供給された行列**の対角化誤差を扱う。
DF snapshotの数学的generatorと供給行列との差は、別のassembly項がcoverする必要がある。
exact full evolutionの摂動に使える`T * error`を、signed-stage・finite polynomial積へ
そのまま適用する根拠は成立しない。unitary factorでも絶対時間が寄与し、finite factorでは
polynomialの感度とprefix/suffixのnormも寄与する。

これは未実証の懸念だけではない。固定Suzuki5係数を使う1×1 synthetic witnessで、
**assembly errorの上界を上向きに丸め、実際のgenerator errorを覆った状態でも**、
既存codeのreported `u_signal`をRe biasの誤差が超えた。

| synthetic条件・観測量 | 値 |
|---|---|
| deterministic側 | 4個のzero 1×1 matrix |
| reference tail / supplied tail | `5` / `5 + 1e-6` |
| formula / T / q / R / K | 固定Suzuki5 / `0.8` / `1` / `5` / `2`（P3） |
| allocation | `[1,1,1,1,1]` |
| assembly bound | `nextafter(abs(supplied - reference), +infinity)` |
| reported `u_signal`（target guardを含む） | `2.593050117669466e-6` |
| Re biasの誤差 | `3.1553663404348953e-6` |
| Im biasの誤差 | `2.400019153858679e-6` |
| 最大axis誤差 / `u_signal` | `1.2168551309262072` |

referenceは80-digitの
`prod_j P3(-i * 0.8 * w_j * 5)`と`exp(-i * 0.8 * 5)`から計算した。
比較対象はcodeが報告した**axis bias**であり、coefficient探索やscience評価ではない。
このwitnessはactual H4での誤差量、feasibility、winner、BF-A/B/Cの証拠ではない。
既存testsで`Task.matrix_assembly_bound`が非zeroの場合のこの摂動をcoverしていなかったため、
33件のpassだけで実行前のguard義務を閉じられない。

## 3. 最小の修正・再review範囲

1. assembly budgetがfull targetと各generatorについて何をboundするかを明示する。
   full-H reconstruction discrepancyだけからindividual generator誤差を推定しない。
2. 各factorのassembly摂動を、絶対時間と有限polynomialのLipschitz項へ戻す。
   例えばoperator摂動`delta_i`とreference norm上界`N_i`を用い、
   `e_i <= N_i * e_(i-1) + delta_i * norm(v_(i-1))`で積へ伝播する。
   必要なgenerator norm、prefix norm、scalar/identity項も同じ定義でboundする。
   提案段階の式であり、実装済み・証明済みとはしない。
3. このwitnessを回帰検査へ加え、限定testとsource-content sealを新しいsource commitへ固定し直す。
   domain、RQ、O/L/F自由度、accuracy/materiality、資源上限を変えない。
   `296ec7e`へのauthorization-only childにsource修正を混ぜず、改訂sourceを別途reviewする。

guardを緩めたり、このsynthetic条件だけを対象外としてpassへ戻したりしない。
現sourceへの正式authorizationを作成せず停止する。研究方針の全面再評価は引き続きBF-1後に置く。

## 4. 再現資料

- [source audit](../../../artifacts/track_b_bf1_preexecution_review/2026-10-05/296ec7e/source_review_audit.json)
- [synthetic witness result](../../../artifacts/track_b_bf1_preexecution_review/2026-10-05/296ec7e/assembly_guard_witness.json)
- [reproducer](../../../artifacts/track_b_bf1_preexecution_review/2026-10-05/296ec7e/assembly_guard_witness.py)

reproducerは固定CPU interpreter、`PYTHONDONTWRITEBYTECODE=1`、四thread設定=1で実行する。
stdoutにJSONを出すだけで、既存artifactを上書きしない。
分子NPZ操作、science input開封、science signal、係数探索、trajectory、circuit、compile、GPUは全て0。
新規追加は本review文書と上記三資料だけで、commit/push、正式authorization、BF-1実行は行っていない。
