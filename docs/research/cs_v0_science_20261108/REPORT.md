# C-S V0限定科学実行：binding型不一致でSTOP

2026-11-08 JST／SERVER1 GPUA100-gleap-01。
**一campaignを実行したが、Stage Bの最初のinput bindingでTypeErrorとなり、mandatory STOPした。完了spec0、strict合成0、科学retry0。局所的な費用・誤差交換は未判定。**

## 固定条件・SOURCE・実行回数

ユーザーの明示実行許可に従い、科学SOURCE `0bd18bb7d982bad32f795558a2caf27d34ba3778`（tree `2513565e8d88f853119c5644c50417ba8bcddf77`）を変更せず、一回だけ固定CLIを呼んだ。

対象は保存linear H4／1.00 Å／STO-3G／legacy DF12／8system／全4電子sector ground参照／L_D3、m0,q1,δ=.02,r1,K2,coefficient_atol0、N1 actual Q／selected full。実stateは使わない。候補はcanonical first D/Rのjoint factor0各1、π保持nearest1/256rad/tie-away、residual r1/K2/root256/proposal160/ηρ10^-12、strict pygridsynth2.0.0/seed0/dps100/10^-6、finite cap10^-12/local cap3e-6、positive keys≤8、campaign1/retry0。

N1 `744e1a09661ebba6841fe5faf456f88c41c47780`、N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`、旧engine `fdf00cee3c87d804845497afc89ac6a71f3b3f59`の固定参照を維持した。旧engine保証を変更後C-Sへ転用せず、N1/N2/DF/CDF/stateを再計算していない。

worktree=`/tmp/cs-v0-science-20261108`、branch=`codex/cs-v0-science-results-20261108`。固定runnerは一回のみ、process exit code=1。正確なargv・環境・tracebackは[RUN.json](../../../artifacts/cs_v0_science/2026-11-08/campaign_run01/RUN.json)と[RUN.log](../../../artifacts/cs_v0_science/2026-11-08/campaign_run01/RUN.log)。

## 取得できた範囲

| 項目 | 結果 |
|---|---|
| Stage A | PASS：新SOURCE/remote/tree、固定13入力bytes、native runtime/tool profile、専用fresh output/cache、実空き資源を照合 |
| Stage B | binding一回開始、RETURNなし、例外STOP。D/R selected specの返却0 |
| local comparison／finite mean/error | 各0 call、未評価 |
| strict backend／qualified keys | 予約0／返却0／qualified0 |
| local native price | 0 call。S0/S1/S2の価格はUNKNOWN（無料/0価格ではない） |
| finite/native/local cap | UNKNOWN・未評価 |
| state/Born/RNG/quantum shot/full build/compile/N/G | 全0 call |
| science retry／追加campaign | 各0 |

[RESULT_SUMMARY.json](RESULT_SUMMARY.json)は保存値の転記、正本は[runner RESULT](../../../artifacts/cs_v0_science/2026-11-08/campaign_run01/RESULT.json)、[CALL_COUNTS](../../../artifacts/cs_v0_science/2026-11-08/campaign_run01/CALL_COUNTS.json)、[journal](../../../artifacts/cs_v0_science/2026-11-08/campaign_run01/journal/)。CALL_COUNTSのbinding state=STARTは成功を意味しない。journal 0001とSTOP.jsonに例外があり、bindingは返却されていない。

runner記録のwallは約1.77秒、process CPU約0.212秒、peak self RSS 37,888 KiB。これらはrunnerの計測点までの古典費用で、Python起動・最終監視/書込み等を含む全経過時間の完全会計とはしない。実測資源snapshotでは空きCPU/RAMがあり、今回のSTOP理由は型例外である。OOMやserver/session障害と解釈する証拠はない。

## 原因：保存schemaと人工fixtureの型差

例外位置は固定`cs_v0.py:74`の `params[0] = -params[0]`。実N1の保存basis parameterは文字列で、最初のinverse側XXPlusYY gateの2parameterもstrだった。一方、adapterは符号反転前に数値型を想定しているため、`TypeError: bad operand type for unary -: 'str'` となった。

対応する従来人工fixtureは `parameters=[.5]` の数値入力で、この保存文字列の境界を覆っていなかった。**前回の人工PASSとTECHNICALLY_READY判定から、実保存schemaでのbinding成功を保証してはいけなかった。** 旧PASSは元の人工条件で保持し、今回の実入力STOPを別の証拠として追加する。

[SCHEMA_DIAGNOSTIC.json](SCHEMA_DIAGNOSTIC.json)は例外後の静的な型・SOURCE確認だけ。保存角の再導出、slot再取得、coarse/residual評価、bindingの再実行はしていない。SOURCE/input/runtimeの不一致ではなく、固定adapterのschema対応不足である。

## 保全・公開・次に必要なこと

固定runnerが作った原本9ファイル（SOURCE/PREFLIGHT/profile/RESULT/call/cost/STOPと2journal）はbyte不変で保全し、実行log/argvの2ファイルだけ追加した。全11ファイルを[ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json)で結ぶ。cacheは空のまま。旧SOURCE・STOPを成功へ書き換えず、prefixを消して再試行していない。

元mainのHEAD/status/3,087既存files、0bd18bb/606c40dの旧コード・準備資料、固定13入力、N1/N2/engine、SCV raw、旧grant/ledger/marker/STOP、C1凍結、他job/process/cacheを保全。共有OS/venv/settings変更0。既存人工testの再実行0、新コード/新test追加0、科学SOURCE修正0。結果・索引/statusだけ新branchへ保存し、GitHub公開SHA/body/blob/tree照合後STOPする。

次に必要なのは**保存parameterの既存codecを使う最小schema修正と、文字列保存fixtureに対する差分人工test**である。今回は実装も再実行もしていない。その新SOURCEを固定した後、必要と判断された場合だけ、変更SOURCEと有限scopeに対する別途明示実行許可を受ける。既存の許可を科学retryへ読み替えない。承認証明/grant発行は不要。

今回のSTOPはC-Sの科学的NO-GOではなく、局所資格計算へ進む前の技術的不適合。S0/S1/S2の局所価格・normalization交換、cap、資源改善は全て未判定。V1や他改善、重要科学レビューへ進まない。

**Mandatory STOP：一campaignは中断終了・原本保全済み。retry0を維持し、追加科学計算を開始しない。**
