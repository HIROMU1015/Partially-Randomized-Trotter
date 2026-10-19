# H6 full m5_best：compile後不一致の静的調査と診断修正案

2026-10-19。**静的調査完了、根本原因は未確定。実装・実計算は行わずmandatory STOP。** 元run01の29/30 PASS・H6 full m5 cosineのSTOPは変更しない。新sampling・signal・statevector・compile・保存値の数値再解析は0件。判定閾値と科学条件の変更も0件。今回の成果は根拠付きの切り分けと、次の診断用実装の設計案である。

調査対象SOURCE：`c7d53f04c3be131078ee91e8bef3c78ca434ed3e`。結果commit：`b38898429af3a2a48d7057d9d828256c057685f8`。調査branch：`track-a-h46-m5-compile-stop-static-20261019`。既存の結果branchはb3889842に保持した。[結果報告](track_a_h46_m5_tol_pilot_run01_results_20261019.md)、[静的所見](../../artifacts/resource_applicability/track_a_h46_m5_compile_stop_static/2026-10-19/findings.json)、[SOURCE・runtime根拠](../../artifacts/resource_applicability/track_a_h46_m5_compile_stop_static/2026-10-19/source_trace.json)、[未承認計画](../../artifacts/resource_applicability/track_a_h46_m5_compile_stop_static/2026-10-19/diagnostic_recovery_plan_DRAFT.json)。

## 1. 確定した範囲

対象は保存tol-only1e-8 H6 DF19、直鎖1 Å/STO-3G、T=.8、P=0、full PF/prefix19/q2、`PF_M5_BEST_20D64C2_BINARY64_V1`、symmetric-directional、cosine full Hadamard wrapper。Qiskit1.3.0・rz/sx/x/cx・opt1・seed17。m5の元binary64係数・11 signed S2 piece・scalar一回は変更していない。

保存STOPの最後の値は `compiled_vector_cosine = 1.504043707445665e-9`。既存基準のfiniteかつ1e-9未満に不適合だった。それ以前のnative低枝、漏出、独立occupation作用（2.0510309204898196e-12）、uncompiled Hadamard cosine（1.1957101975212936e-13）はPASS。これは保存値の転記で、新しい計算ではない。

エラーはnative/occupation照合の後、[transpile後のraw状態ベクトル差](../../scripts/h46_m5_native.py#L267)で発生している。HamiltonianへのPF近似誤差やm5係数残差そのものをこのcheckで測ったわけではない。初期化からtranspile後まで全てが正しいと、この部分PASSだけで認定することもしない。

## 2. SOURCEから確認できたこと

|項目|確認内容|意味する限界|
|---|---|---|
|比較規則|`norm(after-before)`、位相合わせ・再正規化なし|位相、ノルム、非位相の誤差が同じraw値へ入る|
|global phase|保存simulatorがtop-level phaseを回路copyから外し、作用後一回適用。Qiskitはnested definitionのphaseも処理する|phaseの単純な未適用は見つからないが、実値・累積誤差は未保存|
|状態精度|固定Qiskitはdtype=complexとNumPyの行列積を使用|complex64降格の証拠なし。有限精度誤差の上界証明ではない|
|compiler|呼出側はopt1/basis/seedを渡す。近似度を1未満へ指定しておらず、固定transpileのdefaultは1.0|default値は回路全体の誤差1e-9未満を保証しない|
|opt1|固定ソースのdefault translator・1qubit再合成・inverse cancellationを確認|実runのpass履歴を保存していないため、どのpassが差を作ったか未確定|
|Gaussian|明示XXPlusYYを使い、uncompiledの直接matrix作用とbasis分解では演算列が異なる|演算列差・丸め蓄積は仮説。XXPlusYY実装の誤りを立証していない|
|scalar/control|各signed pieceはS2/scalar0、最後にancilla scalar一回、既存Hadamard配置を維持|この静的照合を理由に失敗点をPASSへ変更しない|

調べたQiskit Python file8件は固定1.3.0のdist-info/RECORDのSHA/bytesと一致した。これはその8件の静的整合であり、全package/backendの新しい独立検証ではない。providerと科学SOURCEは元のsealへ照合した。共有環境やpackageを変更していない。

## 3. 特定できた観測情報の不足

[元 `_run_task`](../../scripts/h46_m5_native.py#L255)では、transpile → raw検証 → metrics/cost保存の順になる。raw検証でraiseすると、cost行を保存する前にSTOPへ進む。

失敗taskに残ったfileは `events_PRIVATE.json`、`progress.jsonl`、`signal.json`、`STOP.json`、`fingerprint_cache.json` の5件だった。**wrapper/compiled回路のQPY、before/afterベクトル、実global phase、unvalidated費用は保存されていない。** [file一覧とSHA](../../artifacts/resource_applicability/track_a_h46_m5_compile_stop_static/2026-10-19/failed_task_file_inventory.json)で記録した。fingerprint/cache統計は回路本体やgate数を復元するデータではない。workerも回収済みである。

したがって保存証拠だけでは、次の仮説を識別できない。

- 共通global phaseの累積差が主成分だった。
- 長いgate列でのstatevector演算・ノルムの丸め誤差が主成分だった。
- basis変換・1qubit再合成などに非位相の差が生じた。

これらは併存し得る。閾値付近というだけで丸め誤差と断定せず、opt1を原因と断定せず、係数の再最適化も提案しない。今回確定した修正対象は**失敗時の証拠保存不足**であり、不一致を解消する数値修正を特定したわけではない。

## 4. 推奨する修正案：検証前に未検証証拠を保存

これは設計案で、コードへは適用していない。元SOURCEの `_run_task` を後から上書きせず、将来の新SOURCE・adapterで接続する。

1. 元と同じ回路・compiler・controlでtranspileし、identityを照合する。
2. raw受入判定の前に、wrapper/compiled両QPY・global phase・layout・fingerprint・compiler/input/SOURCEを専用quarantineへ保存する。整数費用も `UNVALIDATED_NOT_FOR_ACCOUNTING` として保存する。
3. low初期ベクトル・before・after、必要な独立expected作用をNPZへ保存する。dtype/shape/byte SHAと、保存段階・保存成功を記録する。
4. raw差・各ノルム・内積・枝別差・compile前後のHadamard値を記録する。共通位相を外した差を算出する場合も**診断専用**とし、受入規則・full-H reference・signalへは反映しない。
5. 元の全checkを元の1e-9未満で評価する。FAILはそのままSTOP。PASS後だけaccepted費用を出し、quarantineを費用会計の入力にしない。

概念上、a=before、b=afterなら、raw差は `||b-a||`。位相診断として `phi=arg(<a,b>)` と `||exp(-i phi)b-a||` を追加できるが、この値が小さくてもraw FAILをPASSへ変えない。元full-H referenceの位相を合わせ直す操作でもない。内積0・非finite・保存失敗では未解決/STOPを維持する。

保存は元runの場所へ書かず、別output/ledger/markerで行う。QPY/NPZは一時file→hash確認→確定の順にし、保存失敗でもaccepted費用を作らない。元の数値エラーを保存エラーで隠さず、両者を記録する。これにより、将来の数値検査を保存回路・ベクトルへ限定し、原因確認のたびにfull compileを繰り返す事態を避けられる。

## 5. 最小の実行案と承認境界

[plan DRAFT](../../artifacts/resource_applicability/track_a_h46_m5_compile_stop_static/2026-10-19/diagnostic_recovery_plan_DRAFT.json)は**未承認・未seal・有効grantなし・実行commandなし**。今回これを起動していない。

新しい診断取得案はH6 full m5/p19/q2の固定1点だけ。最大2軸、cosineを先に一回取得し、元の全検査にPASSした場合だけsineへ進む。cosineが再びFAILなら診断証拠を残して即STOP。成功済み58軸の再取得、PR標本、追加候補、model/費用再会計は0。元full-H referenceは元SOURCE/receipt/NPZ SHAとlineageを結合して再利用し、作り直さない。

旧59予約は返却しない。新scope最大2軸は**追加消費**であり、累積予約は最大61となる。旧60軸計画を遡及的に拡張するのではなく、新しいgrantとして別途承認を受ける。元run01は永久に部分STOPのまま、将来の結果は別runとして結合する。

回路再構築・独立occupation/Hadamard作用など、固定1点の新しい数値正しさ検査も将来の承認範囲へ明示する。新signal探索は行わない。修正案が合意されただけでは、新しい数値検査・compileの承認とはみなさない。

対象1点でcosine→sineの依存があるため、別候補や両軸を無理に同時起動しない。既存reference・preparation/fingerprint cacheを再利用し、58軸を繰り返さず、保存証拠を後続診断へ使うことを高速化の中心とする。fresh CPU/RAM/disk・自分のプロセスを確認し、歴史的worker数を恒常上限へ転用しない。compiler/profile・共有環境を高速化のために変えない。

## 6. 次の実装で必要な人工検証案

- raw FAILでQPY/NPZ/未検証metricsが残り、accepted costが作られない。
- 共通位相だけの人工差でもraw FAILを維持し、診断値で受入を代用しない。
- cosine FAIL時はsine予約0、全worker回収・原STOP保全。
- 保存失敗・非finite・欠測はfail closed、欠測費用を0補完しない。
- 旧grant再使用・SOURCE/input/plan改竄を拒否し、旧58軸や新candidateへscopeを広げない。
- full-H referenceの欠測・SHA/lineage不一致時に、新solverへ自動fallbackしない。
- 全PASS時のみaccepted費用を保存し、後続会計/Coreへ自動移行しない。

これらは今後の人工検証項目で、今回実行済みのtest件数には含めない。

## 7. 今回の確認・保全・次の担当

今回の構造検証38項目は全PASS（人工数値testは0件）。今回実施したのはSOURCE/JSON/file inventoryの読取、AST/行位置確認、選択したQiskit source8件のRECORD照合、封印/原grant/結果のSHA照合、資料構造検査のみ。実分子配列は数値配列として開かず、Qiskit/NumPy等の科学moduleも呼び出していない。原run・authorization・使用sealed fileを629 fileの保全一覧へ記録した。実装SOURCE16 fileは不変。新しいroot causeの数値証明・新公式採用・独立reviewは0。

**Codex（次の技術作業案）**：新しい診断adapter、未検証保存のschema、人工test、別契約・資源見積り・無効grantを用意してsealする。まだ実装していない。今回の静的調査を「数値不一致の修正完了」と呼ばない。

**ユーザー（次の実行判断）**：その準備資料を確認し、固定1点・最大2軸の新診断取得を明示承認するか判断する。m5主比較採用AD03・後続Coreは今回の対象外。

資料はlocal commitで停止する。公開push・旧結果/STOPの変更・新科学計算は行わない。
