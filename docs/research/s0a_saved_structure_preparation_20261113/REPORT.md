# S0a保存構造価格照合の実行準備（2026-11-13）

**S0aのreader・S専用context・固定4path/XY8計数runnerを準備し、新13＋既存9＝22技術テストがPASSした。実pathのS計数・保存Sとの数値比較は未実行。** 保存raw4本とsnapshot/receiptの所在は確認したが、内容hash・event digest・実preparation再構成の整合性は科学実行時の条件として残る。したがって、技術経路は準備済み、実入力での動作成功は未確認である。

実装SOURCE `11cb91a65413bd33f5c02fc43fda40e583d91bd8`（最初の追加`d1d36c33c`とmetadata profile位置の修正を含む）。既存C branch `codex/r-variable-compiled-price-design-20261111`を継続。基点`fcf5b64fac24f3897ab3408b84fd8256f46e8513`。

[EXECUTION_PLAN](EXECUTION_PLAN.json)／[INPUT_AVAILABILITY](INPUT_AVAILABILITY.json)／[SCIENCE_PLAN](SCIENCE_PLAN.json)／[TESTS](TESTS.txt)／[旧9件](REUSED_TESTS.txt)／[HANDOFF](HANDOFF.md)／[CHECKS](CHECKS.json)。

## 固定候補とSOURCE

| 系 | qPF/R | candidate ID | 元role | selector |
|---|---|---|---|---|
| H4 DF10 | 8/8 | `6e1efff1ba1f7ac95a3b` | CALIBRATION | CALIBRATION.json → H4_q8_R8 → index0 |
| H4 DF10 | 16/8 | `201d498a3fa5eaa3b5c5` | VALIDATION | VALIDATION.json → data → H4_q16_R8 → index0 |
| H6 DF19 | 1/32 | `4269eafc8c0ef8922c23` | CALIBRATION | CALIBRATION.json → H6_q1_R32 → index0 |
| H6 DF19 | 2/32 | `24fec6bdd44d6a728f1e` | VALIDATION | VALIDATION.json → data → H6_q2_R32 → index0 |

すべてC1公開`2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`の`docs/research/r_price_phase1_results_20261111/run01/`に固定。元科学SOURCEは`1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`、basis/prepare/signed scheduleはC1 carrier`155529e20a8fac39e00786c40b57830b71203c4f`、providerは`c31ef424175c02379eafebe8fc6c7548c1d9aabe`。各normal cohortは元n8のまま、今回はindex0だけを選ぶ。roleを変更せず、新しい平均法にとっては既知development診断でありfresh holdoutではない。

H4はLinear 1Å/STO-3G・保存tol-only DF10・S2/δ.1/prefix1/K2、H6は同DF19・m5_best/δ1.6/prefix7/K2・元11 signed piece/整数allocation。保存weighted completion/fragment順・H/state/snapshot・control binding・scalar once/P0を維持する。`EXECUTION_PLAN`は元C1 identity/law/profileを無変更copyし、H/state/snapshot hash、T/q/R、signed weights/allocation、formula、分割、元cohort、private raw path、event digestを保持する。

samplingの宣言は元`ideal_finite_RTE_law`と`PCG64_numpy_float64_CDF_saved_cohort`、SOURCEは元rte.py。今回は乱数を再生せず、保存された同一event列を用いる計画。seedはprovenance照合だけでRNGを起動しない。実samplerとideal lawの距離を新たに保証しない。

比較単位は**測定1回のfull Hadamard wrapper（measure付き）に対するliteral structural RZの一path値**。X/YはC1の`cosine/sine`順で保存Sと整数の完全一致を調べる。実compiled C、N×C、平均S、平均compiled価格との比較ではない。compiler profileはQiskit1.3.0/rz,sx,x,cx/opt1/seed17・元hashをidentity照合に保持するが、compilerを呼ばない。

## 実装と再利用

- `s0a_saved_structure.SavedEvents`：既存C1 JSONの`events`だけをtail/event単位で読む。標準JSONDecoderを使い、binary64値、−0.0、event/app順を変更しない。canonical digestは元`r_price_phase1.digest`と同じsort_keys/compact/allow_nan=falseのnested-array表現を逐次SHAへ送る。event列全体は保持しない。seeds/laws metadataは一括保持するため全体memoryがq/R非依存とは主張しない。
- 同じ合法tailを一つのiteratorとして最後まで供給する。前段`stream_structure`、C1 `structural`、H4/H6 `Accounting`、native counterは変更していない。piece/outer間の状態継続、scalar、axis、wrapper、terminal一回は元loopに任せる。
- S専用contextは固定C1 `Backend.context`のうち保存input/prepare部分とcounter/signed head部分だけを再利用する。compile関連のimport/patch、compose、backendのsampling呼出しを実行しない。元式の演算順を保ち、snapshotから元preparation metadata・head primitives・scalarを再構成する処理も**科学実行側に延期**した。SCF/DF/H/state再生成、QC buildは行う経路を設けていない。
- law検査は元`make_rte_config`・cutoff helper・`_make_event`を再利用する。保存selected componentから1eventの合法metadataを決定論的に復元し、元basis operations/hash、signed angle、relative phase、product順/符号、probability、normalization等をcanonical完全一致で照合する。これはsamplingではない。元preparation hashとH/input hashはC1の検査を保つ。
- `parity_one`はまずSを計数せず全event/digest/law/seed/conditionalを検査する。その後、同じrawを各軸で読み直し、計2件の新Sを保存Sと照合する。旧Sの新再計数0、計4path/8新S。検査のための複数回読取はあり、O(qR)時間は残る。
- `run_s0a_saved_structure.py prepare`はmetadataと所在確認だけ。`execute --execute`は別途のユーザー科学実行許可がある場合の入口であり、flag自体は許可を証明しない。fresh outputを要求し、失敗時STOP、上書き/自動retry/代替sampling/model fallbackを行わない。

## 人工テストと初回テストの問題

新13件はreaderのchunk1/7/4096、binary64/負ゼロ・順序/digest、合法tail継続、不足/過剰・duplicate JSON/nonfinite/欠測、law/seed/conditional/basis/phase/angle不一致、正負時間とorder0/2、元Sとの人工parity、X/Y/scalar/wrapper、mock8件の正常STOP・失敗STOP、metadata-only公開fixture照合を検証した。既存9件の11piece/outer/phase/terminal/memory反例テストもPASS。作者自身の技術確認であり独立科学レビューではない。

**初回lawテストでは、既存toy fixtureのimportが広い`trotterlib.__init__`を経由し、Qiskit passmanagerの警告を伴う人工loweringの副作用が発生した。** 実分子S/compileや保存rawは対象ではなかったが、回路経路が一度も起動しなかったとは記載しない。最終テストはroot packageを実行しない純rte metadata fixtureへ変更した。S専用の実行processでも明示submoduleだけのnamespaceを使い、preparation中のQuantumCircuit生成/transpileを例外にする。拒否guardの人工テストは実処理前に停止する。元packageや固定SOURCEは書き換えていない。初回のtoy tolerance/mock field不足、公開profileのidentity内配置の修正も技術debugであり、科学retryではない。

最終新13件は警告なし。実H4/H6のS計数・保存S比較・新実価格・fit・N/Gは0。実rawを開いてevent内容を検査する処理も実施していない。

## 未確認条件・失敗時の扱い

保存4 raw、H4 snapshot/receipt、H6 snapshot/2 receipt、元config/providerの所在を確認した。内容hash・preparation_hash・実イベントdigest/lawを今回PASSにはしていない。providerのNumPy1.26.4/Qiskit1.3.0、保存入力hash、元basis/precisionが合わなければSTOPし、自動更新しない。QC build禁止guardに触れる依存が残っていた場合も失敗STOPし、別SOURCEの技術修正として扱う。

欠測raw/metadata/S/axis、source/candidate/role/compiler/law不一致、allocation過不足、digest差、S不一致は修復・resample・旧S再計数をせず報告する。実入力でのcontext/readerは未実行なので、技術テストPASSを実S parity PASSへ昇格しない。

S0a一致が後に得られても、小さい物理閉状態、全event生成なしのE[S]、全law平均compiled C、長q/RのC−S移送、モデル採用や必要価格精度を保証しない。C1資格false・未知のsystematic/共分散/CI/nullは不変。MeanKernel/connected-cluster/C1は今後の候補として保持し、係数移植や補正の二重計上は行わない。

## 実行前の判断とSTOP

次に必要な明示許可は、**SOURCE11cb91a65の固定4保存pathについて、保存inputからの元構造metadata再構成とlaw/digest検査、X/Y計8件の新stream S計数と保存S照合のみ**。新sampling/旧S再計数/compile/fit/平均価格/残差再解析/N/G/C2/他pathを含めない。追加の科学的必要性が判明したら、黙って拡張せずユーザー判断へ戻す。[実行案](SCIENCE_PLAN.json)参照。

B最終統合`f09af82482aad11460c4bb73d3e13d7ada40be6f`のREPORTを読取参照した。B所有のcontroller/identity/provider会計は変更せず、本成果はC内部leaf。S0aの結果をBの平均compiled価格provider recordへ変換しない。Bの実RMSE/M/κ候補が明らかになってから、平均法の閉包/局所残差・独立検証を最小範囲で選び直す。旧S0全32path/C2の40/16wrapperは自動実行しない。

次：**ユーザー**が上記8件の科学scopeを判断。**Codex C**は許可後にその範囲だけを実行し、失敗・不足も保存してSTOP。**Codex B**は最終統合を担当し、S0a parityを平均価格資格として使用しない。今回の技術成果のcommit/push・remote SHA/主要成果物照合後mandatory STOP。
