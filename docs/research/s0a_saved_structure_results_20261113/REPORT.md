# S0a run01：context設定エラーでS開始前に停止（2026-11-13）

**S0aを承認範囲で1回起動したが、contextのexec scopeに`ast`を渡し忘れており、`NameError`で停止した。新Sは0/8件、保存S照合は0件で、parityは全4path未評価。** 科学条件の不成立ではなく実行adapterの不備である。実H4/H6でのparity PASSは得ていない。

[run01 STOP](run01/STOP.json)／[固定入力PLAN](run01/PLAN.json)／[実行log](EXECUTION.txt)／[SUMMARY](SUMMARY.json)／[修正後人工14件](TECHNICAL_TESTS.txt)／[HANDOFF](HANDOFF.md)／[CHECKS](CHECKS.json)。

## 実行した範囲と到達段階

実行SOURCEは`11cb91a65413bd33f5c02fc43fda40e583d91bd8`、着手時公開commitは`db3598aa52aad7b9db80ad9fdddad4c1f6d0c85b`。既存C branch `codex/r-variable-compiled-price-design-20261111`を継続し、[準備済み計画](../s0a_saved_structure_preparation_20261113/SCIENCE_PLAN.json)どおりのrun01を起動した。

| 系 | qPF/R_outer | candidate ID | 元role | path | 状態 |
|---|---|---|---|---|---|
| H4 DF10 | 8/8 | `6e1efff1ba1f7ac95a3b` | CALIBRATION | normal index0 | X/Y未評価 |
| H4 DF10 | 16/8 | `201d498a3fa5eaa3b5c5` | VALIDATION | normal index0 | X/Y未評価 |
| H6 DF19 | 1/32 | `4269eafc8c0ef8922c23` | CALIBRATION | normal index0 | X/Y未評価 |
| H6 DF19 | 2/32 | `24fec6bdd44d6a728f1e` | VALIDATION | normal index0 | X/Y未評価 |

対象はLinear H4/H6・1Å・STO-3G・保存tol-only DF10/DF19。H4 S2/δ.1/prefix1/K2、H6 m5_best/δ1.6/prefix7/K2・元11 signed piece/整数allocation、元weighted completion/H/state/snapshot、symmetric-directional control、full Hadamard wrapper/measure/P0を変更していない。C1公開`2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`の同じevent列と保存Sを照合する計画であり、元n8/roleはそのまま。平均価格法のfresh holdoutではない。

runnerは固定public PLAN/identity/role/source/compilerと現SOURCEを照合し、fresh outputへPLANを保存した。最初のH4 contextで、C1 carrierのSOURCEを`ast.parse`する段階で失敗した。この処理はsnapshot/receiptの内容hash検査、保存H/preparation再構成、raw reader、law/event digest検査、Accounting生成より前にある。したがって、今回それらの実入力検査も完了していない。rawの所在確認を内容PASSに読み替えない。

- 実分子の新stream S：0、旧S再計数：0、保存Sとの数値比較：0。
- 新sampling/RNG replay、QC build/compile/transpile、fit/価格残差再解析/平均予測、signal/N/G：0。
- 平均S/平均compiled C・物理閉包・compiler残差の外挿資格・model採用：未評価のまま。

実行logと`S0A_FAILED_NO_AUTOMATIC_RETRY`のSTOPを保全し、同じrunを再開・上書きしていない。runnerのRESULTS.jsonは生成されていない。SUMMARYは失敗段階と未評価coverageを記録するもので、科学結果を補完するものではない。

## 技術修正と検証

`s0a_structure_context.load_context`のexec用scopeへ`ast=ast`を追加した。元C1/Accounting/stream_structure/law/source数値式は変更していない。修正SOURCEは`9516351e6`（full SHAはSUMMARY参照）。実行した旧SOURCEはGit固定commit11cb91a65に残す。

新しい人工testは、固定context本体とnested scopeが参照するglobal名をsymtableで取り出し、実際に渡されるexec scopeにすべて存在することを確認する。execをmock化し、実分子context・保存input/raw・新Sを呼ばずに今回の漏れを検出する。修正後14技術testが全PASS。前段の旧counter人工9件は保存記録を再利用し、今回再実行した14件へ加算していない。作者自身の確認で独立科学レビューではない。

修正版の[RUN02_PROPOSED_PLAN](RUN02_PROPOSED_PLAN.json)をmetadata-onlyで準備した。対象・候補・role・law・axis・digestはrun01と同じで、SOURCEだけを修正版へ固定する。**run02は未実行**。input/preparation/digest/lawの実内容確認はまだ残り、修正テストPASSは実parity成功を保証しない。

## STOPと次の判断

固定[HANDOFF](../s0a_saved_structure_preparation_20261113/HANDOFF.md)の「黙って科学retryしない」、[SCIENCE_PLAN](../s0a_saved_structure_preparation_20261113/SCIENCE_PLAN.json)の失敗時STOP・自動retry禁止に従った。今回のユーザー許可はrun01へ適用し、失敗後の修正版SOURCEによる科学再試行を暗黙に追加していない。

次は**ユーザー**が修正版SOURCE・同じ4保存path/XY8件のrun02再試行を承認するか判断する。承認対象は保存input→元構造metadata再構成、law/digest検査、新S8件と保存Sの完全一致照合だけ。追加path・旧S再計数・sampling・QC build/compile・fit/残差再解析・平均価格・N/Gを含めない。**Codex C**は明示許可後のみfresh run02へ進み、成功/失敗のいずれも公開してSTOPする。**Codex B**は今回をmissing parityとして扱い、平均価格資格へ変換しない。

C1/C2/connected-cluster・旧SOURCE/input/raw/結果/cohort/grant/ledger/marker/KEEP/STOP、A/Bのworktree/process/cacheを保全した。本結果と技術修正をcommit/pushし、remote SHA/主要成果物照合後mandatory STOP。
