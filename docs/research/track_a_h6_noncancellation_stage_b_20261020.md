# H6 noncancellation段階B：保存処理エラーによるSTOP

2026-10-20。基点は段階A完了commit `c53dd16ae16413182c9d3080f82f80237231a248`。
科学SOURCEは **`224fbba00686f818a7a037465dcf2f2cc60b0dff`**。
専用branchは `codex/h6-noncancellation-stage-b-20261020`。

**承認済み30 scopeのone-shotは、数値処理後の結果保存でTypeErrorとなり、native exit 1でSTOPした。解析完了・100 scope比較完成とは扱わない。再試行0、worker回収済み。**
既存712費用の完全一致gateと部分JSONは保存されたが、最終数値receipt、summary、CSV、図、完了gateは未保存である。残存JSONを完成結果として採用したり、欠けた監査値を追加計算して補完したりしない。

## 承認範囲と固定条件

ユーザーは段階Aの適格34・不適格86・未解決0を正式に受け入れ、H6 noncancellation30 scope（重複除外10 endpoint）について、旧未確定120行の接続、保存shot込み再会計、PF再選択、PR/PF比較、順位・同率・逆転、数値検証と終了監査までをone-shotで承認した。自動retry・追加科学実行は承認していない。
[ユーザー指示receipt](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b/2026-10-20/user_approval_receipt.txt)と[契約](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b/2026-10-20/sealed_contract.json)を保存した。

Linear H6、隣接1 Å、STO-3G、tol-only DF19、T=.8、P=0。
元のsymmetric_directional/full cosine-sine Hadamard wrapper系列を使用し、candidateごとのprefix（L_D）とq、δ=T/qを保持する。
ε=.05/.01/.005/.001/.0005、allowance倍率1/10、元のnoncancellation bias/shot政策、登録候補集合old S2 32・fine S2 96・S2+truncated S4 136を固定した。
PF費用指標は `saved N_cosine × compiled RZ_cosine + saved N_sine × compiled RZ_sine`。
PRは決定論的prefix・確率的RTE tail・full wrapperを含むH6確認union8 n32の保存平均・SE・cohort・選択を保持した。実量子shot実行、MC標本n本の古典取得合計費用、独立P cohortの比較とは異なる。

新規signal/sampling/compile/モデル予測/refit/shot再計算/適格性再計算/PR統計/旧70再解析/候補追加は0。
既存SOURCE・raw・grant・ledger・marker・STOP、旧70結果、論文主表・Figure2・プロトコルv2本文を変更していない。

## 実装と実行前の検査

[core](../../src/trotterlib/h6_noncancellation_stage_b.py)は旧比較coreの有理数費用・選択・順位処理を再利用し、Stage Aの接続と新baselineの来歴だけを追加した。
`eligible`は明示的なboolを要求する。Noneをtruthinessで除外する経路を使用せず、unknown・欠測・baseline不一致をfail-closedにした。
旧1440行（PF1360・PR80）のうち、元のunknown120行だけにStage Aを接続した。元から確定していた行とPRはliteral不変である。

[人工test](../../tests/test_h6_noncancellation_stage_b.py)14件と旧比較回帰13件の計27件がPASSした。missing/duplicate/extra patch、known行の変更、shot不整合、None、PR不一致、domain欠落、厳密な順位逆転、旧70転記を検査した。実入力の数値再解析はテスト時0。
一方、runnerの全出力を人工値で保存する検査を欠いており、SOURCEを既に持つsummaryを処理する実装ミスを検出できなかった。これらのテストPASSを本番完了の証拠としない。検査はprimary agent自身によるもので、別担当の独立レビューではない。

SOURCEを先にcommitし、7 source file・13入力・算術runtime・旧85証拠file・underlying Git blobのSHAをsealした。専用grant/launch marker/one-shot marker/ledgerを作成した。過去のgrantは再利用していない。

## 到達点と停止原因

[ledger](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b/2026-10-20/ledger.json)では以下7段階の関数復帰を記録している。

1. Stage Aの120行接続。
2. 既存712費用の完全一致検査と、新34適格費用のbaseline構成。
3. 新しく完成させるbaselineでのPF選択。
4. ALL_SAVED458による保存単価置換。
5. 新30 scopeの比較・順位処理。
6. 整数cross-productによる費用・選択・順位・比の数値検証。
7. 旧70 literal参照と新30の結合。

[baseline replay gate](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b/2026-10-20/baseline_replay_gate.json)は既存712費用の完全一致と新34費用の構成を記録した。
元の全domain最良値はnullだったので、新34を含むbaselineは今回の新来歴であり、旧全domain winnerの再現とは呼ばない。

その後、runnerの出力loopで `dict(SOURCE=c['SOURCE'], **value)` を使用した。
先行してSOURCEを付与したsummaryが `value` に到達すると、同じkeywordを二重指定するため、次の例外が生じた。

```text
TypeError: dict() got multiple values for keyword argument 'SOURCE'
```

[stderr](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b/2026-10-20/native_stderr.txt)にtracebackを保全した。入力不一致や科学的な値のFAILは記録されていないが、最終数値監査結果そのものを保存する前に停止しており、未保存の監査をPASS済み成果物として扱わない。

保存済みは、接続行1440・patch来歴120・PF候補費用1360行、selection30、candidate ranks、rank relation changes、旧70+新30の100参照行などのJSONである。
`summary.json`、`numeric_validation.json`、`numeric_completion.json`、5 CSV、図、完了監査は作成されていない。
科学処理の完了call receiptも未保存のため、ledgerの段階完了記録と固定call予算を区別する。
部分結果の費用・比・順位の再計算や、新たな結果集計、作図は行わない。

## 終了・保全監査

[technical STOP audit](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b/2026-10-20/stop_audit.json)は科学関数を呼ばず、保存JSONの構造・literal・bytes/SHAとnative終了のみを確認した。
実行したscriptは[STOP専用監査](../../scripts/audit_h6_noncancellation_stage_b_stop.py)である。正常完了用checker/plotterは用意したが、今回実行していない。

| 監査項目 | 状態 |
| --- | --- |
| SOURCE7・入力13・runtime・旧証拠85の照合 | STOP後の読取監査PASS |
| Stage A接続 | 旧unknown120行だけ、適格性/shot/accounting literal一致 |
| 元のnonnull行・PR | literal不変 |
| combined JSONの旧70行 | 元70行にnumeric_originを付加しただけで完全一致 |
| combined参照 | 100 scope、重複除外40 system/bias/ε/倍率 endpoint |
| 科学完了・最終数値gate | 未達、今回の正式受入0 scope |
| native exit | 1 |
| 同一起動親のwait4/reaped | 確認済み、owned worker残0 |
| grant | CONSUMED_STOP、追加実行権なし |
| retry・追加科学 | 0 |

親PID487730、子PID487784。wall `4.454593515023589`秒、CPU `4.442003000000001`秒、peak RSS `110284800` bytes。
小規模の保存値演算を1 childでindexしendpoint間の重複を省く構成で、CPU/RAM/workerの独自上限、affinity・OS設定変更、他ユーザー/processへの変更は0。
STOP後のSHA inventoryは、保存時点のbytesを今後のために固定したものであり、失われた正常終了gateを遡及的に復元した証拠ではない。

## 100 scopeの扱いと残る限界

旧70 scope（unique30 endpoint）は従来の完了済み証拠のまま保全した。今回30 scope（unique10 endpoint）は部分結果であり、100 scope全体を完了・検証済みと主張しない。
scopeの候補集合は重複するため、100は独立100実験ではない。
今回のbaselineはStage Aの新34会計を含む。旧70の保存baseline完全再現と数値来歴が異なる。

共通プロトコルv2の旧70補足証拠、主表・Figure2・v2本文はそのまま保全し、本資料を新しいSTOP状態の参照としてのみ追加する。
H6のprofile差、H8/model runtime同一性未証明、S4合成仮説、初回572軸の最終gate欠測、旧profile照合待ち48軸等は未解決である。
保存n32の点費用をPR母平均優位・未観測希少event・モデル固有誤差の保証へ変えない。追加signal・sampling・compileは今回の保存処理問題の修正に不要である。

## 最小の次作業と必要な承認

停止したSOURCE/契約/部分結果/grant/ledger/marker/STOPはそのまま残す。
保存loopを、既存SOURCEがあれば一致を検査し、keyが重複しないmapping構築へ修正する案を推奨する。修正は封印済みSOURCEへ適用していない。
全出力種別（SOURCEを既に持つsummaryを含む）を保存する人工テストを加え、数値条件・演算を変えないことを検査する必要がある。

次担当**ユーザー**：保存処理修正と、同一30 scopeを新SOURCE/新packet/新契約/新grantで1回再実行することを明示承認する。旧70再解析・Stage A再計算・新compile等は追加不要。
次担当**Codex**：その承認後に修正・人工/回帰検査・seal・同一範囲one-shot・CSV/図・数値/終了監査を実施する。現grantで再開・retryしない。
公開範囲も別判断であり、新規pushは行っていない。

**mandatory STOP。詳細科学レビュー・最終evidence freezeは未開始。**
