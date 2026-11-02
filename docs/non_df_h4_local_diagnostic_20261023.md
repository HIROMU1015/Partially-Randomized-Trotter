# H4 DF-native局所数値診断L：one-shot結果・終了監査

2026-10-23。専用branch `codex/non-df-h4-local-diagnostic-20261023`、基点 `80147890fa91ad8270ce7643ab671727b3db5de4`。**封印済みL SOURCE `eb195bf2dedd0aed9d80ad7f84948b59ea53d854`を変更せず一回実行した。128 bitで148 forward・148 inverse・11 basis・324 supportの全結果を保存し、局所保証の範囲で検証PASS。** C/L結合は`COMBINATION_BLOCKED`、旧compiler全体は`MISSING_EVIDENCE`、正式certificate受理0・N/G再会計0のまま、mandatory STOPした。

## 固定対象と来歴

Linear H4、隣接1.00 Å、STO-3G、8 system qubit、Nα=Nβ=2、DF tol-only1e-8/rank10、PR prefix1＋one-body、S2/T=.8/q8/r4/K4の既存保存モデルを使用した。今回はDF-nativeの局所gate・basis・event familyだけを扱う。Pauli-JW/Cの40軸やH2 S2/Yoshida4系列の新計算はない。既存5精度ε=.05/.01/.005/.001/.0005と両軸に対する局所operator boundの再利用scopeを保持し、40軸の新総誤差は作成していない。

C科学SOURCEは`e80d7510880f8db993b777c627bb8e95b05a070d`、元H4 SOURCEは`0c329b30a81accb35c212be8defa66fee0dd0f12`。GitHubから基点commit、新L SOURCE commit、必須資料と実装9 fileを取得し、Git blob SHAと全bytesをローカルへ照合した。[取得・照合](../artifacts/non_df_h4_local_diagnostic/2026-10-23/github_verification.json)。前回の認証失敗記録は履歴として保持し、今回GitHub上で基点が取得できた事実を別記録にした。

最新版共通運用指示2026-10-23 §8.4/§10を適用した。[受領SHA](../artifacts/non_df_h4_local_diagnostic/2026-10-23/instruction_receipt.json)。前段階で検証済みのL kernel/controller/CLI、CのAcb・SHA・非返却ledger・owned native witness、T67/binding33/L48テストをそのまま再利用した。**科学実装・入力・runtimeの変更も再封印も0**。追加したpreflight・保存証拠auditorは実行管理・事後検証用で、Lの数値workerから呼ばれない。

## 承認・契約・one-shot制御

[ユーザー実行指示の保存](../artifacts/non_df_h4_local_diagnostic/2026-10-23/USER_AUTHORIZATION.md)、[承認来歴](../artifacts/non_df_h4_local_diagnostic/2026-10-23/authorization_receipt.json)、[今回の契約](../artifacts/non_df_h4_local_diagnostic/2026-10-23/L_execution_contract.json)。前段階DRAFTは変更せず、実行flag・明示承認のpath/SHA・statusだけを新契約で有効化した。科学条件・固定予算・128 bitは同一。

Code13・runtime1,050・input5のSHAとPython実体、SOURCE commit bytesを起動前と終了後に照合した。入力model SHAは`cc60f6dc3de885974cfe74ef822e49380edb6b9ca44777bf12e9654256cd322a`、registry SHAは`aa7c95451d3f46d97cb29ce79586c8e774b27e410abe15906f1f76004e005936`。sealed runtimeは既存private python-flint/Arb/Acb環境を読取利用し、共有package等は変更していない。[事前検査](../artifacts/non_df_h4_local_diagnostic/2026-10-23/preflight.json)。

Fresh namespace `artifacts/non_df_h4_local_diagnostic/2026-10-23/run01`にL科学grantを新規1件だけ発行し、一回消費した。[grant発行記録](../artifacts/non_df_h4_local_diagnostic/2026-10-23/grant_issuance.json)、[grant](../artifacts/non_df_h4_local_diagnostic/2026-10-23/run01/grant.json)、[消費](../artifacts/non_df_h4_local_diagnostic/2026-10-23/run01/CONSUMED.json)。過去grantの再利用・再発行、科学retry、精度昇格は0。

| 科学call分類 | 最大予算 | 予約 | 保存結果 |
|---|---:|---:|---:|
| Forward局所gate包含 | 148 | 148 | 148 |
| Inverse局所gate包含 | 148 | 148 | 148 |
| Ordered basis合成・一様伝播 | 11 | 11 | 11 |
| Symbolic component support | 324 | 324 | 324 |
| 合計 | 631 | 631 | 631 |

[Call ledger](../artifacts/non_df_h4_local_diagnostic/2026-10-23/run01/call_ledger.json)の631予約を、worker PID・分類・順序・631結果へ突合した。SOURCEの`authorize -> budget.take -> evaluate -> emit`とfsync保存によりcall前の非返却予約を確認した。SCF/FCI/DF/state/reference/law/bias/sampling/RNG/compile/N_Gは全0。

## 128 bitで得られた局所保証

[機械可読要約](../artifacts/non_df_h4_local_diagnostic/2026-10-23/result_summary.json)、[局所gate296行](../artifacts/non_df_h4_local_diagnostic/2026-10-23/audit02/local_gate_bounds.csv)、[basis11行](../artifacts/non_df_h4_local_diagnostic/2026-10-23/audit02/basis_bounds.csv)、[support324行](../artifacts/non_df_h4_local_diagnostic/2026-10-23/audit02/component_support.csv)、[一様伝播](../artifacts/non_df_h4_local_diagnostic/2026-10-23/audit02/uniform_basis_transfer.csv)。CSVにはdescriptor・SHA・exact上端・精度・結果SHAを接続した。

| 保存された上界の最大値・scope | 表示用の外向き上端 |
|---|---:|
| Local forward Frobenius/spectral norm上界の最大 | 8.50512327712e-17 |
| Local inverse Frobenius/spectral norm上界の最大 | 8.50512327712e-17 |
| Ordered basis conjugation、control0/1の最大上界 | 2.09409930399e-15 |
| DF-native PF：最大176 conjugationの局所basis伝播上界 | 3.68561477502e-13 |
| DF-native PR：最大192 conjugationの局所basis伝播上界 | 4.02067066365e-13 |

表は丸めを上向きにした表示値で、exact rationalはJSON/CSVを正本とする。局所誤差normの区間は[0, exact upper]であり、上界の値を実際の誤差そのものと解釈しない。296局所結果のうちexact upper=0は2件。未証明・未対応・nonfiniteの局所結果は0件。

保存binary64パラメータをexactに取り込み、RZとXXPlusYYの理想解析gateとの差をAcbで包含した。理想gateのunitarityを、丸め保存行列そのもののunitarityとは扱っていない。Inverseではθ反転・β保持、記録順のforwardと逆順inverse、global/relative phase・control0/1を維持する。局所Frobenius norm上界、外向きtelescoping、f+i+fiの両branch上界、一様伝播の保存exact rational不等式を事後確認した。この確認は保存値の検証であり、局所gate/basisの数値診断をもう一度実行していない。

324 supportは全registryのcomponent ID・basis・sign・qubit support・signed Z/ZZ/Iを対象とする。PR order0/2/4、r4/q8、deterministic S2との組合せを一様boundに接続した。有限の過去MC観測eventを全coverageの代用にせず、DF fragment λA²をPauli involutionに置換していない。

## 保証の限界とC/L ledger

[今回のcoverage ledger](../artifacts/non_df_h4_local_diagnostic/2026-10-23/coverage_ledger.json)は`basis_reuse`の重複を残し、**COMBINATION_BLOCKED**を維持する。Lの独立上界を保存したが、既存C totalから引いたり、C totalへ単純加算したりしていない。Cのbias_distance/semantic allowance/basis reuse/phase accumulation/pi roundingと40軸結果は不変。

Lの保証は、同じexact real angle/signを持つ理想局所gateと保存encoded matrixの差、およびそのbasis/operator伝播まで。実際のdiagonal角度生成、pi/phase蓄積、controlled分解、literal wrapper、compiler変換、physical preparation、B_savedスケールの転送は完了していない。Bを乗じた新measurement boundや40軸の総誤差は作成していない。

旧postcompiled programとpass traceの欠落は今回も未解決で、旧compiler全体保証は`MISSING_EVIDENCE`。L成功はこの欠落を補わず、正式certificate受理・shot十分条件・N/G確定を意味しない。旧20精度行のnull/UNDETERMINEDは維持する。

後続案は、C/Lのprimitiveごとの被覆を証明して二重計上のないjoint enclosureまたは別会計を設計し、diagonal/control/phase/B転送と旧compilerの証拠を別scopeで補うこと。今回は設計変更・追加数値評価・C totalの分解や相殺を行っていない。

## 人工・数値・構造・native終了の検証

[回帰テスト](../artifacts/non_df_h4_local_diagnostic/2026-10-23/regression_tests.json)はT67＋binding33＋L48＝148 PASS、FAIL/ERROR/skip0。人工gate・interval・phase/control・非可換積・拒否/partial/STOP/native異常系を流用し、実H4の科学callを追加していない。主agent自身の検査であり、別担当者の独立科学レビューではない。

[保存結果検証](../artifacts/non_df_h4_local_diagnostic/2026-10-23/verification.json)では、descriptor/registry一致、全結果の欠落/重複なし、finiteな非負dyadic上端、基底順序、両branch、不等式、631予約、SOURCE/runtime/input、native回収、STOP後authority拒否を確認した。

初回の**事後auditor**は、file mtimeと予約UTCを厳密順序として比較する追加assertionで停止した。631件中108件に最大約0.88 msのmtime逆転が観測された。file mtimeだけで実行順序を証明する根拠がないため、その判定を使用せず、封印済みSOURCEの因果順序とdurable ledger/result対応を根拠にした。失敗auditorのsource・4 CSV・観測記録を[audit01](../artifacts/non_df_h4_local_diagnostic/2026-10-23/audit01/failure.json)へ保全した。修正したのは別の保存記録auditorだけで、科学SOURCE・grant・ledger・raw・STOPを変更せず、科学再実行0。厳密なfile時刻順序を回収できたとは主張しない。

Launcher wallは約4.45秒。Launcher exit0、witness exit0/reaped、worker native exit0/reaped、owned descendants終了介入0、全owned回収を保存した。[launcher](../artifacts/non_df_h4_local_diagnostic/2026-10-23/launcher_exit.json)、[native worker](../artifacts/non_df_h4_local_diagnostic/2026-10-23/run01/native/worker_native_exit.json)、[native STOP](../artifacts/non_df_h4_local_diagnostic/2026-10-23/run01/native/STOP.json)、[run STOP](../artifacts/non_df_h4_local_diagnostic/2026-10-23/run01/STOP.json)。このreceiptは新Lだけに有効で、元H4 native/監視消失原因UNKNOWNを変更しない。STOP後はlauncher/worker用authorityがどちらも`L_ALREADY_STOPPED`を返すことを読取専用で確認し、再launchは試していない。

## 保全・資源・公開・引継ぎ

[保全監査](../artifacts/non_df_h4_local_diagnostic/2026-10-23/preservation_after.json)で既知の旧2,486 file SHAを照合し、SOURCE/raw/grant/ledger/marker/STOP、C40軸、K調査を保全した。新run649 fileの[byte inventory](../artifacts/non_df_h4_local_diagnostic/2026-10-23/raw_result_inventory.json)も保存した。これは技術的保全確認であり、最終scientific evidence freezeではない。

実行前に64 CPU affinity、空きRAM約486GiB・filesystem約377GiB、CPU/memory/I/O PSI avg10=0を観測した。重複除去済み148 hashと2×2/4×4 kernel・scalar合成を流用し、短い処理を危険なthread共有や不要なprocess転送へ変更しなかった。CPU/RAM/workerの人工上限、共有環境・他ユーザーへの変更は0。性能一般化は主張しない。

専用commitへ保存し、共通AGENTSによる公開承認に従い新branchだけpushを試みる。公開成否・remote SHAは[publication_status.json](../artifacts/non_df_h4_local_diagnostic/2026-10-23/publication_status.json)へ別記録とする。

**mandatory STOP。** 実行契約内のtrue flagは一回実行時の来歴として保存し、現在の有効authorityはCONSUMED/STOPによりexecution_authorized/grant_valid/dispatch_enabled=false。正式受理A・N/G・新compile・新系列は未認可のまま。

次の担当は**ユーザー**：C/L重複解消とliteral/compiler保証をどの範囲で進めるか判断する。**Codex**：明示承認された後続scopeに限って技術設計・検証を進める。追加数値診断・compile・正式受理A・N/Gには、それぞれ必要な明示承認を受ける。今回のgrant/namespaceの再使用・自動続行はしない。
