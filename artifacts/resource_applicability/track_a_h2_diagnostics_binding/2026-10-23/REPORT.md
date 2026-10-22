# H2 DIAGNOSTICS実行binding準備：完了報告

**DIAGNOSTICSの実SOURCE/input/runtime契約を技術封印し、科学importなしのpreflightがPASS。科学実行は未承認・未実行。COMPARE_REUSE_PRはBLOCKED。ローカルcommit後mandatory STOP。**

専用branch `codex/h2-diagnostics-binding-20261023`、基準d405b880a679ee4a95d233f5535c4e3bf2db53f4。runner SOURCEは **35444654f9159a5d85debdbc577feab0590e2610**。取得4ec26921/provider c31ef424、前段production59e5ddddの役割を維持する。push・PR・mergeを行わない。

## 今回完了した技術作業

- 役割別268 SOURCE fileを固定Git blobと全byte照合。旧snapshot/receiptをhash照合後に専用assetsへcopyし、固定H/state/bindingと結合した。NPZ decode・実preparationは0。
- fresh Python3.11/NumPy1.26.4/Qiskit1.3.0の静的import/ELF許可集合を封印。runtime10,450 file、契約全体10,721 file。隔離launcher、通常import/直接SourceFileLoader、loaded module/executable mapping検査を接続した。[詳細](RUNTIME_INPUT_SEAL.md)。
- 選択Python自身のstdlib ASTで7,560 sealed Python sourceをparseしerror0。[監査](python_syntax_audit.json)。科学moduleのimport成功や数値動作を試したものではない。
- [実契約](DIAGNOSTICS_contract.json)、[exact予算](call_budgets.json)、[preflight](preflight.json)を保存。canonical契約digest `f71e52ce6f42cc17d1a97f0cbdfac311da57c80b49ee8ea69ebe34fb4d45bfd9`。bootstrap1/setup1/prepare2/diagnose4/path_signal8/reference_rebind4を維持し、他kindは全0。
- 将来の診断にvector/basis archiveと完了bundle hash indexを保存する実装を追加。独立診断の外部completion seal・元ledger/marker・exact identityを検査するmetadata interfaceを人工検証した。比較を起動するloaderではない。
- [保存PR67 tasks/134 axes監査](PR_C_IDENTITY.md)と[後続比較の最小案](COMPARE_CONTINUATION.md)を完成。完全event/stream/runtime witnessが不足することを保持した。

## 科学政策の推奨と判断事項

[採択資料](POLICY_DECISIONS.md)の推奨は、共通adjacent QR signed-piece native、共通の経験的adaptive A-rule、旧FP scalar＋combined empirical diagnosticの条件付き再利用。ユーザーの正式採択はまだない。

旧Full/PRのallowance差と新instrumentationの単位を区別し、欠測reference誤差を0にしない。u/headroom≤.01は将来のeligibility guardであり、rigorous保証ではない。DIAGNOSTICSはnative/compiled correctnessや新u/N/G/順位を評価しない。政策・reference条件付きscope・今回のfresh3.11 runtime選択は次の明示承認事項。

## 実行可能性と残る条件

DIAGNOSTICSは**技術契約がsealed-ready**。ただし科学実行権限はなく、grant/科学run_root/CONSUMED/ledgerを作っていない。scientific_imports_observed=false/loaded_closure_complete=falseを保持し、静的許可集合を動的観測と偽っていない。承認後のbootstrapで集合外import/native mappingが要求されれば、非返金でSTOPする。成功する科学runを今回認定したものではない。

次にユーザーが承認するのは[実行契約と証拠要件](DIAGNOSTICS_EXECUTION_CONTRACT.md)の4候補・exact budgetだけ。3政策/計数/経験的uncertainty、fresh runtime、契約digest、新unused grant・新root・no-retry・mandatory STOPを含める。今回はこの科学承認を求めて自動待機/実行へ進まず、指示通り準備後STOPする。

比較にはPRのfull event/rotation/operation-stream、scalar/wrapper/lowering/compiler/runtime/correctnessの完全identityが不足。正確な不足locatorはO00002の原exploration configまたはsender token map。既存Git exporterまで回収してもcanonical private rootは得られなかった。推測探索で補わず[不足条件](blockers.json)に残す。

独立DIAGNOSTICS後に現行COMPAREを使うと診断4/path8/ref4を再実行する。重複を避けるには、受入bundleを封印して読む新SOURCE/別契約/新grant/新rootの専用分岐が必要。旧grantを再利用せず、PR C identity成立後にだけ別承認へ進む。

## 検証・保全・STOP

既存production36＋旧policy/mock36＋追加binding26＝**98件PASS**。[結果](test_results.json)、[全ログ](technical_tests.txt)。SOURCE/runtime/入力不一致、外部承認欠如、grant再使用、予算超過、missing/異単位/0補完、partial結果、無検査import/cache、native mapping混入、元ledger不足、completion/result改変を人工検証した。Codex主担当自身の技術チェックであり、独立科学レビューではない。

旧原本9件と旧完了runの保全manifest314件が当時SHAに一致。既知control-plane23 fileを最終再照合する。旧tracked SOURCE/raw/grant/ledger/marker/STOPの差分がないこと、専用科学run_root/cacheが存在しないことを[preservation_audit.json](preservation_audit.json)に保存する。共有環境・他ユーザー・別worktreeを変更せず、全worktreeの凍結も要求しない。

実H2科学import/simulation/signal/reference/compile/sampling、新u/N/G、科学比較/詳細GPT・Claudeレビューは全て0。liblzmaは選択3.11の旧hash一致を確認したが、別3.12 runtimeで報告された不一致の歴史的影響は未判定のまま保持した。

次の担当は**ユーザー**（政策と限定DIAGNOSTICSの明示承認、必要ならPR原configのcanonical locator提供）。承認後の**Codex**が別指示の範囲でone-shot DIAGNOSTICSのみ実行してSTOPし、その受入結果とPR witnessが揃ってから後続契約へ進む。科学レビューは別途指示まで開始しない。
