# H2 DIAGNOSTICS run01：承認済みone-shot完了

**固定4候補のDIAGNOSTICSが正常終了。strict `<1e-9` の診断gateを全4候補で通過し、exact call予算・SOURCE/input/runtime・結果hash・元ledger/markerの照合がPASSした。比較・allowance/shot/費用評価へは進まずmandatory STOP。**

linear H2、隣接1 Å、STO-3G、tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4。Full prefix3、PR prefix1/K4/ABS_TIME_PER_OUTER32、通常signed bias、symmetric-directional、cos/sin full wrapperの固定条件を保持した。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend/coupling/layout/routingなし。ただし今回はnative/compiled correctnessとRZ計測を行っていない。

## 承認・SOURCE・実行単位

ユーザーは確認質問に **「はい、上記DIAGNOSTICSのみ承認する」** と明示回答し、共通QR native・経験的adaptive allowance・旧FP referenceの条件付き再利用、およびfresh Python3.11 runtimeを採択した。採択scopeはこの固定H2契約。経験的allowanceを厳密boundとして採択したものではなく、allowance評価そのものも今回0回。

- runner SOURCE：`35444654f9159a5d85debdbc577feab0590e2610`
- acquisition：`4ec26921c6872ed6e8667f167074e1e405cf1a3f`
- provider：`c31ef424175c02379eafebe8fc6c7548c1d9aabe`
- 実行前HEAD：`59d936785ccb766c25362fb5d886b751e4c0cd28`
- 契約digest：`f71e52ce6f42cc17d1a97f0cbdfac311da57c80b49ee8ea69ebe34fb4d45bfd9`
- completion SHA256：`cc7ebfee979b2e34eb4327aa8a8c6e20ea53bc46871f97c83d0391db5e7ec90f`

[明示承認記録](authorization_record.json)、[launch前照合](prelaunch_record.json)、[process終了](process_exit.json)、[元completion](completion.json)、[受入監査](acceptance_audit.json)を保存した。新grantを承認後に発行し、科学import前にCONSUMED markerを作成。実科学launcherの起動は1回、子process exit0・親processでwait/reap済み。wallは **44.256679195998004秒**（保存receiptの転記）。retry/resumeは0。

| kind | 承認予算 | 実消費 |
|---|---:|---:|
| bootstrap | 1 | 1 |
| setup | 1 | 1 |
| prepare | 2 | 2 |
| diagnose | 4 | 4 |
| 内部path_signal | 8 | 8 |
| reference_rebind | 4 | 4 |
| native_validation / compose / lowering / wrapper | 各0 | 各0 |
| compile / transpile / simulate | 各0 | 各0 |
| request_factory / rte_event / occupation_trajectory | 各0 | 各0 |
| allowance / accounting / ranking | 各0 | 各0 |

上位entryと内部callを併記しており、表を独立実験数として合計しない。新full-H reference・quantum shots・q探索も0。保存grantファイルは原byteのまま保全し、消費済みroot/markerを消費の正とする。旧grantを再利用していない。

## 実行済み診断の結果

以下は新診断が既に保存した値の転記。新しい差分計算・u/N/G・順位計算ではない。PRの列順はcorrected/rawで、rawは各pathのexp(log_B)でcorrected単位へ戻して比較する。Fullは1列/B=1。

| 候補 | path signal差（corrected/raw換算） | path vector差（同順） | corrected-signal closure |
|---|---|---|---|
| B1_p3_q4_2nd | 4.25086404278222e-15 | 5.0842372423983806e-15 | 0.0 |
| B1_p3_q2_4th | 5.578801654593729e-15 | 6.8603671893258265e-15 | 0.0 |
| H2O32_c7ec590a32a859bc | 1.6946818973100072e-15 / 7.850462293418876e-16 | 1.75480956440342e-15 / 9.888285250772094e-16 | 6.753223014464259e-16 |
| H2O32_8a2ecdd0294ffb74 | 6.753223014464259e-16 / 1.6653345369377348e-15 | 1.7612182362671013e-15 / 2.1788096812823775e-15 | 3.1401849173675503e-16 |

2 pathのlog_B差は4候補とも保存値0.0。log_BはFull2候補0、PR S2は0.00047735001373052684、PR Y4は0.018531064276367512。primary/independentは同じ保存H/state/signed schedule/setup表現を共有するため、これらは全誤差を独立に保証する証明ではない。

[diagnostic_projection.json](diagnostic_projection.json)に4候補のsignals、log_B、signed schedule、reference再結合、work/norm、prep/operations hashをコピーした。全stage traceとvector archiveの原byteは[results](results/)に保存。phase alignment/postcompile correctionはfalse、scalar_applicationsは各候補1。新referenceを計算せず、旧FP scalarと保存combined empirical diagnosticの条件付き解釈を維持した。reference厳密誤差が得られたという認定ではない。

## preparationと後続identityの範囲

prefix3 receiptはprep `98c1beb120c22077b409550a4d61c31638b9a52637246fae4cb1ac55a776f1d9`、operations `4a60ed34ef2e81a88cf5b1ab0672141970a331b509bc902cbe1b111fc233e5a0`、rotation/operation記録4組。prefix1はprep `85b8243d752d3390e214b429a80e3899f0cc8b0166e0958fc1d73a930a77597c`、operations `acb9c34e7a1a801133e68055bb2f7e028c77ed77d7eaf90ec887eaeb48ab5863`、記録6組。

新しい共通prep・rotation byte hash・operation streamのreceiptを得た。各候補のprep/operations hashを対応prefixのreceiptと結合した。ただし、保存PR67 tasksの完全event body、当時のoperation stream、実使用Python3.12 runtime、scalar/wrapper/lowering/compiler/correctnessの完全joinが揃ったことにはならない。**COMPARE_REUSE_PRはBLOCKEDのまま。**

現runnerはCOMPAREで診断も再実行するため、今回の正常終了結果を読み込んでpath8/ref4を省くには、別SOURCE・別契約・新grant/rootの専用分岐が必要。今回のmetadata受入interfaceはcompletion外部seal・結果/marker/元ledger・exact identityを検査してACCEPTED_METADATA_ONLYを返したが、比較を開始するloaderではない。消費済みDIAGNOSTICS grantは継ぎ足し・再使用しない。

## runtime・検証・保全

launch直前の科学importなしpreflightを再確認し、実実行では固定source/array binding/loaded module/executable mapping検査を含むSOURCE guardを完了した。実行後も契約内10,721 fileとGit SOURCE268 fileが一致。静的許可集合を使う契約の `loaded_closure_complete=false` は変更していない。今回の正常終了は当該guardを通過した証拠で、moduleごとの実load一覧のdumpを保存したという意味ではない。

新しい科学結果は10 file（2 prep・4 diagnostic・4 vector archive）とcompletion、ledgerを保存した。[保全監査](preservation_audit.json)では既知旧原本/完了run/control-planeの346照合、および前段準備artifact28 fileのhashが一致。旧SOURCE/raw/grant/marker/ledger/STOPを保持し、共有環境/venv/他worktreeは変更していない。専用未存在bytecode cacheも未作成のまま。

使用SOURCEに対する人工98件PASSは前段59d936785の証拠を継承し、科学実行後に同じ人工テストを繰り返していない。今回追加の検証は実DIAGNOSTICS一回と、終了後のmetadata/hash/ledger照合。Codex主担当のチェックであり、独立科学レビュー・詳細GPT/Claudeレビューではない。

## STOPと次の担当

[STOP.json](STOP.json)を新しい実行記録として保存。元科学runには正常completionがあり、失敗partialのSTOPはない。science_GO=falseは研究上の費用比較や最終採用のGOを与えていないという意味で、今回の明示実行承認・診断完了と区別する。

次は**ユーザー**がPR原exploration config O00002のcanonical locator、またはsender private token mapを提供すること。**Codex**は別作業として保存PR witnessの照合と、受入済み診断を読むCOMPARE専用分岐の技術準備を進める。比較科学実行は証拠・契約完成後の別途明示承認を要する。現在はmandatory STOP、push・PR・merge・追加科学計算なし。
