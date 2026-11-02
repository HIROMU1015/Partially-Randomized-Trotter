# H2 runtime差：保存費用の再利用調査・技術準備

**元PRと同じPython 3.12 binaryで取得した共通Full S2/q4・Yoshida4/q2の保存費用が存在する。PR134軸＋Full4軸の保存cost objectを再利用する政策を先に検討でき、新規compileを要求する根拠は現時点ではない。ただし現行の代替契約は未採択、原Python3.12対採択Python3.11の出力同等性はUNRESOLVED、COMPAREはBLOCKEDのまま。**

2026-10-23 JST、GPUA100-gleap02、Codex主担当のみ。基準 `65085ff7f818fd92af3424b1980f6bd9dbf1c742`、専用branch `codex/h2-runtime-cost-reuse-20261023`。検査SOURCEは[SOURCE.json](SOURCE.json)。旧COMPARE SOURCE `46a3b33e3717c4fadf11846e922caf17d10156d9` と契約digest `a65f3e90b7ed70701f57bcf1519e1d74acdfd1c45c320a5fe91d00d4d44d9ef2`、旧A/B判定を変更していない。科学実行・政策採択・費用再解析は0。

対象はlinear H2、隣接1 Å、STO-3G、signed tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4。Fullはprefix3、PRはprefix1/K4/ABS_TIME_PER_OUTER32、無条件finite-RTE。S2/q4のδ=.2、Yoshida4/q2のδ=.4と元signed pieceを維持する。通常signed bias、経験的allowance、旧FP reference条件付き政策は別の比較条件として保持し、新しいu/N/G・比・順位は計算していない。

## 1. 最初に確認した保存費用

新compileコードを作る前に、3環境索引、旧H2 A/FP/D/P、outer32探索・別確認、共通PFの固定結果を読んだ。対象費用の所在は次の通り。RZは保存整数の転記で、再計測ではない。

| 保存系列・候補 | runtime証拠 | 保存範囲・単価 | 今回の利用判定 |
|---|---|---|---|
| 旧Full `B1_p3_q4_2nd` / `B1_p3_q2_4th` | 旧A/Pと同じ3.11系のruntime契約、D recovery SOURCE `cf99e445…` | 各cos/sin **1367 / 2033 RZ**、旧D result/measurement/control/終了監査 | 旧nativeの別cost object。共通PF単価への無断置換不可。対象compile workerにQPY保存は確認できない。compiled-vector strict gateの同等証拠も不足 |
| 元PR `H2O32_c7ec590a32a859bc` | 原O00002、Python3.12 binary `e50d468e…`、取得SOURCE `4ec26921…` | EXPLORATION33 trajectory/**66軸**。cost/raw/QPY/guard/CT01あり | 原候補・sample・cohortを固定して再利用候補。新平均なし |
| 元PR `H2O32_8a2ecdd0294ffb74` | 同上 | EXPLORATION34 trajectory/**68軸**。同じ種類の証拠あり | 同上。確認cohortとはpoolしない |
| 共通Full `H2_FULL_3_4_2nd` | 固定結果 `f36402f9…`、科学SOURCE `1b35f9ac…`、**同じPython3.12 binary** | 各cos/sin **1655 RZ**、wrapper/compiled/vector archive SHA、guard、saved-state correctness、exit/reap/end audit | 要求Full S2/prefix3/q4に対応する保存候補。旧B1とはID・取得SOURCE・nativeが異なる。別の保存cost objectとしてのみ利用候補 |
| 共通Full `H2_FULL_3_2_4th` | 同上 | 各cos/sin **2463 RZ**、同じ証拠 | 要求Full Yoshida4/prefix3/q2に対応。旧2033を2463へ書換えない |
| 旧A/FP/P PR | 3.11系、旧S2/law/cohort | 旧A n8、FP診断、独立P等 | 指定outer32の2候補ではない。同じPythonでも混入拒否 |
| 元PR-vs-PR確認、再構成PR-vs-PR確認 | 3環境索引が別SOURCE/cohortとして区別 | H2 n24/n28等の別固定比較 | 元33/34探索軌道の欠測費用を埋めるものではない |
| 新共通DIAGNOSTICS / 現COMPARE準備 | 採択3.11系 | DIAGNOSTICSのcompile0、COMPAREのFull compile4は将来予算 | 未実行予算を保存費用と扱わない。今回調査scopeで新3.11共通Full単価は未取得 |

[134原PR cost object](saved_PR_cost_objects_134.json)、[4共通PF cost object](saved_common_PF_cost_objects_4.json)、[旧系列一覧](historical_cost_inventory.json)。原PRは全402 archiveを既存streaming SHA関数で再照合。QPYを回路objectとしてloadせず、vectorを数値decodeしていない。[再照合](PR_archive_identity_recheck.json)。元のcompiled RZ計測はfull controlled Hadamard wrapperの`count_ops()['rz']`であり、理論RZ、T count、depth、RPE総費用へ換算しない。

共通PFの原文入口は[固定README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f36402f90bcfcbcc4d556c7a952837dc5d7dd291/artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/README.md)。`PF_endpoint_costs.csv`の2行と`native_results.json.gz`の2対象receiptの整数が一致する。既存result/archive/hashを使う経路であり、この費用を再計算していない。

## 2. 同じruntimeと呼べる範囲、残る証拠

[runtime封印照合](runtime_seal_comparison.json)で、共通PFの公開projectionから元PR原sealへrole/SHA/byte数を結合した。

- fixed Python binary **1/1**一致。単に3.12というversion名が同じという判断ではない。
- `loaded_runtime_identity` **8/8**、compiler/array binary **21/21**一致。NumPy cp312の19 binary、Qiskit Rust acceleration、SymEngine cp312を含む。対応30件の現在原fileも再hash一致。
- provider **235/236**一致。差分1は`h48_preflight_contract_v1.py`の版差で、科学的出力不一致の証拠ではない。読込・検査役割を保存し、全provider一致とも扱わない。
- scalar guard、`h46_m5_native.py`、`h46_new2_native.py`のSOURCE SHAも一致。関連compiler/correctness関数は[再利用索引](compiler_reuse_plan.json)に固定commit/line/file SHA/AST digestを保存。

公開PF projectionは全Qiskit/NumPy/Python stdlib/native libraryのpath対応・完全load manifestを含まない。SHA集合の一致はpath/origin結合の代わりではなく、現fileの再hashは過去worker実loadの証明でもない。**「同じ宣言compiler/runtimeと一致する主要binaryを持つ保存費用が揃う」と「完全runtime identityの成立」を分ける。**

科学条件はHamiltonian/state hash、1 Å/STO-3G/DF3政策、T、prefix/q/formula、control/scalar/wrapper、compiler設定、測定単位を照合。Full/PRのprefixやlawの方式差は正当な比較自由度だが、別candidate/cohortの費用は移植しない。[比較条件表](comparison_conditions.json)。

共通Fullのpreparation hash `98c1beb1…` は、採択3.11共通DIAGNOSTICSのprefix3 preparation hashと**完全一致**し、native preparation SOURCE bytesも一致した。共通PFは再生成NPZの科学配列一致receiptを持つが、旧NPZ容器SHAの復元ではない。これらは強い再利用証拠だが、原PF QPYの全operation streamと現比較定義へのbinding、原planの関連runtime依存・pathは追加照合が残る。

共通PF原plan/QPY/archive rootはこのhostの既知locatorには存在しない。保存receiptと原raw SHAは固定Gitから取得済みであり、**原raw bodyをこのhostで再hashしたとはしない**。まず保持者からその原本または承認済みprojectionを取得する。科学条件を別回路へ置換したり、raw未取得をcompile欠測と同一視したりしない。[不足判定](same_runtime_comparability.json)。

現行Python3.12 PR対3.11 Fullという旧契約Bの条件は、比較対象を3.12保存PFへ変えるだけで自動的にVERIFIEDにはならない。別のcost-object政策・契約の正式採用が必要。Aの独立absolute diagonal witnessも別作業の未解決条件として保持する。

## 3. 3環境とruntime技術差

[取得receipt](remote_acquisition_receipts.json)、[横断対応](three_environment_crosswalk.json)、[全固定SOURCE一覧](fixed_source_inventory.json)。GPU02 `b8bd6433…`、GPU01 `e0c942fb…`、ユーザー指定ABE `f284c501bd8332fa5e8fc098b5424e34c666174f`と統合索引 `6dafed86cc82149d55b4e6395382953d5f3631d7`を使用した。ABEの旧H2 Pとouter32は別系列で、重複参照を追加標本にしない。所持hostから元実行hostを推定しない。ABE/統合8資料はGitHub本文とblobを照合、PF5資料はGitHub先頭行＋blob OIDを取得し固定local全文と結合。gzipのPF receiptは固定local Git blobを読んだ。全資料のremote全文取得とは記載しない。

[元runtime調査](../../track_a_h2_pr_equivalence_contract/2026-10-23/runtime_relevance.json)を継承し、今回runtimeを変更していない。

| 項目 | 意味と現在の限界 |
|---|---|
| Python3.12/3.11 | interpreter/extension ABI差。版差だけで科学的MISMATCHとしない。同じSOURCE・seedも出力一致証明ではない |
| NumPy ABI/build config | 数値preparation・synthesis経路に関係し得る。元PR対採択3.11では19 ABI binaryとbuild configが異なる。今回の3.12保存PFでは21対象binaryのSHAが一致 |
| SymEngine ABI | guardがParameter/exact_zero_scalarを作り、ParameterExpressionがSymEngineをimportする。numeric保存QPYだから無関係とはしない |
| Qiskit1.3.0・830 byte一致 | 元PR対採択runtimeの既存SOURCE証拠。今回のPF21 binary一致と区別する。全transitive依存や入力の出力感度まで証明しない |
| liblzma歴史seal差 | 過去sealの1差分を保全。調査済みcompiler ELFの直接NEEDEDにないことは非load証明ではない。原zlib archive/raw SHA一致とcounterfactual compileを分ける |
| historical actual-load欠測 | 新runtimeの結果でも過去workerのloadは復元できない。版差・封印差はmetadata差、科学出力影響はUNRESOLVED |

## 4. 政策1/2と必要科学call

[政策比較](runtime_policy_options_DRAFT.json)。採択はユーザー判断として保留し、どちらの政策でも元PR CT01は**single-saved-input-state strict <1e-7 / phase_alignment=false**の範囲を超えない。all-state/operator保証はない。

| 政策 | 使う費用と必要な証拠 | 今後の科学call |
|---|---|---|
| 1：元PR3.12＋新Full3.11 | 原PR134軸固定値。新Fullは現COMPAREの未実行4軸。取得runtime差、cost-object意味、correctness範囲、B出力同等性をどこまで要求するか明示採択 | 原COMPARE案のFull compile4/transpile4/simulate12等は別予算で未承認。runtime診断probe4軸とは別で、加算・実行しない |
| 2a：保存3.12同士を再利用 | **PR134＋共通PF4軸の費用は既に存在**。原PF plan/QPY/receipt/関連runtimeをmetadata/SHA/保存fieldで追加結合。現在の共通診断と科学条件を照合し、cost-object政策を別採択 | 次の費用側技術作業はcompile/transpile/simulation等**すべて0**。その後の科学比較やA不足解消まで0で済むという認定ではない |
| 2b：採択3.11に統一して新取得 | 元wrapper QPY、既存compiler/guard/counter/correctnessを再利用。不足する同一cost objectの軸集合を政策決定後に固定。保存結果と別SOURCE/cohort/測定来歴になる | 真の最小数は未確定。134再compileを無条件の必要条件としない。元134軸を全て新取得する政策ならPR134 compile＋268 before/after simulationがその集合の条件付き数であり、必要性の認定・予約ではない。Full4は別に再利用可否を判定 |

政策2aを優先する理由は、同じ宣言runtime・主要compiler binaryの保存費用が見つかったためである。旧Python3.12/3.11のcounterfactual出力一致を証明せずに研究上比較できる可能性がある。現在のstrict契約を緩めたり、保存PR費用を新runtime費用へ改称したりしない。

## 5. 4軸probeと最小SOURCE

[原4軸案の保全DRAFT](preserved_four_axis_probe_DRAFT.json)。S2 `H2E1__H2O32_c7ec590a32a859bc__0`、Y4 `H2E1__H2O32_8a2ecdd0294ffb74__4`各cos/sin。保存QPY4件、compile4/transpile4/simulate8、bootstrap1/saved-state load1、RZ equality4、setup/prepare/build/lowering/wrapper/sampling/u/N/G/COMPARE0という原契約を保持した。対象4件のSHA・整数保存費用・receipt・correctnessを入力manifestへ結合した。

**政策2aの保存費用利用にprobeは必須ではなく、今は実施を推奨しない。** 3.11の出力感度自体を研究上確かめたい場合だけ価値がある。成功が示すのは選んだ4軸の費用一致と当該saved-state correctnessに限る。全134軸・歴史load・all-state/operator・Aの独立diagonalは証明しない。

新SOURCEはmetadata join/receipt audit/fixture ledgerに限定した。既存`transpile_and_measure_cost`、`_gate_counts`、phase guard、`simulate_statevector`、原`gate`、QPY loadを再利用する計画を固定した。元`run_task`やPF `cost_one`を丸ごと実行するとnative再構築・sampling/追加simulationを伴うため、saved-QPY案の入口にしない。compiler・回路生成・費用測定の再実装はない。

guardは1 guarded transpileの内部で各BasisTranslator passの原`run`を2回呼ぶ。元134軸の保存pass数は各1だが、新実行のpass数をそこから無条件には固定できない。**内部pass ledgerを含むproduction adapter・runtime/input sealは未実装・未承認で、現DRAFTは実行準備完了ではない。** top-level compile4/transpile4を内部8 translator呼出しと混同せず、将来SOURCEの実callが変わるなら契約差分を明示して別承認する。未使用grant/root/cacheは計画名だけで、発行・作成していない。

## 6. 人工検証・保全・STOP

[人工テスト](test_results.json)は新34＋既存代替契約48＝**82件PASS**。新規テストはcandidate/task/axis/cohort/runtime/settings/SHA/保存整数RZの混入、型不正、scope拡張、4→134昇格、既存科学validatorの無断実行、exact call不足/超過とSTOP保持を検査。人工fixture/mockのみで、Codex主担当の技術自己確認である。独立科学レビューではない。

[metadata-only preflight](metadata_preflight.json)は実行不可・政策未採択・COMPARE BLOCKED。`--execute`は科学データを読む前に拒否する。[保全監査](preservation_audit.json)は**16,824ファイル前後一致、変更0**。実使用SOURCE/input/runtime/result/旧grant/ledger/marker/STOPと旧H2関連metadataだけを監査し、無関係worktreeの進行を凍結しない。共有venv/OS/scheduler/他ユーザーprocessの変更0、旧grant再利用0、新grant0。歴史的liblzma差を修復していない。

新科学計算0、実H2 compile/transpile/simulation/sampling0、u/N/G再解析0、COMPARE0。今回の指示に従い専用branchへローカルcommitのみ。private原config/seed/full event/raw QPY/NPZ/archive/内部絶対pathを新成果物へコピーしない。[mandatory STOP](STOP.json)。

次の作業（提案・未実施）：**ユーザー**が政策1か2を選択する。政策2なら、保存3.12費用を使う2aか、新3.11 cost objectを必要とする2bかを指定する。**共通PF原本の保持者**が対象2候補の原plan・QPY/archive・関連runtime/path receiptのlocatorまたは承認済みprojectionを提示する。**Codex**が先に科学call0の静的結合を行い、なお追加科学計算が必要と判明した場合だけ別SOURCE・対象・exact予算を具体化する。A作業、政策採択、科学実行承認、COMPARE実装/実行はそれぞれ別の段階である。
