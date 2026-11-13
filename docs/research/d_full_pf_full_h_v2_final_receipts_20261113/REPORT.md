# Full PF・Full-H RTEのB最新版V2への最終接続と保存価格17件のreceipt照合（Codex D、2026-11-13）

Dの物理request（`0536646b`）をB最新版`f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8`のV2 interfaceへ接続した。あわせて、Full PF保存compiled価格のうち`CONDITIONAL`だった17候補を元のreceiptまで追跡した。

**主な結論**
- **標準Yoshida4**：Bのcanonical ID`PF_STANDARD_YOSHIDA4_BINARY64_V1`と、Dの旧label`YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64`は同じ式である。どちらも同一bytesの`h46_m5_plan.weights('4th')`を使う。V2 IDはBが生成したものだけを使い、Dのnull/pending状態を解消した。
- **κなしFull-H RTE**：B kind`FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1`に接続した（`kappa=None`）。保存light studyのH4 q16/R16/K4/T0.8と同じ物理ID`FULL_H_8d6f23210529d68452dd`を、V2 candidateと対応付けた。
- **保存価格17件**：X/Yとも全件**EXACT**になった。12件は`result_sha256`から元のpf_common native結果へ、5件はpilot proofから元のM5 pilot結果へ到達した。
  - 保存価格coverageはEXACT 73（56＋17）、欠測63。CONDITIONAL・MISMATCH・SOURCE_UNAVAILABLEは0。
  - 既存のEXACT 56件と欠測63件は変わっていない。

作業は技術接続とmetadata照合だけである。保存値はcopyのみで、新しいsignal、reference、N/RMSE、sampling、build/compile/transpile、RZ再計数、fit、Gはいずれも0件。8次PFは接続していない。Full PFの公式・幅・M・比較menuは採択していない。

成果：
- adapter：[full_baseline_v2_requests](../../../src/trottertracks/full_baseline_v2_requests.py)／[test](../../../tests/check_full_baseline_v2_requests.py)
- receipt照合：[full_pf_price_receipts](../../../src/trottertracks/full_pf_price_receipts.py)／[CLI](../../../scripts/trace_full_pf_price_receipts.py)／[test](../../../tests/check_full_pf_price_receipts.py)
- 結果：[RECEIPTS.json](RECEIPTS.json)、[COVERAGE_V2.json](COVERAGE_V2.json)・[COVERAGE_V2.csv](COVERAGE_V2.csv)、[V2_SAVED_JOIN_EXAMPLE.json](V2_SAVED_JOIN_EXAMPLE.json)、[REQUEST_EXAMPLE_ARTIFICIAL.json](REQUEST_EXAMPLE_ARTIFICIAL.json)、[SOURCE_CATALOG.json](SOURCE_CATALOG.json)
- [HANDOFF_A](HANDOFF_A.md)／[HANDOFF_C](HANDOFF_C.md)／[TESTS](TESTS.txt)

## 1. B最新版への追従方法

- **B固定commitのmerge**：D branchに`f0c4bb55`をmergeした。B branch・worktreeは変更していない。
- **B moduleはbyte一致**：controller、pipeline、C/A関連moduleはBのfileそのままである。
- **D adapterの統合**：Bは`eecaa803`版のD adapterを自分のtreeへ取り込み、拡張していた。今回、D側の物理schedule層（`0536646b`）とBのAPIを1つのmoduleにまとめた。
  - Bから取り込んだAPI：`generate`、`requests(config)`、`validate_requests`、`layout_mapper`。
  - Bから取り込んだrank検査：`check_full_h_preparation`。
- **V2 IDの一致確認**：同じ入力から生成されるV2 configurationは、BのD module（`f0c4bb55`版）が生成するものと完全に一致する（testで確認）。
  - 対象：2nd、標準4th、m5_best、κなしFull-H。
- **Bのtestも通過**：B公開のtest（`check_abc_d_v2_integration` 12件、B版`check_full_baseline_v2_requests` 14件）は、統合後のmoduleで全件PASSした。

## 2. 標準Yoshida4（D-1）

| 項目 | 照合結果 |
|---|---|
| 係数 | `weights('4th')`＝(a, −u/(2−u), a)、u=2^{1/3}。binary64値は1.3512071919596578、−1.7024143839193153、1.3512071919596578（hex `0x1.59e8b6eb96339p+0`、`-0x1.b3d16dd72c671p+0`） |
| 順序・負時間 | piece順はa→負の中央→a。piece時間はT·w_j/q。各pieceはone-body＋全fragmentのS2 sweep（往復）で、scalarはTで一回。`h46_long_time_signal.evolve_batch`（`core.weights`）と`h46_long_time_native.compose`（`core.schedule`）が同じ順序で使う |
| SOURCE | `h46_m5_plan.py`のblob `0f7bb708`（sha256 `2ad8cfa8…`）は、次のすべてで同一：D基点`af284d85`、B`f0c4bb55`、保存carrier`155529e2`、pf_common SOURCE（`e2179f68`・`b7662487`・`bdf6772a`）、pilot SOURCE`c7d53f04` |
| B canonical ID | `pf_pr_rpe_pipeline.structure`（`f0c4bb55`）が、同じmoduleの`weights('4th')`に`PF_STANDARD_YOSHIDA4_BINARY64_V1`を付ける。m5_best（`PF_M5_BEST_20D64C2_BINARY64_V1`）とは別ID・別configurationである |
| 全roundの照合 | `_check_against_b`で全roundを照合した。次の不一致はいずれもerrorになる：<br>T、qPF、実PF幅、formula、formula_id、signed weights、scalar方針、H/state/snapshot binding、prefix＝全rank、retained＝FULL_H、K/Rなし |
| 旧D label | `physical_formula_label`として来歴だけに残す。V2では使わない |
| 物理candidate ID | `cid`（formula label`'4th'`）のまま、変更なし |

- pending状態`PENDING_B_FORMULA_INTERFACE`は、`resolved_pending`に記録して解消した。記録内容は解消条件と解消commit（`f0c4bb55`）である。
- 8次PFは`OUT_OF_SCOPE_NO_DF_CONTROLLED_FULL_WRAPPER_PATH`のままにした。D・BともV2を生成しない（testで確認）。

## 3. κなし明示Full-H RTE（D-2）

- `full_h_rte_spec(..., kappa=None)`はB kind`FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1`を使う。
  - round mappingの形はBの`layout_mapper`と同一にした。q、R、event_count＝qR、allocation[R]、event τ＝λ(T/q)/R、K、scalar一回、決定論block 0、`priced_equivalence_claim=false`を含む。
  - one-body randomized、全fragment、全H λ、lawの検査は、BのgenerateがDのvalidatorで再実行する。B3は拒否される。
- κを供給した場合は、従来kind`D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1`（κ付き）になる。
  - κなしとは別のconfiguration IDになり、Bのselectでも別group（`EXPLICIT_LAYOUT`／`COMMON_KAPPA`）として扱われる。
  - Dはκを生成・逆算しない。
- 保存light studyの例：h=0.1・m3（T=0.8）に{q16, R16}、K=4を与えた。
  - 物理IDは`FULL_H_8d6f23210529d68452dd`で、保存`selection.json`の候補（H4 q16/R16/K4/T0.8）と一致した。
  - V2 roundは`GENERATED`になった。layoutを与えないroundは`NOT_EVALUATED`のまま残り、Bのassessは`UNRESOLVED_PHYSICAL_LAYOUT`を返す。
  - 同じevent数256でもq1/R256は別の物理ID・別のV2 IDになる。
- H6の保存light候補（q32/R32/K6）も、物理ID`FULL_H_78f4b605c921e6d3f491`で一致した（[V2_SAVED_JOIN_EXAMPLE](V2_SAVED_JOIN_EXAMPLE.json)）。
- これは新しいFull-H比較menuの採択ではない。

**A/C request**：Dの`requests()`は、bind packetとB V2 configurationのどちらも受け付ける。configurationは再生成して、IDの一致を確認してから使う。
- A requestとX/Y各C requestは、次の項目がすべて同一である：
  - `configuration_id`、`v2_candidate_id`、`identity`、`law_binding`
  - `physical_candidate_id`、`physical_schedule_mapping`、`binding`、`b_commit`
- `validate_requests`は、物理IDやidentityの付け替えを拒否する。
- N/C値はnullで、`price_status=NOT_EVALUATED`である（[REQUEST_EXAMPLE_ARTIFICIAL](REQUEST_EXAMPLE_ARTIFICIAL.json)）。

## 4. 保存価格17件のreceipt照合（D-3）

| 区分 | 件数 | 追跡先 | X/Y |
|---|---:|---|---|
| H4 `NEW_COMMON_NATIVE`（`result_sha256`のみ） | 6 | pf_common H4 `native_results.json.gz`（`0b1886f5`）。native SOURCE `e2179f68` | EXACT 6 |
| H6 `AUDITED_COMPLETE_PAIR_FROM_INTERRUPTED_STAGE` | 5 | pf_common H6 `native_results.json.gz`（`90f1cd55`）。SOURCE `b7662487`（`end_audit`のinherited_SOURCE） | EXACT 5 |
| H6 `NEW_RECOVERY_NATIVE` | 1 | 同上。SOURCE `bdf6772a` | EXACT 1 |
| pilot axis receipt（q=2） | 5 | M5 pilot run01の`result.json`（`3eca4614:…/track_a_h46_m5_tol_pilot_run01/2026-10-19/`、SOURCE `c7d53f04`）。H4 2nd/4th/m5_best、H6 2nd/4th | EXACT 5 |

H4 m5_best q2には、pilotとは別に`m5_dataset`のcandidate receiptもある。両者のfingerprintは一致した。前回のinventoryでは、pilotのaxis記録がこのcandidate receiptを上書きしたため、CONDITIONALになっていた。今回はproofごとに別々に判定している。

**照合の連鎖**（すべてpinned Gitの読取。軸ごとに判定）：
1. **carrier**：長時間研究のreused価格record（`155529e2`）は、`reuse_provenance`が示すpf_commonの`PF_endpoint_costs.json.gz`の同一候補の価格と完全に一致する。
   - 価格が付いた候補はH4 161、H6 245。endpoint間の不一致は0。
2. **native結果**：`result_sha256`は、保存result.json（key順sort、indent 2、末尾改行）のSHA-256と一致する。
   - pf_commonの342 recordすべてで、`native_result_inventory`のSHAがこの形式で再現した。
   - recordの内容：candidate（formula/method FULL/prefix 10・19/q/system）、H・state hash、`native_policy`、status PASS、8 gate check、軸ごとのcompiler・fingerprint・RZ、archiveの復元PASS、prep hash。
3. **pilot**：pilot group IDを、元の`prepare_pf_common_domain.pilot_group`とpinned digestで再計算した。
   - group IDはsystem/FULL/formula/prefix/q=2/seriesのdigestで、これがproof名と一致する。
   - result.jsonのSHAは、pilot `inventory.json`の記録と一致した。task、PASS、native policy、binding（T 0.8、H/state/snapshot、compiler、control、profile）も一致した。
   - 軸ごとのfingerprint・metricsはproofと一致し、RZは保存値と一致した。
4. **SOURCE本文**（text照合のみ、import・buildなし）：
   - pf_common/pilotのcost関数は、`build_native_hadamard_wrapper(..., include_measurement=True)`と`CompilerSettings(rz/sx/x/cx, opt1, seed17, 1.3.0)`を使う。
   - `h46_m5_native.compose`のASTは、pilot・pf_commonの全SOURCEで同一である。
   - 決定論loweringの呼出し`build_deterministic_native(..., formula='2nd', scalar=0., control_policy='symmetric_directional')`は、長時間研究の`compose`と同一である。
   - pf_commonの`source_input_identity`に記録されたSHAは、次のSOURCEと一致する：`pf_common_run_native.py`、`h46_m5_native.py`、`h46_m5_plan.py`。
5. **binding・profile**：pf_commonのbindingとpilotのbindingは、長時間研究のbindingと13項目すべて一致した。
   - 項目：T=0.8、basis、geometry、H、state、snapshot、rank、rank_policy、series、compiler、control、profile hash、system。
   - cost policy（full wrapper、測定込み、P0）は長時間研究と同じである。
6. **補助確認**（private、読取のみ）：長時間研究のFULL trialのprep hashは、pf_common FULLのprep hashと一致した（H4 `bb736dc3…`、H6 `1de5325d…`）。

**更新後の保存価格coverage**（[COVERAGE_V2](COVERAGE_V2.csv)、136候補）

| 区分 | 件数 |
|---|---:|
| X/Y EXACT | 73（既存56＋今回17） |
| X/Y CONDITIONAL・MISMATCH・SOURCE_UNAVAILABLE | 0 |
| 欠測 | 63 |

- 既存EXACTのうち`m5_dataset` receipt由来の7件も同じ方法で追跡し、矛盾はなかった。
- 前回の[COVERAGE.json](../d_full_pf_saved_evidence_coverage_20261113/COVERAGE.json)（`92ff17de`）は上書きしていない。
- COVERAGE_V2の各行には次の項目を付けた：
  - `previous_status`
  - `receipt_trace`（追跡先・native SOURCE・prep hash・軸ごとの判定）
  - `physical_formula_label`
- 4thの`formula_id`はB canonicalへ更新した。旧labelは`physical_formula_label`に残している。

## 5. B V2と保存coverageの結合（D-4）

- `for_packet()`は、3公式ともBのV2 ID・formula ID・B commitを物理IDの横にcopyする。物理identityとV2 identityは別のfieldに保持する。
- 人工的な結合例：H4/H6、h=0.1、δ=0.1、M=4（[V2_SAVED_JOIN_EXAMPLE](V2_SAVED_JOIN_EXAMPLE.json)）。
  - m3（T0.8 q8）は3公式ともsignal・X/Y価格EXACT。
  - m4（T1.6 q16）は、2ndが価格EXACT（H4・H6）、4thとm5_bestは価格欠測（signalあり）。
  - m0–m2は保存gridの外（`OUTSIDE_SAVED_GRID_ALL_MISSING`）。
  - 4thは今回新たにV2対応が付いた候補で、`newly_mapped_by_B_formula_hook=true`。各行に物理carrierとV2の出典を付けた。
- 136候補を比較menuへ自動登録していない。

## 6. 未解決・科学的資格

- **未採択**：Full PFの公式・幅・M・比較menu、Full PF/Full-H RTEのRMSE誤差契約。保存endpointのPASSは旧契約の結果である。
- **reference・signal**：独立reference pairはT=2.0だけ。H6 fixed-delta由来の2件はraw signalがない。
- **価格**：欠測63件と、保存grid外のroundは新規取得が必要（compileはCの担当）。
- **Full-H RTE**：finite-K signal、normalization、十分N、全law価格、cohort期待値は未取得。
- **Bのnative physical hook**：Bにはまだない（`physical_candidate_hook=false`）。物理対応はDの外部request表で保持している。
- **EXACTの意味**：今回のEXACTは、保存価格が同じ物理候補・回路構成・compilerのものだと技術的に確定したことを意味する。RMSE契約やbaseline採択の資格ではない。

## 7. 技術テスト

[TESTS](TESTS.txt)：**152 PASS**。
- **今回のD関連**：37件。
  - adapter 17件：4th canonical ID、m5との区別、係数・負時間・順序、B公開moduleとのV2 ID一致、κなしlayout、q×R、同event数・異layout、B3、A/C identity、8次非接続など。
  - receipt 8件：17件の候補照合、X/Yの独立判定、不当昇格の拒否、binding・SOURCE、既存行の不変、MODEL/DIRECT、V2結合、非起動・bytes保全。
  - inventory 12件。
- **B公開test**：93件。結合12、B版D 14、controller 17、4方式14、統合12、C provider 12、evidence 12。evidenceはBと同じく8次検証を含む1 methodを除外した。
- **関連回帰**：22件（H6 provider 10、pipeline 12）。
- **mutation確認**：主要guardを無効化すると、testが失敗することを確認した。対象はreceiptのfingerprint、result SHA、pilot group、Bとのformula_id照合。
- これらは作者自身の技術確認であり、独立reviewや科学的保証ではない。

## 8. 保全・STOP

- carrier、pf_common、pilot、private context・trial、B/A/Cのbranch・worktree・processは変更していない。private contextのbytesは作業の前後で不変。
- 既存のSTOP・grant・ledger・markerは変更していない。
- 次は**mandatory STOP**。commit/pushとremote照合の後、新しい科学計算やcompileには進まない。
