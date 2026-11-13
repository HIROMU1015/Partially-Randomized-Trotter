# Full PF保存成果のSOURCE照合とcompiled価格coverage（Codex D、2026-11-13）

**更新（2026-11-13）**：B f0c4bb55への最終接続と保存価格17件のreceipt照合は[d_full_pf_full_h_v2_final_receipts_20261113](../d_full_pf_full_h_v2_final_receipts_20261113/REPORT.md)。

H4 DF10・H6 DF19のFull PF（2nd・標準4th・m5_best）について、過去に保存されたsignal、reference、数値証拠、X/Yのcompiled価格を、SOURCE付きで照合した。

**主な結論**
- 保存長時間grid（T∈{0.8,1.2,1.6,2.0}）のFull PF 136候補すべてに、保存signalとreferenceの値がある。
- X/Y両軸のcompiled価格があるのは73候補。残る63候補は価格だけが欠測している。
- signal・reference・価格がすべて欠測するのは、保存gridの外（新しいRMSE scheduleの多くのround）である。

作業はmetadataの読取と照合だけである。保存値はcopyしただけで、再計算・集計・外挿・ランキングはしていない。新しい作用、reference、N/RMSE、build/compile、fit、Gはいずれも0件。

成果：
- [inventory tool](../../../src/trottertracks/full_pf_saved_inventory.py)／[CLI](../../../scripts/inventory_full_pf_saved_evidence.py)／[test](../../../tests/check_full_pf_saved_inventory.py)
- [INVENTORY.json](INVENTORY.json)（SOURCE・carrier・hash）／[COVERAGE.json](COVERAGE.json)・[COVERAGE.csv](COVERAGE.csv)（候補×signal/reference/数値/X/Y）
- [HANDOFF_A](HANDOFF_A.md)／[HANDOFF_C](HANDOFF_C.md)／[TESTS](TESTS.txt)

## 1. 確認したSOURCE

| 種類 | 実在を確認した場所 | 照合結果 |
|---|---|---|
| 長時間研究の公開carrier | `155529e20a8fac39e00786c40b57830b71203c4f:artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/`（signal_records、signal_endpoints、extension_signal_*、exploration/extension/confirmation/fixed_delta costs、reused_compiled_prices、candidate_inventory） | 8 fileが`completion_provenance.json`（B `af284d85`の同じdir）の記録SHA-256と**一致**。candidate_inventoryとreused_compiled_pricesはprovenanceに記録がない（同じcommitのfileとして参照） |
| 研究条件 | 同dirの`scientific_provenance.json`（compiler `1239304c…`、opt1、seed17、cost policy `FULL_COSINE_SINE_HADAMARD_WRAPPER_MEASUREMENT_INCLUDED_P0`、H4 DF10/H6 DF19のbinding）、`signal_provenance.json`（signal SOURCE `e417ffc9`） | bindingはD・A S0のV2 binding（H/state/snapshot/rank policy）と一致 |
| T=0.8の再利用元 | pf_common結果 `0b1886f5`（H4）・`90f1cd55`（H6）の`source_input_identity.json` | H/state/snapshotとtol-onlyが一致 |
| reference context | private `/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/{H4,H6}_context.json`（読取のみ。SHA `8396605f…`／`61d87e2f…`。前者はA S0の記録と一致） | 前後でbytes不変 |
| 他の保存系列 | B基点の`h6_df19_price_provider`（`73dcbe23`/`b8583796`/`a4d743cb`）が読むH6 FULL19・m5_best・δ1.6の価格 | DIRECTとMODELを分けて列挙 |
| 索引 | `track_a_h246_pf_common_adopted/2026-10-22/source_catalog.json`・`paper_manifest.json` | 実在する（figure・evidence単位の索引）。候補単位の照合は上記carrierで行った |

未確認の範囲：
- 公開carrierはcommitに保存されたcopyで照合した。private内の候補別fileは、exploration trialの`result.json`の存在を確認しただけで、中身は読んでいない（`signals/`は件数の確認のみ）。
- pf_commonの`native_results.json.gz`との突合は行っていない（C側の作業）。

## 2. 物理条件の照合

- 136候補の内訳：保存grid 132（2系×22(T,q)×3公式）と、extensionのq=32の4候補（2nd、T=1.6/2.0）。
- 全候補が次を満たす：`method=FULL`、prefix=全rank（H4 10、H6 19）、公式2nd/4th/m5_best、R/Kなし、`cid`がcanonical、δ=T/q、tol-only DF10/DF19のbinding。
- 標準4thとm5_bestは別のformula_id・別の物理IDである。B0/B0I/PRは対象外。
- 公式の係数、signed pieceの順序、S2 sweep、scalar一回は、保存研究のSOURCE（`h46_long_time_signal.evolve_batch`、`h46_m5_plan.weights`、`h46_long_time_native.compose`）とDの物理requestで同じである。
- 旧H4 rank12（DF12）の結果、truncated PF、別SOURCEの価格は、testで拒否されることを確認した。

## 3. 結論1：保存signal・referenceの再利用可能性

| 項目 | 件数 | 内容 |
|---|---:|---|
| signal（EXACT） | 134 | 110：長時間研究の`evolve_batch`（`NEW_SAVED_INPUT_FINITE_MEAN`）。24：pf_common T=0.8を再利用し、bindingを照合済み。raw/corrected/reference/bias/allowance（最大中間norm・closure）のfieldがある |
| signal（CONDITIONAL） | 2 | H6 T=1.6 q8の2nd/4th（fixed-delta summaryからの再利用）。raw signalとpropagation evidenceが欠ける（`saved_allowance_only`） |
| reference値 | 136 | 対象は元DF H（全行列＋constant）の、保存sector stateに対する値。内訳：新しいspectral full-H 90、保存referenceの再利用（T=0.8のreference file、H6 T=1.6）20、再利用recordに内包されたもの26 |
| 独立reference pair | 38 | T=2.0だけ。`expm_multiply`による照合（H4 3.20e-15、H6 1.45e-15。private context）。T=0.8はreceipt付きの保存reference file。その他のTには独立pairがない |
| 数値gate | 136 | 旧契約（経験的allowance、α_axis .025）でのendpoint status（ELIGIBLE/INELIGIBLE）を保持。**新しいRMSE policyのPASSではない** |

## 4. 結論2：保存compiled価格の再利用可能性（X/Yを別判定）

| 区分 | 件数 | 内容 |
|---|---:|---|
| EXACT（X/Yとも） | 56 | 49：長時間研究のexploration/extensionで決定論direct compile（n=1）。7：pf_common T=0.8の価格で、candidate receipt（method・prefix・formula・q・compiler）が一致 |
| CONDITIONAL（X/Yとも） | 17 | pf_common T=0.8の再利用。receiptにcandidate fieldがない（5）、または`result_sha256`しかない（12）。Cが元の`native_results`と照合する必要がある |
| 価格なし（signalあり） | 63 | X/Yとも欠測。T=0.8 q16は全公式で欠測。m5_bestは各系15候補 |
| 片軸だけ保存 | 0 | X/Yを互いに補完していない |
| D-only・uncontrolledの価格 | 0 | carrierはすべてcontrolled full wrapper（cosine/sine、測定込み、P0） |

他の保存系列：
- H6 FULL19・m5_best・δ1.6のdirect価格がある：q1（T1.6）、q2（T3.2）、q4（T6.4）。別研究のため`CONDITIONAL`とし、nativeの構成が同一かをCが確認する。
- 同じ系列のAffine予測（q2/q4）は`MODEL_NOT_DIRECT_PRICE`。
- C providerのH4 PFはB0/p7（truncated）で、Full PFではない。

## 5. 結論3：新しい科学計算・compileが本当に必要な候補

- **保存gridの内側**：
  - 63候補の価格（Cのdeterministic compile）。signalとreferenceは既存のものを使える。
  - 2候補のsignal（H6 fixed-deltaの再利用分）をFull carrierの水準にするかは、Aの判断。
- **保存gridの外側**：保存T={0.8,1.2,1.6,2.0}にないRMSE scheduleのroundは、signal・reference・価格がすべて欠測する。
  - 例：A S0と同じh=0.1・δ=0.1では、m3（T0.8 q8）とm4（T1.6 q16）だけが保存gridに入る。
  - m0–2とm5以降は、新しいmeanとreference（A）、新しいcompile（C）が必要。
  - `for_packet`でD requestと結合すると、`OUTSIDE_SAVED_GRID_ALL_MISSING`と表示される。
- **全候補共通**：全round十分NとFull PF用のRMSE誤差契約（A）。保存の数値gateは旧契約の結果である。

## 6. 結論4：A/Cの次の作業（承認が必要）

- **Codex A**（[HANDOFF_A](HANDOFF_A.md)）：
  - EXACT signal 134件とreferenceを再利用候補として、Full PF用のRMSE誤差契約（PF biasの扱いを含む）を定める。
  - 保存gridの外のroundについて、mean/referenceの新規計算範囲を提案する。
- **Codex C**（[HANDOFF_C](HANDOFF_C.md)）：
  - CONDITIONAL 17件の価格をreceiptで確認する。
  - 価格欠測63候補と、grid外の必要roundのcompile範囲を提案する。

いずれも、新しい科学計算・compileの前にユーザーの明示承認が必要である。

## 7. 結論5：未解決のSOURCE・科学的資格

- 公式・幅・M、Full PFの比較menu、RMSE誤差契約は未採択である。
- 保存signal/endpointのPASSは旧契約の結果である。新しいRMSE N・全RPE decoderの資格はない。
- 独立reference pairはT=2.0にしかない。
- H6 fixed-deltaの2件はraw signalがない。
- pf_common 17件の価格はreceiptの照合待ちである。
- 標準4thのV2 IDは、Bの公式hookが公開されるまでnull（`for_packet`もnullのまま結合する）。

## 8. 技術テスト・保全・STOP

- [TESTS](TESTS.txt)：**80 PASS**。新12件（DF10/DF12、Full/truncated、4th/m5、同じT/qで異SOURCE、signal/reference/価格の分離、X/Y、compiler/policy、MODEL/DIRECT、FAILの保持、V2 nullと結合、非起動・copy、bytes保全）＋D 16件＋関連回帰52件。
- 主要guard（receiptの公式、全rank、X/Yの別判定）は、無効化すると検出されることを確認した。
- 保全：carrier、private context、SOURCE、結果、STOPは変更していない。B/A/Cのbranchも変更していない。
- **STOP**：commit/pushとremote照合の後、mandatory STOP。
