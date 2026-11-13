# D → C/A：Pauli pilotとFull PF/Truncated p7の価格interface

固定SOURCE `8b28dce02cc5b5da1a8f4f1718f0b9f855759978`（branch `codex/d-kappa-s1-price-builders-20261114`）。[REPORT](REPORT.md)／[次の実行](NEXT_EXECUTION.md)。C・A・Bのbranch・worktreeは変更していない。Cのguardとmodel moduleは、`cfc0fae6` の固定blobから読み込んでtestに使っただけである。

## C担当へ

### Full PF／Truncated p7のprice record

- 生成元：`run_full_pf_shortlist_prices.price_bindings`、`run_truncated_p7_prices.price_bindings`（`I.price_requests` の骨格に値を入れたもの）。
- 各recordが持つfield：
  - `quantity=FULL_WRAPPER_MEAN_COMPILED_PRICE`、`P=0`、`normalization_is_price_multiplier=false`
  - identity、law_binding、X/Y measurement、physical_context（`I.price_context`）
  - `cohort`（Full：`D_FULL_PF_SHORTLIST_DETERMINISTIC_V1`、Truncated：`D_TRUNCATED_P7_DETERMINISTIC_T3_V1`）、`n_paths=1`
  - `classification=DIRECT_DETERMINISTIC`、`expectation=EXACT_DETERMINISTIC_CIRCUIT`
  - `mean.rz_count` と全metrics、`price_profile`＝profile＋`lowering_id=H46_LONG_TIME_NATIVE_COMPOSE`
- sidecarの `implementation`：

  | | method | lowering | scalar_quantity |
  |---|---|---|---|
  | Full PF | `FULL_PF` | `NOT_APPLICABLE_NO_RTE` | `HAMILTONIAN_CONSTANT` |
  | Truncated p7 | `TRUNCATED_PF` | `NOT_APPLICABLE_NO_RTE` | `HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY` |

  共通項目：native/wrapper SOURCE `af284d85`、scalar SOURCE、resolved/expected scalar。
- proof：`IMPLEMENTATION_EVIDENCE.json` に、selectorをkeyとして書く。公開後に、`implementation_evidence.commit` へ公開commitを記入する。
- 両方を同じpolicy（`by_method` にFULL_PFとTRUNCATED_PFの2項目、native policyは共通）でguardに通せることを確認した。

### fit入力

- `run_full_pf_shortlist_prices.fit_inputs(prices, route, formula, width_over_h)` が、C `price_model_validation.record` をそのまま呼ぶ。
- 入力の内容：features `one`、`q`（q_PF）。role：m4/m5はCALIBRATION、m6はHOLDOUT。kind `DIRECT_COMPILED_DETERMINISTIC`。implementation `{lowering_id, native}`。
- cell_idは `ROUTE:formula:width:m` の形式とする。fit、holdout判定、許容差はCとユーザーのscopeで、Dは実行しない。

### 保存M5 m4（Full PF）

- `0cd1527c80f7b9841a83`、X/Y。locatorは `155529e20:…/exploration_costs.json.gz#0cd1527c…`（C COVERAGEの `direct_evidence` を転記）。
- 実行時に `SAVED_RECEIPTS_FOR_C.json` として渡す。値は読んでおらず、受入はCの担当である。

### Pauli pilotのSOURCE規約

- sidecar：`method=PR`、`route=PR_PAULI_TAIL`、`rte_rotation_lowering=PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、`rte_lowering_source=9c4006bf:src/trotterlib/h4_d2_pauli_control.py`、`native_policy=NATIVE_DF_HEAD_PLUS_PAULI_D2_TAIL`、native SOURCE＝実行commitの `pauli_tail_price_connection.py`、wrapper `af284d85`。
- scalar：`HAMILTONIAN_CONSTANT_PLUS_SAVED_DF_TAIL_IDENTITY_sR`。SOURCEはA `LD1_DICTIONARY.json`、selector `saved_DF_scalar`。
- physical_context：A packetのconfiguration（binding、profile、split（`tail_dictionary` を含む）、law）、roundのstructure、legacy PR row（`261e184236a8b95246f7`）。
- **legacy physical CIDは辞書を含まない。** CのphysicalID bridgeは、canonical identity（configuration `d905d219…`、candidate `RMSE:62a9a6b6…`）と `tail_dictionary.dictionary_sha256` をkeyにしてほしい。Dは独自のphysicalID生成器を作っていない。bridgeが新しい物理IDを定めた場合は、`legacy_physical_candidate` を差し替える一点で接続できる。
- seed：既存long-time `trial_seed` に `row={'id': canonical candidate ID}` を渡す。bridge後もseed keyは変えない（変えると別cohortになる）。

## A担当へ

- 保存P0（4220f300）のbit順の照合は全項目PASS（[P0_BIT_ORDER_CHECK.json](P0_BIT_ORDER_CHECK.json)）。`Pauli_nonidentity`/`head`/`tail`/`blocks` はmode-0-MSB順で、`reverse_bits` による変換でQiskit labelと1e−15台で一致する。変換を省くと差は4.75（LD1）、3.62（LD2）になる。
- full-2^n oracleは、保存blockを変換したうえで各blockを指数化し、dense Pauli eventを適用する。sector射影は使わない。toyでは、独立なJW構成の保存順headとnative `prepare(ham,1)` の回路が一致した。
- 実H4での確認は、実行時のindex0 gate（block[0]=one-body、block[1]=fragment0の順序を含む）に委ねる。
- s_R（`saved_DF_scalar`）は、実行時にnativeのDF tail identityとのbit一致を必須にした。`pauli_identity` との差（LD1 3.9e−16）は、Aの規約どおりidentityへ変換していない。
