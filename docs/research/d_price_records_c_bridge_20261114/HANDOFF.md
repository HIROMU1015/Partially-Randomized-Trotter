# D → C/B：bridge接続済みのPauli pilotとB kindの価格record

branch `codex/d-price-records-c-bridge-20261114`（C `c628e9e8` 基点）。[REPORT](REPORT.md)／[次の有限compile](NEXT_EXECUTION.md)。Cとbのbranch・worktreeは変更していない。

## C担当へ

- **Pauli**
  - Dのsidecarの `physical_context` は `I.price_context(req, m)` と同一である（`req` はCの `I.plan` で、`implementation_sources` 付き）。recordの土台はCの `pauli_requests` のrowで、Dが付けるのは価格値、cohort、分類、SOURCEだけ。
  - D側の照合規則は、`expected_route/lowering/native_policy/scalar_quantity` がDの定数と一致すること、canonical候補IDがrowのidentity candidate IDと一致すること。どちらかが違えば拒否する。
- **seed**：Dの `seed()` が使う `row={'id': canonical candidate ID}` は、Cのbridgeのphysical IDと同じ文字列である。bridge対応によるseedの変化はない（固定値 `14788159701244282917` と照合）。
- **価格の分類**
  - Full PF／Truncated p7：`DIRECT_COMPILED_DETERMINISTIC`／`DETERMINISTIC`（B PRICE_KINDS）。
  - Pauli B0：`DIRECT_COMPILED_FINITE_SAMPLE`／`FINITE_COHORT_ONLY`、`n_paths=1`。
  - 旧名はrecordの `diagnostic` に残した。分類の読み替えは不要である。
  - `record_schema=D_PRICE_RECORD_B_KINDS_V2` で、旧版（8b28dce0）と区別できる。
- **fit入力**：Full PFとTruncated p7の価格recordは、tree内の `price_model_validation.record` へ `fit_inputs()` で変換できる（features `one`/`q`、m4/m5は較正、m6はholdout）。
- **保存M5 m4**：所在情報の形式は前回と同じ（`SAVED_RECEIPTS_FOR_C.json`）。受入はCの担当である。

## B担当へ

- Pauli B0の価格は配線証拠であり、Gには入れない想定である。入れる場合でも、tierは `FINITE_COHORT_SCENARIO` に留まる（`price_policy='qualified_expectation'` では不適格）。
- Bのguard再pin（C A11）とPauli bridgeの取込み（C A12）は、Bの担当範囲である。Dは関与していない。
- 新decoder bridge（C §4）で、Pauliのcanonical IDが変わった場合：
  - 既存cohortのseed keyは旧IDのまま記録に残す。
  - 新IDで価格を使う場合は、(a) Dが新identityで固定assertionを再発行する（再compileはしない）か、(b) Cのguardに写像を追加するか、のどちらかが必要になる。

## Full-H P-1について（今回の修正範囲外）

`04bead11` のP-1 callerのrecordも旧名（`DIRECT_SINGLE_PATH_WIRING`／`NOT_A_LAW_MEAN`）で、G会計では拒否される。配線用途だけなら現状でよい。Gに使う場合は、Pauliと同じ方針（`DIRECT_COMPILED_FINITE_SAMPLE`／`FINITE_COHORT_ONLY`、旧名はdiagnosticへ）で新版を作る必要がある。変更するかどうかはユーザーが判断する。
