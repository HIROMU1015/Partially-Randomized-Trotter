# M1-S Gate 0 v5：正式採択記録の固定

## 目的

GPT独立数学レビューの条件付き受理を踏まえ、ユーザーは**Gate 0 v5最終実験契約を、条件付きengineering評価として正式採択する方針を承認しました。**

今回の作業は採択記録・契約固定・GitHub公開までです。**科学計算の実行は承認しません。**

## 固定資料

Repository：`HIROMU1015/Partially-Randomized-Trotter`

基点commit：`47da54279838a6b3e1497f36204ba7e53c4857f9`

SOURCE固定commit：`69de022dc8142f703423b1ce0bd42f05e123954e`

証明固定commit：`b5a185cc0c3aae0e477bfc52b93061e86feb7dec`

必須資料：

- `docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v5_candidate_20261016.md`
- `docs/tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md`
- `artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/execution_contract_v5_candidate.json`
- GPT独立数学レビュー：`track_b_m1s_gate0_v5_gpt_independent_mathematical_review_20261016.md`
- 最新共通研究指示：`research_project_instructions_unified_2026-10-16.md`

## 作業

1. ユーザーの今回の発言「その方針で進めて」と直前のGPT回答を、採択の文脈が追跡できる形で保存する。
2. 次の条件を正式採択記録に固定する。
   - H4 1.00 Å／STO-3G／DF rank12、G3/G6/G9
   - 新係数上界 `alpha_R=gamma_(713+r)*S_R`
   - 数値精度17項目と従来の精度要求
   - 絶対oracle残差品質 `1e-8 Ha+B_res`
   - 条件付きIEEE／BLAS／fsum誤差モデル
   - `CONDITIONAL_ENGINEERING_NOT_CERTIFIED`
   - 既存runtime・resource caps
   - 全3分割の報告、one-shot、failure分類、mandatory STOP
3. 独立した新しい採択済みmachine契約を作成する。`adopted=true`と`coverage_model_adopted=true`は、ユーザーが**条件付きモデルを採用したこと**だけを示す。プラットフォーム数値保証の認定と混同しない。
4. SOURCE・proof・入力・契約のhashとcommit順序を照合する。
5. 静的検査と非数値の契約検査を実施する。
6. 専用branchでcommit・pushし、remote SHA、公開bytes／SHA256、旧成果保全を確認する。

## 禁止事項

- 新規科学計算、算定式の実入力評価、W/v/ρ計算
- Pauli係数生成、数値correctness test、simulation、compile
- 新しい科学実施authorization／grant／markerの作成
- v4のTECHNICAL_STOP、null結果、消費済みmarkerの変更
- 精度閾値・算定式・runtime制限の独断変更
- v5実行への自動移行

## 完了時

正式採択記録、固定契約commit、静的検査、GitHub公開照合、旧成果保全を報告してください。

**採択記録を固定した時点でmandatory STOPしてください。**

次のGate 0 v5科学計算には、ユーザーからさらに別途、明示的な実施承認を受ける必要があります。
