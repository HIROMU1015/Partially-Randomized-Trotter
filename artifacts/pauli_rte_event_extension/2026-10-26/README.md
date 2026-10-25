# H4/H6 pure RTE event extension — 2026-10-26

入口は [技術報告](../../../docs/pauli_rte_event_extension_20261026.md)。実行SOURCEは
`e755ecf581600192fd2745d7066e410cebd933d3`、準備commitは
`c6d42d925b258939768a8c6a34ee3f48db27b964`。
実際の件数・比較結果は `result_summary.json` と `verification.json` を参照する。
研究は `STUDY_STOP.json` で完了し、科学runを再開しない。

## 比較データ

- `candidate_status.csv`：12計画候補の取得状態。未実行を0価格・不適格にしない。
- `candidate_precision_table.csv`：新候補の5精度×cohort。Exact B/bias/u/N/C/G、paired G SE、保存Hybrid/旧RTEとの比、rho guard。
- `event_cost_trends.csv`：旧保存RTEを引用し、新候補を別origin/source/cohortで追加。旧行は再会計せず保持する。
- `best_by_precision.csv`：H4/H6×5精度の確認集合minと保存Hybridのfamily比較。探索nominee・その確認値・事後minを区別。
- `selected_cost_breakdown.csv`／`comparison_details.json`：選択行の費用内訳と元出典。
- `candidate_results.json`：新候補の各軸shot判定・guard・paired価格のmachine-readable詳細。任意精度整数/有理数対応のreaderを使用する。
- 保存Hybrid/旧RTEの選択行は [準備baseline](../../pauli_rte_event_extension_preparation/2026-10-26/saved_baseline_quotes.csv)。旧matched比較はH6 fairness/H4 comparisonの報告を参照。

主評価は固定linear H4/H6、隣接1Å/STO-3G、S2/T=.8、純RTE L0/q8、central-control/Qiskit1.3 opt1 seed17、状態準備を除くRZ×shot、仮想ρ=0。
Actual compiled rho・母平均価格tail・大域最適性は未保証。5精度は同じ価格標本を共有し、探索/確認をpoolしない。
新r>8のmatched Hybridは未取得。保存Hybrid familyとの比較を同一q/r/K対照と呼ばない。
旧H6 guard未確定値はnullを保持する。旧H4監視証拠や局所Arb certificateのscopeも変更しない。

## 図

`figures/H4_G_vs_s.*`／`H6_G_vs_s.*` は5精度のlog-s/log-Gと保存Hybrid水平線。
`H4_B_N_C_vs_s.*`／`H6_B_N_C_vs_s.*` はB、cosine単回価格C、5精度のK2必要shot数。
`H6_G_vs_s_extension_zoom.*` はs>=64だけを拡大し、s512近傍の最小とs1024での上昇を読みやすくした保存値の図。
PNG/SVG/PDFを保存。Exploration/confirmationとK2/K4を分ける。誤差棒は名目2SE、線は視覚的ガイドで、tail/逐次選抜補正の保証ではない。
旧H4のBがdecimal引用のみの行は、そのdecimal表示を作図に使い、exact Bへ格上げしていない。

## SOURCE・rawの復元

`CURRENT_SOURCE.json` と `SOURCE_*.json` は実際に使うcode/input/runtimeだけを固定する。
`analysis_SOURCE.json` は最終整理コードと入力snapshotのhash。
`active_scope.json` は新出力の実行scope、元の `execution_plan_DRAFT_DISABLED.json` は無変更・無効のまま保全。
scope内candidate descriptorの初期statusは計画時の値で、実際の状態は `candidate_status.csv`／target/native記録が示す。
`engine_completion_observations.json` は実際に回収した親engine終了値。

`data_archives.json` と `*_evidence.tar.xz.part*` に元rawを可逆保存する。
casesにはtarget/spec/law、raw event/seed、両軸input/compiled QPYを収録。
stages/confirmationsには先行固定plan、資源観測、native log/exit/reap、延長判断と各STOP、inputsには元bytesの入力コピーを収録する。
元rawを削除・変更せずGitではarchiveを公開する。既存restorerを再利用する。

```bash
python3 scripts/restore_pauli_h2_h6_evidence.py \
  artifacts/pauli_rte_event_extension/2026-10-26/data_archives.json \
  /tmp/pauli-rte-event-extension-restored
```

出力先は未存在directoryを指定する。これはbytesの復元・SHA検査だけで、科学runの再開ではない。
`archive_restore_verification.json` は全member元bytesの復元検査結果。
技術検査はprimary Codexによるもので、独立科学レビュー・主要claim採択・正式certificate受理ではない。

次の担当：ユーザー/GPTが科学的な扱いを判断する。追加科学計算・主要claim採択・論文作成には自動移行しない。
