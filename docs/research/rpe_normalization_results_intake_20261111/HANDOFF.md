# A0実結果受入とC1診断共有HANDOFF

[REPORT](REPORT.md)／[SUMMARY](SUMMARY.json)／[受入原記録](INTAKE.json)／[N/C coverage metadata](COVERAGE.json)／[C1共有2件](C1_LINKS.json)。固定A0公開commitは`4745bb676568bb11d46552be42ffc6f4fba75385`、科学実行SOURCEは`6a75ebf967aaae2698ce6d295bf6fbec9d5e9929`。既存受入実装`529e81fc20969bd8265ae98ba71ad18fa8f3eca7`を変更せず使用した。

## 受入済み状態

元root `PHASE0_NORMALIZATION_RECORDED`、24件正常/欠測0/STOP0/数値未資格0、SOURCE/input保全true、`normalization_evidence_valid=true`、`normalization_evidence_usable=true`。全24件の診断消費資格が有効で、C1との共有対象は以下2件だけ。

- `2d2af6ca830dff14a695`：H6 m4/T1.6/qPF1/δ1.6/R16/prefix7/K2。
- `4269eafc8c0ef8922c23`：同条件のR32。

C1 consumerはC1_LINKSの`request`/`normalization_source`と実際のconsumer request/SOURCEを照合し、`share_eligible`と元statusを保持する。C1_LINKSは診断のcopyであり、C/Nレコードとして使わない。残る6セルにdiagnosticを埋めず、H4 R8/q8・q16の明示laneを保持する。C1実行の前提としてA0へ追加計算を要求しない。

## 再現する受動的コマンド

既存B CLIをfresh B output directoryに向けて使用する：

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  scripts/prepare_rpe_normalization_intake.py \
  --handoff-commit 4745bb676568bb11d46552be42ffc6f4fba75385 \
  --handoff-path docs/research/pr_r_normalization_phase0_results_20261111/run01/HANDOFF.json \
  --result-path docs/research/pr_r_normalization_phase0_results_20261111/run01/result.json \
  --output-dir <fresh-B-intake-directory>
```

今回の保存出力/coverage/C1接続を再照合する場合：

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  docs/research/rpe_normalization_results_intake_20261111/VERIFY_INTAKE.py
```

上記は固定Git JSON/SOURCEとB保存出力だけを読む。通常検査は出力を書き換えない。`--write-derived`は初回に新規派生metadata3件を書き出すためだけの引数で、既存ファイルがあれば拒否する。入力の実数値の再解析、signal/N/C/G、R選択を行うoptionはない。

## 追加成果の受入条件

1. **Codex A：signal/N**。別の許可範囲で取得した同一candidate/full identity/lawのsignal/phase/radius・十分N・allowance政策を公開commit/pathで渡す。A0の24件はこの計算を行っていない。同一candidateの旧固定R N6件は別出典を保持し、他Rへ移植しない。新R18件とbaseline共通R-aware allowance、全RPE成功は別の科学的不足である。巨大Nは既存int/decimal string形式を守り、H4補正済みN_HANDOFFを正本にする。
2. **Codex C：価格**。C1の校正/独立検証/rareを元scopeで区別し、full domain（T/qPF/δ/formula/prefix/K/R/allocation/law/H/state/snapshot/compiler/wrapper）とSOURCE/cohort/paired axisを持つ既存共通C recordを渡す。direct、model、有限cohort、全law期待価格、未検証外挿、nullを混同しない。C1小qや固定Rモデルを長q/大Rへ無条件に流用しない。
3. **Codex B：技術接続**。今後も明示された公開SHA/pathから既存provider/coverageへ取り込み、raw law宣言とB law bindingを保全する。未知の共有校正誤差/cross-R共分散/long-q systematic/CIや全RPE資格はnull/未資格を維持する。実G/最適化の前に必要なN/C/全law/不確かさのdomainが揃ったことを確認する。
4. **ユーザー：次の科学scopeの判断**。新signal/N、R依存C、長q/全law/共分散、実G/R最適化が必要なら対象・有限範囲を明示する。今回の受入指示やA0の完了を、その科学実行許可へ拡張しない。

本受入でN/C/G資格、科学的PASS/FAIL、推奨R、全RPE成功は生成していない。A元86候補（A0外62）とC1元8セルは保全した。47技術test＋公開fixture検査PASS、科学kernel・pipeline・テスト実装の変更なし。remote公開照合後mandatory STOPし、新しい科学実行やR最適化へ自動進行しない。
