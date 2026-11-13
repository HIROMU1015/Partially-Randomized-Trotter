# 2026-11-13：PRリソース評価の実装・公開ライブラリ監査

## 確認済み（技術）

- A `169d3316`、B `3216d27a`、C `83d219cc`、D `e7dca5f7`/`af6f27d7`の現行pipelineと、共通の物理・回路SOURCEを読解した。
- 人工入力・公開metadataで技術計測を行った。
- コード変更・ライブラリ導入・科学計算・compileは0件。

## 所見

- **制御ZZ回転**：RTE tailの`RZZGate.control(1)`は、Qiskit 1.3.0でCCX·CRZ·CCXになる（gate定義で確認）。等価な中央だけの制御形より、CXが10個多い。
- **`candidate_id`**：毎回parseしており、1回7.3 msかかる。memo化でtest時間が75 s→16.5 sになった。
- **RPE decoder**：vectorized分岐選択がO(反復·q)のmemoryを使う。閉形式にすると180倍速い（人工位相、結果は一致）。
- **その他**：固定SOURCEをexecする箇所に、自由変数の事前検査がない（S0a run01のNameErrorが実例）。共有moduleが3版に分岐している。

## 判断

- 物理核・compiler profile・保存値の扱いは現状維持とする。
- 改善計画R1〜R6はいずれも未承認・未実装。R1は価格定義に関わる科学判断である。
- 詳細は[REPORT](../pr_resource_implementation_audit_20261113/REPORT.md)。mandatory STOP。
