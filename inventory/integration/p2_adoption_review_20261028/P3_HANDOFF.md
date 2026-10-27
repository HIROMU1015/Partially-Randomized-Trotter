# P3 Research B/B2-Dへの設計引継ぎ（未開始）

固定アーキテクチャ `1b649896393a4a0c5f2e3bcf41b2853090ef2053` のINTEGRATION_ROADMAP P3を継承する。
今回P3のcode実装/新branch/科学計算/現行環境調査は0。

## 既存入口

代表候補は `codex/research-b-final-pause-handoff-20261026` @ `e87375fa9a998dce80792be0f15597c747f00d83`。
これは既存計画の固定参照で、今回最新remote tipや再開許可を新しく確認したものではない。
旧 `codex/research-b-b2d-static-cost-outer-20261025` @ `3f5d60122f158b3320fadab86498963950f00734` は履歴比較元。
既存evaluator/backend/recipe、outer/Exact-R/Full-DF/q4は後続に既継承という既存監査を再利用する。

|固定版で所在確認した入口|用途|
|---|---|
|[b2d_evaluator README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e87375fa9a998dce80792be0f15597c747f00d83/src/trottertracks/b2d_evaluator/README.md)|stage/source/result/STOPの探索入口|
|src/trottertracks/b2d_evaluator/evaluation.py|既存evaluator/backend境界|
|src/trottertracks/b2d_evaluator/outer.py|outer固有評価/条件|
|src/trottertracks/b2d_evaluator/exact_r.py|Exact-Rの別scope|
|src/trottertracks/b2d_evaluator/full_df_q4.py|Full-DF/q4の別allowlist|
|src/trottertracks/b2d_adapter/README.md|既存adapter入口|

今回はpath所在metadataと既存計画のみを使用。B2-D code/科学データや現行jobを再調査していない。

## P2から引き継ぐ設計

- 成果の固定refと今後のcode開発tipを区別する。専用の完了/実行branchを汎用開発へ黙って転用しない。
- CLI→composer→研究adapter→既存core。kernelやbackendを同名の新実装で作り直さない。
- 全branch mergeでなく機能・file・依存closureを特定する。既継承のevaluator/input/backendは移植不要と明記する。
- contextの完全bindingに加え、stage record内のmethod/order/metric等の表示も照合する。同bindingでも矛盾値を許さない。
- stageごとにqualification/missing/failed/STOPを伝播し、未取得u_num/gamma/rare probability/価格を0へ埋めない。
- 実Science providerとmock routeを分け、no-site技術PASSやAPI同名だけで数値同等性をclaimしない。
- fixed SOURCE/cohort/seed/one-shot/retry/output layout/旧失敗と後続成功を版別保持し、固定SOURCEへ新codeを適用しない。
- global initializer変更や共有venv/alias loaderを依存不足の回避策にしない。新しい承認証明/汎用grant制度を作らない。

Pauli context/法則/会計/候補allowlistをB2-Dへコピーしない。S2/Y4 signed schedule/adjoint/processor、native compiler範囲、
A0 reference/B0 native/B1 IID/Exact-R/q2/q4の独立契約はB2-D側のadapterに残す。

## 次回の判断対象

ユーザー：P3の対象stageと代表開発入口、資料整備/技術実装の範囲を明示する。
Codex：その指示後、固定final-pauseと最新正式handoffの関係だけを限定確認し、既存packageのstage表と薄い配線の必要差分を設計する。
tests/management/test_b2d_full_df.py・test_b2d_full_df_q4.py等はsourceからpure/mock範囲を選ぶ。
高精度数値/operator/build/compile/N/Gや停止研究の再開は別の科学実行許可が必要。
final-pause、B3、Research C、他jobへ自動介入しない。P2代表統合が終わってもP3開始許可にはならない。
