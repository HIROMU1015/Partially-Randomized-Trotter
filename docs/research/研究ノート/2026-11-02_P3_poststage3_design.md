# 2026-11-02 P3 Stage 3後：非実行の次作業設計

ユーザーの方針継続指示を受け、Stage 3完成結果`90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5`から、
[独立価格確認とStage 4接続の案](../h4_p3_poststage3_cost_and_stage4_design_20261102.md)を具体化した。
新32 trajectory/arm、Full64 body/128 compile、独立参照半指数1回・保存half照合・Full64 oracleを提案。
新平均/SE・cos/sin共分散を対象とし、旧pilotをpoolせず、追加標本数による精度達成や母平均tailの保証はしない。
Stage 4のN/G・実装誤差仮定は別scope。finite biasの高精度化は価格確認の必須条件にしない。

今回の科学call/再集計/回路生成/sampling/compileは0。Markdownのみ変更し、コード・科学SOURCE・artifact・旧manifest・全STOPを変更していない。
次の担当はユーザー：上記の追加価格確認を実行するか判断。Codexは範囲の明示許可後に実装から公開までまとめて進める。
