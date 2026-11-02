# 2026-10-10 Track A 独立レビュー後の準備

利用者が[GPT独立レビュー](../track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)の方針を共有し、続いて「作業を進めて」と指示した。
[準備追補](../track_a_ax2b_post_independent_review_amendment_v1.md)と[H6契約追補](../track_a_ax2b_h6_pilot_preparation_contract_v2.md)へ反映。
RQ-R/補助RQ-P1を維持し、technical agreement・empirical u・certified boundを分ける。
H4-N/A/E/Mの具体化、独立occupation構成/mpmath fixture、u-aware/logN会計、H6 tol-only adapter、7/36 schedule、
before-call solver/reference caps、bounded writer、synthetic-controller/dummy-watchdogを別sourceで追加した。
H4 v5 source/結果/原稿/M1〜PM-2/旧v3/v4の失敗記録とdirty差分は保存する。

初回testsは43 pass/1 fail：metadata CLIがtrotterlib packageのeager importでnumpyを要求した。
既存pure-stdlib PF moduleだけを読み込む経路へ修正し、科学package不要のCLIを検査した。
log drain/terminal/phase/large-scale算術の合成検査を加えた最終49 testsはpass、fail/error/skip0。
これはlocal synthetic evidence。新しい分子signal/sampling/circuit build/transpile/compile、input/state生成は0。
H4-N/Aの分子runner/全stage precision port、H6 molecular backend/science launcher、入力/資源/別実行認可は残る。
完全準備済み・起動可能とは報告せず、H6_NOT_AUTHORIZED/DRAFT_NOT_AUTHORIZATION、mandatory STOPで終了する。

[準備inventory](../../../artifacts/resource_applicability/track_a_ax2b_post_review_preparation/2026-10-10/preparation_inventory_v1.json)とsource-bound記録から保存test・hashへ辿る。
