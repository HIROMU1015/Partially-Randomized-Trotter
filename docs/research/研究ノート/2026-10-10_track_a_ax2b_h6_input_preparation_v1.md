# 2026-10-10 Track A：H6入力生成準備 v1

H4補完後、ユーザーの「次の作業に進んで」を受け、直前に説明したH6入力生成の準備・契約固定を進めた。
今回は分子入力生成・H6 pilotの実行を認可する指示とは扱わない。

[準備・契約・source/監査索引](../track_a_ax2b_h6_input_preparation_v1.md)。新source commit67312f3。
既存tol-only adapter・bounded solver・matrix-free・H6 loaderを再利用し、新versionを追加。
新grant gate、3 phase watchdog、original grant bytes保持、snapshot保存/stdlib auditを接続した。
87 local synthetic/metadata tests pass（新38＋既存49）。分子を生成せずfake provider/solverとdummy processで検査。
入力生成予算をphase900/300/900秒・total2100秒・AS8GiB・output128MiB・solver共通10000作用として結果前に提案固定。
実rank/収束/正しさ/性能は未確認。小residualをground-state証明やu certificateとしない。

source183/validation3をlocal/Git bytes照合。既存H4 science180/旧結果/freezes、dirty/untracked、rootレビュー、Track Bを保全。
prepared manifestはCPU null・unsealed・新grantなし・科学output未作成。
H6_INPUT_GENERATION_NOT_AUTHORIZED / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/Gnull、UNDETERMINED、mandatory STOP。
次は別指示後にresource/seal/grantを固定して一回入力生成し、hash保存・監査後STOP。
H6 pilotの7 cell/36 wrapperとlaunchには、その後の別固定・別認可を要する。
