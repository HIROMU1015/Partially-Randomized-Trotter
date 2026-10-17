# A2 Phase B：科学実行の明示承認

ユーザーは正式契約Cの固定・公開完了後、会話で次のように明示承認した。

> 計算に進んで大丈夫です　中断された場合は修正して計算が終わるまで続けて

続けて「続きを行って」と指示した。これを固定科学条件のPhase B実行と、技術的中断があった場合の原因調査・必要最小限修正・継続の権限として記録する。科学条件・保証対象・比較条件の変更や、科学的不成立を成功へ変える権限とは解釈しない。

初回はSOURCE S `8d483f9bfa5983bb52fef55887e2f9f73a790511`、正式契約C commit `a7f8307123f6ba5bbb8d8b84e124400ac04a60ce`（公開最終 `6ea2d559e461bea4e12c4c367b9a4b29a2864c6f`）をそのまま使用する。[authorization](../../../artifacts/qc_a2_phase_b_execution/2026-10-18/authorization.json) と [外部pins](../../../artifacts/qc_a2_phase_b_execution/2026-10-18/external_pins.json) を実行前にcommit固定する。

初回のC・SOURCE・one-shot・retry禁止は書き換えない。将来技術的中断の修正が必要な場合、消費済みclaim・旧SOURCE・失敗・契約を全て保全し、今回の継続指示を根拠とする技術修正版／別試行の変更理由・hash・provenanceを別途固定する。旧claim再利用・削除、無診断のretry loop、科学条件の緩和は行わない。科学的な受理条件を満たさない問題なら不成立を保存し、その科学判断はユーザーへ返す。

今回の自発的なツール中断時点では読み取り調査だけで、科学call・authorization・claimは未実施だった。このauthorizationは中断後に新規発行する初回用であり、過去の消費済み権限の再使用ではない。

実行場所は固定canonical checkout `/tmp/prt-qc-a2-pilot-phase-a-20261018`。管理用checkoutは別にし、科学ledgerは移さない。実行後は結果・失敗・費用・hashを保全して公開照合しmandatory STOP。A2研究NOT_ADOPTED、A1 CLOSED、K1-F未承認、SPEC-08/QPE/native/state preparation/新科学レビューは範囲外。
