# N3統合準備 v1：GitHub公開と保存照合

2026-10-12 JST。branch `n3-theory-task-integration-20261012`、repository `HIROMU1015/Partially-Randomized-Trotter`。
source `370c42a6d76ab034714c065182143d0aa112c0ff`、統合payload `2de3b26fc36df9e99b7504631c97635f1473cb51`。

## GPT判断の入口

- [統合準備・到達点・次の判断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/docs/research/n3_theory_task_integration_v1_20261012.md)
- [claim map・証明・独立照合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/docs/research/n3_theory_claim_map_v1_20261012.md)
- [一次文献・利用情報・保証・費用の対応](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/docs/research/n3_literature_correspondence_v1_20261012.md)
- [幅指向generatorと一mode縮約taskの仕様案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/docs/research/n3_width_reduction_v2_spec_20261012.md)
- [exact保存照合結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/artifacts/n3_theory_task_integration_v1/2026-10-12/saved_analytic_checks.json)
- [14項目のレビュー付録対応](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/artifacts/n3_theory_task_integration_v1/2026-10-12/review_correspondence.json)
- [30 synthetic・source identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/artifacts/n3_theory_task_integration_v1/2026-10-12/technical_validation.json)
- [受領した完了済みholistic review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de3b26fc36df9e99b7504631c97635f1473cb51/artifacts/n3_theory_task_integration_v1/2026-10-12/review_intake.md)

GitHubから別bareへfetchし、上記資料・34 frozen inputs/source・過去数学契約等66 blobs、7,725,968 bytesをexport。
全bytes/SHAが専用worktreeと一致。local object copy／alternatesは使用していない。
取得したsourceと保存入力でstdlib checkerを実行し、全幅排除二例の出力がlocal archiveとbyte一致した。
これは同じlocal環境での保存帰結の再照合であり、科学producer再実行・外部環境での再現・immutable CIではない。
[取得inventory・receipt](../../artifacts/n3_theory_task_integration_v1/2026-10-12/publication_receipt.json)、[remote checker出力](../../artifacts/n3_theory_task_integration_v1/2026-10-12/remote_saved_check.json)、[完了確認](../../artifacts/n3_theory_task_integration_v1/2026-10-12/completion_check.json)。

元科学source/input/結果/gate/STOP、manifestの既存102 entryと他field、rootの編集を保全。
root baseline1362 filesのうち1360件はbyte不変。中断中に外部更新された`docs/DOCUMENTATION_INDEX.md`と`docs/progress/README.md`の現bytesも保持し、receiptへ差分SHAを記録した。本taskからのroot編集0。
root/BentoSlide／別Trackをstage・commitしていない。既存global manifest description欠落FAILは[原出力](../../artifacts/n3_theory_task_integration_v1/2026-10-12/global_manifest_check.txt)へ記録し、新entry schema／リンク検査PASSと分ける。

ScopeはS1 star、一体＋非負density、scale1.25、ν4、full10→active9、一外部mode8、元δgoal=.005、保存OCC有理上端、同じU/c。
geometry/basis/DF rank/L_D/PF時間窓は適用なし。今回の解析的帰結は同じS1の事後developmentで、独立移送・量子資源改善ではない。
SQuISH全文を補充したが、文献原典二件の全文未取得と新規性未確定は記録に残す。
幅指向v2 generatorは未実装／未実行。新入力、eigh、optimizer、certificate探索、δ scan、U/c再取得、compile、science batchは0。

次はGPTによる、既知原理との研究差分、強い対照、縮約task・取得費用、独立検証の必要性の判断。
holistic review第11節の統合後の科学判断に到達したため停止する。新しいGPTレビューの開始と計算実行の認可は別。
中心テーマ未採択、next_stage_authorized=false、N1停止、N2保留、Mandatory STOP。
