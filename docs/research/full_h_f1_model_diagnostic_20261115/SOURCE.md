# F1固定実行SOURCE

ユーザーが明示したF1の26軸適用・条件付き総額・両PF基準・m6部分額の範囲を実行する。元計画61f249e6eと凍結モデルb5165f11bは変更しない。[EXECUTION_PLAN.json](EXECUTION_PLAN.json) に対象・gate・最大件数・STOPを固定。

§10.5：C/D1の公開branch、関連Git履歴、Cのresearch索引、同名F1/weighted-cost系の自分のユーザーのPython job metadataを限定照合した。同じF1結果・進行中jobは確認範囲で0。既存の計画、4凍結予測、D1の4比較、BのQ3、PF固定総額を再利用する。非公開の未知archive/他ホスト全体の不在証明ではない。他担当のworktree/private raw/processへ変更なし。

再利用：Aの7a0c9468b `finite_price_design.features(FULL_H_GROUP_V1)`、同moduleの `c_model` が固定C c628e9e8から読む `price_model_validation.predict`、既存C `freeze_full_h_d7_models.metadata_api` と `price_implementation_guard.evidence`。fit/validate/科学runnerは呼ばない。X/Yモデルを別に保持。

正式な価格会計器へ未資格モデルを適格packetに見せて投入しない。26個のexact scalarに対するF1固有の和と差だけを薄いrunnerで扱う。Bのdecoder/optimizer/provider/正式G会計は変更しない。

効率：保存JSONのGit reader cacheと一回の元APIロードを再利用。26行・22回の小さい有理数予測、4保存予測転記に限定し、qR event列・raw展開・巨大回路を生成しない。26個の独立scalar処理にprocess起動を増やす必要はない。D1の並列化資産と適用条件は固定D branchの `docs/PERFORMANCE_METHODS.md` を参照済みだが、本F1にnative/compile workerは不要。

人工テスト：新6件＋保存受入回帰6件＝12 PASS。負値の保持、全総額null、m6の独立継続、保存4予測のpredict非呼出し、SOURCE/axis/特徴量不一致、重複軸の拒否を確認。metadataだけの実入口確認で26行・B_join・compiler/law/scalar・PF2基準の結合を確認した。科学評価前にこのrunnerと計画をcommitし、そのSOURCEからfresh run01を一回実行する。段階開始・各軸完了・終了をPROGRESS.jsonlへ保存。異常時は部分STOPを保存し、自動retryしない。
