# 次の科学実行への接続：未採択条件を保全

2026-11-13 JST。正本はB `91d730ac3f49bbbd8ab16b6aceda33cd44701b1f`の[FINITE_SCIENCE_PLAN](../pr_ld1_ld2_rpe_risk_shot_plan_20261113/FINITE_SCIENCE_PLAN.md)および[RISK_AND_N_EVALUATION_PLAN](../pr_ld1_ld2_rpe_risk_shot_plan_20261113/RISK_AND_N_EVALUATION_PLAN.md)。今回その科学条件や有限範囲は変更・採択していない。

候補案はLD1/LD2×共通whole-shot vector1本＝計2 MC call、または同2本＝計4 MC call。両構成は各全13roundを一括decoderへ渡す。どちらも未実行。旧S3のN/反復数/結果をLD2へ移植しない。

ユーザーが実行前に採択する事項：

1. **evaluation_target**：値・意味・SOURCE・qualification scope。solver参照proxyを採用するか、ground energy認定と区別する。元targetを保持する。
2. **risk claimと全事象loss**：条件付き経験的診断、統計区間付きmodel risk、元energy契約への資格を分ける。zero/tie・未解決・branch/alias/tailの扱いを明示する。
3. **whole-shot vector**：全13roundのN_X/N_Yとallocation SOURCEを選ぶ。候補本数・選択根拠を固定し、roundごとの独立最適化としない。
4. **samplingと数値scope**：ideal finite-K rawに条件付くfresh IID binomial、実sampler/保存state/数値allowance/物理biasの適用限界を明示する。
5. **MC precision/confidence/反復数・seed**：採択loss supportと必要なMSE精度から既存trial_budget等の適用可能な方法で有限数を決める。旧43,819や95%、Dの価格精度を自動採用しない。
6. **有限scopeとSTOP**：対象2構成・vector本数・反復数・SOURCE・結果先を固定する。欠測/不一致/数値domain外/未知status等で未確認結果をPASSにせず停止する。再試行や探索への自動拡張はしない。

技術的なtarget欠測とsignal bindingの接続は[今回のcaller](HANDOFF.md)で対応した。評価に使ったtargetと全signal_inputを結果へ記録し、R3 hookを明示選択できる。これはN資格producerの完成ではない。十分N/global boundはnullを維持し、正式Gに必要な適格E[C]もD/C側の別証拠を待つ。今回の成果はD T2 pilotの完了を必要としない。

ユーザーの明示採択と有限科学実行許可後に、Bが既存MC/R3を用いてその範囲だけ評価する。計算資源は過去のworker capを継承せず、独立call単位の並列化・既存batch処理を利用し、共有環境や他担当を変更しない。Cはsignal_input/資格付きN、Dは同identityの価格を接続する。承認証明書/grant等の追加発行は不要。

今回は人工・回帰テストのみで終了する。**mandatory STOP。新MC/保存再解析/N/RMSE/G評価へ進まない。**
