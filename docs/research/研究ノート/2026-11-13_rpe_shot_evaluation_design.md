# 2026-11-13：供給signalからの全RPE精度・shot評価

B cdf6a806の既存統計branchを継続し、S3/R3・A97146481・V2 menu/価格会計とshotモデルを確認した。prepare/simulate_prepared/summarizeはwhole-shot vectorを既に扱えるため、production評価器を重複実装せず、設計と入出力HANDOFF・有限科学案を作成した。

元decoderは逐次branchで全roundが結合し、整数Nの偶奇・zero/tieもある。Nの独立round最適化や無根拠な単調二分探索は仮定しない。低次元の有限whole-vector familyをまず供給する案とし、nominal/L1経験的値・conditional RMSE・有効上側risk/十分N資格を分ける。L1具体SOURCEが未指定のため、そのラベルをnormや既存科学policyへ読み替えない。

最終目的はG=N×Cであり、mock例でもshot総数最小と価格付き費用最小は異なる。現selectはunique configごと1評価なので、将来はallocation IDで内側有限表を管理し、qualified配分1代表＋coverageを外側へ渡す必要がある。未確定policyのためoptimizerを先行実装しない。

新人工7＋既存decoder33＋controller/evidence29＝69件PASS。巨大Nのexact保全/int64拒否、source/law/target、欠測/未資格、MC診断のN昇格拒否、scalar/batch同値を確認。実H4 MC/保存数値再解析/新N最適化/RMSE/実G0。他担当に非介入。[REPORT](../rpe_shot_evaluation_design_20261113/REPORT.md)／[HANDOFF](../rpe_shot_evaluation_design_20261113/HANDOFF.md)。次はユーザーのclaim/loss/shot-family/有限scope採択・許可とsignal担当の全round入力後にBが評価する。公開照合後mandatory STOP。
