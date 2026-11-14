# 決定論PF Stage1価格モデルの凍結

決定論PF Stage1価格モデル（2026-11-15）：D b190bd815の保存4＋新規20価格からFull/p7×S2/Y4/M5×X/Yの12モデルをexact fit、係数とm6予測を凍結。SOURCE 79c951b38、結果 12372fa6b。新7＋既存15＝22技術test PASS。m6独立holdout未取得・未開封・未判定、qualified=false、長q外挿/N/Gなし、mandatory STOP。

固定D indexの各C_INPUTSだけを使用し、2較正点への補間一致を独立検証と扱わない。元EXPLORATION/receipt/sourceを保持。Dの既存readerが全12およびroute別6で結果を受理した。次はDの別作業によるm6独立holdout。比較規則は観測値基準1/50・12モデル個別全PASS。ロードマップ本体は統合担当が更新する。

[REPORT](../pf_stage1_model_freeze_20261115/REPORT.md)／[D HANDOFF](../pf_stage1_model_freeze_20261115/HANDOFF_D.md)。旧SOURCE/raw/較正/STOPは不変。
