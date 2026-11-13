# S3 run01：条件付き結果の引渡し

2026-11-13 JST。[REPORT](REPORT.md)／[RESULT](RESULT.json)／[採択条件](EXECUTION.json)。公開計画ee1dの推奨セットを採用し、code ab12・caller7486・input bf515で固定43,819反復/seed20261113を1回完了した。過去の未採択計画/受入packetは保全し、新runでのみ採択・実行を記録した。

## 受け渡すもの

- [PRIMARY.json](PRIMARY.json)：経験的な解決条件付きRMSE=0.0006847202619833286 Ha、未解決0/43819、97.5% CP上端9.999788982664716e−5。主population_RMSE_CI=null。
- [LOSS_ENVELOPE.json](LOSS_ENVELOPE.json)：同じtrialの[0,4096] Ha²未解決loss感度。97.5% population MSE envelope=[0,28.963512732533932] Ha²、RMSEスケール上端5.381775983124338 Ha。**特定の全試行推定器のRMSEではない**。
- [SIMULATION.json.gz](SIMULATION.json.gz)：全43,819trialの元出力、各13roundの位相/branch/距離・energy・status/SOURCE、入力rawと誤差参照を保持。[ARTIFACTS](ARTIFACTS.json)にサイズ/SHA256、[VERIFICATION](VERIFICATION.txt)に読戻し/保全確認。

identityはcanonical `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、H4 DF10/S2/L_D1/ΔPF=h=.1/K2/M12・元exactκ/H-state/snapshot/prior/profile/P0。各軸nominal64,60,56,52,47,43,38,34,30,25,21,17,12を変更していない。

targetは保存solver energyの条件付きproxy（ground未認証）、lawはideal finite-K/fresh IIDのplug-in binomial、count callerはexact整数和→binary64、batchはR3 scalar互換。数値allowanceやactual law誤差を伝播した評価ではない。

## downstreamで昇格させないもの

全trialがDECODEDでもbranch正解のoracleはない。未解決0観測は真の率0ではなく、prior外0観測もalias資格ではない。原APIの全観測RMSE/点端点をpopulation保証として使わない。条件付きRMSEのCI、十分N、global RMSE bound、scientific_RMSE_PASS、適格compiled C、Gはnull。

副評価のCI field名が`population_RMSE_CI`でも、意味は採択したloss域の感度envelope。二つのconfidence familyは各97.5%でjoint名目95%以上だが、FP認証やAの物理系統誤差を含まない。幅が広いため目標精度達成の証拠ではない。

元入力の`target.energy_ha=null`、`truth_proxy_adopted=false`、物理qualification=falseは保存来歴として維持した。今回のproxy採用と条件付き科学実行は新EXECUTION/RESULTで示す。AのN packetやB controllerの価格/十分N評価へ、自動でQUALIFIED packetとして渡さない。

## 再現と次の担当

callerは`scripts/run_h4_s3_statistics.py`。コードSOURCEと入力SOURCEを分けて記録し、既存出力への上書きを拒否する。再現のための再simulationも新しい科学実行なので、今回の1実験を無断で繰り返さない。artifact検証やmetadata読取りにsimulationは不要。

**ユーザー**：次の有限研究範囲を決める。**Codex B**：別許可後に必要なMSE精度/未解決loss/比較契約を扱う。**A**：必要な物理L2/target/law証拠、**C/D**：価格/残差は既存scopeで継続。今回BからA/C/Dへ新しい科学実行を指示していない。

固定run完了・結果公開照合後mandatory STOP。追加trial/seed/N探索/C/G/4方式最適化へ自動進行しない。
