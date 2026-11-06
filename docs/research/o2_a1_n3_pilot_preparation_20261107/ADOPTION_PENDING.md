# 指定済みscopeと採択待ち事項

2026-11-07 JST。参照契約は5c27684d904aed9f4f9f2787752c917bb46d684cの[N3_PILOT_CONTRACT_DRAFT.md](../o2_a1_n3_scope_redesign_20261106/N3_PILOT_CONTRACT_DRAFT.md)。原草案は上書きしない。

| 区分 | 状態 |
|---|---|
| 現行新Bの正規停止・保全 | 今回明示承認され、完了 |
| 研究方向・予算 | A1対B1-OL、X/Y、160 train＋80 probe＋512 evaluation＝752、B0/B1-O除外を準備する指示を受領 |
| 固定科学対象 | 保存H4/1.00 Å/STO-3G/legacy DF12、8system qubits/全4電子sector/L_D3/m0/q1/δ.02/r1/K2/atol0、N1 actual Q/selected policy、N2保証ground、認証済exact engine、3-control/全5正層を維持 |
| 新752-shot実測 | **今回は未承認・未実行** |

[PILOT_DRAFT_SETTINGS.json](PILOT_DRAFT_SETTINGS.json)はコードから生成した設定表示で、実RNGや量子処理は生成していない。
次は下記を科学実行SOURCEへ確定する前に明示採択する。条件を証明する追加packet/grant/承認ファイルは要求しない。

- campaign候補 `A1-N3-PILOT-H4-m0-q1-20261106-v1` とseed候補 **20261108**。草案から引き継いだ候補で、消費済み旧20261106/20261107と分離する。今回はseedをRNGへ投入していない。
- 32 sorted domains/64子stream、PCG64/SeedSequence、独立uniform53-bitモデル、path64word/別measurement word1。数学的PRNG独立性の証明ではない。
- B1-OL配分proxy `(1+min(s²,1))/2`、価格score floor1、quota floor4、残り108のlargest remainder、既存128-bit root中点、固定tie順。
- rank4、range guard D_CV≤4、再fit/追加shotsなし。A1 trainの16/層とB1 probeの8/層、評価128/方式/軸は752内訳に対応する。
- 予備診断score、zero/all-equalのINCONCLUSIVE、両軸一貫性の探索的分類。十分N・正式ε/α・期待費用保証を採択しない。
- guard-firstのX→Y/phase順、fresh output/cacheと管理入口、入力/source/environment/profileの最終固定。

候補outputは`artifacts/o2_a1_new_target/2026-11-07/n3_pilot_752_run01`、cacheはその配下`cache`。この科学output/cacheは今回作成していない。
旧397件、新Bの1339 bit/係数/word/cacheは参考資料・支出履歴に限定し、新pilotには読み込まない。

PREPARATION_PROFILE.jsonとSTATIC_INPUT_BINDING.jsonは静的な準備検査であり、正式な実行profile lockや新しいengine認証ではない。
採択・科学実行許可後、activation差分を最小限で接続し、人工検証・科学SOURCE/profile/remote本文/blob/treeを固定してから初めて実入力を投入する。
今回の正規停止許可や以前の技術修復許可を、新752-shotの実行許可として扱わない。
