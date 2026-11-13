# 2026-11-13：PRの全RPE資源探索設計

ユーザー指定により、Track Aを全RPE最適化の研究と解釈せず、旧点最良/Paretoを採択しない設計を作成した。基点b72ea6b30・既存A branchを継続し、B公開3216d27aの正本API、Track A155529eの候補/有限RTE/native価格処理をSOURCE確認した。[REPORT](../pr_rpe_search_design_20261113/REPORT.md)が正本。

探索自由度は既存の元順prefix、S2/Y4/M5、q/幅、R、K、共通paperκ・Mを整理。実際に試す集合は未採択。BのPR-Y4未対応・K横断select制約・native ceilとA fixed-gridの違いを明示した。直接head expmと既存finite-K/exact-tail/raw/correctedを全roundへ接続する方針で、追加のexpm正当性研究・新候補管理器は作らない。

`pr_rpe_signal_plan`はB validator/exact_scalarを注入するmetadata-only helper。全13roundのMOCKで7新人工test、旧V2接続10回帰、計17PASS。実λ計算・実H4新signal・保存数値再解析・新実candidate menu/順位/N/C/Gは0。finite-mean providerの数値拡張はまだ設計段階。

prefix1/2/3/5/7/9の保存tail descriptorと全13round reference pairはfield存在だけを確認した。次の最小案はbaseline＋ユーザー選択の別prefixの2全schedule、最大26候補・78complex出力・normalizer26・新reference0。split/κ/policy/実行SOURCEはユーザーの明示選択・科学承認後だけ固定する。旧SOURCE/raw/結果/KEEP/grant/ledger/marker/STOPと他担当作業を保全し、公開・remote照合後mandatory STOP。
