# 有限R候補を持つ総費用pipeline（2026-11-11 JST）

[REPORT](../rpe_r_selection_20261111/REPORT.md)／[interface](../rpe_r_selection_20261111/INTERFACE.md)。利用者がR依存N/Cの受入れ・費用選択の技術実装を指定した。89c91c692から独立branchを作り、Cの現行H6統合は別treeで継続。旧B task/identity/plan/price policy、H6 m5 allocation、元会計/比較を再利用し、R選択層だけを追加した。

巨大Nを旧float積和へ渡さず、既存会計へexact有理数の任意optionを追加。人工12・旧会計15・旧pipeline12の39件PASS。実N/C生成・保存数値再解析・実G/実R最適化0。固定Rモデルを新Rに適用せず、full R domain/lawを不足のまま保持。round PASSと選択scheduleの全RPE成功を分離し、共有fit・未知cross-R covariance/選択後CIを維持した。

次の有限案はH4全13round R4/8/16/32とH6 m4..12 R16/32/64（早期固定）。保存基準Nが使えれば新PR N候補57、別R価格pilot15条件というmetadata案で、科学的採択・実行許可ではない。ユーザーが科学範囲を決め、AがN/C担当が価格を別許可内で供給し、その後Bが実比較する。commit/push・remote照合後mandatory STOP。
