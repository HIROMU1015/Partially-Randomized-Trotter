# H6 m5 compiler位相処理の追加調査

SOURCE `9fe4981ac21da17888c7f5dcadc094257c282e4e`。固定runtimeの人工最小例で、parameter付き等価則の数値定数global phase0.123が0になる挙動を再現した。parameter式θ/2とゼロは一致し、既取得の公式Rust sourceの分岐と整合した。H6の約1.5e-9差への因果接続は未確定。

P-only案は人工compileでFAILしたため単独修正に採用しない。XXPlusYY/RY等を明示basisへ展開してscalarをまとめる候補は、人工3qubitのraw全演算・制御付き演算で一致。18 test method PASS。失敗経路を再現するtest PASSと科学受入PASSを区別した。

実分子QPY/NPZ復元・compile・再生・有効grant0、旧STOP/58軸/AD03保留を維持。固定cosine一軸・最大1compile/2再生のDRAFT計画のみで、adapter/launcher/実行SOURCE sealは未完成。[結果と次の担当](../track_a_h6_m5_phase_repair_design_20261019.md)。local commit後mandatory STOP、push0。
