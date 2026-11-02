# 2026-11-02 P3の区切りとH6純RTE対Hybridの読取調査

ユーザー指示に沿い、P3のPauli–DF matched比較を固定H4の条件付きcase studyとして一区切りとし、追加のnative/compiled保証は保留した。純RTE研究を継続し、元のDF B2・決定論的PF・RPE全roundの目的も維持する。

[調査書](../h6_pure_rte_hybrid_implementation_evidence_inventory_20261102.md)でH6純RTE q8/r64/K2対Hybrid prefix78/q8/r32/K2、同純RTE対Hybrid q8/r64/K2を対応付けた。既存H6 targetは53-bit実効lawとbinary64角度のtight L2平均を含むため、P3の有限bit評価を再実装することを優先しない。
P1はbias原因の限定診断、P2は固定input意味論・個別価格の証拠であり、literal native/compiled作用誤差や全lawρの保証ではない。保存ρ_guardと価格SE/希少event coverageを引用し、実装精度と価格tailが異なる主張を制限することを整理した。どちらが実際の順位を変えるかの数値感度は新規評価していない。

既存P2のselected_members/recover_memberを再利用して固定21member・12 QPY raw/gzip bindingsを確認した。QPY deserialize・状態作用、sampling/compile、新誤差・上界・感度・N/G計算、人工数値testsは0。保存3候補×2軸・最大15状態作用の局所診断案は未実行で、今後の明示実行許可の対象である。
次はユーザーが報告の利用範囲と局所診断の実施要否を判断する。許可後の最小実装・人工検査・SOURCE固定・限定実行担当はCodex。旧SOURCE/raw/結果/全STOPは変更しない。
