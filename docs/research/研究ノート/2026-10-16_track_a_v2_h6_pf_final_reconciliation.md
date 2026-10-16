# 2026-10-16：H6探索PFを最後に限定照合し、現行V2残差設計をNO_GO候補へ

利用者の指示で、旧H6探索PF20候補（測定16・未測定4）だけを追加調査した。15測定候補はpackage/version宣言までで固定profile receiptへの結合がない。full S4/q8の1候補は旧profileへ結合できたが、cost taskの全prepared basisは独立に記録されていない。共通native codeとarchiveの「deterministic preparation unchanged」宣言だけで比較可能と認定しなかった。

比較可能PFは0で、共通fitの必要件数を満たさない。profile確認済み1候補もtrain側にあり、仮にbasisの宣言を採用してもtrain1/holdout0で不足する。分割・閾値・最小件数を変更せず、広域回収を終了した。

現契約の実行可否を`NO_GO_CANDIDATE_CURRENT_RESIDUAL_DESIGN`として整理し、別設計を採る価値の判断を利用者へ残す。PR・確認・H8・B0I台帳と旧config/grant、V1/H5/H7は保全。新fit・残差・compile・新grantは0。[報告書](../track_a_v2_h6_pf_final_reconciliation_20261016.md)と[判定表](../../../artifacts/resource_applicability/track_a_v2_h6_pf_final_reconciliation/2026-10-16/README.md)を公開し、remote照合後mandatory STOPする。
