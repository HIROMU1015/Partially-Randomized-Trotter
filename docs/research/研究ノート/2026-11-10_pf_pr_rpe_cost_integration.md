# PF/PRエネルギー推定総費用の統合（2026-11-10 JST）

利用者の新方針により、H6 m8..12を逐次追加する優先度を下げ、N/C・全RPE会計をまとめる構造を先に実装した。[REPORT](../pf_pr_rpe_cost_integration_20261110/REPORT.md)。旧H6 branch14c4f871fは維持し、新しい統合branchで科学SOURCE・保存N/C・STOPを保全する。

既存のsupplied-axis積和・比較、H6 D2 action、m5 schedule/canonical IDを再利用。共通record/binding、静的保存adapter、欠測batch、H6 context/reference/cache共有、全mean-price covarianceと供給済み同時区間を接続。人工/mock12件PASS。固定資料内のH6 N16候補/m0..7とH4 N2候補/m3を再計算せず読取。H4 m4の同candidate保存signalとPF直接Cも確認した。実新signal/N/C予測・実N×C/G・fit/sampling/build/compile/詳細科学レビュー0。

H6残5時点・10候補を一括N範囲として提案し、H4は保存m4 scalar2と未取得11時点22候補を分けた。H4 fresh N leafは未接続で、既存backendの接続・人工資格が必要。A/Cの新価格SOURCE/domain資格・共有不確かさと全G評価は未完了。旧H4 V2 S1/H6 V1を全RPE providerへ自動採用しない。現在の方針承認を科学実行許可に読み替えず、公開照合後mandatory STOP。
