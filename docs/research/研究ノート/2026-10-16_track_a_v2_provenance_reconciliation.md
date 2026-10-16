# 2026-10-16：V2来歴証拠の回収と固定holdout件数の不足

利用者の指示に従い、保存receipt・compile経路をprimary agentで読取専用照合した。PR158 entriesのseed来歴を回収した。H6探索17 PR候補と確認8 PR候補は、全paired groupsの固定archive・旧profile・保存law記述へ結合できたため、条件付きmetadata結合済みへ更新した。237 entriesは未解決を保持した。

来歴が結合できた25件にPFはなく、共通fitの必要件数を満たさない。また来歴を無視した件数上限でも、H8探索72/確認6とH6確認8のPR holdoutは0だった。分割条件を変えてfit対象を作る処理は実施していない。H6探索だけは全測定候補の件数上限を満たすが、旧PF basis/profileとPR mixed lineageの証拠が残る。

未測定B0のorigin16件、未測定B0Iのwrapper7件を実行SOURCE・新fingerprintで補わない。更新台帳を新private出力へ保存し、旧registry/config/grantは不変。新科学計算・fit・残差・compile・law正規化・新grantは0。[報告書](../track_a_v2_provenance_reconciliation_20261016.md)と[機械可読証拠](../../../artifacts/resource_applicability/track_a_v2_provenance_reconciliation/2026-10-16/README.md)を公開し、remote照合後mandatory STOPする。
