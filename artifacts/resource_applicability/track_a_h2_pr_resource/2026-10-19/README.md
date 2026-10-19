# H2専用の準備・結果root

隔離準備のmetadataだけを保存する。H2科学結果は未取得。

将来の各stage/attemptでは、このrootの下にH2固有の公開用要約・証拠を置く。raw、入力snapshot、runtime、承認receipt、grant、ledger、one-shot markerはH2私有rootで別々に管理し、公開範囲を確認する。新しい科学outputは承認前のsetupで事前作成しない。

引継ぎ：`docs/research/track_a_h2_pr_resource_handoff_20261019.md`。
