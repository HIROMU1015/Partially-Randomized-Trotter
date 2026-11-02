# V2 H6探索PF：最終限定照合・NO_GO候補

[報告書](../../../../docs/research/track_a_v2_h6_pf_final_reconciliation_20261016.md)。旧PF20候補（測定済み16・未測定4）だけを追加照合した。比較可能PFは0、fit適格stratumは0。profile結合済みの1候補もtrain側のみで、basisの独立結合は未完了。現行残差設計をNO_GO候補として整理し、広域回収を停止した。

|ファイル|内容|
|---|---|
|[summary.json](summary.json)|限定scope、必要条件の上限、更新private台帳SHA、実行0、利用者判断・STOP|
|[PF_entry_audit.json](PF_entry_audit.json)|20候補のprofile/preparation/native構成と未解決理由。費用値なし|
|[source_receipts.json](source_receipts.json)|Git blobのSOURCE/path/bytes/SHA。code一致を全basis同値へ昇格させない|
|[audit_saved_h6_pf.py](audit_saved_h6_pf.py)|実行したstdlib監査手順。新規exclusive metadata出力だけを作る|
|[technical_validation.json](technical_validation.json)|141入力・旧667保護ファイル・SOURCE/worktree保全、scope guard|
|[publication_manifest.json](publication_manifest.json)|公開ファイルのbytes/SHA台帳（自己参照を除く）|

PR/確認/H8/B0Iの台帳entry、旧config/grant、既存科学結果は不変。fit・残差・新compile・新grant0。公開後mandatory STOP。別設計の実装・実行はこの成果に含めない。
