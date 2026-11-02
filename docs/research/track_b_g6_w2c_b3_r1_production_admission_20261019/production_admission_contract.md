# Production admissionのbindingとfail-closed契約

新SOURCEは`g6_w2c_b3_admission.py`。`ProductionAdmission.admit_once(driver_contract_digest)`は旧`Boundary.enter`のinterfaceに対応する。
科学を起動するCLI・env switch・実approvalの初期値はない。旧driverを変更していない。
`ArtificialAdmission`は同じ検査・SQLite engineを使うが`admit_once`を持たず、ARTIFICIAL claimで旧Boundaryのreceipt条件も満たさない。

## Binding

正式契約形式`b3_formal_admission_v1`のdigestは旧`D.identity`（exact canonical hex整数/有理形式）のSHA256。
ファイルbytesのSHA256とは別の座標である。正式契約はnested driver candidateを含み、driver digestと正式契約digestの両方を予約・receiptへ残す。
固定旧candidate原本SHAを照合し、入力・policy・全16cell/N・call budget・seed/setup/cache/保存/二pass・受入条件・one-shotを比較する。
R1/A6、正式採択記録のpath/完全SHA、採択元指示hashを結ぶ。科学SOURCE不変、driver/guard/admissionと全science pathをrole別に完全SHAで検査する。
保存入力はbytes/SHAのみ検査し、保存科学値を再解析しない。outputは専用root内・正規path・symlinkなし・新規を要求。
release40桁commit、F01統合済status、failure-provenance回帰hashが必須。既知の未修正driver SHAはproductionで明示拒否する。
本番では読み込み済みdriver/adapterの実pathも固定rootと照合する。coreにboolを渡すだけの承認pathはない。

`PinnedApproval`には候補JSONとは独立した`TrustAnchor`が必要。信頼された将来hostが、実在するユーザーの科学実行承認を独立に確認した後、
承認の参照と完全bytes hash、元ユーザーstatementの参照とhash、最終release commit、正式契約digest、固定ledger path/store IDを構成する。
承認documentの全bindingはscope/claim/output/adoption/source/input/digestに完全一致しなければならない。
予約rowにはさらに承認document自身のhash・statement hash・store IDを保存する。自己hash循環を作らない。
ファイルを編集して`approved=true`にしても、候補の自己申告、CLI flag、envだけでは独立pinを満たさない。

**TrustAnchorの信頼できる発行・hostへの配置は、今回実装済みengineの外側に残る将来の統合条件**。
Pythonコード・trust anchor自体を書き換えられる同一ownerの任意コードに対するsecurity sandboxではない。
署名済み外部承認サービスを実装したと主張しない。今回real anchor/approvalは存在せず、準備metadataからは生成できない。
本番ledgerも未作成。未来の実承認参照・release・契約digest・scope/claim/storeはnull。

## 順序とSTOP

全SOURCE/input/contract/authority検査 → output/path確認 → scope+claimのSQLite atomic reservation → 全bindingと予約rowの再照合 → receipt。
scienceはportの責務に含まれず、一切呼ばない。将来hostは`with port:`で既存`execute(..., approval_port=port)`を包む必要がある。
旧`execute`のguard.stopは外部portを通知しないため、このhost接続はF01統合時に固定・人工回帰する。
既存executeがcorrectnessを先行させ、不合格時は例外でfull-Nへ進まず、成功時のみ二passへ進む構造を保持する。
今回実correctnessへの移行は0。人工テストでは無承認executeがload_saved/load_modulesより前に拒否されることを確認した。

receipt後の障害でも予約を解除しない。context正常return・例外でSTOP、異常exitならRESERVED/ADMITTEDのspent rowが残る。
STOPログ・failure保存が失敗しても再取得を許さない。同一portの再入・fork後reuse・retry/resumeを拒否し、保存処理も自動retryしない。
公開candidateが未認可でも、独立anchorなしでpermissionを得られない。今回のproduction releaseはF01未統合でも閉じる。
