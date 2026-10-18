# Track A Mechanism：Figure 2統合準備とA82限定回収

2026-10-19 JST。**保存値の論文統合準備とA82原signalの限定回収が完了。新数値再解析0、mandatory STOP。**

独立branch `track-a-mechanism-paper-integration-20261019`を説明結果0edd62403c35c5ee18b2a2a152158faca5c2ebd8から作成した。
coreは2bf5d83293b3a47f672805fae956ff2e50df2878、H4 Dは4e8ad1936363266d1b55bb689c36846e6c3dfcdb、論文契約は0a8ac1d2d03c5a9aef2acab0295226f8c88b38ce、claim matrixはbf8c19057fd82c900c6234152e152e79cb52e707へ固定した。
共通／worktree AGENTS、研究概要全体と関連契約・SOURCE・結果・STOPを確認した。
既存科学SOURCE、元結果、使用済みgrant/ledger/marker/STOPは変更していない。

## 成果物

|成果物|今回保存した内容|
|---|---|
|[A82回収報告](A82_signal_recovery_report.md)|D0 receiptから限定的に原signalへ到達。全byte/SHA、candidate、SOURCE、DF/T、旧n32のmetadata同一性|
|[新binding metadata案](A82_recovered_input_binding_projection_DRAFT.json)／[再解析契約案](A82_reanalysis_contract_DRAFT.md)|回収2 fieldを別入力案へ転記。未承認・未実行|
|[図表manifest](data_manifest.json)／[Figure 2仕様](figure2_table_specification.md)|32比較の費用・比率・SE、64軸exact分解、31比較の62軸説明因子を保存文字列で転記|
|[主表CSV](data/main_table.csv)|旧H4_stage、元core説明status、現説明status、新D provenanceを別列に保持|
|[Claim–Evidence更新案](claim_evidence_matrix_update_DRAFT.md)|旧6 claimを保全した追補案、H4 D／A82回収の2項目を追加|
|[論文evidence readiness](paper_evidence_readiness.md)|H4 D完了・PF代表10/10不変・P未完了。最終freezeなし|
|[技術検査](technical_validation_report.md)|人工26 PASS、2群の人工描画。科学値の表示図は未生成|
|[次承認一覧](next_approval_list.json)|担当と別承認の単位を明示|

A82は`/signals/32`に存在し、normalization／signed biasを回収できた。
回収後のG・headroom・shot/log/ceil・因子計算は0なので、既存結果は引き続き63 side完了／1 side欠測、31比較完了／1比較欠測。
Figure 2はこの保存statusをそのまま使い、回収済みというmetadataを別注記にする。

H8独立n48の2比較は探索n8と分離して主要実証例へ配置する仕様とした。
保存された2.935924%／3.302597%の点削減率は母平均優位を証明しない。高次event未観測の制約を維持する。
H6のPFは.05/.0005が直接実測、.01/.005/.001がmodelという区分を全行に保持した。

H4 Dの保存PF代表は通常signed5＋経験的非相殺5の10/10条件で不変。
旧32比較のH4数値・subset16・H4_PRE_D_P_VALIDATIONを改訂せず、D_COMPLETE_P_PENDINGを別provenanceとして付した。
独立確認Pは未完了。H4 S_STATS/Pのworktree・結果・grant・jobには干渉していない。

## 保全・未解決・承認境界

保存数値を文字列で転記し、同名fieldの等値を確認した。新しい比・統計量・順位、保存値の数値再解析、sampling、
signal、compile、追加212比較、実結果描画、最終evidence freeze、公開pushは0。
empirical allowance、有限標本／選択効果、model残存、H8原profile bytes等の旧制約を継承する。
母平均優位性、rigorous error、FT資源削減へ主張を拡張していない。

原signal・raw field・seedを含むselection・入力mapping・完全SHA一覧はlocal専用監査領域へ保全した。
repositoryには保存科学値の図表、出典、仕様、回収metadata、契約案を置く。公開pushは行わない。

**mandatory STOP。** 次担当はユーザー：A82追加検証、Figure 2仕様／claim採択と実図描画、local commit公開の要否を判断する。
希望があればCodexが別指示・別実行承認の範囲で進める。H4 Pと最終evidence freezeは未完了として保持する。
