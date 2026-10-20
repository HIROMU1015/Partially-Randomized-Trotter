# 2026-10-20 H6 PR pilot：cutoff修正と軽量高速化

[報告書](../track_a_h6_pr_pilot_cutoff_repair_20261020.md)と[公開bundle](../../../artifacts/resource_applicability/track_a_h6_pr_pilot_cutoff_repair/2026-10-20/README.md)を追加した。

旧H6 run01の74/396完了・STOPを保全する。停止片は同じλでもτの演算順序差で一方の残差が1ULPの余裕を超えることを、保存metadataとfactoryのscalar式で再現した。新SOURCE `e7595e8ff543a0d0f5e1ea91dec14283b61f9c32` は演算順序を統一し、H6-only opt-inで親context共有、fresh並列数、SHA照合付きJSON cacheを追加した。遅かったcall予算cache案は採用しなかった。

新規16件・既存回帰161件を作者自身が検査した。人工例と既存結果のbyte/metadataを対象とし、新しい実H6 sampling/compile/state作用は0。保存74本の再利用と固定未完了322登録の取得は提案であり、受入・継続専用adapterと実行承認は未完了。旧grant/ledger/marker/STOPは変更しない。

次の担当はユーザーの再利用・再実行判断、その後のCodexによる専用契約結合。統計評価・n_E・shot・会計・AD03へ自動進行しない。local commit後mandatory STOP、pushなし。
