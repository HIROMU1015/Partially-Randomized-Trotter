# 2026-10-20 H4/H6 outer32標本設計pilot準備

基点`489422d38bad4d5892cf98c580d93cc4b602bf0c`から、PR24群の同law設計pilotを別branchで準備した。[報告書](../track_a_h46_pr_design_pilot_readiness_20261020.md)を参照。

正規IIDの分散精度を名目参照に、33本/群（相対SE25%）を推奨案、9本/群（50%）を粗い案として提示。いずれも未採択。希少層を観測しきる保証や母分散/母平均保証ではない。n_Eは全群null。

元adapter/launcherを再利用し、DESIGN_PILOTの独立cohort・契約、law記録とstratum、固定n・非混合・台帳・STOPを接続した。封印準備で原signal config結合と全scratch予算gateを追加し、前のDRAFTを保持して最終SOURCEの無効DRAFT2契約を保存した。科学実行は0件。

新20件・回帰49件を検証。nativeは合成4-mode入力だけで、H4/H6配列を復元しない。既存SOURCE/raw/grant/ledger/marker/STOP、AD03/shot政策保留を保持した。次はユーザーがn_Pと資源予算・系別実行の承認を判断する。pilot終了後の数値評価と本探索は別承認、mandatory STOP。
