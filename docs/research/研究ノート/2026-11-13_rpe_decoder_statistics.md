# 2026-11-13：decoder shot統計の担当分離と技術準備

基点B3216d27a6、専用branch codex/rpe-decoder-statistics-20261113。ユーザーの役割分担に従い、A=物理mean/normalization/局所誤差、B=shot統計/decoder/branch-alias/全loss、C/D=compiled価格/残差とした。旧統合branchと科学SOURCEを保全する。

確認済み（技術）：元decoderはpriorを入力せず、zero/tieにはenergyを返さない。新count adapterは元wrapperを直接呼び、未解決lossを勝手に決めない。条件付きRMSE・解決分の全trial寄与・全risk interval・finite-MC CI・十分Nを分離。count-gridとexact整数和ratioの丸め経路差も記録し、科学的count callerは未採択。新16＋元decoder1＋旧V2 106＝123技術test PASS、人工解析分布と一致。

実H4 DF10の統計simulation/保存mean再解析は0、全13meanを統計へ渡していない。nominal64/60/38と残10nullをmetadataだけcopy。D e7dca5fの73/0/63を既存受入へ追加し、旧56/17/63と値/sourceを保全。136候補のmenu採択・実Gなし。

未確認（科学）：未解決trialのloss、joint law、aliasとpriorの正式契約、物理mean誤差のdecoder lossへの移送、十分N/global RMSE。次の[有限計画](../rpe_decoder_statistics_20261113/SCIENTIFIC_PLAN.md)はloss support/精度/confidenceから反復数を設計し、固定trial数を根拠なく決めない。ユーザーが条件とscopeを判断・別許可した後にBが実行する。今回公開SHA/tree/artifact照合後mandatory STOP。
