# 2026-11-11：A P1を受けた長q・大R価格検証計画

[REPORT](../r_price_longq_largeR_preparation_20261111/REPORT.md)／[PLAN](../r_price_longq_largeR_preparation_20261111/EXECUTION_PLAN.json)。C1公開2e6a0cdf5を基点に既存C branchを継続。A公開71ba49bbの全12件の同policy条件付きNを整数copyし、元候補identity/statusを静的照合した。Nの再解析・比較比率・Gは計算しない。更新されたユーザー指示に従い、N取得済み中時間候補に対応するCの計画を先に置いた。

第一案P1_DIRECTはH4 m6 q64/R4,R128とH6 m8 q16/R16,R128の4セル、Q baselineは各n4・QRは各n6、最大40 full wrapper。誤差原因分離が必要な場合だけC1低qでR128を取るR-only別16を検討し、SHORTも別任意16とする。nは有限pilotの予算案で、精度やpowerの保証ではない。R256や巨大m12を自動対象にしない。C1係数と校正/独立検証役割を保持し、新fitは提案に含めない。

Linear H4/H6/1Å/STO-3G・保存DF10/DF19、H4 S2/δ.1/p1/K2、H6 m5_best/δ1.6/p7/K2/11signed piece・元integer allocation、元compiler/full wrapper/P0を保持。NativeCounter/Accounting、2/3-event connected increments、window多重度、cache/並列を方法として再利用するが、別DF/制御の係数を移植しない。現C1 backendはSのcounter自体が小メモリでも全eventsを保持するためO(E) memory。streamingや局所境界の閉包は今後の技術/有限科学確認が必要で、巨大価格を取得済とは扱わない。

metadata専用生成器SOURCE f0daeb96bと8テストPASS。B724dの共通pipelineと同blob、B menu/requestとC descriptorを再利用し、BのC1受入やA/B作業領域を変更していない。今回はsampling/build/compile/S計数/fit/実価格予測/残差再解析/signal/N/G/詳細科学レビュー0、追加rare調査・設計・実験0。C1 raw/固定SOURCE/係数/旧STOPは保全。

公開・remote照合後mandatory STOP。ユーザーが次の有限package・pilot予算を判断し、Cがbounded runner/SOURCEを技術固定後、具体的な科学実行許可範囲だけを進める。Bは自身のscopeでN/C取込みと将来の感度評価の準備を担当する。
