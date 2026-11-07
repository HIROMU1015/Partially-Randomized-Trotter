# Track C：same-basis runの共同isometry

基点10bf5a4813559cea807fcd8c50f3b966407747d1。Linear H4/1Å/STO-3G/DF12、Partial LD3/Full LD12、保存元H exact ground、18round/δ0=.02、energy1.5936e-4Ha、失敗上限.05、位相.4rad、状態準備除外、Track C compiler A/Qiskit1.3.0/seed17を保持する。旧SOURCE/raw/結果/grant/ledger/marker/STOPを変更しない。

元の最大same-basis run内で列要求の和集合Sを作り、既存stable partial QRで先頭|S|列を元basisのS列に対応させる。個々の列のprojector、Z/ZZ、積とrotation、controlled相対位相を保存する。列集合を回転して同じspanにするだけの置換は認めない。元のevent/sampling law/return aggregationは不変更。

前回holdout384標本を今回のtrainingと明示して再利用する。観測列集合と全singleton supportからcatalogueを固定し、native basis RZが小さいものを登録。未登録・高価な共同basisにはfull共有を使い、元full/singleton whole circuitもfallbackとして残す。個別partial化で共有を失うalways-partialの負結果を保全する。

まず保存trajectoryのrun長/列rank/頻度と実価格を分析し、full Fock作用素を確認する。有望なら局所native区間証拠・controlled/X/Y・modular加算性を確認し、新たな独立価格holdoutを行う。既存D調整係数と選択済みh/resolutionは維持し、cost-sensitive αと整数shotを再評価する。新旧価格は同じtrajectoryでpaired比較し、Fullにも既存の同等配分・basis機会を保持する。

CPU affinity/実空きRAMを観測し、独立workerを使う。共有設定・他ユーザーは変更しない。科学的に合理的な同一仮説の修正・検証まで連続実行し、成果commit/pushとremote SHA/tree/内容照合後mandatory STOP。別Track・分子・RTE法則変更・詳細GPT科学レビューへ進まない。
