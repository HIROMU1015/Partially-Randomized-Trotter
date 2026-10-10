# 2026-10-11 Track A H6精度一致資源比較 v1の実行準備

利用者がH6 pilot後GPTレビューの採用を確認し、同じ方向の再レビューを挟まず契約・実装・synthetic検証・固定公開まで一括準備するよう指示した。追加科学計算のlaunch認可とは別である。

[実行契約](../track_a_h6_matched_accuracy_execution_contract_v1.md)に、同じH6/rank19/sector400/T0.8/保存state、92有限signal候補、4精度、全適格候補の費用探索、精度別B2上位2の独立確認、empirical u、shot-inclusive RZ、paired統計とrare-order限界を固定した。B0共通prefix5/10/15、B1 S2/S4、B2 q/r/Kを独立に探索する。B3168 scalar normalization診断はfamily除外を意味しない。

旧pilot32/36 wrapper・原STOP・欠測・u/N/G状態を変更しない。旧標本をfresh meanへpoolしない。native lowering/sector/reference/finite law/shot会計を再利用し、コンパクトstage、prefix/spectral cache、2 disposable cost processを新sourceで準備した。phase/total wall capはnull。資源guardとmandatory STOPを維持し、H8は未認可。

検証はtoy、injected compile/cost、dummy子processのみ。real保存NPZをdecodeせず、new molecule signal/sampling/compileを実行していない。テスト・source/input/env sealとremote照合は[準備資料](../../../artifacts/resource_applicability/track_a_h6_matched_preparation_v1/2026-10-11/preparation_audit_v1.json)へ保存する。未公開の旧dirtyを一括stageしない。

準備後は `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION` で停止。次は固定v1全run一回に対する具体的な明示認可。主要GPT科学レビューは結果map/確認後、H8検証設計前に戻す。
