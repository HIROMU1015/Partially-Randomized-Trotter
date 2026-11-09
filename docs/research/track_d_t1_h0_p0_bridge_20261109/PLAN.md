# T1〜H0一括技術検証：固定scope

2026-11-09 JST。利用者の一括指示に従い、T0 c583abbaを基点に同じリソース評価枝を継続する。既存SOURCE・STOPを変更せず新namespaceを使う。新grant／承認証明は不要。

1. 固定C7-2 709d6ed9の54 caseを全件列挙し、native取得可能なraw/compiled全列をT0で検査する。最初の3 caseはcx_adjacent／cx_commuting_control_rz／cx_noncommuting_target_rz。native不適格は欠測分類し、別fixtureを生成しない。
2. Qiskit1.3のinverse-cancellation SOURCEと保存証拠に基づき、同方向CXが両wire frontierで隣接する場合だけ消す最小bridgeを実装する。RZ可換移動・1q再合成・Gaussian合成・P0全体の再実装はしない。chunk不変性と可変履歴量を小人工wordで検証する。
3. 既存54 traceには全体／二chunkの3出力を同じ固定native wordで比較する記録がないため、連鎖界面を切り分ける指定3qubit列 CX01,CX12 | CX12,CX01 のfullと各halfを各一回、計3 P0 compileする。既存Qiskit1.3／opt1／seed17／rz,sx,x,cx／approx1／initially_zero=False／topology-free／MPFR160位相保持経路を使う。小8×8 Operatorで位相付きcorrectnessを確認し、component abs1e−10・rtol0を緩和しない。H4 simulationはしない。
4. 保存H4はC6a q4_sample00、C8-1 q64_sample00の2 path、C1追加融合OFFに固定し、Strong/Reference X/Yの計8 compiled QPYをread-only計数する。元archive／保存hash／価格／compilerのidentityを先に照合する。rawから取得できるnative連続segmentには同じbridgeを適用し、未展開gateを越えた全P0予測とはしない。取得不能は当該HOLD、他pathへの置換・再生成なし。
5. compiled列の再計数、pre-P0 native部分の局所変換、全P0への移送未資格を別に記録する。主A・比較F/S・N・全RPE会計は更新しない。

出力はartifacts/track_d_t1_h0_p0_bridge/2026-11-09/run01。科学実行前にrunner/bridge/PLANをcommitし、実行SOURCEと実際のinput hashを同namespaceへ記録する。原本QPY/NPZ/eventは追加公開せず、人工3 P0のraw/compiledもlocalに保全し、公開はscalar/hash/比較/文書のみとする。

SOURCE/compiler/input/correctnessの不整合は依存実行を止め、partial/STOPを保全する。通常技術バグはscope内修正可能だが、P0の追加call、閾値緩和、fixture変更、H4再compile／state-action、追加path、q≥128、sampling、fit、N／13-round、詳細科学レビューへ進まない。公開SHA/tree/bytes照合後mandatory STOPし、ユーザー/GPTへ判断を返す。
