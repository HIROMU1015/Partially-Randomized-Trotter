# N3保存帰属v1：GPT科学判断への引継ぎ

2026-10-13 JST。新しい数値事後結果が得られたため、研究上の判断をGPTへ戻す。**GPTレビュー開始は別認可・Mandatory STOP**。
[結果](n3_saved_attribution_v1_results_20261013.md)、[独立保存scalar/timer audit](../../artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1_audit/independent_saved_scalar_timing_audit.json)、[3証拠再認証](../../artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1/selected_existing_proof_recheck.json)を一組で扱う。
通常のsource/gate/testを再度詳細レビューする要請ではない。

## 今回追加された判断材料

固定6stress H、full5/6→active4/5、ν2/3、非負density、一候補一外部mode、候補2/H、元ε_proj=.01。geometry/basis/DF/L_D/PF窓は対象外。
216保存path/432event/696候補を事後解析。一回実行/retry0、新science/eig/solver/探索/入力0。
全候補でroot側が活性。12 same-modeのkernel反復0で同じU/cのOCC最終q・幅は同値SDPより狭い。しかし最終κと修復前情報の欠落により、改善原因の完全分離はできない。
SOS banded6/external5の選択候補でeが最終qの約98.9%。eは複合残差、raw/pre-repair/rは未保存で、solver品質だけの因果帰属は未確定。
保存phaseの費用内訳を得たが、同値SDPのstrict dual反証は追加出力で専用timerなし。同機能・同品質費用を新たに確定した結果ではない。
同modeのSOS正/理想OCC負と他modeOCC正の既存3証拠を厳密再認証した。同mode条件族分離を確認し、追加H縮約や物理的不能とは区別した。

## GPTへ返す科学判断

1. 未保存情報による識別限界を踏まえ、現prototypeの幅・費用結果をどの強さの研究claimへ限定するか。
2. 理論と厳密証拠・有限pool結果を残してN3を縮小/技術記録化するか、継続する場合はどの具体的な狭幅taskに価値があるか。
3. 次の同機能・同品質比較が必要なら、得る情報と研究判断の関係を定義するか。追加benchmarkを今回の認可の続きとして走らせない。
4. 既知Fantope/SOS/v2RDM/検証付き下界との違いに関する、既存文献確認範囲の下でのclaimを選ぶか。未取得原典全文の確認範囲を格上げしない。

原目標でNORM等も6/6、追加H縮約0。旧NO_ADDITIONAL_ACCEPTANCE_OBSERVED・旧2-base未達を保持。中心研究採択・最適性・量子総資源優位の結論なし。
新しい科学batch、solver tuning、機能/精度変更、新入力/ε、拡張、N1再開、N2探索は未認可。
全体manifestの既存description欠落FAILは未解消。個別保存audit PASS、公開/retrieval PASS、CI/外部再現は区別する。
