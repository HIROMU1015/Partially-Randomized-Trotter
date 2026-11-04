# B-1準備完了後の判断

2026-11-03 JST。**第一B-1aを実行できるadapterと人工検証・固定計画を準備した。科学実行は未承認・未実行。**

1. 新たに識別できる範囲は、保存A以降のtail構成算術をHPへ変更した場合のmean変化である。
   Task A保存factor固定meanと同じHP後段を使い、SOURCE replayのdriftも分ける。
   scalarとmatrixの個別原因、元float入力の表現誤差、定義差を一意に確定する検証ではない。
2. 保存A/F/R/tail/ψ/π/τ/BでB-1aは構成可能。θ64/cos64/sin64/normalizerは原本にないため、
   将来のSOURCE replayで初めて生成し、新しい値として記録する。
   保存component別係数・確率・basisを完全には取得できておらず、B-1bには不足する。
3. B-1bは現時点で準備・実行する必要はない。B-1aが収束しても差が残り、
   固定成分入力の取得と追加分離の合理性を示せた場合に別提案する。
   B-2も定義式からπ/B等を再生成する別候補に留める。
4. 元1e-10は数値診断閾値のまま。差を必ずこれ以下へ減らすことをRPE適合性の必要条件にしない。
   保存統計残余約8.60e-5に対し、真の周辺平均への誤差を予算内で評価する保証が科学判断の変更に必要。
   収束や数値一致だけではu_numを確立しない。
5. 元INCONSISTENT、u_num_bound=null、RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを維持する。
   B-1a/B-1b/B-2、他round、sampling、独立詳細科学レビューを開始せずmandatory STOPする。

次の担当は**ユーザー**：公開SOURCEの[第一B-1a計画](B1_SCIENTIFIC_EXECUTION_PLAN.md)について実行を明示許可する。
その許可後の担当は**Codex**：同じ固定input/条件でB-1a（必要時だけ差のあるreplay tailの基準、最大224 bit）を1回実行し、
新しい原結果・収束・差・限界を報告してSTOPする。承認証明ファイル等は不要。

成果物： [入力境界](B1_INPUT_BOUNDARIES.md)・[実装報告](B1_IMPLEMENTATION_REPORT.md)・
[実行計画](B1_SCIENTIFIC_EXECUTION_PLAN.md)・[人工検証記録](VALIDATION.txt)。
branch：`codex/rpe-m17-b1-construction-precision-preparation-20261103`。
実行SOURCEは本フォルダを含む公開commitで固定する（最終チャットにfull SHAを記載）。
