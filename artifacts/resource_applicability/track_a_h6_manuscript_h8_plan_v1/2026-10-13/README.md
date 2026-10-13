# H6原稿統合・H4共通panel・H8計画：資料入口

2026-10-13 JST。保存結果を用いた原稿と実験設計。新科学計算/trajectory/transpile/input生成/grantは0、共有環境・既存venv・H4 runの変更0。

- [H6本文v0.2](../../../../docs/manuscripts/track_a_h6_resource_study_v0_2.md)：探索592、独立確認256、条件付きpilot8を分離。事前K4と確認後点選択、54–63%点RZ削減、数値allowance・母平均tail・state preparationの限界。
- [H4/H6採用と最小補完panel](../../../../docs/research/track_a_h4_h6_common_comparison_plan_v1.md)：旧M1/PM1/PM2/M2/AX2B/稼働geometryごとの採用範囲。共通16構成528compile案は旧H4全再計算ではなく不足情報の取得案。
- [H8の科学・高速化・事前検証・予算案](../../../../docs/research/track_a_h8_resource_comparison_plan_v1.md)：比較generator、sparse/matrix-free依存、hash/preparation再利用、最大8workerとheavy weighted admission。
- [完成済みGPTレビュー原文](../../../../docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md)：利用者の外部fileをbyte同一コピー。再科学レビューを行っていない。
- [claimと証拠対応](../../../../docs/manuscripts/track_a_h6_claim_evidence_v0_2.md)。
- [metadata入力SHA](input_metadata_receipt.json)、[表示4精度CSV](h6_precision_results.csv)、[事前8panel CSV](h6_fixed_panel.csv)、[pilot局所更新CSV](h6_pilot_effect.csv)、[pilot層CSV](h6_pilot_strata.csv)。
- SVG：[精度と選択](FigH6_1_precision.svg)、[shotと回路](FigH6_2_cost_factors.svg)、[pilot](FigH6_3_pilot.svg)。標準Matplotlibから作成したexport可能な図で、science計算図の新規測定ではない。
- [H8 machine-readable draft](h8_plan_draft.json)、[古典時間の保存anchorsと仮定付き見積り](classical_planning_anchors.json)、[local検証](verification.json)。
- [公開後remote照合](publication_receipt.json)はGitHub公開後に追加する。

## 固定identityと検証scope

base e0e6473d7047158a7f0f788d026a7a89a01898b8、科学SOURCE5a903aea2492c2d211c1e15e7737df0d50694c12、確認SOURCE87b6b9f46b7e8ba9abe132ee21478c114b6f46f7、pilot SOURCE843265b624eac8fbb9acea47ef21cab1207e1880。科学224を含む旧runtime closure257はbyte不変を検査する。原稿/計画commitとscience SOURCEを混同しない。

local検証は公開CSVの32paired標本からのG/SE独立再計算、選択union8、pilotの既知確率による点更新、2SE内のK順位差、budgetの独立列挙、carry/予約、weighted admission mock、memory算術、linkと入力SHA。H8回路・性能・signalの動作確認、厳密母平均bound、独立再現/CIのPASSではない。

stdlibで表示CSV/JSONを再生成：`python3 -B build_evidence.py`。検証：`python3 -B verify_evidence.py`。図は既存Matplotlibを用い、thread1・home配下MPLCONFIGDIR/TMPDIRを指定して`python -I -P -B render_figures.py`。install/upgrade不要。旧NPZをdecodeせず、公開metadataだけを読む。

## 未決事項と開始許可

利用者の「準備ができたら開始してよい」は受領済み。対象（H4補完、H8事前検証、本格H8）を確認中。H8の科学比較はまだdraft、実input/rank/source/profile未固定、primitive/reference予算未確定、技術sealなし、実行commandなし。したがって現在は準備完成や実行開始を宣言しない。

GPTへの次の判断はH8 primaryをresource-mapか固定transferにするか、prefix/q16 tier、state/reference/u、tail estimator範囲、H4補完panelの必要性。実装上の小修正はCodexで完結する。計算を行うときは科学条件を維持したhash/preparation再利用とgroup並列化を先に組み込み、heavy実測からworker数を決める。

known compile1716/trajectory842＋予約20/10＝1736/852を保持。core追加1788/872はcarry込み3524/1724案、occurrenceは7792案。input/validation/partial予約・transfer追加は別。予算・CPU/RAMの提案を実行grantと混同しない。
