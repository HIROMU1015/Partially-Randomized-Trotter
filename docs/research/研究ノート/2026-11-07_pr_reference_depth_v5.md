# 2026-11-07 新A V5：全18round depth

基準224440f740051ed5794e8db87a57908fc02dd02f。V5-A〜E限定scopeを完了した。[REPORT](../pr_reference_depth_v5_20261107/REPORT.md)、[SOURCE・結果・STOP](../../../artifacts/pr_reference_depth_v5/2026-11-07/README.md)。

B′の保護D/frame依存鎖下界とreference2 laneの期待max上界を用い、全18round shot加重native DAG depthはreference上界<B′下界、削減率≥40.995952%。正確な期待depth・最適depth・実機時間・全資源優越ではない。旧CX/RZ/回転分類・shot・誤差を不変更とし、追加8qubit・総T未取得を維持する。

保存referenceには8 resetが存在する。新コードの最初のmetadata仮定を修正し、初期Hとの並列性を人工検査した後のQPY照合・会計を採用した。5284 context-record・268 fence・256価格・8 QPY・人工13検査。科学対象の新build/compile/sampling/simulation/T合成0。旧SOURCE/raw/結果/STOPを保全してmandatory STOP。次の科学判断はGPT／ユーザー。
