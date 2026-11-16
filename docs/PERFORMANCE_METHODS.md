# 並列化・高速化手法の再利用入口

既存の科学計算用の主索引は、`codex/d-full-h-p2-caller-prep-20261114` の同pathを参照する。
2026-11-16読取基点は`481c4b7abffbf070347f697980f2ec051e6a6538`。利用時は対象・固定SOURCE・条件を照合する。

## PR-Pauliの必要時準備

[Assets / prepare](../src/trottertracks/pauli_candidate_preparation.py)は、固定Git成果の辞書・基準構成を
再利用し、B2 generatorをprocess内でcache、保存信号のphysical-key indexをrequest batchで一度だけ作る。
辞書cacheはH/state/snapshot・分割・辞書規則に対応し、κ/幅の変更でP0を再実行しない。
local metadata構成と公開fixed locatorを区別する。

[受入記録](research/pauli_candidate_preparation_20261116/REPORT.md)は保存JSONと人工系で確認。
新H4の辞書生成やsignal科学計算での速度倍率は未測定。stage1の一時metadata差替えはprocess-localなので、
独立splitを並列化する場合は別processと個別出力/cacheを使う。
P0 commit後のHEAD進行でcache再利用を妨げる不具合は、新科学SOURCE確認をleaf内へ移して修正し、
実materializer factoryとfresh-entry cacheの人工回帰で確認した。
