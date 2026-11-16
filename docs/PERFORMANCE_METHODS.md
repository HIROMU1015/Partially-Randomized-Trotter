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

## PR-Pauli較正要求の接続

[候補固有request](../src/trottertracks/pauli_calibration_request.py)は固定driver moduleをprocess内で再利用し、
既存価格・較正cellをSOURCE付きで探して重複取得を避ける。取得coreは既存C2の段階log・2 path並列・
event-matrix memo/同値fingerprintを使用する[split driver](../scripts/run_pauli_split_cal.py)へ接続する。
新requestではspecをJOB/workerへ運び、SPECS限定経路へ戻らないことを人工確認。
[検収](research/pauli_calibration_request_20261116/REPORT.md)は保存metadata・人工fixtureのみ。
新H4 pathの速度倍率・新runtimeでの取得性能は未測定。旧sourceのBLAS設定は旧経路だけ保持し、
新requestはjob環境を継承する。SOURCE/input/gate/STOPは保持する。

[複数実step拡張](research/pauli_multistep_calibration_request_20261116/REPORT.md)はcanonicalのexact実stepでroundをまとめ、複数候補間では同じ物理request・law・cohortのcellだけを共有する。異なるTの同名Q8R1は物理hashで出力先を分ける。旧LD3/LD5 metadata再現、通常CLIとbatch→driver/worker入口を人工検査。取得性能の速度倍率は未測定。同stepのgroupingは後続audit・価格model適用資格の代用ではない。

## P0係数追跡の保存入力・局所JW再利用

[係数trace](../src/trottertracks/pauli_p0_factor_trace.py)は既存M1関数とJW bytecodeを局所globalsで再利用し、library globalを変更しない。保存operator3件とfactor復元1回を使い回し、LD9→8→7を順に追う。SOURCE655e74212の実H4診断全体5.38秒、人工12件で順序・SOURCE・部分STOPを確認。[結果/適用条件](research/pauli_p0_factor_trace_results_20261116/REPORT.md)。他LDや本番辞書での性能・資格は未検証。
