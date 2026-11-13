# 2026-11-13：PF power安定性の人工調査

ユーザーの「その方針で進めてよい」を、直前の人工入力による数値安定性調査の許可として実施した。実H4追加診断の承認へ拡張していない。基点699178305・同A branchを継続し、元科学SOURCE256caad34、run01/run02、B/C/Dの作業を保全した。[REPORT](../pf_power_stability_artificial_20261113/REPORT.md)を参照。

確認済み：既存spectrumは再構成を調べるがGram defectを調べない。既存`algorithm_codesign.numerics.Spectrum` observerとsigned schedule/powerを再利用し、36次元・11人工block・固定seed2種の192条件を比較した。元EVR7/48、EVD4/48、EV1/48でunitarity gateを失敗し、直接expmは48/48通過。手動注入のzero-eigenspace対照でも再構成0からpower失敗を作れたが、これは実solverの出力ではない。

`python -B tests/check_pf_power_stability_artificial.py -v`の新8件、既存`check_h4_four_method_screen.py`の9件、計17技術test PASS。NumPy1.26.4/SciPy1.14.1。旧依存SOURCEのinvalid-escape warningは保持。実H4の状態・固有値・signalを用いる新計算や保存値再解析は0。

未確認：run02の実原因と直接expmの実H4資格。driver変更だけを解決と呼ばず、production・gate1e−9は不変。次はユーザーが代表F2-COARSE/m12の二経路・最大2 mean・新reference0の診断可否を決定し、Aが承認後のみ実行する案。公開・remote照合後mandatory STOP。
