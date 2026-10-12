# N3幅比較v2の固定pilot：2026-10-13

利用者「次の作業に進んで大丈夫」により、直前の認可待ちだった固定6入力pilot一回を認可。
[認可記録](../n3_width_comparison_v2_execution_authorization_20261013.md)にsource/input/plan/auth SHA、scope/capsを結合した。
source13 blobを維持し、CPU32/available約51GiB、2workers/thread1の凍結配置を適用。
既存STOP履歴・他Track・root差分を保全。attempt1/retry0、全結果・監査公開後のSTOPを維持する。

## 一回結果と保存監査の追記

全6H/504 rows完了、wall71.111秒、sampled aggregate RSS約329.9MiB、budget violation0。
独立saved audit1224 certificate・36反証record PASS。NORM/OCC/同等SDP/SOSは6/6H・12/12modeで目標.01を受理し、追加H縮約0。
OCCの有限H幅はcheap ORより5/6で狭く、同等SDPより全6Hで狭い幅と短いcold medianを得た。
この有限費用と幅の価値をCodexで中心テーマ採択へ変換せず、[一次結果](../n3_width_comparison_v2_results_20261013.md)・[GPT引継ぎ](../n3_width_comparison_v2_results_review_handoff_20261013.md)を保存した。
SCS上限20000iteration到達18call（optimal_inaccurate）も記録し、有理証拠チェックを通した。追加solver/入力/再試行0。
認可消費済み・Mandatory STOP。GitHub公開/再取得後にGPT科学判断を待つ。rootのdocs/progress/README.mdに外部編集を観測し巻き込まず保持した。
