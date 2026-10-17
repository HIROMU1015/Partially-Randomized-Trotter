# H8 Phase C 一回完了

[結果・監査](../track_a_h8_phase_c_run01_results_20261017.md)。

SOURCE931755c4、公開REVIEWb357c77a、plan79d624b9、grant21775ba6。利用者承認に基づき通常signed biasの1900行を一回再会計。driver/launcher native exit0、全回収、gate PASS。本体0.355秒、new signal/compile/sampling/fit/retry0。

旧32のPR削減率（倍率1）は28.77/52.72/52.73/56.12/23.83%、184基準は−7.14/33.09/31.31/22.96/−51.14%。184最良は全条件モデル推定で、.05はprefix11外挿、.0005は未測定S4。PRは確認n32有限標本平均、未解決/欠測0だが全格子実測・厳密bias・母平均保証なし。

この条件付き変化はユーザーの科学判断へ戻す。旧結果・STOP・他worktreeを保持し、次stage/追加科学レビューは開始せずmandatory STOP。
