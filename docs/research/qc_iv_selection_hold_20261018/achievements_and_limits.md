# 保存する成果と限界

[正式方針](decision_record.md)／[再開条件](handoff.md)／[固定証拠](evidence_index.json)。以下は既存の技術調査・方法差検証の短い要約であり、科学判断・証明・SOURCE検証・文献調査を再実施していない。

| 保存する成果 | 固定記録の結論 | 限界・認定していないこと |
|---|---|---|
| A2後の有界な候補調査 | 停止済みのPF分割・A2・encoding・THRIFT等を再包装せず、DFと固定粒子数を使う共分散縮約の狭い問いを1件だけ条件付きで残した | 新アルゴリズムを構成・認定したものではなく、網羅的なpriority searchでもない |
| 既知式・SOURCEの対応 | Koizumiほか2606.30601v2 S8、著者SOURCE206aa441の1-/2-/mixed covariance、charge/incidence、符号・scalar・高次RDMを対応づけた | 全式・全class・著者コードの数値correctness、端点filling、実分子の保証を認証したものではない |
| 同情報の既知対照K* | DF factorization・標準縮約・stream/cache・同じstate情報を許すと同じ演算列・費用記述に達する。指定候補はREDUCES_TO_KNOWN_METHOD | 全fermion測定の最適性や、将来の異なる縮約規則・情報取得法の不存在を証明したものではない |
| 費用・情報の区別 | 算術演算量、live tensorのmemory、保存量、相関取得費を分離し、巨大な全共分散行列を強制する弱い対照を排除した | 最適縮約順序・tight bit complexity・実測speedupは未取得。高次相関・state準備を無料にしない |
| 結果解釈 | 同じ測定ensembleとestimatorの古典縮約を速くしても、測定分散・必要shots・量子gate数は自動的に改善しない | ground-energy推定の総資源改善・論文化GOを認定していない |

調査：[既存成果と停止理由](../qc_track_iv_question_survey_20261018/existing_research.md)、[候補比較](../qc_track_iv_question_survey_20261018/candidate_comparison.md)。方法差：[具体手順・費用比較](../qc_iv_covariance_method_audit_20261018/method_comparison.md)、[固定技術判定](../qc_iv_covariance_method_audit_20261018/technical_verdict.json)。過去の候補検討・技術判定を、今回の正式非採択と区別して保全する。

未検証のSOURCEや未完成の実装が残ることは、この候補を独立算法として再開する根拠ではない。engineeringとしての既知法実装の価値一般は否定しないが、今回その別目的の採択・準備も行わない。

既存のA2理論・認証SOURCE・固定DF pilotと負の結果、M1-Sや他研究の成果・STOP・消費済みclaimはそのまま保持する。これらを今回の新しい実験・独立再現・研究テーマ採択に数えない。
