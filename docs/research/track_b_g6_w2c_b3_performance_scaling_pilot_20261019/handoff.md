# 技術ハンドオフ・Q1–Q8

**人工性能検証完了、全候補非採択。科学実行・科学SOURCE統合なし。公開確認はpublication_receipt.jsonを参照。**

1. **Q1 重い処理**：静的にはFull packet/子law、有理数、B guard/requery/angle、canonical/save/profilerが候補。人工heavyではangleが支配し、mixedではevidence/generator/saveも無視できなかった。実G7の順位は未確定。
2. **Q2 scaling**：8 pairのlight/mixed/heavyで8 workerは6.79/6.02/3.95×。64人工taskの2/4/8/16/32/64 workerは約1.95/3.81/7.26/13.53/23.41/36.78×。
3. **Q3 資源と効率**：64 taskのCPUは1 worker約11.643秒→64 worker約12.645秒、sampled aggregate RSS約53→1577 MiB。最高throughputは64 worker、並列効率重視なら2 worker。32 workerはCPU維持とwall短縮の折衷。
4. **Q4 非並列改善**：native canonical micro約3.04×、Stream全体約1.58×。exact GMP級数micro約1.18–1.43×。heavy end-to-endでGMP候補wall約1.34×/CPU約1.30×。漸化式は小さく安定した利益。mixedの全体改善は小さい。大payload hashはpreencode+incrementalが有効。
5. **Q5 本番時間**：人工scalingを31710848へ条件付き換算した表まで。実G7の数値予報や上下界は出せない。instrumented費用、実path/operand/guard、実出力が未測定。
6. **Q6 互換性**：fused/native canonical、冪漸化式、exact rational adapter、同じbytesのincremental hashはA候補。process/pass並列、pool、buffer、profiling省略はB。8 worker failureでは通知後の新規開始0でも、開始済みtrialが中央値7件完了した。逐次R1のSTOPを維持したと主張しない。
7. **Q7 実G7 pilot**：実用的な時間予測には実入力観測が必要。計画案は新しい専用one-shotでcorrectness16/cell=256、その後baseline64/cell=1024、instrumented同数1024、合計2304 callsを上限固定する小pilot。coverage不足でも追加しない。必要ならさらに縮小できるがprecision/setup以外の費用推定は弱くなる。F01を含む新しい公開SOURCE commit、R1 math-only τ=10^-6/Kφ=32/J=32、固定input/policy/N、正式pilot契約digest、fresh namespaceと実承認参照/atomic admission、retry0/resumeなしを先に固定し、別途ユーザーの科学実行承認を受ける。今回はSOURCE統合・contract採択・claim発行・pilotを実施しない。既存本番one-shotを消費しない。
8. **Q8 導入候補の優先順位**：下表。性能だけで採択せず、現在のR1を変更するものは別判断。

| 優先 | 候補 | 推奨理由・条件 |
|---|---|---|
| 1 | fused canonical＋必要ならnative backend | bytes/F01を人工検証。依存なし版から導入検討でき、nativeの単独micro効果も明確 |
| 2 | 角度級数の冪漸化式 | 厳密式を維持し外部依存なし。GMP追加より統合範囲が小さい |
| 3 | GMP exact adapter | heavy全体のCPU/Wall改善。型境界と全domain互換性・専用依存の固定が必要 |
| 4 | 8 pair並列 | wall利益は最大級だがB。失敗予算・prefixと費用比較条件を変更する判断が必要 |
| 5 | 大payload incremental hash | 同じbytesを維持できる場合。小rowでは利益が小さく優先度を下げる |
| 6 | pass overlap/pool | B。full-Nへの利益は未確定、tiny task倍率は流用しない |
| 見送り | buffer・GPU・pair分割 | bufferは小利益とprefix保証低下。GPU/pair分割は未測定、今回のCPU候補より優先しない |

次の担当（未着手）：**ユーザー**がA候補の統合技術準備を依頼するか、Bの契約変更検討や別途実入力pilotを認めるか判断する。依頼された**Codex**だけが、Codex Aの公開固定成果とF01のbindingを確認して新しい独立scopeで準備する。現行admission準備へ自動merge/rebaseしない。公開・保全確認後mandatory STOP。
