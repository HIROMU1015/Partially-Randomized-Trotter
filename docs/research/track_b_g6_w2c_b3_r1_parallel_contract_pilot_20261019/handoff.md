# Q1–Q10と次の担当

**人工実装・測定完了。非採択。本番SOURCEへの統合・science release・実authorityは未実施。** 公開/保全はpublication_receipt.jsonとverification.jsonを参照する。

1. **Q1 実装**：統合公開baselineから実装した。8 fresh pair、X/Y共有、3phase barrier、固定identity集約、親global state/bytes/start管理、fake host/SQLite経由が動作する。本番のparallel host/schemaや実LegacyEndpoint接続は未提供で、人工prototypeの完成を本番release完成と混同しない。
2. **Q2 性能**：下表。元逐次S0比P1_8のwallは3.84/3.63/3.89倍短縮、CPUは1.62/1.56/1.31倍。全設定のmin/max、S、P2、CPU内訳はparallel_scaling_and_fault_results.md。CPU増加には親管理・IPC・追加observer対象等が含まれ、単純な科学計算kernelの増加とは断定しない。
3. **Q3 in-flight**：同期人工障害でP1は8 ticket（失敗1＋後に完了7）、P2 cap2は2 ticket（失敗1＋完了1）。STOP→全joinは約113.68/104.04ms。STOP後worker CPU推定は約810.94/197.96ms、tick誤差目安80ms、親CPU約7.64/6.12ms。開始済み費用を無視せず各workerのprefix/失敗/unknownを保持する。これは障害専用busy loopの実測で、通常失敗の予測ではない。親crashはSTART_RESERVED1件の結果不明を保存し、実開始回数はnull。
4. **Q4 one-shot**：人工fixtureでは1 global claim、SQLite atomic spent、reserve-before-start、一意ID、call上限、retry0/resumeなしを確認。二重hostは1つだけadmit、commit ACK不明・worker/parent crash・STOP保存失敗後もscopeを再利用しない。実authorityでのparallel契約bindingは未採択・未接続。
5. **Q5 F01/output**：原run_pair/Streamを変更せず、current/last/candidate/committed/stream prefixを分離。event取得前後・angle・write・flushの人工障害を確認。親のglobal書込前予約でresult64GiB/log256MiBを管理し、per-worker固定分割なし。状態fileは全予定callでも30.24MiB。未知writeは返却せず、保存不完全をPASSにしない。
6. **Q6 公平性**：同時実行によるcache/CPU/I/O競合、観測対象、管理費用が変わる。concurrent pair CPUと旧逐次pair CPUを同条件と扱えない。coarse/fineや重なるwallは足さず、親＋全worker CPU、全体wall、pair費用を分ける。どの費用を科学比較の主estimandにするかはユーザーの別採択事項。
7. **Q7 最小変更**：6項目。dispatch、start/call、STOP/prefix、費用estimand、global保存/予算、SOURCE/schema/host/admission binding。数学的law/N/seed/event/angle/error/call予算を変更する提案ではない。
8. **Q8 推奨**：wall優先の技術候補は**P1・8 worker**。P2 cap2は完了in-flightを7→1へ減らせるが、light/mixedではS0より遅く総CPUも大きい。P2 cap4は折衷だがP1_8の速度に届かない。採択には6項目の判断、実N非uniform計画/最大log・SOURCE/guard、正式schema/host/独立承認binding、全人工回帰が必要。必要なら別途、予算固定の実入力pilotを提案してユーザーの科学実行承認を得る。今回は行わない。
9. **Q9 後の統合**：公開固定したローカル候補を受領後、fused canonicalのdriver/audit/encoder（実際の変更fileを固定commitから特定）、冪漸化式のprecision/angle（同様に実差分で特定）、scheduler、host、admission、readiness、manifest/schema/全pinを再検証する。未公開SOURCEは推測しない。canonical bytes/Unicode/整数有理数/compact/hash/F01全障害、級数partial/remainder/端点/J/τ/Kφ/σ、source/input/policy/preflight/law/state guard、event/packet再query、bit消費・A/B/pass trace、24 constructor/prepare3/preflight9、global budget/claim/fork-owner/STOP競合、coarse/fine費用を確認する。GMP/native serialization・F3科学SOURCE変更は今回取り込んでいない。
10. **Q10 実行境界**：新科学計算、実G7/F3 import/sampling、実authorization/grant/claim発行消費、実correctness256、本番31,710,848 callsはいずれも0。全て明示的fake fixture・人工endpoint・人工scope。既存science/R1/結果/authority/ledger/marker/STOPを変更していない。

P1の1/2/4/8 workerにおける中央値（秒）。S0は元逐次実装、P1_1は新管理機構の1 workerで異なる。

| 負荷 | S0 wall / CPU | P1 wall 1 / 2 / 4 / 8 | P1総CPU 1 / 2 / 4 / 8 |
|---|---|---|---|
| light | 10.3422 / 10.3538 | 15.4965 / 8.5339 / 4.9241 / 2.6914 | 16.0719 / 15.9789 / 16.2614 / 16.7601 |
| mixed | 5.8032 / 5.8166 | 8.5546 / 4.7461 / 2.8284 / 1.5997 | 8.8676 / 8.6184 / 8.9482 / 9.0969 |
| heavy | 5.7055 / 5.7202 | 7.2547 / 4.0060 / 2.5804 / 1.4685 | 7.4484 / 7.3003 / 7.5195 / 7.4692 |

優先順位は、(1)P1_8の契約判断と管理費用の技術改善、(2)失敗後の追加処理を減らす必要があればP2 capの選択、(3)別公開のローカル最適化との統合検証。CPU研究の主比較へP1を無条件に採用する推奨ではない。

**次の作業（未着手）**：**ユーザー**が6項目の採否、特にconcurrent CPU estimandとP1/P2・capを判断する。採択する場合は、依頼を受けた**Codex**が新しい独立scopeで正式schema/host/SOURCE接続と必要人工再検証を行う。ローカル最適化はその固定公開commitをユーザーが指定した後に別依頼で統合検討する。実入力pilot・実authority・本番は別承認まで実施しない。今回の公開保全確認後、**Codexはmandatory STOPして待機**する。
