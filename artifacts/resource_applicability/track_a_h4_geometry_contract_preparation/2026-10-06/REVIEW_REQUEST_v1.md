# 契約review依頼

対象は[CONTRACT_DRAFT_v1.md](CONTRACT_DRAFT_v1.md)と[zero_compute_plan_v1.json](zero_compute_plan_v1.json)。
本計算・science source実装・authorization作成・commit/pushを依頼する文書ではない。

確認済み：追加6距離、登録218 template/点、32 paired-axis trajectories、
74,784 logical/actual invocation上限、外側workers最大12・内部threads/process1。
修正済み：wrapper key/数値回路fingerprint、baseline null seed/index、
外部completion digest、独立identity/numerical registryとowner検査、
消費済み予約・AMBIGUOUS STOP、cacheでsample weightを削除しない。

[4判断の一覧](review_decisions_v1.json)をreviewする。

1. D1：RHFの全明示SCF/積分/MO規約と非収束STOP、DF rank12の生成条件。
2. D2：旧M1と同じ生成順のprefix、DF tie/eigenspace縮退・sector/state/solver/gate案。
   Frobenius再ソートを選ぶなら旧M1との同一split-policy主張を撤回する別契約が必要。
3. D3：master seed候補20261006。今回の合成seed7は科学seedではない。
4. D4：worker AS8 GiB・合計104 GiB/available64 GiB、wall停止72h、
   output10 GiBと固定run ID/rootの案。static array内訳はscience RSS/ETA保証ではない。

生成前null hashと生成後input freezeのseal/認可順、およびproduction Qiskit serializer・
round-trip意味論gateはfuture source reviewで必ず明示する。
今回のJSON検査はfield/binding/mutation拒否を示し、物理回路やatomic runtime実装の検証ではない。

未解決を閉じるまでstatusは
`H4_GEOMETRY_CONTRACT_REQUIRES_DECISION_SCIENCE_NOT_AUTHORIZED`。
合意後も別指示でscience sourceを実装し、そのcommitを固定してからplan/別authorizationへ進む。
今回のSTOP、science_execution_authorized=false、next_stage=false、research_decision=nullを維持する。
