# 旧run05 native終了proof：不足する最小追加証拠

受領bytes/freeze/carry/P1はPASSだが、technical再sealには旧run05の全owned終了証拠が不足している。
旧SOURCE `6d365257770e99022b91d6a38dbee49ee0077503`、UID30038、driver/session8932。

| role | PID | `/proc/<pid>/stat` starttime ticks |
|---|---|---|
| driver | 8932 | 307348817 |
| workers | 9090–9092 | 307349254 |
| workers | 9093–9099 | 307349255 |
| workers | 9100–9101 | 307349256 |

旧host担当へ渡す依頼：

> 既存の同identityに結合したfinal execution/exit-status/remaining-owned-zero監査があれば、そのJSONとbytes/SHAを提供する。
> なければ旧hostで上記UID/PID/starttimeを停止後にread-only確認し、hostname/boot identity/観測UTC、対象13 identityのABSENT・REUSED・EXITED_REAPED・LIVE・UNKNOWN判定、own旧run/sessionの残存監査を一つのnative JSONへ保存する。
> stat/status読取を前後で照合し、欠測・権限不足・観測中identity変化はUNKNOWNとして残す。同じPIDでもstarttime/UIDが違えば旧identityの終了と現在processの再利用を区別する。
> 旧identityのLIVE/未reap/UNKNOWNがあれば合格扱いしない。任意のprocessへsignalせず、他job・共有設定・既存venvを変更しない。
> 新しいJSONは旧hostのhome配下へexclusive保存し、bytes/SHA sidecarと一緒に既存の認証済み転送方法で新hostのincomingへ提供する。内部SSH接続情報・credentialはJSONへ入れない。

新hostの同PID確認は旧host proofの代替にならない。handoffのall_owned_processes_ended metadataだけでも合格にしない。
受領済みold_owned_run_stopped_v1.jsonはrun05起動前のrun02監査なので流用しない。
run05 terminal tracebackとexact旧sourceからworker/executor/budget cleanup到達は推認できるが、上記terminal identity証拠の代替にはしない。

追加proofをbyte受領・identity/host/run/時刻へ結合し、native条件がPASSとなった場合だけstop receipt/plan/auth/reviewのdigestを再結合してtechnical sealと独立最終整合reviewを行う。
approval/runtime flags=false・allowed_cpus=[]は別途明示承認まで維持する。
