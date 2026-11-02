# 2026-10-16 H8 B0 anchor run01の起動前STOP

利用者の継続承認を受け、固定4構成8軸の取得一回を試みた。SOURCE572a337d…、新scope grant、driver CPU0/observer2/workers5,6,7,10を結合した。事前2秒観測ではidleだったが、launcherの1秒fresh idle checkでPHYSICAL_CORE_BUSY。fresh gate未発行、control/output/lock未作成、driver/worker/科学予約/compile0でSTOPした。

失敗CPU・counter差分は原receiptに記録されず、原因jobは未確定。detached launcherのnative exit codeも未捕捉でunknownとして保全する。自動retry・CPU再割当・SOURCE変更・追加compileは行わなかった。

新grant原本、stderr、STOP receiptを保存し、旧676保護/115worktree・旧V2未追跡scriptの不変とSOURCE14/provider265/runtime34,273/input8を照合した。結果は専用worktreeに公開する。旧B1 STOP/V2 NO_GOとB1/B2/Cの別承認を維持し、公開後mandatory STOP。

[現在のSTOP状態と証拠](../track_a_h8_b0_anchor_run01_stop_20261016.md)。
