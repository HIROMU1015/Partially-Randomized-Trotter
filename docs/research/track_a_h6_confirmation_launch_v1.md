# H6独立32確認：一回起動と将来の高速化版への引継ぎ

利用者の「途中からハッシュ再利用と8ワーカー実装を入れた計算に引き継ぐことを想定して開始」と「840への変更も承認して開始」を根拠に、新exact grantを発行し、固定4worker版を一度起動した。

SOURCE `73018cbcc39803890974c7ee51401063e6c8daf6`、探索結果 `df99d6ee28942df8d12aac3c4db7cd202df55d77`。
[準備契約](track_a_h6_independent_confirmation_preparation_v1.md)、[認可・直前検査・起動観測入口](../../artifacts/resource_applicability/track_a_h6_confirmation_preparation_v1/2026-10-12/launch_authorization_v1/README.md)。
固定union8×32 fresh trajectory＝256 group/512 X/Y wrapper、原task/seed/compiler・Hamiltonian/DF/state・経験的allowance/shot会計を維持。探索8と確認32はpoolしない。

trajectory累積上限832→840だけを承認済みgrantに反映した。旧prepared manifestと科学planの原bytesは保持し、exact grantがproposed840を認可する。compile1712、他の上限・既知carryと停止予約は維持し、未知actual totalはnull、返却/resetなし。

worker4はCPU20/21/24/25、coordinator22/23。各AS32GiB/RSS9GiB・aggregate40GiB/jobRAM64GiB・内部thread1、phase/total wall cap=null。SOURCE/profile/input/全592探索・selection/seed/carry、fresh CPU/RAM/PSI/OOM/FS/inode/quota、未使用output/one-shotを直前照合し合格。69件の既存限定testsや完了済みproofは繰り返していない。running SOURCE・共有設定・既存venv・H4を変更しない。

run ID `h6-independent32-confirmation-v1`、coordinator PID20241。初期native journalは4worker稼働・first_stop_reason=null。起動は完了や科学的PASSではなく、完了/STOP後mandatory STOP、追加sample/H8/次stage・自動retryなし。

完了whole-trajectory groupだけを、task/seed/source/profile/input/compiler/event digest、両軸wrapper、worker TASK_COMPLETE、元source用admissionで検証後に未来の新runへ再利用できる。seedはSOURCE SHAに依存しない。部分groupを完成品として再利用しない。切替先は独立worktree/new SOURCE・新output/lockに固定し、旧source成果を保全、同じ残task/seedだけを処理する。

本runはgraceful drain requestを実装していない。明示切替時のSTOP_REQUESTは自分のactive group最大4を終了するため、部分groupの追加予約は最大compile8/trajectory4となり得る。費用を実消費と予約に分けて再計算し、必要な契約変更を切替時に判断する。現時点で余分なcallや8workerの新資源予算を認可したものではない。

hash再利用・8worker実装は今回のSOURCEに含まれず、移行実装/動作検証も未完了。将来のsourceはcanonical bytes/digest・mutation安全性・元回路/compilerを維持して限定検証し、CPU/RAMを再見積もる。自動切替なし。

Git公開は既存のGitHub認証不足で保留。認証設定を変更せず、利用者の開始指示に従いローカル固定SOURCEで起動した。remote公開済みとは主張しない。source closureは準備SOURCEと同一で、新commitは認可・軽量起動観測・資料のみ。
