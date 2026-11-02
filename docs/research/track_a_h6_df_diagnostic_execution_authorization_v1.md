# Track A H6：固定済みDF診断一回の認可 v1

2026-10-10 JST。ユーザーの明示指示「固定済みのH6 DF診断を一回実行することを承認します」を、別grantで今回だけの実行へ結合した。
これは[診断準備](track_a_h6_df_hermitization_diagnostic_preparation_v1.md)の実行認可であり、H6 state/pilot・DF表現/閾値/rank政策変更の認可ではない。

- 元結果：`df3b1f694ceb72a198ab6e3e89706b239e56e1da`。元実行source：`67312f3195aede26e8ba4f5727d89c236772f82e`。
- 新診断source：`ff24de4bc410234472a416186b773fc7875ae373`。準備公開HEAD：`29c175b24182172d3e858fdec32106de05b43aa3`。
- [固定manifest](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/sealed_diagnostic_preparation_v1.json) digest：`59ff0f13202e25731ad63cb0c47bc5f9600aff9f671b412999279508b3631bd0`。
- [新grant](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/authorization_v1.json) SHA-256：`5249805956780614c2421994a701caaff300ab05cfe50ecfd9431b5d290ec4e4`。
- [実行前照合](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/prelaunch_authorization_audit_v1.json)：source187件、保存入力・旧raw/source・環境・CPUを再照合。数値library未import、runner未起動。

入力は旧保存integralsだけ、linear H6 / 1.00 Å / STO-3G / 12 spin orbitals。
明示kwargsはtruncation_threshold=1e-8のみ。final_rank追加・rank fallback・fragment削除・許容1e-10の変更なし。
別execution identity `h6_df_diagnostic_20261010_launch_v1`、一回だけ、retry/resume false。
予算はphase60/300/120秒・total480秒・AS8GiB・output32MiB・log64KiB。
論理CPU ID 2・worker1・BLAS1。今回の診断はPF prefix/delta/signal比較を含まない。
future output repository pathは`artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/`。
この認可資料とsource/manifestをGitHubから再取得・照合してから一回起動する。

返されたrawを要約/検査より先に保存し、lambdaと非Hermiticity・weighted係数影響を別記録する。
新しいrawを旧失敗runの未保存bytesと同一とは主張しない。
成功/失敗/欠測をそのまま保存・stdlib監査・公開後mandatory STOPし、GPTへ科学判断を戻す。
診断記録完了はH6入力受理・Hermitization政策PASS・H6 GOではない。
H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION、N/G null・u未認定・UNDETERMINEDを維持する。
