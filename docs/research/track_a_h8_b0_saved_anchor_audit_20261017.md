# H8 B0：監視・終了監査の修正と保存8軸の再利用判定

**8軸を再compileさせる技術的欠損は見つからなかった。保存測定は、B1の明示承認判断に使える固定入力候補として構造的に再利用可能。** 元run02の取得契約はSTOPのままであり、このauditは最終gateを遡及的にPASSにしない。

利用者の「その方針で進めて」により、監視・管理用保全確認を修正し、保存証拠を読取専用で照合した。新規科学計算・compile・signal・費用fit・残差・再会計・有効grantは0。元のSOURCE/raw/config/grantと他worktreeを保持した。

## 読取専用で確認したこと

対象はH8直線鎖、隣接1Å、STO-3G、16 modes、Nα=Nβ=4/sector4900、保存tol-only DF rank28/tol1e-8/cutoff0、signed generation order、weighted Hermitian projection。B0 S2のp14/q1・p21/q1・p14/q8をTRAIN、p21/q8をHOLDOUTとし、各cosine/sineの8軸。T=.8、δ=.8/.1、L_D=p14/21。元constant一回/no tail identity、symmetric-directional、state preparationを除くnative full measured Hadamard wrapper。元Qiskit1.3.0/compiler profileを維持した。

保存測定そのものは[run02報告](track_a_h8_b0_anchor_run02_stop_20261016.md)にある。今回は値を転記して新しい予測や総費用を作らず、以下を照合した。

- 元公開commit `af10b55420d31e227905f7762971e4b39c6c6a05`のbyte inventoryを起点に、29公開ファイルと53 private original filesを確認。
- config/grantのSHA、実行SOURCE `db559a518214fd27fa96e2f1451911df05c4f4a8`の15 files、provider265、runtime34,273、入力8を確認。元grantを変更・再使用していない。
- TRAIN6/HOLDOUT2の軸集合・分割、4 control/preparation receipt、wrapper before/after fingerprint、profile/compiler/control/scalar・prefix preparationを結合。保存control誤差が1e-8未満であることを確認し、再simulateしていない。
- 8 task_resultのSHAと各wave受入行、worker identity/exit0/reap、reserved8/settled8/unknown0、許可call数と禁止call0を確認。旧B0の未結合16軸やB0+Iの測定を混合していない。
- 元末尾のSOURCE/input/runtime検証後、`r.protected`の`git status`待ち中にSTOPしたtracebackを確認。全8軸受入・測定保存済みと契約の正常完了は区別する。
- 元の保全対象678 files/117 worktreeを管理用Git設定で再照合。さらに今回の開始baselineの681保護files/118既存worktreeのHEAD/status、旧V2 untrackedも不変。

機械判定は `SAVED_EIGHT_AXES_STRUCTURALLY_REUSABLE_PENDING_B1_APPROVAL`。未取得・不整合を埋めるための再compileは不要と判断する。ただし、過去の全期間thread遵守を認証したり、未記録の違反PID/exeを復元したりする判定ではない。

## 修正した動作

修正SOURCEは **`e5f3fb07e3719c9472487729a18b9d000fb8cf18`**。元実行SOURCEとは別で、元run02への差し替えや再起動をしていない。

管理用Git呼出しは、コマンド単位で`--no-optional-locks`、`index.threads=1`、`core.preloadIndex=false`、`core.fsmonitor=false`、`core.untrackedCache=false`を指定する。グローバル/ローカルGit configを書き換えず、保全確認でindexを任意更新しない。既存status bytesとの一致を確認した。

observerは今後、thread違反を検出したrowのPID/start/parent/uid identity、実行ファイル、process名、thread数と観測process一覧をcleanup前に`process_thread_violation.json`へ保存し、terminalにも残す。元の理由文字列`RuntimeError:PROCESS_THREADS`を維持する。driver/direct childの上限1を維持し、Gitを監視対象から除外する例外は設けない。

契約schemaはV3、SOURCE closureは新helperを含む16 files。候補・資源・call上限・2 wave・4 worker・原取得計画のbytesは不変。新しい実行config/grantは発行せず、旧run02 grantが新契約では起動前に拒否されることを確認した。

## 技術検証

主エージェント自身による人工テスト・ローカル照合であり、独立レビュー・CI・外部再現とは呼ばない。

- 新14件：metadata欠測/重複/誤分割、SOURCE/profile/preparation/scalar/fingerprint改変、control境界/NaN、worker失敗・未settle・禁止call・STOPのPASS化を拒否。Git status同値・index/config bytes不変、CLIの科学library未importも確認。
- 実OSの人工2-thread子processでSTOPを発生させ、違反identity/executable/thread数の保存、元理由文字列、全回収・retryなしを確認。実H8/native compileは起動していない。
- 既存83件を含め、固定Python3.11.0rc1でも計97件PASS。資源上限、CPU gate、wave barrier、owned cleanup、既存observer動作を確認。
- 現在の管理用Git metadata probeでexit0、9 samplesの最大観測thread1、status bytesが開始baselineと一致。これは今回のサンプル結果で、run02の違反原因の再現や連続thread証明ではない。

## 残る制限と次の段階

元run02の違反PID/thread数/executableは未記録のまま。Git待ち中というtracebackだけでGitが違反processだったとは断定しない。原STOP・driver exit−9・launcher exit1・未生成の最終gateを保持し、事後auditで契約を閉じない。

次は、この保存8軸を用いたB1の実行準備・明示承認判断。B0 TRAIN3点と独立HOLDOUT1点を各軸で扱い、B0+Iは既存の来歴結合済み16軸を再利用する。2%以上ならSTOP、holdoutのfit混入・条件緩和・H6係数移送・自動refit・追加compileを認めない。今回そのfitも2%判定も実行していない。旧B1 STOPを自動解除せず、B2 S4 compileとC再会計も別承認。V2 CLOSED_NO_GOとV1/H5/H7を保持する。

[公開evidence/checker](../../artifacts/resource_applicability/track_a_h8_b0_saved_anchor_audit/2026-10-17/README.md)をcommit・pushし、remote SHA確認後mandatory STOP。
