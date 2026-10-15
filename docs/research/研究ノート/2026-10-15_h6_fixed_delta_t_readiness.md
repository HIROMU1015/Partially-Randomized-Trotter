# H6固定δ T依存性の準備

利用者の訂正版仕様と「計算を実際に開始できる直前まで」を受け、固定34efc914から新worktreeを作成。H6/H8/共有venvを変更せず、旧T=.8を再利用しT=.4/1.6の新scopeだけを実装。PR各32・最大8group並列、heavy S4単独。46人工検査とmetadata/API binding合格。SOURCE90a3c06d16559d132d8fcb04364868c919c2d546。新grant/科学0、独立review0。承認前STOP。

[現行準備資料](../track_a_h6_fixed_delta_t_dependence_readiness_20261015.md)。旧消費・unknownは保持し、compile1910/trajectory916/sampling3584は未承認案。

## 2026-10-15 利用者の開始指示・実行中

利用者「計算を開始して」を、直前のscope/資源/累積上限案の実行承認として固定した。SOURCE `669842c85e2c04a90dcdcfb774b323a2737cb8ba`、run `h6-fixed-delta-t-panel-20261015-run02`、coordinator1603119／signal workers1603135,1603136。承認済みCPU予算内で空いているworker34/35/36/37/39/40/41/43・driver44へ自runだけを配置した。

初回CPU admission停止はscience0/one-shot未消費。後のbootstrapはsnapshotパスを文字列で渡し、NPZ array読込前のAttributeErrorで停止した。両workerのexit1/reap・native wait1と予約を保持し、Path変換の1行修正＋人工47件PASS後に新SOURCE/新outputへ結合した。数学・input/compiler・64trajectory/144compile・累積1910/916/3584は不変。

最後のfresh CPU/memory/pressure/OOM/filesystem/inode/quotaとSOURCE230provider/34273environment/input/carry byte bindingに合格して起動した。[開始確認](../../../artifacts/resource_applicability/track_a_h6_fixed_delta_readiness_v1/2026-10-15/execution_start_v1.json)。2signal worker/thread1/CPU時間増加・STOP理由なしを確認。完了証拠ではない。runtimeはprivate homeのmaterialized05、native logはlaunch02。旧grant/証拠を保持し、共有設定・他job・venv変更0。runnerの自動retry/次stageは無効のまま。
