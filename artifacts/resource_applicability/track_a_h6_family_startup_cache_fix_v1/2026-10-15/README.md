# H6 family startup cache/diagnostic fix

SOURCE `84d236cf9bbdd4f0b990dbb8f67c97d312a08ac7`。旧run01はsnapshot/scalar/overlay/build/probe/compile予約全0、native wait2/driver退出。元証拠を保全しrefund/resetなし。

[停止](run01_stop_projection_v1.json)／[修正source closure](source_closure_v1.json)／[人工46件](artificial_test_result_v1.json)／[主担当照合](primary_launch_review_v1.json)／[binding](binding_projection_v1.json)／[import診断](import_diagnosis_summary_v1.json)。

Numba cacheをrun/worker専用tempへ閉じ、thread数/affinityの停止理由を別々に保存。元の曖昧な停止理由の内訳は補完しない。math/compiler/旧6/最大追加18/資源/5秒/thread1を維持。独立reviewではなく主担当だけの検査、subagents0。ユーザーの実行・停止後修正指示の範囲で、新grant/outputへ結合しfresh gate後に一度起動確認する。
