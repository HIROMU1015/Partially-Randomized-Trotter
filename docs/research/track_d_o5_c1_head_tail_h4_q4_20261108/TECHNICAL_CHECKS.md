# 実行前の技術検証

2026-11-08 JST。新規8-mode人工guard6件PASS（physical/ordinaryのfull phase、同basis、π/7 analytic gauge一回、非隣接、mode/register/rotation/gauge/wire拒否）。分子入力・P0は使用していない。

再利用するnofusion公式Aer adapter4件、保存archive guard5件、旧wrapper再利用guard4件、計13件PASS。合わせて19件。syntax・固定依存version・git diff --checkもPASS。H4のcorrectnessや資源削減をこれらの人工テストだけから主張しない。

既存共有Pythonにはthreadpoolctlだけが欠けていたため、既存3.6.0配布物を今回専用private依存directoryへcopyしPYTHONPATHで選択した。module SHA256はPLANへ固定。共有venv/package/cacheは無変更。初回の追加回帰driverは依存欠落でimport時に停止し、科学入力・build・compile呼出しは0。その後private依存を指定して上記13件が完了した。

実行方法は同じ既存Pythonにprivate threadpoolctlを指定し、専用runnerへ固定archiveとprivate原本copyのrootを渡す。PLAN.jsonのsource_filesをcommitから照合し、SOURCE.jsonをevent decode前に保存する。
