# PR-LD2 m12：T2単一path実行

2026-11-13 JST。基点39678f5a0のT2について、ユーザーが明示承認したPR-LD2/m12/normal fresh 1 path/X,Y/最大2 compileだけを実行する。現在はcaller・人工確認・固定PLANを公開する実行前段階。実科学結果はまだない。

対象：Linear H4/1.00 Å/STO-3G、保存tol-only DF10/sector(2,2)、deterministic one-body＋prefix2/tail2…9、S2/ΔPF=.1/K2/h=.1/M12/元exactκ/P0。A a9ab固定configuration、q4096/R2559/T409.6/10,481,664 logical event。cohort D_LONG_TIME_FEASIBILITY_V1/index0。元λ/scalar・fragment順、full measured symmetric-directional X/Y、新ZZ、Qiskit1.3.0/NumPy1.26.4/rz,sx,x,cx/opt1/seed17・元compiler hashを保持する。

[PLAN](PLAN.json)／[技術確認](TESTS.txt)／[HANDOFF](HANDOFF.md)／[NEXT_SCIENCE_PLAN](NEXT_SCIENCE_PLAN.md)。既存ld12_price_requests/candidate_price_adapter、保存入力loader、元normal sampler・Accounting/native compose/wrapper/fingerprint/cache/compilerを使う。fast価格処理にはstage observerを任意追加しただけで、新compiler・modelを作っていない。

新callerのscope/context/source/例外保存と、observer付き人工回路の旧新fingerprint/価格同値性、既存fast/candidate adapter回帰を確認した（distinct40項目PASS）。SOURCE・PLAN・出力先をcommitで固定した後、この承認済み一pathのみ科学実行へ進む。追加の承認証明書/grantは不要。

自己workerを10秒間隔で観測し、実際の空き容量が直近の自己RSS増加又は自己書込みの次の二観測期間分を下回る場合のみ、枯渇防止のresource STOPを行う。固定RAM/CPU/worker/time上限を追加せず、PSIだけでは停止しない。可逆module差替えがあるため独立worker内でX/Yを順に処理する。monitorは別processであり科学pathを追加しない。他者process・共有設定を変更しない。

T2完了又は不一致/失敗/実資源不足時には部分結果を保全し、無断retry・別seed/path/round/candidateへ進まない。個別path C/S/Dと実行可能性までを報告し、mean/CI/全law資格・fit/外挿・N/RMSE/G/winnerは供給しない。最終成果のcommit/push・remote照合後mandatory STOP。
