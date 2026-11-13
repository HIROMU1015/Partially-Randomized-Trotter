# LD1/LD2初回2診断の引渡し

2026-11-14 JST。[REPORT](REPORT.md)／[固定入力](INPUTS.json)／[完了記録](run01/COMPLETION.json)。利用者が採択した5e8f1664dの推奨条件を各LD1件だけ適用した。科学実行は完了しており、自動再実行・追加配分は行わない。

## 正本と読み方

- [RESULT_LD1.json](run01/RESULT_LD1.json)、[RESULT_LD2.json](run01/RESULT_LD2.json)の`record.diagnostic.deterministic_risk`が既存評価器のexact結果。
- `rounds[m].unsafe/zero/reference_oracle_tie.exact`は分子・分母を保存する。`all_good/first_unsafe/any_zero/unresolved_probability_bounds`も同形式。小さい確率をfloatの0で置き換えない。
- `derived.any_unsafe`と`derived.zero_free_global_tie_bounds`は上記保存値からexact算術した派生量。`derived.global_tie_probability=null`は全履歴tieの厳密値が未評価という意味。
- `MSE_upper`は採択したloss familyに対するrawモデル上界。`RMSE_upper_display`は既存の外向き平方根表示で、RMSEの点推定ではない。confidence=None、MC_calls=0。
- 各recordの`deterministic_input`は元configuration、全vector、明示target、統計契約、実際に消費したC signal_input、options、SOURCEに結び付く。binding成立を物理資格としない。

exact整数は大きいため、実行・確認には固定の`/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B`（3.11.0rc1）を用いた。安定版Pythonのdecimal整数桁数制限がある環境では、読み取り側の局所I/O設定が必要になり得る。元exact整数をfloat化する解決はしない。今回serializer・共有runtime設定は変更していない。

## UNKNOWN/nullの保持

有限good遷移検査は両方成立し、モデルMSEにUNKNOWN理由はない。実全体tieの厳密値、物理target/state/数値誤差/actual samplerへの適用資格、元Minkowski各項への正式接続は未取得。`scientific_energy_qualification=UNKNOWN`、`n_packet/sufficient_N/global_RMSE_bound/G/winner=null`を保持する。

評価targetは元full-H solverの条件付きproxyであり、raw riskは時間発展近似の影響も含み得る。AのδEを足す・引く処理はない。未解決energy=Noneの既存規則も変えず、loss familyだけを指定した。

## C/D・後続費用評価

この結果だけを正式N packetとして受け入れない。将来の費用評価は既存D.compare→W.compare/B.selectと外部N資格・適格価格を用いる。今回、これらの会計APIは呼出し禁止とし、価格要求一覧も新規生成していない。A/C/Dの物理signal・価格研究を再実行していない。

sourceは[EXECUTION](run01/EXECUTION.json)のcaller/INPUTS commit `43898ada12e405eacda27b5bccf1d464a0362ed2`、数値moduleはd71721f30。既存C純metadata helperを再利用し、src/testsを変更していない。保存結果の再照合は`VERIFY_RESULTS.py`だけででき、risk計算を再起動しない。

次は**ユーザー**がこの条件付き診断を踏まえ、別の上界改良・配分評価等が必要か判断する。**Codex B**は次の明示指示まで**mandatory STOP**。新しい科学実行や正式N/G評価へ自動進行しない。
