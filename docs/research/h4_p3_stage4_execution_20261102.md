# P3 Stage 4：承認済み条件付きscalar会計の接続

2026-11-02 JST。ユーザーは公開実行案`e4a9b1345a46f1bccccbf451f4e2d26521350fe3`の
[H4・2方式・5精度](h4_p3_stage4_conditional_accounting_contract_20261102.md)を明示許可した。
旧案の未実行表記は当時の履歴として保全する。実行範囲・仮定・丸め・STOPは同案を維持する。

[scalar kernel](../../src/trotterlib/matched_partition_stage4.py)は既存fixed_signal_shotsと
h4_conditional_cng.trans_boundsだけを再利用し、保存nominal/reference端点からbiasを直接評価する。
保存Bのengineering上界候補と保守側log、ε_a下向き・α_a下向き・bias上向き丸めを記録する。
ceilは既存関数と有理数によるscalar上側の大きい方を採り、2^52の既存domainは維持する。
既存1%guard/rho探索は使わない。u_num等はbias側に一度だけ含め、u_remaining=0は実装gap仮定。

Gは保存32本のexact平均を用い、分散はcos/sin共分散を含む式と同じ32本の重み付き価格から独立照合する。
P/DFは独立標本、旧pilotはpoolしない。主RZと補助CX/shot加重depth、価格標本SEのみ。
母平均tail保証・実compiled/finite-bit誤差保証・RPE全round総費用は取得しない。

[runner](../../scripts/run_h4_p3_stage4_accounting.py)は指定8JSONだけを読み、
旧来歴からH4/state/ν・time・辞書・測定境界を接続する。QPYや分子データの再構築は行わない。
[人工tests](../../tests/test_matched_partition_stage4.py)でscalar端点・誤差一回計上・B候補・ceil/domain・
負の軸間共分散・独立方式差・nullを検査し、新SOURCE固定・公開後に一度だけ10行/20軸を評価する。
SOURCE固定前の人工7件は全PASS。保存8JSONと15件の共通元入力bindingの技術照合もPASSし、
この事前確認では科学evaluateを呼んでいない。
保存先は`artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01`。
小さいscalar処理なので逐次実行し、不必要なprocess/管理layerを作らない。

新sampling/compile/Gaussian/reference/signal/finite bias高精度化は0。
失敗時は原因と途中結果を保全して自動retryせずSTOPする。
成功時は結果・数値仮定・価格SE・未保証事項を公開し、remote照合後mandatory STOPする。
