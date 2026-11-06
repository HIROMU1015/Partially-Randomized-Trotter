# 強PFの固定3 PASS arm：保存18roundのbranch・shot会計

2026-11-07 JST。Codex B、主エージェントのみ。ユーザー指示によって、固定3 armの保存18roundの数値再解析・shot会計・実装/人工確認/公開が許可されている。別チャットのSOURCE/状態を変更しない。

- 保存信号：`2583ce4957ed44e914ce6571e3e676830439dfa7:artifacts/track_a_b2_p1_strong_pf/2026-11-07/run01/result.json`。
- 原科学SOURCE：`a7940b58c6cb939d77e9f8e44d214b3787b7e101`。
- 対象は`Full_S2_s1`、`Full_S2_s2`、`Yoshida4_s1`のみ。5 STOP armの信号を再評価しない。
- 元H/state/DF12、sector/JW/order、P1 `7bc952772e451f27c1654dab6803192f2b204c9f`、ε_E=.00015936001019904 Ha、β=.4、δ=.02、m0..17/Q=2^m、prior[-128,128]/shift0、95%目標、各軸α=.05/36、位相配分.02/.06/.04/.24/.02/.02、u=1e-8・独立閾値1e-9を維持。
- `b2_p1_branch_validation.decode_raw_means`から既存`reconstruct_rpe_phase`とscalar/vectorized照合を使う。3 armのdecoder完了/STOP後にだけ元H energy参照を開き、`posthoc_correctness`で照合。枝を選び直さない。
- 決定論raw=z_PF、A=B=1、disk=0、ρ_PF,lb=abs(z_PF)-u_radius。`b2_p1_shot_accounting.account_saved_round`→`account_radius_round`→既存Hoeffding式を使用。RTE .04を予約したままstat=.24。`aggregate_shots`とJSON/CSV再読込検査で整数和を確認。
- B2 branch `d424bb8bec1aaf1224a33513a06e64801749a69f`・shot `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`は共有条件・公開来歴を照合し、既存7,805,784 shotsを引用するだけ。B2信号/branch/shotを再評価しない。
- 新規は薄い保存JSON adapter、runner、人工test。本来のsignal/decoder/shot式・基本step精度は不変更。人工試験後にSOURCE commitを固定してから実行する。
- 出力先：`artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01/`。各armのJSON/CSVと全体summary。成功した科学実行を再試行・追加しない。
- 共有SOURCE/input不一致なら全体STOP。branch/energy/正半径/shot不適格なら該当arm STOP、残りは継続。出力は`not_certified`で、実機95%・厳密数値包含・全長回路のcorrectnessを認定しない。
- private配列読込、新PF信号/演算子構成、5 STOP arm再検証、新SCF/DF/state、回路build/compile、価格MC/総費用、最適化、詳細科学レビューは行わない。

公開JSON/CSV・簡潔REPORTを専用branchへ保存し、remote SHA/tree/公開内容を照合後mandatory STOP。後続科学実行には別途の明示承認が必要。
