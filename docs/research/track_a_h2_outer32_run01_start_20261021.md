# H2高次PF outer32：signal完了・pilot並列起動

利用者の中間承認省略・H4/H6同段階までの実行指示をSOURCE `4ec26921c6872ed6e8667f167074e1e405cf1a3f` に結合した。

H2直鎖1 Å/STO-3G/tol-only1e-8 rank3、保存Hamiltonian/state、T=.8/P=0、prefix1、S2/Yoshida4/m5_best、q1/2/4/8、K4/outer32。旧H2のr系列とは別cohortで、標本を混合しない。

観測時刻 2026-10-21T02:49:12.096199+09:00：12候補の新signalと独立forward/normalization検査がPASSし、120軸の経験的適格性labelを保存した。pilotは25 workerで167/396本の両軸を完了、STOP/結果エラー0。これは全pilot完了や最終exit/reapを意味しない。

[起動snapshot](../../artifacts/resource_applicability/track_a_h2_outer32_run01_start/2026-10-21/start_snapshot.json)にplan/result/reference SHAと起動状況を保存した。新14件＋既存100件の人工・回帰検査はPASS。private原本・seed・内部pathは公開しない。

準備attempt01はregistry参照の引数誤りを静的処理で停止し、科学call・有効grantは0。実configで固定されたregistry_pathへ参照を修正し、attempt02だけを起動した。旧STOP・SOURCE・原本は変更していない。

H4/H6のprocess・設定・grant・SOURCEは変更せず、空きCPU/RAM/scratchからH2を25 workerで起動した。H4/H6の既存容量計画は再推定せず固定予約を保全した。H2では既存process-local compiler位相guardを保持し、全gate/角度/wire列が不変であることを検査する。

次の担当Codex：pilotの終了監査後、H2自身の保存pilot統計から2%/f4/群別下限33の本探索数を決め、新seed/cohortの単価取得・終了監査まで進める。中間再承認は不要。新探索後の統計・shot・総費用会計・順位・P・AD03は含めない。終了時mandatory STOP、次の解析判断はユーザーが行う。

## pilot終了監査・本探索の開始

2026-10-21T02:52:38.541224+09:00の追跡観測。pilot396本/792軸とcompiler guardの終了監査がPASSし、H2のpilot統計から群別n_Eを決めた。

| 公式 | q1 | q2 | q4 | q8 |
| --- | ---: | ---: | ---: | ---: |
| 2nd | 96 | 71 | 33 | 33 |
| 4th | 60 | 34 | 33 | 33 |
| 4th(m5_best) | 33 | 33 | 33 | 33 |

本探索は新525 trajectory/1050軸を25 workerで起動した。観測時の両軸完了は42本、結果エラー0。pilotを混合しない。

これはpilot観測分散に基づく名目2%/f4設計で、母平均・母分散や未観測tailの保証ではない。観測分散0による未確定群は今回は0。

[H2標本設計と本探索起動記録](../../artifacts/resource_applicability/track_a_h2_outer32_run01_start/2026-10-21/main_exploration_start.json)。新しい探索の統計・shot・順位を計算せず、取得・終了監査後mandatory STOPする。中間再承認は不要。
