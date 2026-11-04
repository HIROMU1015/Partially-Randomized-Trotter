# 2026-11-04 RPE D：理想branch・literal energy条件の限定検証

公開SOURCE6942e5176d2dec115ac3c271e48f84377abda915に対する別途実行許可後、
linear H4/1Å/STO-3G/保存legacy DF12、Partial L_D3/δ=.02、保存physical state ray/sector70、
保存q/r/K/N/α/β/ε・E_ref、m0..17を160 bit・1 passで評価した。
旧36軸包含を再利用し、全18phase条件・全17理想branch条件・alias/lift・literal energy目標が成立した。

保存decimal αのexact和は0.050000000000000004626684で、保存.05基準には僅かに収まらなかった。
1−Σα=0.949999999999999995373316を保持し、固定SOURCEの総合判定をUNKNOWN/STOPとする。
丸め・予算/decoder変更・retry/精度追加によるPASS化は行わない判断とした。
これは利用した十分下界の不足であり、実際の失敗率>5%の証明ではない。

実装Float64 decoderの一様誤差、η_ref、ground/state preparation・実測定bridgeは未保証。
詳細・原RESULT・全round表示は[実行報告](../rpe_branch_energy_execution_20261104/REPORT.md)。
旧SOURCE/raw/STOP、V2 CONDITIONAL_SUCCESS、global false/UNDETERMINED・元bridge U=nullは保全。
次の範囲・α基準の扱いはユーザーが指定する。別指示まで追加科学実行・重要レビューなし。
公開照合後mandatory STOP。
