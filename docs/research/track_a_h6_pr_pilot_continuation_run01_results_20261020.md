# H6 PR DESIGN_PILOT：残り322本の取得・終了監査完了

**新322 trajectory・644 compile軸が正常終了し、native/byte/終了監査がPASSした。** 独立受入した保存74本/148軸と合わせ、元396登録/792軸が揃った。旧run01全体のSTOP・exit1・172予約は保持し、過去のSTOPをPASSに変更していない。

科学SOURCE：`0ff0f70491423d2ef7cb99b006e6b054a8b17b3b`、plan SHA：`2eae0538667a648a139f75edb8b9787e2141c6b08554617ba211ea08993166db`。[固定継続契約](track_a_h6_pr_pilot_continuation_contract_20261020.md)、[起動時の記録](track_a_h6_pr_pilot_continuation_run01_start_20261020.md)、[完了summary](../../artifacts/resource_applicability/track_a_h6_pr_pilot_continuation_run01_results/2026-10-20/completion_summary.json)、[構造検証](../../artifacts/resource_applicability/track_a_h6_pr_pilot_continuation_run01_results/2026-10-20/structure_validation.json)を参照する。

H6直鎖1 Å/STO-3G、tol-only1e-8/DF19、保存Hamiltonian/state/reference、prefix5、T=.8、P=0。S2/Yoshida4/m5_best、q1/2/4/8、K4、outer32、12群×33本の元登録・seed/cohortを維持した。symmetric-directional full Hadamard wrapper、Qiskit1.3.0/rz-sx-x-cx/opt1/seed17、CT01 1e-7未満・その他native/oracle/Hadamard等1e-9未満は変更していない。

| 項目 | 保存された終了記録 |
| --- | --- |
| 新規完了 | 322/322 trajectory、644/644軸 |
| 新規失敗・未完了 | 0 |
| 並列・終了 | 30 worker、全exit0/reap、native launcher exit0/reap |
| launcher wall | 3485.801772秒（約58分6秒） |
| 外部wait4 wall | 3493.031333秒 |
| 新archive | 1932件、全native検証・原byte SHA復元PASS |
| 新archiveの原byte / 圧縮byte | 14,292,667,536 / 946,971,163 byte |
| 新run全体の論理 / allocated容量 | 3,176,496,597 / 3,258,966,016 byte |
| worker最大RSS | 3,720,052 KiB |
| process RSS合計の観測peak | 61,436,710,912 byte（共有page重複を含む） |
| 所有raw scratchの観測peak | 124,481,536 byte |

RAM/scratchは約5秒間隔の観測値で、厳密なpeak上界ではない。process RSS合計を固有の物理RAM使用量とは呼ばない。約58分は今回の取得時間であり、旧runと条件を揃えた全体高速化倍率は測定していない。

親context共有・SHA付き入力cache・fresh並列数の修正版で完了した。前回cutoffで停止したm5/q2/sample0も両軸native/compiled/byte検証までPASSした。終了時に新322本の全call台帳・結果SHA・worker/launcher回収を検査し、保存74本/444 archiveを再監査した。参照表は旧74本と新322本のSOURCEを分けて保持し、元396登録を一度ずつ被覆する。原seedやprivate原本、QPY/NPZを公開しない。

H4は元pilotで396本/792軸が完了済みであり、今回はH4を再実行していない。H4/H6の取得が揃ったことと、標本設計の数値評価やn_Eが確定したことは別である。今回は平均・分散・MCSE・SE・希少層の数値評価・shot・費用会計・ランキングを行っていない。正式shot政策・AD03は保留、最終evidence freezeもしていない。

**mandatory STOP。** 次の担当：ユーザーがH4/H6 pilot保存結果の分散・MCSE・希少event等の数値評価と、本探索n_E設計へ進める範囲を別途承認する。Codexは承認範囲で評価契約を準備・実行する。自動top-up、新しい探索・独立確認・会計へは進まない。
