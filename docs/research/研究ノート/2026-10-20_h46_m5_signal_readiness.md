# H4/H6 m5 signal readiness / 2026-10-20

基本88条件の範囲を確定して、signal-onlyの技術準備を実施した。旧plan・grant・STOPは書き換えない。旧pilot q2の同一raw6件とreference2件をbyte joinし、最大82新signalとする。prefixごとのtail/scalar/contextを分けてcacheし、H4/H6独立process・候補threadで重複を減らす。

SOURCE `480ca4507394e35185ec535c9f488bd90538de25`。経験的allowance案は厳密reference誤差上界を含まず、reference error=nullの条件付き診断。実行承認でその限定的使用を判断し、正式shot政策への自動採択はしない。新実分子signal/sampling/compile/会計0、人工14＋回帰16PASS、metadata17/改変拒否7。保護2899file不変、2046seal。

[報告](../track_a_h46_m5_signal_readiness_20261020.md)。ユーザーがsignal一回実行とallowance使用を判断し、承認後Codexがfresh gate・新grantで実行する。今回local commit後mandatory STOP、pushなし。

## 一回実行のSTOPと別SOURCE修正

ユーザー承認によりrun01を起動。2系ともfull prefixに打切り専用helperを呼ぶ不具合で初期化STOP、候補予約・評価0。旧SOURCEとSTOPを保持し、別worktreeでfull empty-tailを明示、元pilot検証は元SOURCE pathのまま行う。修正前回帰失敗を再現、修正後18件・metadata37項目PASS。新SOURCE `bec018c64e7a8458dbf13b1c0e9655d628292857`、新planは未承認・無効grant、自動retryなし。

[STOPと修正報告](../track_a_h46_m5_signal_run01_stop_and_fix_20261020.md)。ユーザーが新しい一回実行を判断、承認後Codexがfresh gateで起動する。local commit後mandatory STOP、pushなし。

## 修正版run02の新承認・正常終了

ユーザー「修正を行って再実行して」により、SOURCE bec018c64e7a8458dbf13b1c0e9655d628292857へ新しい一回run02を承認・実行。旧run01 STOP/元grantは保持。固定88候補を正常終了し、82新signal＋6raw再利用、880軸別適格性行を保存した。2系各30候補thread、wall57.6758秒・外部61.4078秒、全exit0/reap。47終了監査PASS、2065seal・旧2899file/STOP18file不変。保存reference条件付き経験的診断で厳密bias証明なし。

[結果と次の担当](../track_a_h46_m5_signal_run02_results_20261020.md)。shot/cost/ranking/compile/samplingは0、AD03保留。次はユーザーの費用取得計画・n/seed設計準備への判断。新sampling/compile/会計は別承認。local commit後mandatory STOP、pushなし。
