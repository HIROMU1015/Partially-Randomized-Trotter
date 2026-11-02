# 人工scaling・障害実測

全て人工。実G7/F3は未import・未実行。3 workload × 8設定 × 3反復 = **72測定**、別に24 warm-up。設定順は各反復でseed20261019のshuffle。CPU-boundのinteger/Fraction/人工law sort、Bの冪による級数surrogateを用い、sleepによる性能測定ではない。正式scienceの処理頻度・N・operand幅を再現したものではない。

性能SOURCEは `b1dba4b6f282ffab413719aa000f18267caeb1d7`。測定中のSOURCE hash不変をsuite末で検査。最終人工SOURCE `ef57714b35bbdd3912fadf13ee522e6a4ff491e6` は、初期化失敗のfile close、dispatch途中の親例外に対する協調join、0件状態の明示、実Nの非実行plan、追加fault/監査を加えたもの。最終revisionの性能を72回再測定したとは扱わない。両revisionはGit bundleで保全。

S0=元逐次host/driverを人工依存で実行。S=新管理機構の逐次。P1_n=n worker。P2_n=8 worker・outstanding cap n。時間はhost外側STOPまで含む。表は中央値 [min–max]、speedupはS0比。

## light（人工profile N=128/axis）

| 設定 | wall秒 [min–max] | 総CPU秒 [min–max] | wall短縮 S0比 | CPU S0比 |
|---|---:|---:|---:|---:|
| S0 | 10.3422 [10.2597–10.3497] | 10.3538 [10.2722–10.3628] | 1.00× | 1.00× |
| S | 15.5507 [15.5151–15.5548] | 16.1118 [16.0602–16.1178] | 0.67× | 1.56× |
| P1_1 | 15.4965 [15.3511–15.6222] | 16.0719 [15.9016–16.1943] | 0.67× | 1.55× |
| P1_2 | 8.5339 [8.5216–8.6370] | 15.9789 [15.8572–16.0618] | 1.21× | 1.54× |
| P1_4 | 4.9241 [4.8345–5.0689] | 16.2614 [16.2432–16.4181] | 2.10× | 1.57× |
| P1_8 | 2.6914 [2.6513–2.7582] | 16.7601 [16.7592–16.8243] | 3.84× | 1.62× |
| P2_2 | 14.5015 [14.2076–14.5566] | 31.2583 [30.7203–31.4789] | 0.71× | 3.02× |
| P2_4 | 6.2989 [6.2938–6.3599] | 25.0232 [24.9458–25.7897] | 1.64× | 2.42× |

## mixed（人工profile N=64/axis）

| 設定 | wall秒 [min–max] | 総CPU秒 [min–max] | wall短縮 S0比 | CPU S0比 |
|---|---:|---:|---:|---:|
| S0 | 5.8032 [5.8014–5.8691] | 5.8166 [5.8148–5.8825] | 1.00× | 1.00× |
| S | 8.5683 [8.5273–8.6261] | 8.8645 [8.8392–8.9397] | 0.68× | 1.52× |
| P1_1 | 8.5546 [8.5469–8.5981] | 8.8676 [8.8615–8.9179] | 0.68× | 1.52× |
| P1_2 | 4.7461 [4.7119–4.8257] | 8.6184 [8.5564–8.7673] | 1.22× | 1.48× |
| P1_4 | 2.8284 [2.8051–2.9381] | 8.9482 [8.9077–9.2447] | 2.05× | 1.54× |
| P1_8 | 1.5997 [1.5281–1.6638] | 9.0969 [9.0665–9.1981] | 3.63× | 1.56× |
| P2_2 | 7.2352 [7.0419–7.2709] | 15.5007 [14.9247–15.7392] | 0.80× | 2.66× |
| P2_4 | 3.2666 [3.1118–3.3937] | 12.6992 [11.7825–13.6066] | 1.78× | 2.18× |

## heavy（人工profile N=32/axis）

| 設定 | wall秒 [min–max] | 総CPU秒 [min–max] | wall短縮 S0比 | CPU S0比 |
|---|---:|---:|---:|---:|
| S0 | 5.7055 [5.6584–5.7391] | 5.7202 [5.6715–5.7537] | 1.00× | 1.00× |
| S | 7.2560 [7.2001–7.2638] | 7.4445 [7.3915–7.4551] | 0.79× | 1.30× |
| P1_1 | 7.2547 [7.2243–7.2672] | 7.4484 [7.4085–7.4591] | 0.79× | 1.30× |
| P1_2 | 4.0060 [3.9852–4.0296] | 7.3003 [7.2384–7.3350] | 1.42× | 1.28× |
| P1_4 | 2.5804 [2.5410–2.6241] | 7.5195 [7.4599–7.5211] | 2.21× | 1.31× |
| P1_8 | 1.4685 [1.4679–1.4771] | 7.4692 [7.4671–7.4942] | 3.89× | 1.31× |
| P2_2 | 5.3593 [4.9142–5.6430] | 11.3765 [10.4164–12.0599] | 1.06× | 1.99× |
| P2_4 | 2.2897 [2.1182–2.3141] | 8.8619 [8.0524–8.9246] | 2.49× | 1.55× |

## Memory・I/O・負荷偏り（P1_8）

| 負荷 | sampled RSS和 MiB | worker HWM最大 MiB | pair出力 MiB | phase別 max/mean CPU |
|---|---:|---:|---:|---|
| light | 254.1 | 27.9 | 2.97 | 1.15 / 1.14 / 1.08 |
| mixed | 266.0 | 29.1 | 2.18 | 1.26 / 1.21 / 1.12 |
| heavy | 281.0 | 30.3 | 4.43 | 1.72 / 1.88 / 1.32 |

RSS和は共有pageを重複計上し得るsample値。parent/workerの/proc I/O全counter、起動費、各pair CPU/wall/setup/coarse/fine/RPCはruns.jsonとrun_reports.jsonl.gzに保存。pair wallやRPC wallの和は総wallではない。P2のCPU増加を全て単一原因へ帰属せず、追加同期・実行の交錯・fine profilerを含む実測差として扱う。

P1_8のS比wall短縮はlight/mixed/heavyで 5.78/5.36/4.94倍。元S0比は3.84/3.63/3.89倍、総CPU増は約62/56/31%。旧pilotの6.79/6.02/3.95倍は異なるSOURCE/観測/単一passで、今回の全phase・admission・状態/出力管理・fine観測へ転用しない。16/32/64人工task増殖やGPU測定を今回追加していない。

## 障害保全（17ケース）

| ケース | ENTER観測 | reserved | 親STOP時ticket数 | completed | failed | unknown | STOP→join ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| sample | 17 | 20 | 7 | 16 | 1 | 0 | 9.63 |
| event | 2 | 3 | 3 | 1 | 1 | 0 | 12.68 |
| after_event | 3 | 4 | 3 | 2 | 1 | 0 | 14.73 |
| parent_dispatch | 0 | 0 | 0 | 0 | 0 | 0 | 15.74 |
| angle | 3 | 3 | 3 | 2 | 1 | 0 | 12.75 |
| write | 5 | 5 | 3 | 4 | 1 | 0 | 12.57 |
| flush | 219 | 221 | 6 | 219 | 0 | 0 | 10.02 |
| worker_exit | 3 | 3 | 2 | 2 | 0 | 1 | 15.84 |
| reserved_exit | 3 | 5 | 5 | 3 | 0 | 1 | 11.50 |
| result_cap | 2 | 3 | 3 | 1 | 1 | 0 | 10.90 |
| log_cap | 214 | 216 | 6 | 214 | 0 | 0 | 7.75 |
| correctness_zero | 84 | 85 | 6 | 84 | 0 | 0 | 10.08 |
| stop_save | 2 | 2 | 2 | 1 | 1 | 0 | 12.41 |
| P1_inflight | 8 | 8 | 8 | 7 | 1 | 0 | 113.68 |
| P2_inflight | 2 | 2 | 2 | 1 | 1 | 0 | 104.04 |
| simultaneous | 8 | 8 | 8 | 0 | 8 | 0 | 8.65 |
| parent_crash | 不明 | 1 | 不明 | 0 | 0 | 1 | 不明 |

同期rendezvousによるP1では失敗callを含む8 ticketが残り、**7 trialが完了**。P2 cap2では2 ticketで**1 trialが完了**。障害専用のCPU busy loopを使ってin-flightを確実に作る検証で、通常workloadの失敗後費用予報ではない。

| 構成 | STOP後の親CPU ms | STOP後のworker CPU推定 ms | tick量子化誤差目安 ms |
|---|---:|---:|---:|
| P1_inflight | 7.64 | 810.94 | 80 |
| P2_inflight | 6.12 | 197.96 | 80 |

費用/結果を破棄せずraw prefixとfailure reportへ保存した。simultaneousでは8失敗全てを保存。parent crashは1 START_RESERVEDを保守的な不明枠として残し、実開始数/STOP後CPUをnullにした。8 childはEOFで終了、scope再利用を拒否。正確な物理関数entry時刻・process間例外発生全順序は保証しない。

sample結果前・event取得時・event取得後invariant・angle/requery・partial write・flush・global result/log cap・worker/reserved/parent crash・親dispatch例外・correctness不合格・STOP保存失敗を測定。output前/atomic admission commit直後/同scope二重host/STOP後retry/claim保存失敗等は36件のtestと既存275件に含む。詳細はtestsの個別ログ。

最終 **既存275＋新規36 = 311 PASS、skip0**。初回33件には初期化失敗時のResourceWarning、追加36件の初回には未発生COMPLETED key省略による1 errorがあり、修正して全36件を再実行した。初期ログも保存し、最終結果と混同しない。

artifact_verification.jsonは全72run/190,464人工trialのsummary・bit・費用partition・hash対応を照合。代表rawと全fault rawの355 chain/canonical file、24,365行、partial tail1件を再読。全runの正規化SHAが対応S0 rawと一致し、ID重複0。原科学結果の監査ではない。

再現：`PYTHONDONTWRITEBYTECODE=1 python3 scripts/tracks/algorithm_codesign/run_b3_parallel_artificial.py benchmark --output <fresh-directory> --repeats 3`。性能再現にはbundleの測定commit b1dba4b6を専用領域へ復元する。最終sourceのfaultsは同runnerの `faults`、保全監査は `verify_b3_parallel_artifacts.py <report-directory>`。共有package・OS・cache設定を変更しない。
