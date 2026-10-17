# A2固定DF pilot Phase B：初回計算完了・STOP

**`COMPLETED_FIXED_DF_CERTIFICATE_COMPARISON`**。固定科学SOURCE／正式契約Cの初回one-shotで、SPEC-01〜03とSPEC-04〜07の独立checkerが受理した。[raw result](../../../artifacts/qc_a2_phase_b_execution/2026-10-18/attempt_1/ledger/result.json)、[保存値summary](../../../artifacts/qc_a2_phase_b_execution/2026-10-18/saved_results_summary.json)、[byte監査](../../../artifacts/qc_a2_phase_b_execution/2026-10-18/attempt_1/evidence_byte_manifest.json)、[実行承認](execution_authorization_record.md)、[証拠索引](evidence_index.json)、[公開照合](publication_receipt.json)、[保全](preservation_audit.json)。これはローカルの固定DF certificate比較結果であり、immutable CIや外部による独立再実験ではない。

## 対象・固定条件

H4／1.00 Å／STO-3G／保存DF rank12。対象は保存binary64を正確係数としたfull DF Hamiltonian、total N=4、保存順8 modes、全12 DF blocks。PRのtail splitやδ-window探索は実施せず、one_body→block0〜11の固定順positive palindromic S2を使った。trialはoccupied modes 0,1,2,3のdeterminantでU専用。ground状態を計算・準備したことを意味しない。

d=1/1000 Ha、α_spec=1/10、sqrt_bits=96、root_width=2^-40 Ha^-1、max_bisections=256、ENTRYWISE_RE_IM_ABS_SUM_V1、RF0/RF1/EW2対centered・grouped構造付きCbar2を維持した。α_specは固定保証のspectral parameterであり、新たな測定confidenceではない。record各1 TiB／log各1 GiB、CPU/RAM/wall人工capなし、共有設定変更なし。保存容量は上限の予約・利用目標ではない。

SOURCE S `8d483f9bfa5983bb52fef55887e2f9f73a790511`。正式C commit `a7f8307123f6ba5bbb8d8b84e124400ac04a60ce`、raw C SHA256 `c3b13d5007636f83def99bb04bbb00d31dbbdeaff9a6474a4a4319f9ea403c35`。初回authorization commit `2d39c1cd75a8853043ae322d228363739a98fbcb`。入力commit `34d9a2f1a25bd35c03566ce0635e8b66665ad996`、container SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`、input manifest `c70a49aa6adf13b03f3821c4ecb546f0d570ca4db17a097fb371658aef0d139b`。SOURCE31 filesと採択済みCは実行前後にbyte/hash保全を確認する。

## 保存された検証値

以下はcheckerのexact Rat出力の**表示用丸め**であり、十進表示を新たなcertificateや不等式判定へ用いない。exact値・parent context・全probesはrawとsummaryに保存した。

| 値 | 表示値 | 単位 |
|---|---|---|
| L | -13.6185006472 | Ha |
| U | -2.098545937 | Ha |
| W | 11.5199547102 | Ha |
| Lambda | 71.2833149076 | Ha |
| K_SOS_upper | 551.865627899 | Ha² |
| Cbar2 | 0.444112532037 | Ha³ |

| route | 十分step境界のenclosure（Ha^-1、表示丸め） | 分類 | bisections |
|---|---|---|---|
| RF0 | [1.29926081615e-05, 1.29926089781e-05] | ENCLOSED | 32 |
| RF1 | [0.000170959270173, 0.00017095927099] | ENCLOSED | 32 |
| EW2 | [0.000170959270173, 0.00017095927099] | ENCLOSED | 32 |
| CENTERED_CBAR2 | [0.016858378763, 0.0168583787639] | ENCLOSED | 35 |

step全体の分類は `ENCLOSED`、比較は **`COUNTER_LARGER_SUFFICIENT_STEP`**。これは同じ固定DF・証拠・d/α・保証適用域での**十分step上限の比較**であり、実際のPF限界・最大安全step・量子資源の勝敗を示すものではない。両routeへ同じ検証済み証拠を与え、強いCbar2対照を粗く弱めてはいない。

## 実行・独立性・費用

実行前に既存人工test134 methodsを再実施し、失敗0件。人工testは実H4に対する追加科学callではなく、テスト内のauthority/claimも人工memory storeに限る。初回の実行はcanonical checkout `/tmp/prt-qc-a2-pilot-phase-a-20261018`、SOURCEの既存symplectic可換pair除去・mask cache・exact Walsh batch traceを使用した。証拠受理に依存するpipeline順序を保ち、float/GPU/BLAS置換・共有環境変更は行っていない。実測比較のないspeedup倍率は主張しない。

独立checkerは生成結果を受理flagだけで流用せず、保存binary64からのexact intake、独立bit/CAR/occupation-matrix経路、K0辞書とsqrt不等式、grouped交換子とstep不等式を再検査する。generatorはJW Pauli、commutator checkerはoccupation行列とexact traceを使う。schema/hash/NPZ adapter/Python整数・Fraction・既知数式は共有し、完全に独立開発された形式証明やプラットフォーム認定とは呼ばない。NPZ→raw mapperは共通trusted前処理である。

開始 `2026-10-17T18:47:35.334795+00:00`、完了 `2026-10-17T18:53:17.838351+00:00`。process peak RSSは `721304` KiB。取得費用は無料とせず、以下に生成・検査・保存を区別した実測を示す。実行監視の原記録は [resource observations](../../../artifacts/qc_a2_phase_b_execution/2026-10-18/resource_observations.jsonl)、[supervisor結果](../../../artifacts/qc_a2_phase_b_execution/2026-10-18/supervisor_final.json)。

| stage | wall秒 | CPU秒 | その時点までのprocess peak RSS KiB |
|---|---|---|---|
| OUTPUT_SERIALIZATION_AND_WRITE | 0.000602750 | 0.000195640 | 19712 |
| INPUT_IO_AND_PINNED_BYTES_CHECK | 0.001474604 | 0.000178060 | 19712 |
| RAW_SNAPSHOT_PERSISTENCE | 0.000491485 | 0.000108731 | 19712 |
| adapter | 0.001892956 | 0.001889312 | 19712 |
| OUTPUT_SERIALIZATION_AND_WRITE | 0.000632268 | 0.000246752 | 19712 |
| SPEC01_03_generator | 0.034065480 | 0.034004443 | 21248 |
| OUTPUT_SERIALIZATION_AND_WRITE | 0.007869322 | 0.006649156 | 21248 |
| SPEC01_03_checker | 0.031941346 | 0.031916049 | 22784 |
| OUTPUT_SERIALIZATION_AND_WRITE | 0.001479466 | 0.000254873 | 22784 |
| SPEC04_07_generator | 0.044870670 | 0.044850514 | 23040 |
| SPEC04_07_generator/TRIAL_AND_POLICY | 0.005841239 | 0.005840980 | 23040 |
| SPEC04_07_generator/SPEC04_UPPER | 0.002511394 | 0.002511380 | 23040 |
| SPEC04_07_generator/SPEC05_KSOS | 0.029035078 | 0.029007598 | 23040 |
| SPEC04_07_generator/SPEC06_COMMUTATOR | 218.457223173 | 217.972016404 | 342956 |
| SPEC04_07_generator/SPEC07_STEPS | 0.534835510 | 0.534696786 | 344436 |
| OUTPUT_SERIALIZATION_AND_WRITE | 2.465175925 | 2.453224450 | 344436 |
| SPEC04_07_checker | 0.580474107 | 0.580117291 | 344704 |
| SPEC04_07_checker/EVIDENCE_STRUCTURE | 1.513057902 | 1.507306761 | 582188 |
| SPEC04_07_checker/SPEC04_UPPER | 1.704099742 | 1.695547312 | 582188 |
| SPEC04_07_checker/SPEC05_KSOS | 0.021150763 | 0.020982044 | 582188 |
| SPEC04_07_checker/SPEC06_COMMUTATOR | 116.447777066 | 116.111933075 | 721152 |
| SPEC04_07_checker/SPEC07_STEPS | 0.006132496 | 0.006128018 | 721152 |
| SPEC04_07_checker/HASH_SOURCE_LIMITS | 0.605481627 | 0.605126202 | 721304 |
| OUTPUT_SERIALIZATION_AND_WRITE | 0.003149869 | 0.001643011 | 721304 |

SOURCEのruntime計測は最終log/result書込みを除き、RSSは累積process peakでstage単独の増分ではない。表の親stage名の行はobserverが細分stageへ移る前の区間で、子行の小計ではない。supervisor wallはpoll周期と最終化を含む。量子query／T／CX／depth／state preparation／native費用は未知のまま保持し、0や古典CPU時間へ換算しない。

## 試行・保存・解釈の限界

科学pipelineは初回1回、技術修正0回、再試行0回。guard後input IO前のclaimは消費済みでcanonical ledgerに残した。ユーザーの技術的中断時の修正・継続指示を記録したが、今回それを使う追加試行は不要だった。過去の契約・SOURCE・失敗・grant・markerは変更しない。全ledger原bytesを元の場所に保持し、公開copyとSHA256を照合した。raw result SHA256 `1c66113d8e18626a7e682879682fdde221ff1495e16e8d4271ffaa5d0b69f281`。ledgerの合計保存bytesは `67523176`、管理logや公開copyなどの別容量を含まない。

exact fixed DFを対象とした受理であり、元電子Hamiltonianへの厳密保証、DF/入力生成誤差のcertificate、実際のground preparation、QPE成功、native総費用・総資源改善は未確立。結果から新しい数学保証・研究GO/NO-GO・A2正式採択を認定しない。A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認・未実行を維持する。SPEC-08・新分子・新しい対照・QPE/native・科学レビューへ進まない。

**公開・保全確認後mandatory STOP。次の担当はユーザー／GPTで、保存された証拠取得費用と十分step比較の科学的意味を判断する。重要な詳細科学レビューの開始・研究方針採択・追加計算は、それぞれ別途承認が必要。**
