# H4 DF10 N側batch adapter整備（2026-11-10 JST）

H4固定13-roundの不足Nを`pf_pr_rpe_pipeline.run_missing`へ一括供給する入口を完成した。今回の証拠は人工行列・mock・保存metadata確認に限る。実H4の新signal/N、保存signalの位相・半径・N再解析、価格予測、実N×C/Gは実施していない。H6全13-round結果も再計算していない。公開照合後mandatory STOP。

## 再利用とH4差分

| 役割 | 使用実装・保存SOURCE | 今回の変更 |
|---|---|---|
| 共通task/N/C record・欠測呼出し | `pf_pr_rpe_pipeline.py`、B統合`cb2b1d6b4309ca7d705f397e25e27d14e8ba125a` | 不変。N callbackのみ接続 |
| H4 phase/radius/disk/Hoeffding | `h4_df10_rpe_d1.analyze_bound_point`＋`h4_df10_rpe_connection` | 算術・decoder・旧D1 guard不変 |
| 保存H4 loader | `155529e20a8fac39e00786c40b57830b71203c4f:scripts/h46_m5_tol_saved_input.py` | 保存layout専用の`array_record/load`関数本体を再利用。旧launcher/contractは起動しない |
| 行列作用・2経路reference・finite RTE | 既存`h6_df19_rpe_d2`の汎用作用部分 | H4の8-mode/36-sectorと保存tailを渡す。H6の`analyze`・m5・DF19条件は使わない |
| 一括制御 | 新`h4_df10_rpe_batch_provider.H4NBatch` | 保存m3→コピー、m4→保存signalのscalar、他→共有context/reference/cacheから新規伝播 |
| N専用入口 | 新`scripts/run_h4_df10_rpe_batch.py` | 既定はmetadata計画。`--execute`は別途承認後の科学入口。fresh出力、再試行/再開なし |

PFは**B0/prefix7**であり、保存Full-Hをreferenceにしてtailを捨てた決定論prefixを伝播する。汎用kernelの`FULL`分岐は渡されたprefixのループなので、内部だけB0をこの分岐へ対応づけ、外部candidateはB0のまま保持する。scalarは保存constant `c`のみを総Tで一回適用し、PR用tail identityを足さない。PRはprefix1、保存tail scalar、R_outer4/K2、S2の1 piece/allocation `[4]`、event数`4*q_PF`、既存finite RTE normalizationを維持する。corrected/rawは別columnで伝播し、closure診断を残す。PR exact-tailは内部対照である。

新しいsolver、Hamiltonian/state生成、固有値問題、SCF/DF fit、sampling、回路build/compileは追加していない。process内BLAS設定と資源観測に応じた既存worker選択を再利用し、共有環境設定は変更しない。half-step作成と候補伝播でそれぞれ実task数に応じて並列数を選ぶ。

## 固定条件と保存データ

Linear H4、隣接1.00 Å、STO-3G、tol-only 1e-8 DF10、保存weighted completion/fragment順、8 mode、Nα=Nβ=2・dimension36。対象は保存DF-Hの同sector基底エネルギー、保存数値stateを使う条件付き評価、P=0。S2、δ=.1、h=.1、m=0..12、`T=.1*2^m`、`q_PF=2^m`、PF B0/p7対PR p1/R4/K2を固定する。q_RPEとq_PFは今回は数値が等しいが別fieldで保持する。

RPE条件は既存D1からコピー：精度0.00159360010199040 Ha、各方式αtotal=.05、各軸α=.05/26、βtotal/stat=.4/.2、prior `[−4.994249624417489, −2.0985459369977186]` Ha、energy shift0、既存consecutive dyadic decoder。prior/horizon/枝選択を再計算せず、state energyは診断にのみ用いる。

| binding | SHA256 |
|---|---|
| Hamiltonian | `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0` |
| Snapshot | `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96` |
| State | `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b` |

保存入力は公開SOURCE索引`7a8bc7f132e166390c943e6b32c92cc6abe1baf4:artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json`のH4項を使う。今回、receipt/context JSONとファイルhashを照合した。NPZ配列は読み込んでいない。歴史的binding内の`T=.8`は変えず、実roundのTはcandidateで照合する。

| m | T / q_PF | PF / PR candidate ID | 既存N・入力coverage |
|---:|---|---|---|
| 3 | .8 / 8 | `49709717a6c3d405c60d` / `14bd81a375400c9b0305` | 各軸705 / 977を既存結果からそのままコピー |
| 4 | 1.6 / 16 | `8e50c5560bb8f47f026f` / `64e4f7e9ec7fba28815e` | reference/corrected/raw/bias/logB/u/marginを静的選択。RPE Nは未評価 |
| 0,1,2,5..12 | .1,.2,.4,3.2..409.6 / 1,2,4,32..4096 | 全固定IDは[PLAN](PLAN.json)のtask | 新reference・finite meanとscalarが必要 |

m3は`cefc01622a3cd75dd16d30de7b9fb68274f3eddd:docs/research/h4_df10_rpe_d1_20261110/result.json`、科学SOURCE `311255aaac39dc1c3321108c296bde7dc93dd543`を参照し、ideal/actual laneを値変更せず保存する。

m4は`155529e20a8fac39e00786c40b57830b71203c4f:artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/signal_records.json.gz`と`signal_endpoints.json.gz`の同IDを使う。既存D1と同じendpoint selector ε=.0005から数値allowance/marginだけを選ぶ。**旧有限時間Nやαaxis=.025は移植しない**。旧endpointのELIGIBLE/INELIGIBLEは旧εの判定であり、RPE round PASS/FAILではない。m4の保存referenceはfull-H spectral経路で、独立expm_multiply検査は旧T=2での証拠である。T=1.6の新しい2経路検査が済んだとは扱わない。

## N計算入力・資格とFAIL処理

各roundには固定candidate/time/recipe、H/state/snapshot、reference/corrected/raw/total signed bias、logB、座標allowance u、logB margin、保存数値state energy/residual、β/α/prior/decoderが必要。新時点は既存`expm_multiply`/`expm`のreference差gate1e-10、norm/closure gate1e-9を保つ。H4ではT=.4も新referenceであり、H6保存T=.4の特例は移植しない。

新しいH4 allowanceは旧H4 `qualify_record`方式（stage数`q_PF*2*(prefix+1)`、roundoff、closureの10倍、floor1e-9、logB margin floor1e-12）を再利用し、reference二経路差を加える。これは経験的数値allowanceで、rigorous上界ではない。保存m4は保存allowanceを維持する。既存H4 scalarがbiasを一度だけ会計し、reference offsetとT×保存residualを加えてdisk/phase/radius/Hoeffding十分Nへ接続する。

厳密ground同定、actual実装/law誤差、rigorous誤差は未資格。actual laneの不明量とNはnullのままである。ideal数値laneの条件付きPASSは、数学的ground保証・最小shot・全RPE成功保証を意味しない。

保存NがFAILなら共通pipelineがprovider前にSTOPする。m4入力不足・binding不一致は未資格として残し、新規伝播へfallbackしない。保存signal scalarが不成立ならfresh科学計算へ進まない。新reference gate失敗は例外を保存してSTOP、候補gate失敗はFAIL recordと残存診断を保存し、閾値変更・候補変更・科学retryは行わない。coverageは未取得・FAIL・source失格・ideal条件付きN・actual不明を区別する。

## Codex Cとの接続

共通pipeline・既存価格providerを一切変更していない。N callbackは`provider(task, rows) -> list[N record]`、`run_missing(..., n_provider=provider, c_provider=None)`で一度呼ばれる。共通recordは従来の`identity/plan/actual/diagnostics/source/origin`である。identityにcandidate ID、m、q_RPE/q_PF、signed recipe、scientific binding、configuration、compiler/control/full-wrapper profileを保持する。

将来、実行完了statusとSOURCE/input保全が成立した`result.json.records`を既存pipelineのshots入力に渡せる。実行後のSOURCE/input不一致では新recordのNをnullへ失格化し、計算時laneは診断として保存する。`COVERAGE.json.rows`にも同identityとN/status/source/normalizationを出す。C record schema・価格モデル・cohort/expectation/domain/共有不確かさの扱いは変更不要。BやattenuationをCへ再乗算しない。今回はC lookup/予測・費用会計を呼んでいない。

## 技術確認

[TEST_LOG](TEST_LOG.txt)：新13テストPASS。静的import/JSON選択にnumpy/scipy/qiskitのロードなし、m3 N lane完全一致、m4保存入力availability、固定条件拒否、人工非可換2-levelのB0 prefix7とPR S2/R4/K2の既存spectral経路照合、scalar一回会計、独立raw closure、unknown actualを確認した。full26 mockではm3コピー2・m4 scalar2・fresh22・reference11×2・共有half-step8と候補22のworker選択・C形式受入れを確認した。欠落入力/FAIL/例外/再試行なし、出力保護、SOURCE保全も検査した。実H4 signalを使ったN算術のテストではない。

共通pipeline、C provider、H6数値kernel/全13-round結果、過去H4結果・固定SOURCEは保全した。別AIによる独立科学レビューは実施していない。旧SOURCE/KEEP/grant/ledger/marker/mandatory STOPは変更していない。

## 次の最小H4科学実行案（未実行・別途承認が必要）

1. m3のPF/PR既存N2件をコピーし再計算0。
2. m4保存signal2候補についてphase/radius/条件付きNをscalar評価する。
3. 他の11時点 `.1,.2,.4,3.2,6.4,12.8,25.6,51.2,102.4,204.8,409.6` にreference各2経路、固定PF/PR計22 finite mean、PR内部対照11を一括評価する。
4. 新scalarは計24候補・X/Y48軸。FAILを含め13-round coverageへ統合し、機械可読recordを価格担当へ渡す。C/G/最適化/再試行なし。

技術計画だけのコマンド：`python -B scripts/run_h4_df10_rpe_batch.py --output-dir <fresh-plan-dir>`。

上記範囲の明示実行許可後にのみ：`python -B scripts/run_h4_df10_rpe_batch.py --execute --output-dir <fresh-H4-results-dir>`。このコマンドは今回実行していない。実行環境は既存project venvを用いる。

担当：ユーザーがこの固定24候補の科学実行範囲を判断し、許可後にCodex Bが一括実行する。Codex Cは既存C interfaceで別途価格側を担当する。本成果公開・remote照合の後はmandatory STOPし、科学計算へ自動進行しない。
