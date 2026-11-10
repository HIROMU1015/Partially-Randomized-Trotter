# H4 DF10：PF/PRのRPEエネルギー推定総資源への接続準備

2026-11-10 JST。**保存値の読取・接続実装・人工検証まで完了。実H4の新signal、N、G、sampling、build/compile、RPE simulationは0。公開照合後mandatory STOP。**

主研究は[H4/H6長時間PF・PR、155529e2][MAIN]。今回はH4 DF10だけを対象とし、B3・Full-H pure RTE・新A Referenceを比較armへ追加しない。並行[shot調査][SHOT]と[価格調査][PRICE]を引き継ぎ、両調査を再実施していない。

## 完成物と到達点

- [保存候補とRPE入力の対応表](CANDIDATE_INPUT_MAP.md)、[読取専用の保存欄projection](SAVED_POINTS.json)。T=.8と2.0の既存代表PF/PR計4候補を、旧best_configurationsの行から転記した。新しい候補選択・順位付けではない。
- [薄いadapter](../../../src/trottertracks/h4_df10_rpe_connection.py)、[人工test](../../../tests/check_h4_df10_rpe_connection.py)、[15件PASSの結果](ARTIFICIAL_TEST_RESULTS.txt)。既存のraw幾何・Hoeffding・decoderを再利用。旧会計がX/Y等Nに限定されるため、軸別Nの小さい会計関数だけ追加した。
- [SOURCE・コード索引](SOURCE_INDEX.md)、[D1の具体案とD2/D3の条件](D1_PLAN.md)。新しい汎用solver・科学runner・approval証明機構は作っていない。

**保存データを渡すための実装は揃ったが、DF10のRPE phase/radius/reference資格と全round条件は未採択・未評価である。** 実装PASSをエネルギー成功保証に読み替えない。

## Phase A：既存の正式条件と、今回ユーザーが決める条件

| 項目 | 既存の正式条件・実績 | 今回の選択肢・推奨案（未採択）／計算への影響 |
|---|---|---|
| 1. energy target | 長時間研究のreferenceは、保存DF10の一体項・全DF fragment・scalarを含むfull-H。コードのfull-Hは未分解の元積分Hと同一とは限らない | **最小D1は保存DF-Hをtargetとする案**。元HamiltonianをtargetとするならDF/projection差をsignal又はenergyへ一回だけ接続する追加証拠が必要 |
| 2. state・reference・radius | 同じH4保存state/hash、8qubit／Nα=Nβ=2 sector36。元receiptは`ground_state_certified=False`。保存residual・state_energy・reference信号はあるが、新しいeigenpair評価はしていない | 数値固有状態を使う**条件付き診断pilot**か、厳密groundへの保証を要求するか。前者でもreferenceのphase/radius差を誤差diskへ含める。後者は追加の状態・reference誤差資格なしにNを認定できない |
| 3. εE | 既存ε=.05..0005はcomplex signal精度。DF12の約1.5936e−4 Ha、Dの.00159 Haは別taskの実績 | 共通εEを別途選ぶ。D1は単一時間の接続診断なので、選んだεE達成・最終energyは認定しない。全round数はD2で決める |
| 4. αtotal | 主研究はRe/Im各.025。これは全RPEの配分ではない | 各algorithmの全RPE失敗上限（例.05又は.01）、D1用αX/Yを明示。D1だけ各.025とする案は二軸pilot限定。PF/PR両実行のjoint保証まで要求するならさらにfamily配分を明示 |
| 5. β・branch | 既存geometry/Hoeffding、dyadic nearest-branch decoderあり。旧β=.4、Dのπ/6・stat π/12を無条件移植しない | βtotal・βstatとreference/決定論誤差の余地、測定位相符号、外部energy prior・初期alias条件を選ぶ。保存E0をdecoderのbranch選択oracleにしない |
| 6. base time・horizon | 保存時間は.8/1.2/1.6/2.0。q_PFはその時間内のstep数。RPE q=2^mとは別 | **D1 T=.8、base h=.8、m0の案**。そのD2なら保存.8/1.6は同schedule候補、1.2/2.0は入らない。M/TmaxはεE・β・priorに従い既存horizon関数で後に評価。13/18roundは未採用 |
| 7. PF/PR候補 | 原研究はFull/B0/B0I/PR、S2/Y4/m5、q/prefix/R/Kを有限menuで比較済み。今回の4候補は保存ALL best行の参照例 | 最小D1案はT=.8のS2 B0/p7/q8対PR/p1/q8/R4/K2を固定。B0は切捨てPFで、捨てた項のbiasを既存total biasに含む。Full PFのみをbaselineとするか、B0/B0Iも認めるかはユーザー判断。新最適化はD1に含めない |
| 8. 資源・価格domain | 代表4候補にRZ/CX/三depth/sizeの保存直接価格あり。PRは探索n9と独立確認n33を別cohortで保存 | 主指標RZ、CX等を副指標とする案。D1では価格の対応確認まででGを計算しない。D3は同時間・構成・compilerの直接価格又は明示的モデルだけ。D CDF8係数・改善率は移植不可 |

上記は選択肢の技術整理で、科学条件の正式採択ではない。H6 DF19への拡張は今回の範囲外。価格・state・法則の条件差を隠して科学的同一性を作らない。

## Phase B：保存有限時間点で確認できたこと

共通bindingはLinear H4／1Å／STO-3G／tol-only1e−8／rank10、保存H/state/DF fragment順。Hamiltonian hash `c1a6c428…`、state `9bcc0b0e…`、snapshot `12219b7d…`。完全値はSAVED_POINTSと主研究scientific_provenanceに保存。

| 時間 | 保存PF候補 | 保存PR候補 | 接続に使える保存欄 |
|---:|---|---|---|
| .8 | `49709717a6c3d405c60d`：B0、S2、prefix7、q_PF8、δ=.1 | `14bd81a375400c9b0305`：S2、prefix1、q_PF8、R4/K2 | 各candidateのreference/corrected/raw/total_signed_bias、log B、u/旧margin、直接価格。PFは再利用直接価格、PRは独立確認価格もあり |
| 2.0 | `e9c904a137c2f96a5bf0`：B0I、m5、prefix7、q_PF1 | `fb134f11827f3e240843`：m5、prefix3、q_PF1、R32/K6 | 同上。signed piece・整数R配分・normalizationは元SOURCEのまま。今回は追加armとして採択せず代替例を提示 |

PF/PRは同じ保存H/state/評価時刻のfull referenceへ比較されている。ただしT=.8のreference数値は元保存経路の違いで末位が異なる。各値をそのまま残し、一方へ上書きしたり、新たな共通referenceを計算したりしていない。PFの切捨てとPRのD/R分割は近似の構成差であり、target自体の差へ置き換えない。

保存Bは全interrogationのlog Bとして取得できる。`raw=corrected/exp(log B)`は元SOURCEの規約。今回B、attenuation、phase、radiusを再評価せず、対応表の新数値欄はnull。既存PFのlog B=0、PRの正log B、既存log_B_margin/uは減らさずコピーした。古い有限時間Nも専用欄に保持し、RPE Nへ代入していない。

## Phase C：adapterの範囲と人工検証

- `inspect_saved_point`は保存値をcopyし、H4/DF10・candidate/価格・compiler/full-wrapper scopeを照合するだけ。B/A・raw補完・新N・Gの算術を呼ばない。
- `bind_saved_round`は明示したh/mに保存Tが一致する時だけ対応付け、q_PFとq_RPEを別欄へ置く。別時刻への補間・冪乗はしない。今回の実データには未採択h/mを付与していない。
- `assess_supplied_round`は、将来許可された評価から渡す**state-specific signal disk**、reference/stateの対応、raw単位のnormalization/implementation/law誤差、β・軸別αを要求する。既存`c1_stage3_accuracy._assess_raw_signal`と`rpe_resource_accounting._hoeffding_shots`を呼ぶ。保存軸biasを`OriginalHRoundEvidence.operator_error`へ渡さない。未知誤差・reference欠測はN=null。現adapterは数値pilotの条件付き結果だけを返す。
- `failure_budget`は全軸α和を確認。`aggregate_supplied_axes`は旧等N集計の未対応部分だけを軸別に拡張し、ΣNX CX＋NY CYを算術評価する。共通target/base time、全round/両軸coverage、価格candidate/compiler、direct/model、モデルqの定義を維持する。価格trajectory nやBを再乗算せず、幅はpeak。不確かさ・energy成功は自動認定しない。

人工15件は、α和、phase/radius、B一回、軸別ceil N、q二種、time mismatch、candidate/価格ID、direct/model/外挿metadata、欠測・未知をnull、共通schedule、旧集計との等N一致を確認した。toy decoderも元関数で検査した。数値入口を実H4に適用した回数は0。既存helperの依存library importはあるが、原研究runner/context/evolveは起動していない。

## 新compileなしで進められる範囲と追加計算

**D1は新compileなしで実施可能な候補がある。** 同じ4保存pointにはsignal・経験的allowance・価格が揃う。許可後、保存scalarのreference/phase/radius/bias会計と十分N接続をまとめて行える。ただしstate/energy reference条件が不足すれば、未資格として止まり、追加計算を自動で始めない。

**新しいsignal計算が必要なのは**、採択T_mに一致する保存値がないround、構成やstate/lawを変える場合、又は必要なreference/state誤差が既存値で閉じない場合。既存`h46_long_time_signal.evolve_batch`を必要な点だけ使う計画とし、保存Hamiltonian/stateを再生成しない。固定構成と時間が一致する点は再evolveしない。

**D3の主な不足は**、全roundの十分N・共通energy/branch条件と、そのT/δ/q_PF/R/Kで有効な価格。D C6〜C8やDF12の価格係数では埋められない。直接価格からの新fit・外挿資格・新Gは、別途範囲を定める。価格modelのqがq_PFかq_RPEかも明示する。

## branch・SOURCE保全・公開とSTOP

既存`codex/resource-eval-rpe-shot-error-connection-20261110`を継続した（作業前local/remote `fd76a3ed`、clean）。同目的の接続準備を進めるため新branchは不要。主研究155529e2、前回価格調査caa592a8、D固定成果と旧reader/guardは変更しない。新実装・保存欄projectionは独立namespaceに置き、原runのSOURCE/input/raw/grant/ledger/marker/KEEP/STOP、他worktree・他process・共有設定を保全する。

共通AGENTS、worktree AGENTS/PROJECT_MAP、正式共通指示2026-11-09を適用。研究概要は今回の新しい追補を読み、既読141fa7624との同一suffixを照合して再利用した。主研究と再利用コード・テストを関連範囲だけ確認し、私有receiptはstate metadataのみ読んだ。eigenpair・NPZ再解析や全raw監査はしていない。作業用sparse-checkoutは自分のworktreeのsrc/tests/必要READMEだけ展開し、既存runtimeを変更せず利用した。

新たな詳細科学レビュー、実H4のsignal・N・G数値再解析、価格fit、trajectory、回路build/compile、全RPE simulation、Main Claim変更は行っていない。保存データprojectionは欄の選択・転記とmetadata照合だけ。人工テストのPASSは分子性能・ground資格・energy成功の証拠ではない。

GitHubへ公開し、remote SHA/treeと成果物本文を照合して**mandatory STOP**。次の担当は**ユーザー**：Phase AとD1_PLANの科学条件・最小実行範囲を判断・承認。**Codex**：承認後のみD1の保存scalar評価をまとめて行い、それまでは新しい科学処理を開始しない。

[MAIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[SHOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fd76a3edaf5b6fee9cadb556ba57efc7c46ed778/docs/research/rpe_shot_error_connection_20261110/REPORT.md
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/REPORT.md
