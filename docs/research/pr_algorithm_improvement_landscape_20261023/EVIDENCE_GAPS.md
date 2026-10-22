# 不足資料と判断への影響

2026-10-23 JST。公開GitHubの全branch inventory、現在の取得済みGitオブジェクト、登録worktree内の既知UWC候補を確認した範囲で整理する。他環境に存在しないとは断定しない。A/B/Cの判定は[カタログ](STUDY_CATALOG.json)へ研究単位で付与し、ここでは具体的な不足を扱う。

**GAP-001とGAP-008は当該系列の成否判断を止める。** 他のGAPは主に既存限定scopeの独立照合・historical来歴を制約する。新しい全RPE保証、strict native、実状態準備、独立新規性が未確立であることは、既存ファイルが欠けていることと同一ではない。新しい証拠を要する場合は別科学作業として返す。

優先度は資料回収の依存関係であり研究候補の正式順位ではない。回収の実行は今回開始しない。[別環境向け指示案](OTHER_ENVIRONMENT_HANDOFF.md)をGAPごとに用意した。

<a id="gap-001"></a>

## GAP-001 旧BLISS/UWCのrun原データ

| 項目 | 内容 |
|---|---|
| 対象研究 | REP-03 |
| 確認できたbranch/commit/path | [REP-03 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/notes/uwc_current_implementation_and_results.md) |
| 不足資料 | grouped_uwc_pf_qpeの比較JSON、H3 sector/time-grid診断、theta sweep JSON/CSV、baseline入力・実行commit/command/environment。 |
| 不足判定の根拠 | UWC本文がprose-only / raw artifacts missingを明示。公開branch横断のpath inventoryで対象rawを確認できず、現環境の登録worktree 211件の既知artifacts/grouped_uwc_pf_qpeディレクトリ候補も0件。 |
| 判断できないこと | 小さな改善・悪化訂正の妥当性、同じHamiltonian/sector/grouping/time-fitでの比較、PR転用可能性を定量判断できない。 |
| 所在の可能性 | 過去にUWC GPU batchを実行した環境。具体的host/SSH接続先の裏付けはない。 |
| 探す対象 | H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json、H3_bliss_sector_scaling_diagnostics.json、grouped_uwc_pf_qpe配下の既存theta sweep・診断・run log。 |
| 必要な来歴・条件 | 原ファイルSHA256/bytes、生成source commit、元/変換後H hash、target particle sector、basis/geometry、theta/power、grouping、PF label/time grid、fit規則、backend/依存版、commandとUTC/JST日時。 |
| 回収後の照合 | 旧H3誤sector結果と修正版を別runにし、表とrawの対応を技術照合する。新fit・再ランキングは実施しない。 |
| 優先度 | 必須（この系列の改善成否を判断する場合） |
| 他研究の継続 | 他系列の調査は継続可能。UWC数値を有効結果・成功例にしない。 |

<a id="gap-002"></a>

## GAP-002 旧H4のevent履歴・未丸めnormalization・親exit来歴

| 項目 | 内容 |
|---|---|
| 対象研究 | EVAL-02 |
| 確認できたbranch/commit/path | [EVAL-02 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md) |
| 不足資料 | O05の実event履歴、172件の未丸めnormalization原記録、O06のB0+I親driver native wait/exit数値。 |
| 不足判定の根拠 | Track A 2026-10-18 handoffがNOT_SAVED／未回収／UNKNOWNを区別して列挙している。後続のH8層別採択はこの旧H4来歴を補う証拠ではない。 |
| 判断できないこと | 旧H4のevent→価格join、厳密なnormalization来歴、親processの正常回収を一次記録から全面確認できない。 |
| 所在の可能性 | 当該H4 geometry/B0+I runを実施した環境。公開handoffのrun identityが手掛かりで、hostはここで推定しない。 |
| 探す対象 | O05/O06のrun別manifest、worker/nativeログ、親driver wait/return receipt、丸め前canonical normalization、event spool。 |
| 必要な来歴・条件 | 固定source/plan/result commit、candidate/axis/trajectory ID、compiler/runtime、表の各行へのjoin、原ログbytes/SHAとrun時刻。 |
| 回収後の照合 | 既存完成statusとは別に保存範囲を確認し、NOT_SAVEDと未回収を更新案で区別。worker成功から親exitを推測しない。 |
| 優先度 | 重要（旧H4の独立監査を強める場合） |
| 他研究の継続 | 限定benchmarkとしての既存結果は条件付きで利用可能。新samplingで履歴を代用しない。 |

<a id="gap-003"></a>

## GAP-003 M1-S旧Gate 0 v5のfull-H／HDと一部review原本

| 項目 | 内容 |
|---|---|
| 対象研究 | RB-15 |
| 確認できたbranch/commit/path | [RB-15 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md) |
| 不足資料 | v5実行に対応するfull-H/HD行列と完全入力mapping。一部旧Claudeレビュー原本は統合記録でも未取得。 |
| 不足判定の根拠 | M1-S統合summaryは、保存tail/state等の照合はPASSだがfull-H/HD未保存のため全scope UNDETERMINEDと明記。後続finite-stepモデル差会計を旧入力の直接一致へ読み替えない。 |
| 判断できないこと | 旧Gate 0を入力全体から独立照合できず、二次引用された旧reviewの完全な帰属も確定できない。 |
| 所在の可能性 | Gate 0 v5実行時の保存領域、またはユーザーが保持するreview原本。存在を保証しない。 |
| 探す対象 | v5のSOURCE/input manifestで指すH_full/HD、ordered basis/Pauli registry、保存stateとsolver条件。必要ならレビューinner.md、レビューinner(1).md、レビューinner(2).mdの原本。 |
| 必要な来歴・条件 | v5 source/contract/result SHA、行列dtype/shape/order/hash、sector・scalar・mapping・projection、state hash。reviewは作者/作成日/当時の対象commitと原bytes。 |
| 回収後の照合 | 既存の独立照合範囲と欠測範囲へbindする。数値再生成・新固有状態取得・新reviewは別承認とする。 |
| 優先度 | 重要（旧Gate 0全scopeの独立照合を要する場合） |
| 他研究の継続 | 適用域付き理論・保存signal・Stage Sの記述は継続可能。全scope再現済みとはしない。 |

<a id="gap-004"></a>

## GAP-004 FS-C0.5/C0.6のhistorical H01原入力

| 項目 | 内容 |
|---|---|
| 対象研究 | PF-06 |
| 確認できたbranch/commit/path | [PF-06 固定資料](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md) |
| 不足資料 | N2/COのhistorical H01 pickle、指定Hamiltonian SHAとordered-group SHAに一致する元payload、CISD/state・removed scalar・basis/sector来歴。 |
| 不足判定の根拠 | source_missing_followupはNO_GO_SOURCE_MISSING、C0.6はH SHA不一致でNO_GO_RECONSTRUCTION_MISMATCH。別new-source R0/R1が後続し、historical復元路線は終了している。 |
| 判断できないこと | 旧sourceをsame-input実験として再利用できない。new-source prediction/truthと旧truthの混合を避ける必要がある。 |
| 所在の可能性 | 当時のH01/H02 server cache。既存記録ではユーザーもSSH alias/host/IPを特定できていない。自動探索しない。 |
| 探す対象 | historical_fingerprint_registry.json、source_contract.md、candidate_source_audit.jsonが列挙する原pickle/Hamiltonian/CISD/group payload。現同名H02を一致済み扱いしない。 |
| 必要な来歴・条件 | registryの期待SHA、各objectのbytes/dtype/order、basis/geometry/active-space/sector、removed constant、solver/BLAS/依存版、実行commit/時刻。 |
| 回収後の照合 | read-only identity照合のみ。微小なenergy差・functional近似でhash契約を緩めず、旧NO-GOと新R1を保存する。 |
| 優先度 | 重要（historical実行の同一性が必要な場合） |
| 他研究の継続 | 第一研究完了やnew-source Phase Aの記述は可能。回収しても旧STOPの解除・Phase B実行許可にはならない。 |

<a id="gap-005"></a>

## GAP-005 HF/HCl・prospectiveの絶対energy identity tuple

| 項目 | 内容 |
|---|---|
| 対象研究 | PF-05 |
| 確認できたbranch/commit/path | [PF-05 固定資料](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/second_study_v2/evidence_integration_20261006/approved_closure.md) |
| 不足資料 | 同一H、sector、energy origin/scalar、physical branchを結ぶabsolute PF/H-reference tuple。HF/HCl 12件とprospective 39件の閉鎖は既存監査でも未確定。 |
| 不足判定の根拠 | 有限時間完結性監査U6がH-chainの訂正済みclosureと他分子のindeterminateを区別している。互換branchの診断は絶対対応の確認ではない。 |
| 判断できないこと | 絶対PF errorとH-reference errorを分離した結果として外部条件を引用できない。 |
| 所在の可能性 | 該当HF/HCl/prospective runの元保存cache/manifest。保存済みか、そもそも未取得かは未確定。 |
| 探す対象 | 各prediction/truth freezeのabsolute_energy、reference energy、removed constant、Hamiltonian/sector hash、physical phase lift/branch metadata。 |
| 必要な来歴・条件 | row/coordinate ID、元source/freeze/result SHA、PF係数版/time/rank、同H/originのjoin根拠と各file SHA。 |
| 回収後の照合 | 原記録だけでtupleが閉じるか判定する。新ground/target情報が必要なら未実施検証として返し、取得しない。 |
| 優先度 | 重要（Direction Cを既存H-chain scope外へ解釈する場合） |
| 他研究の継続 | H-chain限定Direction C成果と既存negativeは継続して整理可能。 |

<a id="gap-006"></a>

## GAP-006 旧PR-2 pilotの完全入力snapshot

| 項目 | 内容 |
|---|---|
| 対象研究 | PR-01 |
| 確認できたbranch/commit/path | [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md) |
| 不足資料 | 旧pilot Hamiltonian hashに一致する完全snapshot、DF生成順・state・環境と旧数値のjoin。 |
| 不足判定の根拠 | PR-2 V0監査で完全旧入力を回収できず旧S0 hash STOPを保持。別snapshot系列でV1–V4/S2/M1/M2へ進んだ。後続transfer成功は旧入力の回収ではない。 |
| 判断できないこと | 旧pilot数値を新seriesのsame-input baselineや独立確認として扱えない。 |
| 所在の可能性 | 2026-09-27 pilot実行時のcache/保存領域。hostの根拠なし。 |
| 探す対象 | 旧pilotのinput fingerprintと一致するHamiltonian/DF/state payload、generation-prefixとweight-prefixの原ordering/weights。 |
| 必要な来歴・条件 | 旧pilot/source commit、H/DF/state hashes、rank3/6/9の構成、basis/sector、全scalarと依存版、実行command。 |
| 回収後の照合 | 旧pilotと新系列を明示的に区分し、旧S0を成功へ書き換えない。M1/M2を再実行しない。 |
| 優先度 | 補助（旧pilotをsame-input基準に戻す場合のみ） |
| 他研究の継続 | 新系列M1/M2の限定成績と再利用は可能。旧snapshot回収は本調査完了の前提でない。 |

<a id="gap-007"></a>

## GAP-007 旧C1 partition payloadとP5突然終了の一次来歴

| 項目 | 内容 |
|---|---|
| 対象研究 | RB-17 |
| 確認できたbranch/commit/path | [RB-17 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md) |
| 不足資料 | 旧expected partitionの完全payload/環境、旧C1-5A P5 workerの処理済trial数・直接exit/signal。 |
| 不足判定の根拠 | C1成果総括はweight bit差を特定したが元完全payload不足で唯一原因としない。旧P5はsetup後STOPでtrial/exit UNKNOWN、別P5Rは別認可で完走。 |
| 判断できないこと | 旧partition差の唯一原因や突然終了の完了prefixを確定できない。新R1/P5Rを旧runのresume成功と扱えない。 |
| 所在の可能性 | 旧C1-1/C1-5A producer/worker実行領域。具体hostは未確認。 |
| 探す対象 | D0 hash_identityと旧partition計算の入力bytes、旧worker監視/OS exit receipt・event spool・最後のpersisted trial ID。 |
| 必要な来歴・条件 | 旧source/plan/result、candidate/method IDs、ordered coefficient hex、clock/monitor scope、原ログbytes/hash。 |
| 回収後の照合 | R1/P5Rとの条件差・別認可を保持し、UNKNOWNを推測で埋めない。追加P5 runはしない。 |
| 優先度 | 補助（旧失敗の根因確定・来歴監査が必要な場合） |
| 他研究の継続 | 現在のC1限定logical成果と凍結方針の整理は可能。 |

<a id="gap-008"></a>

## GAP-008 stale DF screeningのCgs入力表・来歴

| 項目 | 内容 |
|---|---|
| 対象研究 | REP-02 |
| 確認できたbranch/commit/path | [REP-02 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/VALIDATION_STATUS.md) |
| 不足資料 | 削除済み元Cgs集約表、全split表/index、修正済みcanonical入力とmolecule/PF/L_Dの対応、input hash/source commit/command/environment。 |
| 不足判定の根拠 | VALIDATION_STATUSは旧JSONをDO NOT USE / staleとし、元Cgs表削除・再生成不能・来歴欠損を記録。内部加算の整合と科学的入力の正しさを区別する。 |
| 判断できないこと | 旧best/rank選択が妥当か、どの旧入力で得たか判断できず、現在のDF policyへ転用できない。 |
| 所在の可能性 | 旧DF screening/Cgs計算の保存先または既存backup。場所は確認できていない。 |
| 探す対象 | VALIDATION_STATUSのDF screening節が指すcanonical Cgs、split table/index、旧screening JSON生成log、既存の修正履歴。 |
| 必要な来歴・条件 | molecule/basis/geometry、PF係数版、rank/split L_D、H/state hash、time-fit窓と定義、生成source/command/environment、各表hashとjoin。 |
| 回収後の照合 | 見つかった旧/修正表のidentityを比較し、stale statusを保持。新fit・screening再実行・best再選択はしない。 |
| 優先度 | 必須（旧DF screeningの成否・再利用判断を行う場合） |
| 他研究の継続 | 他研究は継続可能。旧screeningを肯定/否定の性能結果として数えない。 |

## 今回の欠損確認の限界

UWCについては既知の相対保存先を211登録worktreeでfilename確認したが、全ユーザー領域・共有cache・別host・SSH設定は探索していない。他のGAPの「未保存」「UNKNOWN」は固定一次記録の帰属であり、全filesystem不存在の証明ではない。別環境で見つからない場合は、調べた所有領域と対象名・理由を報告すればよく、欠損を再計算して埋めない。

非公開payloadを公開repositoryへそのまま追加する依頼ではない。回収側は原本をread-onlyで保全し、公開可能な相対path/opaque ID・SHA・sanitized manifestを分離する。
