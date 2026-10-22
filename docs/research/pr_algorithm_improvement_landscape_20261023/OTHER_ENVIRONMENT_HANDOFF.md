# 別環境Codexへの資料回収指示案

以下はGAP-IDごとに独立して渡せる**未実行の指示案**。ユーザーが対象環境のCodexへ依頼する段階で適用する。このCodexはSSH探索・別環境作業・資料の再生成を開始していない。

各依頼の共通境界：自分にアクセス権限がある既知領域をread-onlyで確認する。SOURCEをimport/実行しない。新科学計算・数値再解析・simulation・回路build/compile・再合成・solver・新非自明証明は行わない。旧SOURCE/raw/契約/grant/ledger/marker/STOPを保全し、別ユーザー領域や共有設定・cache・jobを変更しない。新しいgrantを作らず、STOP/retry規則を解除しない。

見つけた原本のbytes/SHA・来歴・条件と公開可否を分けて報告する。private raw、認証情報、内部絶対path、非公開review/grant原本を公開しない。公開が必要ならユーザー指定の許可範囲内で別branchへ最小sanitized資料だけを作り、HIROMU1015/*以外へcommit/pushしない。見つからない場合は探索範囲と理由を明示して停止する。

## GAP-001 向け依頼

対象：旧BLISS/UWCのrun原データ。必須（この系列の改善成否を判断する場合）。固定入口：[REP-03 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/notes/uwc_current_implementation_and_results.md)

回収目的：小さな改善・悪化訂正の妥当性、同じHamiltonian/sector/grouping/time-fitでの比較、PR転用可能性を定量判断できない。

既知の手掛かり：過去にUWC GPU batchを実行した環境。具体的host/SSH接続先の裏付けはない。

探すもの：H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json、H3_bliss_sector_scaling_diagnostics.json、grouped_uwc_pf_qpe配下の既存theta sweep・診断・run log。

必須identityと条件：原ファイルSHA256/bytes、生成source commit、元/変換後H hash、target particle sector、basis/geometry、theta/power、grouping、PF label/time grid、fit規則、backend/依存版、commandとUTC/JST日時。

回収後に行うこと：旧H3誤sector結果と修正版を別runにし、表とrawの対応を技術照合する。新fit・再ランキングは実施しない。

提出物：`GAP-001_recovery_report.md`、`GAP-001_provenance.json`、見つかった公開可能資料への固定参照一覧。原本の無断公開や再生成をせず、回収/未発見・限定照合の報告後mandatory STOP。

## GAP-002 向け依頼

対象：旧H4のevent履歴・未丸めnormalization・親exit来歴。重要（旧H4の独立監査を強める場合）。固定入口：[EVAL-02 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md)

回収目的：旧H4のevent→価格join、厳密なnormalization来歴、親processの正常回収を一次記録から全面確認できない。

既知の手掛かり：当該H4 geometry/B0+I runを実施した環境。公開handoffのrun identityが手掛かりで、hostはここで推定しない。

探すもの：O05/O06のrun別manifest、worker/nativeログ、親driver wait/return receipt、丸め前canonical normalization、event spool。

必須identityと条件：固定source/plan/result commit、candidate/axis/trajectory ID、compiler/runtime、表の各行へのjoin、原ログbytes/SHAとrun時刻。

回収後に行うこと：既存完成statusとは別に保存範囲を確認し、NOT_SAVEDと未回収を更新案で区別。worker成功から親exitを推測しない。

提出物：`GAP-002_recovery_report.md`、`GAP-002_provenance.json`、見つかった公開可能資料への固定参照一覧。原本の無断公開や再生成をせず、回収/未発見・限定照合の報告後mandatory STOP。

## GAP-003 向け依頼

対象：M1-S旧Gate 0 v5のfull-H／HDと一部review原本。重要（旧Gate 0全scopeの独立照合を要する場合）。固定入口：[RB-15 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md)

回収目的：旧Gate 0を入力全体から独立照合できず、二次引用された旧reviewの完全な帰属も確定できない。

既知の手掛かり：Gate 0 v5実行時の保存領域、またはユーザーが保持するreview原本。存在を保証しない。

探すもの：v5のSOURCE/input manifestで指すH_full/HD、ordered basis/Pauli registry、保存stateとsolver条件。必要ならレビューinner.md、レビューinner(1).md、レビューinner(2).mdの原本。

必須identityと条件：v5 source/contract/result SHA、行列dtype/shape/order/hash、sector・scalar・mapping・projection、state hash。reviewは作者/作成日/当時の対象commitと原bytes。

回収後に行うこと：既存の独立照合範囲と欠測範囲へbindする。数値再生成・新固有状態取得・新reviewは別承認とする。

提出物：`GAP-003_recovery_report.md`、`GAP-003_provenance.json`、見つかった公開可能資料への固定参照一覧。原本の無断公開や再生成をせず、回収/未発見・限定照合の報告後mandatory STOP。

## GAP-004 向け依頼

対象：FS-C0.5/C0.6のhistorical H01原入力。重要（historical実行の同一性が必要な場合）。固定入口：[PF-06 固定資料](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md)

回収目的：旧sourceをsame-input実験として再利用できない。new-source prediction/truthと旧truthの混合を避ける必要がある。

既知の手掛かり：当時のH01/H02 server cache。既存記録ではユーザーもSSH alias/host/IPを特定できていない。自動探索しない。

探すもの：historical_fingerprint_registry.json、source_contract.md、candidate_source_audit.jsonが列挙する原pickle/Hamiltonian/CISD/group payload。現同名H02を一致済み扱いしない。

必須identityと条件：registryの期待SHA、各objectのbytes/dtype/order、basis/geometry/active-space/sector、removed constant、solver/BLAS/依存版、実行commit/時刻。

回収後に行うこと：read-only identity照合のみ。微小なenergy差・functional近似でhash契約を緩めず、旧NO-GOと新R1を保存する。

提出物：`GAP-004_recovery_report.md`、`GAP-004_provenance.json`、見つかった公開可能資料への固定参照一覧。原本の無断公開や再生成をせず、回収/未発見・限定照合の報告後mandatory STOP。

## GAP-005 向け依頼

対象：HF/HCl・prospectiveの絶対energy identity tuple。重要（Direction Cを既存H-chain scope外へ解釈する場合）。固定入口：[PF-05 固定資料](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/second_study_v2/evidence_integration_20261006/approved_closure.md)

回収目的：絶対PF errorとH-reference errorを分離した結果として外部条件を引用できない。

既知の手掛かり：該当HF/HCl/prospective runの元保存cache/manifest。保存済みか、そもそも未取得かは未確定。

探すもの：各prediction/truth freezeのabsolute_energy、reference energy、removed constant、Hamiltonian/sector hash、physical phase lift/branch metadata。

必須identityと条件：row/coordinate ID、元source/freeze/result SHA、PF係数版/time/rank、同H/originのjoin根拠と各file SHA。

回収後に行うこと：原記録だけでtupleが閉じるか判定する。新ground/target情報が必要なら未実施検証として返し、取得しない。

提出物：`GAP-005_recovery_report.md`、`GAP-005_provenance.json`、見つかった公開可能資料への固定参照一覧。原本の無断公開や再生成をせず、回収/未発見・限定照合の報告後mandatory STOP。

## GAP-006 向け依頼

対象：旧PR-2 pilotの完全入力snapshot。補助（旧pilotをsame-input基準に戻す場合のみ）。固定入口：[PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)

回収目的：旧pilot数値を新seriesのsame-input baselineや独立確認として扱えない。

既知の手掛かり：2026-09-27 pilot実行時のcache/保存領域。hostの根拠なし。

探すもの：旧pilotのinput fingerprintと一致するHamiltonian/DF/state payload、generation-prefixとweight-prefixの原ordering/weights。

必須identityと条件：旧pilot/source commit、H/DF/state hashes、rank3/6/9の構成、basis/sector、全scalarと依存版、実行command。

回収後に行うこと：旧pilotと新系列を明示的に区分し、旧S0を成功へ書き換えない。M1/M2を再実行しない。

提出物：`GAP-006_recovery_report.md`、`GAP-006_provenance.json`、見つかった公開可能資料への固定参照一覧。原本の無断公開や再生成をせず、回収/未発見・限定照合の報告後mandatory STOP。

## GAP-007 向け依頼

対象：旧C1 partition payloadとP5突然終了の一次来歴。補助（旧失敗の根因確定・来歴監査が必要な場合）。固定入口：[RB-17 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md)

回収目的：旧partition差の唯一原因や突然終了の完了prefixを確定できない。新R1/P5Rを旧runのresume成功と扱えない。

既知の手掛かり：旧C1-1/C1-5A producer/worker実行領域。具体hostは未確認。

探すもの：D0 hash_identityと旧partition計算の入力bytes、旧worker監視/OS exit receipt・event spool・最後のpersisted trial ID。

必須identityと条件：旧source/plan/result、candidate/method IDs、ordered coefficient hex、clock/monitor scope、原ログbytes/hash。

回収後に行うこと：R1/P5Rとの条件差・別認可を保持し、UNKNOWNを推測で埋めない。追加P5 runはしない。

提出物：`GAP-007_recovery_report.md`、`GAP-007_provenance.json`、見つかった公開可能資料への固定参照一覧。原本の無断公開や再生成をせず、回収/未発見・限定照合の報告後mandatory STOP。

## GAP-008 向け依頼

対象：stale DF screeningのCgs入力表・来歴。必須（旧DF screeningの成否・再利用判断を行う場合）。固定入口：[REP-02 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/403204edf22e5712a71ac9ac7eebdbcd18626421/VALIDATION_STATUS.md)

回収目的：旧best/rank選択が妥当か、どの旧入力で得たか判断できず、現在のDF policyへ転用できない。

既知の手掛かり：旧DF screening/Cgs計算の保存先または既存backup。場所は確認できていない。

探すもの：VALIDATION_STATUSのDF screening節が指すcanonical Cgs、split table/index、旧screening JSON生成log、既存の修正履歴。

必須identityと条件：molecule/basis/geometry、PF係数版、rank/split L_D、H/state hash、time-fit窓と定義、生成source/command/environment、各表hashとjoin。

回収後に行うこと：見つかった旧/修正表のidentityを比較し、stale statusを保持。新fit・screening再実行・best再選択はしない。

提出物：`GAP-008_recovery_report.md`、`GAP-008_provenance.json`、見つかった公開可能資料への固定参照一覧。原本の無断公開や再生成をせず、回収/未発見・限定照合の報告後mandatory STOP。
