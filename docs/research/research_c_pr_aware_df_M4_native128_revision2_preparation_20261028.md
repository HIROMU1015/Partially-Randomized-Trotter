# M4-Native128：記述修正revision2と初回実行準備

2026-10-28 JST。ユーザーの一括許可に基づく最大332-call、Proposedのみ固定32 trajectoryの初回科学実行を準備した。承認証明・原文保存・新grantは不要。結果は別の実行報告に保存する。

起点4887d6af81d4b26a31c2f85cb0430670b620f4e6、GitHub/local tree013958b7476d61aec02d40c0250203708f22f2eb一致。旧native SOURCE87、input63、M4 98-call、M3 run01技術STOP/run02 471-call、M2/B1/B2/C1等を照合し保全する。

[独立revision2契約](research_c_pr_aware_df_M4_native128_contract_revision2_20261028.json)は、旧契約のtrajectory_sample/kernel_build/trajectory_reference説明だけを64から128 eventsへ修正した。旧契約・旧SOURCEは上書きしない。実装・数値条件・compiler・sampling・call内訳は不変。SOURCE88は元87ファイル＋metadata専用回帰test1ファイルであり、新しいrunner/CAR/compiler/samplerは作らない。新SOURCE digestを既存のseed導出へ渡す。

歴史的`unapproved` filename/statusやCLIのSEPARATE_USER_PERMISSION_REQUIRED表示は、承認機構ではない。今回のチャット指示が固定範囲の実行許可である。独立revisionのpathを既存runnerの`--contract`へ指定する。

対象は保存直鎖H4・1.00Å・STO-3G、通常number Proposed CDF8、sector36/full512、元順0..7、LD2、元g補正κ・nuclear scalar一度。Partial ordered symmetric S2、δ=h=.1/s1/r128/K0、m0..12、own-DF oracle/状態準備除外。Head内部relative融合・同basis tail共有・cross-head/tail一般融合なし。P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/degree1/initially-zero false、MPFR160 declared phaseを維持する。

PCG64 master2026102805/cohort M4_r128_native_fixed32_v1/axis M4_R128_NATIVE_ONLY、SOURCE digestとJ/U identityをseedへ結び、32独立substreamで各128 signed K0 eventsを使う。旧cohortとpool/paired扱い/適応追加なし。

既存B2/M3 pipelineのFull256 CAR/216 signed atoms/head exponential/Gaussian cache、shared-Qiskit build/compile/inverse mutex、並列full512 verifierを再利用する。全512入力出力、位相fitなし、raw/reference・P0/reference/raw・unitarity・actual adjoint/return・X/Yを確認する。Component1e-10、complex signal2e-10、rtol0。単一trajectoryのsignalへattenuationを掛けない。

332=attachment7＋nativecore324＋保存M4会計1。QPY128/compile32/reference32/actualinverse64、価格が不利でも32件を完了する。Correctness FAILでは後続停止・開始済み回収・partial証拠保存、retry/resume0。実装を実行開始後に変更しない。

保存M4の2scenario・W/S・36閾値を再利用し、C=Wμ+S、記述的SE=W·SEkernelを算定する。13roundの独立再校正ではなくsqrt13縮小なし。新PF/RTE/shot/CA/eigensolve/fit/全law評価なし。有限cohort点比較、仮定付きshots、実装law/synthesis未資格、observed extrema非supportを維持する。

[人工・回帰ログ](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/technical/tests.log)、[passive preflight](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/technical/preflight.json)、[SOURCE88](../../artifacts/research_c_pr_aware_df_M4_native/2026-10-28/technical/SOURCE.json)へ試験数・hash・依存環境・空き資源・plannerを保存する。新metadata testと旧native5/M3/B2/互換mutex/M4 testsを実行する。人工入力のみで事前の実H4sampling/build/compile0。

32trajectory並列、各BLAS1はoversubscription回避の実装構成で人工CPU上限ではない。実affinity/quota/空きRAMから既存plannerがworker数を選ぶ。Task-local venv/cache/output、依存read-only、共有設定・他job操作なし。完了後は成果公開・remote照合・mandatory STOP。別r/s/K/γ・追加標本・fit・厳密保証・重要科学レビューへ進まない。
