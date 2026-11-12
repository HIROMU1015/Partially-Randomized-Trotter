# A/B/C共通インターフェース V1（技術案・2026-11-12）

本仕様はA/Cの並行実装向けの最小契約である。科学的採用・科学実行の許可ではない。B baselineは`7dd61438b`（d5d9823eの後続C1受入済みbranch）。既存policy/recordを上書きせず、新schema `PR_RPE_RMSE_V1`を使う。

## 科学的に未確定な点

[原論文v2](https://arxiv.org/pdf/2503.05647v2) Appendix E.3/E21は全回転数の目安を r=κλ_R²T²とし、E1/E23ではQPE誤差とTrotter誤差の二乗の扱いに仮定を置く。v1とv2があり、採用版、hとPF幅の関係、finite-K補正、signed-piece配分、state/targetと誤差契約をA側の科学仕様で確定する必要がある。Bは式をN solverや保証として採用しない。解析的κ最適値をcompiled価格の最適値と同一視しない。

## Configuration → 各ラウンド

共通configurationは次を必須とする（[field一覧](INTERFACE.json)）。

- `binding`：H/state/snapshot、DF分解/順序/rank、system、SOURCE。L_Dに対応するsplit SOURCEとrandomized_scopeを別に指定する。
- `method`：`PF` / `PR` / `FULL_H_RTE`。純RTEは全非scalar H（one-bodyを含む）をランダム化する。DF B3のone-body保持型はこのラベルで受け入れない。
- `L_D`, `delta_pf`, `formula`, `h`, `M`。T_m=h·2^m、m=0..M（Mは最大index）。PF/PRのqPF=ceil(T_m/delta_pf)、実幅=T_m/qPFを**明示した整数化規則**で生成する。h/formulaは固定入力で、Bが探索変数へ追加しない。FULL_H_RTEはPF幅・qPFなし。
- `kappa`：kind/value/definition_source。PFではnull。PR/純RTEではconfiguration全体で1個を共有する。round別の独立R選択で置換しない。
- `event_mapping`：kind/source。A/B共通関数を使い全roundの回数を同じκから生成する。Bの候補変換案`PAPER_TOTAL_UNIFORM_OUTER_DRAFT`はr_target=ceil(κ(λ_R T_m)^2)、PRはR_outer=max(piece数,ceil(r_target/qPF))、既存ABS_TIME配分でevent_count=qPF·R_outerとする。これは整数候補の技術案であり、signed負時間やfinite-KのB≤exp(1/κ)保証を与えない。Aが別定義を使う場合、明示SOURCE付きmapperを供給し、kind/definitionを変更する。無名κを受け付けない。
- `finite_K`, `law`, `rmse_policy`, `profile`。有限K・signed weights/allocation/scalar規則を全identityへ保持。lawはideal/実samplerとSOURCEを区別する。

configuration IDは上記全条件と全scheduleを含む。row identityにもconfiguration ID、M/h/κ、m/T/qPF/実幅、event/allocation、binding/profile/lawを含む。元の固定R candidate IDへ付け替えない。共通κの異なるscheduleを混ぜたrecordは不一致である。

## A → B：Nと全scheduleのRMSE証拠

`N_packet`はconfiguration ID、全row identity、law_binding、SOURCE、`policy`、`status`、`rounds[{m,N_X,N_Y,...}]`を持つ。N_X/Yは**各軸の量子Hadamard shot数（正の整数）**。両軸分を再乗算せず、古典価格標本数やnormalization係数を追加乗算しない。

policy.familyは`PR_PAPER_RMSE`と`HIGH_PROBABILITY_RPE`を別IDで区別する。A-P1等の旧高確率型NをRMSEとして再ラベルしない。A0は診断だけで、N_packetへ変換しない。

RMSE evidenceは全scheduleについて、policy/SOURCE/target RMSE Ha、供給されたRMSE bound、finite-K、signed-piece/κ対応、state/target、alias/prior/decoder/全round等を含む誤差契約の資格を明示する。Bは供給値と資格を照合し、数値的N/誤差導出を行わない。未確定は`NOT_EVALUATED`/`UNQUALIFIED`/`NUMERICAL_UNSTABLE`/`INFEASIBLE`として残す。技術testは`SYNTHETIC_ONLY`として科学資格と分離する。

## C → B：価格record

既存recordのfieldを再利用：`identity`, `law_binding`, `axis`, `measurement_axis`, `observable`, `outcome_convention`, `units`, `classification`, `cohort`, `n_paths`, `expectation`, `mean.rz_count`, `source`, `model`, `uncertainty`。

DIRECT / MODEL、決定論単価 / 有限cohort / 全law期待値 / 未検証外挿を分ける。MODELはfull candidate/domain・binding/compiler・評価変数・校正/検証SOURCEと`qualified_for_candidate`が一致する必要がある。C1のfalseを変更しない。異なるh/M/κ/配分の旧recordを再ラベルしない。既存C1利用時は元SOURCE/価格を保全し、別の適用証拠なしに新identityへ移植しない。

既定は`qualified_expectation`。`finite_cohort_scenario`は明示したcohort条件付き比較だけで、全lawの最適性を示さない。null/未資格の値を0へ補完しない。X/Yはpaired cohort、未知shared calibration/cross-round/cross-κ共分散/systematic/CIはnullのまま。

## Bの実装APIと機能の分離

`trottertracks.pr_rpe_rmse_controller`：

1. `generate(spec, *, event_mapper=None)`：構成とscheduleだけを生成。Aと同じ`paper_event_mapper`または明示A mapperを使う。生成成功は適格性ではない。
2. `assess(config, n_packet, prices, *, cohorts, price_policy)`：同一identity/law/RMSE/finite-K/価格資格/欠測を判定。N/C/provider呼出し・G計算なし。
3. `cost(config, n_packet, prices, *, cohorts, price_policy)`：明示した適格な供給値だけを既存exact数値primitiveで積和。巨大Nをfloat化しない。
4. `select(configs, evaluations)`：固定外側(L_D,delta_pf)ごとのM/共通κ比較と外側比較。**schedule単位で選択**し、roundを他κと組み替えない。有限menu・欠測/未探索範囲・同値・不確かさを保持する。

PF/PR/純RTEを同じtarget/policy/価格scopeで比較する。方式別menu/coverageを表示し、PFや純RTEが欠測ならPR優位と宣言しない。h/formula/追加変数は自動探索しない。全APIに実データの自動取得・再compile・fit・N計算fallbackは置かない。

## A/Cへ共有する範囲

Aはκのkind/mapper、RMSE error contractとN_packetを、Cは上記full identityに対する価格種別/domain/cohort/資格を実装できる。仕様公開を先行し、Bはその後に人工test付きcontrollerを実装する。変更が必要なら新しい明示definition/sourceを渡す。現在の数学的仮定をBが正式採択することはない。新しい実N/C/Gは別のユーザー実行許可が必要。

## 実装後の補足

- scalarは既存会計のint/有限binary64/`{numerator,denominator}`を受け、h/Δ/κ/目標と実時刻・実幅をexact rationalに正規化する。κは現mapperでは正値のみで、κ=0境界は扱わない。
- ceil整数化によりactual_delta_pfがroundごとに変わる場合がある。同じeffective Hamiltonianの冪であるとは仮定しない。hとΔの整合、control/時間半減規則を含む全scheduleの誤差契約をAが資格するまで科学利用しない。Bは追加の時間半減を行わない。
- N_packet.statusは`RMSE_SCHEDULE_EVALUATED`、evidence_kindは`SYNTHETIC_ONLY`または`SCIENTIFIC`。後者はpolicy.scientific_status=`EXTERNALLY_QUALIFIED`と明示科学SOURCEが必要。Bはこのstatusを発行・採択しない。
- rmse_evidence：`source`, `status`, `error_contract`, `rmse_bound_ha`, `covers_state_target_alias_prior_decoder_all_rounds`。ランダム化方式では`finite_K_status`と`signed_piece_kappa_status`も必要。供給されたboundと目標だけを比較し、二乗和・N・normalizationを導出しない。
- N roundは`m`, `identity`, `law_binding`, `N_X`, `N_Y`。Cのq_variableはPF/PRでは`q_PF`、純RTEでは`event_count`。measurement/単位は既存Bと同じ。`cohorts[(candidate_id,axis)]`が選ぶ1 scenarioだけを使う。
- selectへ渡すevaluationはcostの出力。assessだけの結果はG未評価としてpending。各方式の有限menu coverageと価格policyを照合するが、PF baseline最適性・科学的優位性・global最適性は資格しない。finite_K、κ定義/mapper、h/formulaは隠れた探索変数にしない。

具体的な人工I/O：[SYNTHETIC_IO_EXAMPLE.json](SYNTHETIC_IO_EXAMPLE.json)。実装・残る科学判断：[REPORT](REPORT.md)／[HANDOFF](HANDOFF.md)。
