# A→B/C：RMSE policy・共通κ mapperの引渡し

2026-11-12 JST。B公開草案 `d562bc38d3b63df51f775aa4eb0998afbf1873e2:docs/research/pr_rpe_rmse_controller_20261112/INTERFACE.json` に接続。[REPORT](REPORT.md)、[人工例](ARTIFICIAL_EXAMPLE.json)、[module](../../../src/trottertracks/pr_rpe_rmse.py)。今回の実分子N/保存数値再解析/C/Gは0。

## 共通κの唯一の意味

`KAPPA_DEFINITION=PAPER_V2_E21_KAPPA_ROTATIONS_OVER_LAMBDA_R_SQUARED_TIME_SQUARED`。

`paper_event_mapper(kappa_paper,lambda_R,T,q_PF,signed_weights)`（keyword-only）は、r*=κpaper(λR T)²、r_target=ceil(r*)、R_outer=max(J,ceil(r_target/qPF))、既存ABS_TIME整数配分、event_count=qPF R_outerを返す。kindはB草案と同じPAPER_TOTAL_UNIFORM_OUTER_DRAFT。κ_Bやroundごと独立Rをここへaliasしない。

paired finite-K eventには抽象回転が1個あり、order2のproduct2個を回転3個と数えない。compiled RZ数へ置き換えず、full wrapper価格はCの既存SOURCE/単位を使う。involutory tail・λ/scalar/lawの対応資格は未知のまま明示し、event数からB≤exp(1/κ)を保証しない。

H6 m5_bestの11 signed piece/負時間・元整数tieを保全。初期m0..3は保存prefix5/R32/K2/qPF1/δ=Tをcopyし、κ_applied=false。regularはprefix7/K2/δ1.6、H4はprefix1/K2/S2/δ.1。

B草案のqPF=ceil(T/δpf)と、Aの既存固定δの維持は非整数gridで違う。Aは非整数を拒否し、Bへpf_grid=EXACT_FIXED_DELTA_WITH_EXPLICIT_SAVED_EARLY、H6にはPAPER_TOTAL_UNIFORM_OUTER_SAVED_EARLY_V1とSOURCE/early_overridesを渡す。h/formulaを勝手に探索しない。

## API順序

1. `rmse_policy(...)`：target/qpe/evolution RMSE、paper_quadratureまたはminkowski、paper_kappa_envelopeまたはfinite_B_diagnosticを明示。
2. `design_from_task(base)`：旧物理metadataだけをcopy。旧N/α/β/M12を新policyへ採用しない。
3. `schedule(M=...,kappa_paper=...,design=...,tail_norm=...,policy=...)`：Mは最大index、全M+1roundにPF/PR request。既存canonical candidate IDに加え、全schedule/M/h/κ/λ SOURCE/policyをconfiguration/request IDに含む。
4. 将来の科学許可後、`existing_finite_mean(ctx,request,bound_reference,source)`またはSOURCE付き既存mean adapterを供給。ctx/half-step/reference生成はcallerの科学範囲。新referenceへfallbackせず、旧N solverを呼ばない。
5. `evaluate(schedule,mean_provider=...)`：経験的model Nと適用条件。provider無しはNOT_EVALUATED。旧N recordをmean_evidenceへ渡すと拒否。
6. `b_handoff(schedule,evaluation,source,evidence_domain=...)`：B草案のPF/PR別configuration/N_packet。人工例はSYNTHETIC_ONLY。

`demo_pr_rpe_rmse.py <fresh-json>`はTOYだけ。実分子を起動する引数/科学optimizerはない。実データに対するevaluateや既存mean再解析は新しい科学callで、今回の技術許可に含まない。

## B recordと未資格の保持

N_packetはconfiguration_id/policy/source/status/rounds/rmse_evidenceを持つ。policy.family=PR_PAPER_RMSE、policy_id=PR_RPE_PAPER_V2_EMPIRICAL_RMSE_V1。旧HIGH_PROBABILITY_RPEと別lane。

roundsには完全identity、law_binding、source_request_id、nominal_N_X/Y、N_X/Yを保存する。**現在N_X/Yはnull**、nominalだけがモデル値。RMSE bound・schedule_error_contract_qualifiedもnull/false。全適用条件が報告された人工例であっても、論文の経験的定数はfinite-Kの厳密保証ではなく、実Gへ適格なNにしない。

NOT_EVALUATED、UNQUALIFIED、NUMERICAL_UNSTABLE、INFEASIBLE、SYNTHETIC_ONLYを保持する。complete model inputという内部statusを科学PASSへ読み替えない。Bのgenerate/assessコードはこの草案SHAでは未公開なので、`B_generated_identity_verified=false`。必須field/mapper式は人工照合済みだが、今後のB生成configとの完全identity一致をB側で確認する。

旧Bのevaluate_menuへ新packetを直接投入せず、新schema PR_RPE_RMSE_V1で受ける。A-P1の旧NやA0 normalizationからRMSE N_packetを作らない。旧C1の価格も新configへ単に再ラベルしない。FULL_H_RTEは本A実装の対象外で、DF B3をFULL_H_RTEにするadapterはない。

## 科学的に残る項目

- Nmは各軸、2Nmが両軸。exp(2 logB)をNへ一回だけ含め、Cへ追加乗算しない。
- E22のexp(2/κ)配分と、実finite Bmを代入したdiagnosticを別identityにする。H6負時間・整数配分/早期規則で前者が成立するとは推測しない。
- Cmax=.08(PF)/.10(PR)、terminal11/slope4はpaper経験的資源モデル。RMSE target・M・誤差budget・state/effective unitary/alias/finite-K移送/actual lawの資格が必要。
- error compositionの二乗和は不偏性等の仮定を伴う。有限Kのdeterministic bias、branch失敗、保存数値stateを無断で独立ゼロ平均誤差にしない。
- Cは同candidate/R/qPF/配分/H-state/compiler/law/axis/cohort/domainの価格と不確かさを供給。全law期待値やcross-κ/round covarianceが未知なら保持する。今回価格fit/実G/κ最適化は0。

## 次の担当・必要な許可

**Codex B**：この共通mapper・外側/early override・N_packetをcontrollerへ接続し、SOURCE付き全schedule単位のidentityを照合。Cや旧Nを未資格のまま補完しない。

**ユーザー**：1系・固定外側条件・(M,κ)・RMSE budgetと、最初の2round科学pilot（保存再利用の新policy評価、欠測mean最大2/対照2、必要時reference最大4 action）の可否を判断。全RMSE simulation/最適化は別範囲。

**Codex A**：許可後だけその有限範囲を実行。今回は人工テスト・公開照合後mandatory STOP。P2/P3と新実N/C/Gへ進まない。
