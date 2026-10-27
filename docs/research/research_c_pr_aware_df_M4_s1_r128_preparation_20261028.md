# 研究C M4：保存PF再利用・r128 RTE/shots・未知価格の技術準備

2026-10-28 JST。起点`e61e79f68a29d0ff52a9d0a9448a569fc98e0e86`、独立branch `codex/research-c-M4-s1-r128-20261028`。ユーザーの一括許可に従い、人工試験・SOURCE/契約固定後に追加承認なしでM4科学実行へ進む。実128-event sampling/build/compile/価格測定は対象外。

追加は[M4 adapter](../../src/trottertracks/c_pr_aware_df_m4.py)、[runner](../../scripts/run_research_c_pr_aware_df_M4.py)、[人工test](../../tests/management/test_research_c_pr_aware_df_m4.py)の3 SOURCE。旧M3/B1/B2/C1/M2の科学SOURCEを変更しない。

## 再利用と科学条件

保存直鎖H4・隣接1Å/STO-3G、通常number CDF8 Proposed、Nα=Nβ2/sector36、元g補正κ・核scalar一回・frame0..7/LD2、partial ordered symmetric S2、δ=.1/s1/h=.1/K0/m0..12/n=2^m、own-DF ground oracle/状態準備除外を維持し、rだけ128とする。CA=.00159/DF reserveCA/8、βtotalπ/6・βstatπ/12、failure.01/axis.01/26も維持。

- M3の保存PF/generators/H/sector mapping NPZ、13round PF誤差、同じh/g/J/U/κ/scalar、ground/alias、DF bound、旧SOURCEと数値精度を照合する。rはideal PF branchに入らないため、PF kernelと全13 PF誤差を再利用する。新PF tail exponential/13exact-time evolution/PF-error norm/ground eigensolveは0。
- `prepare_arm/configure`と修正済み`catalog_gate`で216 atomの順序・係数・符号・basis・probability・identity/λを確認し、新τ=旧τ/2を確認する。JSON保存・再読込を人工fixtureへ含める。
- `reference_kernels`のRTE分岐で使う既存`_ordered_exponentials`/有限RTE helperだけを呼ぶ薄い接続でcorrected kernelを作る。`reference_round`の同じmatrix_power/norm2/attenuation処理のRTE部分を使い、PF-exact再計算部分は省く。新fitter/PF engine/PR sampler/compilerは作らない。
- 既存B1 `classify_round/summarize`、C1 `qualification/assess_round/energy`、AccuracyTarget/Hoeffdingを再利用。Evidenceは13round未資格/null、2シナリオのsynthesis/law誤差配分は旧C1と同じ。旧scenarioのB2価格proxy仮定は保存比較対象だけに属し、r128価格へ移していないことを明記する。
- 未知kernel価格μについて`C_k(μ_k)=Wμ_k+S_k`を6指標別に算定する。M3/同J/U M2/旧PRの保存条件付き総費用に対して閾値を求める。μ/実総費用はnull。0以下の閾値・不適格・未資格を別に扱う。
- 高m10..12の効果は、旧M3 shotsから当該roundだけを新shotsへ置換した会計上のcounterfactualを事前登録する。別候補・新PF/RTE計算ではない。追加64eventsに許される差は閾値−M3平均であり、測定event単価ではない。

## テストと固定call

[63 PASS log](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/technical/tests_final.log)：新M4 18＋M3 22＋C1 23、34.75秒。人工で既存reference_kernels/reference_roundとのcorrected-RTE一致、PFのr独立性、全13round、PF/exact経路禁止、JSON round-trip、7種類のPF再利用前提不一致、shots・shell・一次式・tie/strict/負閾値・null・partial STOPを検証した。実H4試験実行は0で準備した。

[固定契約](research_c_pr_aware_df_M4_s1_r128_contract_20261028.json)：**A39＋B48＋C10＋summary1＝最大98 units**。

- A：saved evidence/identity/target/input/fit/model/sector/schema/law/identity/τ/kernel各1＝12、13round reference/classification各13、A summary1。
- B：3lane qualification3、accuracy39、energy3、shots/group summary3。
- C：affine3、2scenario×3基準の各6指標threshold6、summary1。
- program summary1。

内部の36次元matrix powers・6metrics・13round group会計をunit内に明記する。Sampling/QPY/native Operator/newcompileのcallは存在しない。

SOURCE84・input58・旧raw564を保全してfresh `artifacts/research_c_pr_aware_df_M4/2026-10-28/run01`を一度実行する。Retry/resume0。SOURCE/入力/PF再利用/atom/RTE/schema/会計のcorrectness FAILで依存処理を停止しpartial保存。科学的不適格は正当な13roundと可能なRPE判定を保存するが、未資格価格閾値や総費用を作らない。条件の事後変更・自動再実行はしない。

## 資源と後続準備

既存resource plannerを13sector36独立roundへ適用し、BLASをworkerと協調。Scalar会計は小さく、不要なprocess/native巨大行列は起動しない。CPU/RAM/worker/wall人工cap0、task-local venv/cache/output、read-only依存、共有設定・他job変更0。承認証明・原文保存・grantは作らない。

科学結果の閾値から128-event native校正の情報価値を判断し、必要な既存B2/M3-B処理・32非paired cohort・full512/adjoint/X/Y・P0・SOURCE/input/call/fresh/STOPの未承認契約案を準備する。M4から実native科学計算は開始しない。公開remote照合後mandatory STOP。
