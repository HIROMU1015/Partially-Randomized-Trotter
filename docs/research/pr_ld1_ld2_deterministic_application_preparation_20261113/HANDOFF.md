# 保存rawから既存決定論的riskへの引渡し

2026-11-13 JST。[REPORT](REPORT.md)／[未採択条件・有限案](CONDITIONS_AND_FINITE_PLAN.md)／[SOURCE一覧](SOURCE_AND_CONDITIONS.json)。今回はmetadata確認だけを完了した。D.evaluateを実入力で呼んでいない。

## 再利用する入口

実装と数式はB `d71721f30c7b7fc13ba0789eefffd6238aca3ebc`の[既存HANDOFF](../rpe_deterministic_risk_20261113/HANDOFF.md)を正本とする。

1. `K.accept_saved(root, bundle_source=..., integration=I)`でA91349のLD1又はLD2 bundleを読む。元configuration、raw carrier、law、normalization、数値uncertainty、C physical_context/provider_bindingを照合し、元bundleを変更しない受入viewを返す。
2. `D.evaluate(config, signal_packet, allocations, evaluation_target=..., statistical_contract=..., integration=I, source=..., options=...)`を、科学条件の選択と別途明示実行許可の後にだけ呼ぶ。内部のK.prepare/W.prepareは既存全round・整数shot・SOURCE検査を再利用する。今回新callerは追加していない。
3. 初回案はLD1/LD2各1whole-vectorだけ。D.compare、W.compare、B.selectは初回では呼ばない。将来、適格N・価格が揃ったときに既存会計へ接続する。

IはCの既存`plan/price_context/signal_input`契約を提供する。今回確認したのは既存保存fixtureが読む固定Cのmetadata関数であり、Cの全runtimeや科学providerを新規実行・配備したことを意味しない。

## 外部から渡す条件

| 入力 | 既存fieldと注意点 |
|---|---|
| evaluation_target | `energy_ha/kind/source/qualification_scope/binding/original_target_source`。bindingと元target sourceはpacketと同一。新proxy採用が必要な場合も元targetのflagを変更しない |
| allocation | W.prepareが受ける`allocation_id/source/shot_policy/shots`。shotsは全13roundの`m/N_X/N_Y`。physical configuration IDは元IDのまま |
| statistical_contract | 元IID policyを渡す既存`simulation_policy`、明示`allow_conditional_unqualified`、`summary_options.unresolved_loss`。field名は既存MC資産と共通だが、Dは標本生成しない |
| options | `source/probability_rule/phase_radii/maximum_round_outcomes/maximum_verification_pairs/model_loss_contract`。全値・loss familyは未採択。model_loss_contractの既存adopted fieldは科学的条件の宣言であり、ユーザー許可の証明を求める機構ではない |

決定論的評価はMC反復数・seed・confidenceを必要としない。exact有理数の確率・loss集計であり、数値rawとtargetそのものの誤差や、実数decoderとの包含は別の未資格事項である。`corrected`やBをrawへ掛け戻さない。

## 出力とUNKNOWN

`diagnostic.deterministic_risk`に各round good/unsafe/zero/reference_oracle_tie、有限遷移certificate、全zero・未解決率の範囲、採択したloss familyのMSE上界、UNKNOWN理由を保持する。unsafeは誤branchの確率そのものではなく、安全集合外の確率である。reference_oracle_tieも実全RPE tieとは異なる。

certificate不成立/範囲外はUNKNOWN。loss未定義ならMSEはnull。元SOURCE、canonical ID、law、target、全shot vector、optionsはdeterministic_inputへ結び付く。MC_calls=0、scientific_energy_qualification=UNKNOWN、n_packet/sufficient_N/global_RMSE_bound/Gはnullを維持する。初回の診断だけから新しいN資格を発行しない。

元full-H proxyに対するrawモデルriskは時間発展近似等を含み得るため、AのPF H_eff由来δEを純shot項として別加算しない。元Minkowskiの正式接続には対象・誤差成分の分離根拠が別途必要。Bの条件付き診断はその統合の完成を主張しない。

## A/C/Dと次の担当

- **ユーザー**：条件一覧の未採択target/loss/診断scope/確率rule/IID/whole-vector/phase集合・有限検査範囲を選び、初回2評価の科学実行を明示許可する。
- **B**：許可後に指定された2評価だけを既存Dで実施し、Nとrisk上界の対応・UNKNOWNを報告する。許可なしに配分探索やC/Gへ進まない。
- **A**：元raw・数値allowance・state/law等の資格を担当する。今回Aへの追加signal/δE計算要求はない。
- **C/D**：今回追加の価格取得要求はない。将来Gへ接続する場合は、今回のcanonical/physical round IDとSOURCE/law/compiler/lowering/full-wrapper/P0を照合し、適格価格だけを既存adapterへ供給する。

既存raw・結果・STOPは保全する。技術成果のcommit/push・remote確認後は**mandatory STOP**。
