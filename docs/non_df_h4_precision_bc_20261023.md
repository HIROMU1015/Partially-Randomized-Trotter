# H4非DF/DF：Bの数値精度診断とCの保証手法設計

2026-10-23 JST。利用者のB＋C一括指示に基づく専用branch `codex/non-df-h4-precision-bc-20261023`。基点は調査commit `ebe20403bcb5c4439162614f64d991a769d7ed68`、旧科学SOURCEは `0c329b30a81accb35c212be8defa66fee0dd0f12`。

Bは保存scalar40軸の必要誤差予算だけを計算する。Cは手法設計と人工テストのみ。実H4の高精度/区間信号・certificate生成、shot/G再計算、旧grantのreuse、旧native終了証拠の補充は承認されていない。

## Bの固定仕様

`h=epsilon_axis-abs(saved_bias)-u`、`u/h<=rho`、`rho=.01`を使う。正の `s=epsilon_axis-abs(saved_bias)` に対し、実数代数では `u<=rho*s/(1+rho)`。理想decimal rhoなら `u<=s/101`。s<=0では非負uをどれだけ改善しても正のheadroomを確保できない。

保存JSONのbinary64を `Fraction.from_float` 相当のexact rationalとして解釈し、Decimal100の別式 `u/rho+u=s` と照合する。元のfloat閾値0.01と理想1/100を分け、両者の小さい予算を必要条件の表示へ用いる。表示は80桁を内向き丸め。これらは新たな実測誤差上界・達成可能性・certificate・正式適格判定ではない。

Bの実行前にtechnical SOURCEをcommitし、そのSOURCE/code/runtime SHA、入力10 file SHA、40軸registry、承認原文SHA、B専用grant、固定namespaceと非返却claimを作成する。40軸と独立算術40件、R/H集約2件が予算。異常時は部分結果を保持し、自動retryしない。新しいnative終了記録はB自身のみに限定する。

## 実装と人工検証

- `src/trotterlib/h4_precision_budget.py`：Bのexact算術・独立Decimal検査・必要誤差予算。
- `src/trotterlib/h4_precision_control.py`：stdlibのB専用authority・one-shot・SHA・保存監査。
- `scripts/run_h4_precision_b.py`：prepare/launch/run/verify。Cの実H4実行機能はない。
- `src/trotterlib/h4_enclosure_prototype.py`：人工入力専用のrational区間・有限RTE scalar law・行列指数包含・誤差伝播prototype。H4 loader/assemblerはない。
- `tests/test_h4_precision_bc.py` / `tests/test_h4_precision_b_authority.py`：算術境界、旧会計の人工回帰、partial/no-retry、未承認C拒否、区間包含等。最終33 PASS、FAIL/ERROR/skip0。

人工入力専用のFLINT接続検査は `tests/check_h4_precision_c_backend.py`。private環境のpython-flint0.9.0/FLINT3.6.0で最終13 PASS。前述の33件とは別の検査であり、開発中の旧ログの件数を合算しない。実H4入力の読込・Cの科学call・certificate生成は0。

## Bの結果：40軸の必要誤差予算

対象はlinear H4/隣接1.00 Å/STO-3G、8 system qubit、共通非DF保存状態Nα=Nβ=2。Pauli-JW184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body。S2/T=.8/q8/δ=.1、PR r4/K4、準備P=0、ordinary-controlled wrapper。同じε_sig=.05/.01/.005/.001/.0005、各cosine/sineを固定した。

| 群 | 軸数 | 現在のu / 必要u上限 | 必要削減率 | 必要u上限の範囲 |
|---|---:|---:|---:|---:|
| R：h>0、ratio guard超過 | 22 | 約1.533〜90.108倍 | 約34.777〜98.890% | 約5.9533e-6〜3.4991e-4 |
| H：保存h≤0 | 18 | 約107.042〜2823.398倍 | 約99.066〜99.965% | 約2.4528e-6〜6.9871e-5 |

最大の改善倍率は **Pauli-JW PF、ε_sig=.0005、cosine**（約2823.40倍、必要u上限約2.6490e-6）。最小の絶対誤差予算は **DF-native PR、ε_sig=.0005、cosine**（約2.4528058205e-6、約218.71倍の改善）であり、同じ軸ではない。

全40軸で固定点biasに対するsは正で、u改善だけでは不可能となる軸は0。ただし保存bias自体の不確かさは未証明である。必要uは現在の経験的allowanceを変更した結果ではなく、現行guardを満たすための必要予算である。全20行のN/G=null、全40軸のUNDETERMINEDは維持した。

全数値・exact rational・独立算術検査は [CSV](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/diagnostics.csv)、[JSON](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/diagnostics.json)、[Markdown](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/diagnostics.md)、[R/H集約](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/summary.json)に保存した。表示の近似値で正式な適格判定を行わない。

## BのSOURCE・one-shot・終了証拠

- 実行前の技術SOURCE：`2bd212288e8530e519c4a8135257df37f7608662`。6 code/test fileとPython/stdlib runtimeをSHAで結合。
- 入力10 fileの結合SHA：`c2931f6604405a7324bf04c096c2d2fe1cc950acc396500c3b75f606b3951180`。
- B専用grant：`6220c5585734e3adf31e0ba04468d163b716ad1a427ca2e34e8fcb764d6539bb`。新namespace B_run01で一回消費済み、再実行・返却・reuseなし。
- 実予算：40軸診断、独立算術40件、closed boundary identity40件、R/H集約2件。科学signal/SCF/FCI/DF/state solve/sampling/compile/N/G会計/C実H4callは0。
- Bの新しい子processはnative exit0、親がdirect waitして回収。保存hash/registry/旧N/G不変検査PASS。元H4 caseのnative exit/reap/STOPは未証明、監視消失原因UNKNOWNを維持する。

[固定SOURCE](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/SOURCE.json)、[契約](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/contract.json)、[検証](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/verification.json)、[B自身のnative記録](../artifacts/non_df_h4_precision_bc/2026-10-23/B_run01/native_process.json)。旧科学grant/marker/STOPを変更していない。

## Cの設計と次回の最小範囲

[保証手法設計書](non_df_h4_numerical_enclosure_design_20261023.md)に、高精度float一致と厳密包含の違い、Arb/Acbのball演算・外向き包含、forward/backward error解析、state/operator/biasの対応誤差、有限RTE degree5の期待値とnormalizationの保証方法を整理した。

有力な経路は、保存binary64をexact値で取り込み、reference・固定S2/有限law mean・signed biasを包含する方法。PRでは全component/lambda/実装CDFを復元し、有限weights→B_K→log_B→累積Bを包含し、保存log_B_upperがその上端を覆うかを検証する。通常floatの再計算や複数手法の一致だけをcertificateにしない。元state/preparation/basisの対応が取れなければ停止する。

[40軸の手法対応CSV](../artifacts/non_df_h4_precision_bc/2026-10-23/C_precision_method_map.csv) / [JSON](../artifacts/non_df_h4_precision_bc/2026-10-23/C_precision_method_map.json)は、Bが既に算出した各必要予算の1/2を**設計目標**として転記した。実H4で達成した精度ではない。64→96→128 bitsの有限scheduleで、包含幅だけが不足した対象を進め、モデル不一致・例外後のretryはしない。

[次回契約完成案](../artifacts/non_df_h4_precision_bc/2026-10-23/C_execution_contract_DRAFT.json)の最大範囲は、operator復元2 batches、確率対応2 batches、reference包含6、nominal law mean12、実装PR law mean6、PR normalization6。dense expは最大39 requests。保存入力を使うため新SCF/FCI/DF生成/状態最適化/sampling/compileは0、shot/G再会計も0で別承認となる。

**実H4 production assembler/certificate runnerは未実装。** [実装計画](../artifacts/non_df_h4_precision_bc/2026-10-23/C_implementation_plan.json)と契約に実装・回帰・新SOURCE封印を起動前条件として明記した。execution_authorized/grant_valid/dispatch_enabledは全てfalse、C grantは未発行・未消費。今回の実装済みprototypeと人工検査を実H4の実行可能性・成功証拠に読み替えない。

## 保全・引継ぎ・STOP

元SOURCE738 file、保存結果/claim1147 file、旧調査17 fileを作業前後でSHA照合する。Bの封印済みcode/runtime/input/outputも再照合し、[終了監査](../artifacts/non_df_h4_precision_bc/2026-10-23/completion_audit.json)へ保存する。保存データの整合性と元H4終了監査の欠落は独立した状態を維持する。

専用branchでSOURCE commitと結果commitを分けて保存し、pushは行わない。詳細科学レビュー・最終evidence freezeは未実施。[引継ぎ](../artifacts/non_df_h4_precision_bc/2026-10-23/CONTINUATION.md)と[今回専用STOP](../artifacts/non_df_h4_precision_bc/2026-10-23/STOP.json)を残す。

次の担当は**ユーザー**：C production実装を進める範囲、および新SOURCE完成後の実H4包含・certificate診断の実行範囲を明示承認する。承認後の**Codex**：実装・回帰・SOURCE固定と、別grantによる承認済み範囲だけを実行する。N/G再会計と公開pushの承認は別。今回はmandatory STOPし、自動移行しない。
