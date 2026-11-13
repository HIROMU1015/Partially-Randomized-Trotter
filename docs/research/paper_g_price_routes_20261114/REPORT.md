# 原論文型G（第1水準）の価格経路・route guard・nominal N接続

2026-11-14 JST。Claude Code（主agentのみ。独立レビューではない）。

- branch：`codex/paper-g-price-routes-20261114`。基点は`5fb2c32e`（C guard `3a8b37f1a`とTruncated T2を含む）。並行するC/Dのbranchとは独立させた。
- 実施範囲：既存SOURCEの調査、技術実装、人工・回帰テスト、有限計画の作成。

**実施していないこと**：実価格の取得、compile、sampling、保存値を使ったN・G・価格の新しい集計、新しい科学計算、契約の採択。

**成果物**：本書、[有限価格検証計画](PRICE_VALIDATION_PLAN.md)、[Bへの引渡し](HANDOFF_B.md)、[テスト記録](TESTS.txt)。

**変更したcode**
- `src/trottertracks/price_implementation_guard.py`：route拡張
- 新規`src/trottertracks/paper_nominal_cost.py`：第1水準の会計
- 新規`src/trottertracks/price_model_validation.py`：較正とholdoutの検査
- 新規`tests/check_paper_g_price_routes.py`

**テスト**：新規18件と、既存の回帰56件（guard 14、mean price 11、pipeline 12、D接続14、T2 caller 5）。計74件すべてPASS。

## 0. 要旨

1. **共通の価格量**：P0の測定付きfull controlled Hadamard wrapperのcompiled全RZ数（`FULL_WRAPPER_MEAN_COMPILED_PRICE`）。
   - X/Yは別軸で扱う。scalarは総Tで一回。compiler profileは共通。
   - routeはSOURCEで識別し、欠測を0価格にしない。
   - 既存のguardとPriceProviderは、上記を変更せずに守る。
2. **guardのroute拡張**：PRを「DF tail（新ZZ）」と「Pauli tail（Pauli D2）」の2つの合法routeに分けた。
   - Pauli tailは、tail辞書のkindから`PR_PAULI_TAIL`と判定する。DF用の`PR`項目へは戻らない（fallbackしない）。
   - Pauli D2の発行元（emitter）は、P3 native `9c4006bf`の`h4_d2_pauli_control._rotation/central_control_tokens`のASTで照合する。
   - DF系の新ZZ・旧ZZとPauli D2は、互いのlowering IDを使えない。
   - 旧Full-H汎用RZZを新ZZとして扱うことの拒否は、従来どおり維持した。
3. **第1水準（原論文型）の会計**：`paper_nominal_cost`を追加した。
   - nominal Nは、Bの既存`nominal_axis_shots`で`⌈B_m²[11+4(M−m)]⌉`として求める。Bには各方式自身のfinite log Bを使い、決定論PFでは`B=1`を強制する。
   - Nは`nominal_N_*`、`sufficient_N_*=None`、status `PAPER_NOMINAL_NOT_SUFFICIENT`として、十分Nとは別に保持する。
   - G_paperはguardを通した後だけ、exact有理数で`Σ N·E[C]`を計算する。B.cost・B.selectは呼ばない。
   - 価格の質に応じたstatusを付ける：決定論direct／資格付き期待値／資格付きmodel／finite cohort／未資格model／X/Y不一致。
   - 欠測があればG_paperはUNKNOWN。`G_qualified`と`sufficient_N`はnullのまま。
   - nominal packetをBの資格付き経路（B.assess）へ入れると拒否されることも確認した。
4. **価格modelの較正とholdout**：`price_model_validation`を追加した。
   - 構成のroundではないcell（Full-Hの単一groupや、PFのm4/m5点など）も、SOURCE付きのcalibration recordとして扱える。Dが求めていた形式にあたる。
   - fitは事前に決めたdesignでのexact解（未知数と同数の点では連立解、それより多い点では正規方程式による最小二乗）。
   - holdoutは事前に宣言した許容差で判定し、失敗してもrefitしない。
   - 係数は既存の`quote_model`のADDITIVE termへ渡せる。新しい価格評価器は作っていない。
   - qualificationの昇格は外部の採択事項として残した。
5. **価格がまだないもの**：Full PFの新しい定常幅（保存directはS2・Y4のm3とM5のm3/m4のみ）、Truncated PF（direct 0）、PR Pauli tail（direct 0）、Full-H κ2（direct 0）、PR DF tailの全roundの期待価格（modelは未資格）。正式なG_paperとG_qualifiedは、どの方式もまだ計算できない（§1）。

## 1. 方式ごとの価格評価経路と不足している証拠

| 方式・route | native/lowering（SOURCE） | 既存の価格証拠 | 期待価格E[C]の経路 | 不足している証拠 |
|---|---|---|---|---|
| Full PF | 決定論PF native、`NOT_APPLICABLE_NO_RTE` | D V2 receipt（`e7dca5f7`）は旧候補のみ。最新3候補（S2 16h/7、Y4 16h/3、M5 16h）の保存directは、S2/Y4のm3（実幅は2h/4hで定常幅ではない）とM5のm3/m4（m4は定常16h）（C `3a8b37f1a` COVERAGE） | 決定論なのでE[C]=C。定常幅ごとのq-affine model `a+b·q`（Cの`quote_model` ADDITIVE、本件の較正検査） | 定常幅での新しい較正点とholdout。m7〜m12（qは最大1792/768/256）への外挿の妥当性。保存M5 m4の受入 |
| Truncated PF（B0I） | 決定論PF native（保持block＋tail identity scalarを総Tで一回）、`NOT_APPLICABLE_NO_RTE`。guardでは`TRUNCATED_PF`として既に対応 | 0件（D receiptなし）。旧B0I p7 m3（`155529e2`）は旧backend・旧Y4/.4用で、別domain | Full PFと同じq-affine。block数の効果は同じ幅のFull PFとの対比較で分離する | Dのtruncated native builder・receipt（並行するD作業で整備中）。p7とFull PFの同条件価格。B0とB0Iの価格同値 |
| PR・DF tail | DF controlled ZZ、`CONTROLLED_ZZ_CX_CRZ_CX_V1`（`01e068af`） | C1のfinite cohort 32 record（`2e6a0cdf5`→`7dd61438`、model未採択）。D P1のC−S（m6） | `STRUCTURAL_PLUS_RESIDUAL`：Cの構造kernelによるE[S]と、Dの残差（C−S）を`combine_mean`で結合 | 全roundの残差model資格、law期待値（finite cohortとの距離）、新しいR依存のN |
| PR・Pauli tail | **Pauli D2**（新route `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、`9c4006bf:h4_d2_pauli_control.py`） | 0件。P3 run03/run04とO1-M1は別の分割・別lowering・別domain | Pauli辞書に合わせたevent law（K2、order確率）の期待値。DF用のE[S]・Accountingは流用できない | Dのtail dispatch（`controlled_tails`への接続）。B0 pilot（LD1/m0、1 path、X/Y）。辞書に依存する特徴量（support重み、X/Y局所基底、product順）のmodel |
| Full-H pure RTE（κ2） | 新ZZ（PRのDF tailと同じdomain）。scalarはc_H＋抽出identityを総Tで一回 | 0件。旧Full-H価格（light、T2.0）は旧`RZZ.control(1)`で、guardが拒否する | group分解 `a+b·s_m+c·(q_m−1)`（Cの`quote_model` FULL_H ADDITIVE。q変数はevent_count）。巨大なevent列は作らない | D P-1（κ2 m3、2 compile）。P-2の単一group calibration（R∈{566, 4528, 36223}）とm5/m6のholdout。group間の独立性と境界の局所性の検証 |

各方式に共通するルール：
- **E[C]は期待値。** 単一pathのcompiled価格を期待値として使わない。finite cohortの平均は、tier `FINITE_COHORT_SCENARIO`として、law期待値と区別する。
- **外挿は資格が付くまで`UNQUALIFIED_MODEL`。** 短い回路の価格を長いroundへ外挿する場合、係数の意味、加法性、境界、holdout、適用範囲が検証されるまで、このtierに留める。

## 2. 共通の価格量とguardの接続結果

**共通の価格量**
- `FULL_WRAPPER_MEAN_COMPILED_PRICE`：P0、`full_Hadamard_with_measurement`、`single_hadamard_interrogation_without_state_preparation`。
- X/Yは`B.p.measurement`で軸ごとに照合する。scalarは`ONCE_AT_TOTAL_T`。
- normalizationは価格の乗数ではない（`normalization_is_price_multiplier=False`）。
- compiler・control・wrapper・P0のprofileが一致すること。
- これらは既存guardの検査をそのまま使う。

**routeの判定**：`route(B, spec)`。

| 判定 | 条件 |
|---|---|
| `FULL_PF` / `TRUNCATED_PF` / `FULL_H_RTE` / `PR` | 従来どおり。`PR`は`tail_dictionary`がない場合で、DF tailを意味する |
| `PR_PAULI_TAIL` | `split.tail_dictionary.kind ∈ {GLOBAL_COLLECTED_JW_PAULI_OF_SAVED_DF_TAIL}`（Aの`714247619a`の形式） |
| 拒否 | 未知の辞書kind、またはPR以外の方式に辞書がある場合 |

**policyの規則**
- `by_method[route]`でpolicyを引く。Pauli tailには明示的な`PR_PAULI_TAIL`項目が必要で、DF用の`PR`項目へは戻らない。
- 同じpolicy内で、`PR`（DF）と`FULL_H_RTE`は同じZZ loweringを使う（従来どおり。旧ZZと新ZZの混在を拒否）。
- ZZ系にPauli D2を宣言したり、Pauli routeにZZ IDを宣言したりすると拒否する（`separate registered lowerings`）。
- sidecarの`implementation.route`は、DF系では省略可能（methodと同じ扱い）。Pauli tailでは`PR_PAULI_TAIL`の明示を必須とした。
- 発行元AST：新ZZと旧ZZは従来の`_append_central_application`。Pauli D2は`_rotation`と`central_control_tokens`を、`9c4006bf`の固定blobと照合する。D最新の`26d4f23e`とも同じblobである。

**拒否・受理を確認した組合せ（人工）**

| 組合せ | 結果 |
|---|---|
| 同じpolicyでのDF PR（新ZZ）＋Full-H（新ZZ）＋Pauli PR（D2） | 受理 |
| Pauli PRをDF用`PR`だけのpolicyで評価 | 拒否 |
| Pauli sidecarがrouteを宣言していない | 拒否 |
| Pauli routeにZZ ID、またはZZ系にD2 | 拒否 |
| Pauli IDなのにZZの発行元 | 拒否 |
| 旧Full-H RZZ | 引き続き拒否 |

**保証の範囲**：guardが保証するのは「供給されたSOURCE付き実装証拠と比較条件の整合」までである。producerがそのemitterを実際に有効化したことは、receipt側で確認する（従来と同じ）。比較policy（scenario）の科学的な採択はしていない。

## 3. 原論文型nominal Nと価格の接続仕様

**入力**
- Bのcanonical configuration。
- その方式のsignal packet：`I.signal_input`と同じcarrierで、各roundの`normalization.log_B`を含む。
- SOURCE。

**規則**
- `N_m,X = N_m,Y = nominal_axis_shots(M, m, log_B_m)`。Bの`pr_rpe_rmse`（固定`b9b9ffa6`）を再利用し、計算式を複製しない。
- normalizationの規則は`finite_B_diagnostic`だけに接続した。論文のκ envelopeへの置換はしていない。
- Full PFとTruncated PFは`log_B=0`を必須とする（`N=11+4(M−m)`。M=12ならm0で59、m12で11）。
- PR（DF・Pauli）とFull-Hは、producerのfinite log Bを使う。Full-H R448ではlog Bがroundとともに倍増する（D記録：m12で40.4）ため、nominal Nは事実上使えない規模になる。κ型ではlog B≤1/κ。
- 論文の適用条件（Cmaxのhorizonと予算）は`paper_applicability=NOT_EVALUATED`と記録した。必要なら、PR/PFについてはBの既存`pr_rpe_rmse.evaluate`で別途確認する。

**出力packet**
- `evidence_kind=PAPER_NOMINAL`、`status=PAPER_NOMINAL_NOT_SUFFICIENT`。
- `nominal_N_X/Y`と`sufficient_N_X/Y=None`を別fieldで持つ。round identityとlawを照合し、`signal_input`のbindingを持つ。

**G_paper**：`paper_cost(...)`
- `comparison_policy`は必須。最初にguard（`G.check`）を通す。
- その後、round/axisごとに、cohortとidentityで価格を選び、exactに`Σ N·E[C]`を計算する。
- 価格区間があれば`[Σ N·L, Σ N·U]`を併記する。点の値が区間外なら拒否する。
- 欠測があれば`G_paper=None`（`NOT_EVALUATED_MISSING_PRICE_OR_N`）とし、0では埋めない。
- statusは`PAPER_NOMINAL_G_<最も弱いtier>`。`G_qualified`・`sufficient_N`・`mse_qualified`はnull/false。

**比較**：`paper_compare`
- 同じpolicy、共通のH/state/h/M/profile、configurationの重複なしを要求する。
- 点の値で並べるが、`qualified_winner=None`、区間の分離判定は`None`のまま返す。

**第2水準との分離**
- nominal packetは、Bの資格付き経路（B.assess）では`UNKNOWN_N_EVIDENCE_KIND`で拒否される（人工で確認）。
- 第2水準（`G_qualified`）は、従来どおり十分N（Z1＋T1/Q3の採択後）とBのguard付き`compare/cost/select`で求める。

## 4〜6. 有限価格検証・Full-H group・Pauli tail

[PRICE_VALIDATION_PLAN.md](PRICE_VALIDATION_PLAN.md)に、件数・SOURCE・STOPを記載した。要点：

- **Truncated p7とFull PF**：同じ3 template（S2 16h/7、Y4 16h/3、M5 16h）・同じround（m4/m5で較正、m6でholdout）で対にして取得する。block数の効果を直接分離する。p8は任意。
- **Full-H κ2**：P-1（2 compile）の後、P-2の単一group calibration recordを、本件の検査で受け付ける。m7〜m12は未資格のscenarioとして扱う。
- **Pauli tail**：Dのtail dispatchとB0 pilot（LD1/m0、1 path、X/Y、2 compile）を最初に行う。平均event単価の資格（B1 pilot以降）とは分けて扱う。

## 7〜8. Bへの引渡しと承認範囲

[HANDOFF_B.md](HANDOFF_B.md)に、Bが必要とする入力、guardの再pin、第1水準の会計の取込み方法を記載した。新しい科学実行とcompileに必要な承認範囲は、PRICE_VALIDATION_PLANの§5にまとめた。

## 9. 検証と限界

**テスト**：[TESTS.txt](TESTS.txt)。
- 新規18件：Pauli route 7、第1水準 8、model検査 3。
- 既存の回帰56件はすべてPASS。旧guardの全14件は無変更でPASSしており、既存の拒否動作は保たれている。
- fixtureは架空の行列・価格・Nだけを使う。H4の配列、保存価格、compilerには触れていない。

**限界**
- Pauli D2 routeの登録は、発行元の照合までである。D2 routeの価格が科学的に資格を得たわけではない。
- nominal Nは論文の経験的な名目値であり、MSEの保証ではない。
- 第1水準のGも、価格model・finite cohortのtierに応じた条件付きの値である。

**保全**
- 固定SOURCE・input・raw・結果・grant・ledger・marker・KEEP・STOP、B・C・Dのbranch・worktree・processは変更していない。
- Bのguard binding（固定`3a8b37f1a`）は引き続き旧guardを実行する。新routeを使うには、Bでの再pinが必要（HANDOFF参照）。

**mandatory STOP**：価格の取得、compile、G計算へは自動で進まない。
