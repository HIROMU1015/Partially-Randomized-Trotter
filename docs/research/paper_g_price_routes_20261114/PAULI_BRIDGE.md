# Pauli physical-ID bridgeの完成と共通価格評価経路への接続

2026-11-14 JST。Claude Code（主agentのみ。独立レビューではない）。

- branch：`codex/paper-g-price-routes-20261114`。基点は`cfc0fae6`（Pauli route・第1水準の会計・model検査）。
- 関連：[REPORT](REPORT.md)（§2のguardに追加した検査は本書§2）、[有限価格検証計画](PRICE_VALIDATION_PLAN.md)（本件で§6・§7を追加）、[Bへの引渡し](HANDOFF_B.md)（本件で§6〜§8を追加）、[テスト記録](TESTS.txt)。

**実施していないこと**
- 新しいH4/H6の計算、sampling、compile。
- 価格・N・Gの計算。保存値の科学的な再解析。
- 契約・許容差・比較policyの採択。

実データについては、A・B・Dが公開済みの**metadata**（configuration、辞書のbyte列とhash、carrier recordのphysical context・SOURCE・ID）だけを読んで照合した。signal値は読んでいない。

**変更したcode**

| file | 変更内容 |
|---|---|
| `src/trottertracks/rpe_optimization_pipeline.py` | Pauli bridge（`plan(..., implementation_sources=, pauli_dictionary=)`、provider `A_PR_PAULI`）、`provider_binding_check` |
| `src/trottertracks/price_implementation_guard.py` | Pauliのcanonical physical IDを検査。D2の発行元照合に、Dが実際に呼ぶ`central_control_body`と、同じcommitの`append_pauli_event`を追加 |
| `src/trottertracks/rpe_d_price_connection.py` | `pauli_requests`：Dの`round_request`とCのphysical contextを照合し、Dのroute・lowering・SOURCE定数とCのguardが一致することを確認 |
| `tests/check_paper_g_price_routes.py` | 18件→34件（Pauli bridge・実LD1/LD2のmetadata・Dの実sidecar関数との整合・Pauli価格の第1/第2水準） |

**テスト**：計90件すべてPASS。内訳は本fileの34件と、既存の回帰56件（guard 14、mean price 11、pipeline 12、D接続14、T2 caller 5）。

## 0. 要旨

1. **Pauli bridge**：Cの正式planner `rpe_optimization_pipeline.plan`が、A・BのPauli canonical configurationをそのまま受け付けるようになった。
   - physical IDは**Bのcanonical candidate ID**（`RMSE:`hash）である。旧DF形式のphysical ID（`candidate_id`。辞書を含まない）への置換は、plannerとguardの両方で拒否する。
   - 辞書は、固定SOURCEのbyte列のSHA-256、Aの`dictionary_identity`、λ_P、保存s_R、λのSOURCEをすべて照合する。splitとlawの辞書が異なれば拒否する。
   - roundごとのnumerical実装SOURCE（全roundのmap）を必須とする。
   - Aの`job`（`714247619a`）を固定blobから無変更で再利用した。Bのcanonical ID生成器は複製していない。B側で行うのは`B._validate`による検証だけ。
2. **実データでの照合（metadataのみ）**
   - 対象はLD1 `d905d219…`とLD2 `d7dd4f52…`の各13 round。
   - Cのplannerが作るphysical context（canonical ID、辞書identity、round別のnumerical recipe）は、Aの保存carrier record 26件とすべて完全一致した。
   - round別SOURCEは、m0・m6の再利用4件が`05447d1d`、残り22件が`7084fbc9`。
   - 旧形式IDとcanonical IDは、全26 roundで異なる。
   - Dの`round_request`（Dの固定SOURCE `8b28dce0`）の結果（T、q、R、K、prefix、candidate ID、λ_P、identity、law）も、CのLD1/LD2各26行（13 round×X/Y）と一致した。
3. **guardとDのprice contextの整合**
   - Dは最新の固定SOURCE `8b28dce0`（branch head `f2a2c3af`）で、Cの`cfc0fae6`の規約に合わせた。lowering ID `PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、`route=PR_PAULI_TAIL`、`rte_lowering_source=9c4006bf:h4_d2_pauli_control.py`である。Cもこの規約を維持した。
   - **Dの実関数で確認した**：Dの`implementation_facts`と`price_records`の出力をそのままCのguardに通すと、physical candidateにCのbridgeのcanonical候補を渡した場合だけ受理される。Dの`legacy_physical_candidate`（旧形式ID）では拒否される。
   - 発行元の照合を強化した。sidecarが指すcommitで、D2の`_rotation`・`central_control_tokens`に加えて、Dの`controlled_tail`が実際に呼ぶ`central_control_body`と、Pauli eventを発行する`append_pauli_event`のASTを、`9c4006bf`と照合する。
   - Dの単一pathの記録名（`DIRECT_SINGLE_PATH_WIRING`）と、Full PF/Truncated p7の記録名（`DIRECT_DETERMINISTIC`／`EXACT_DETERMINISTIC_CIRCUIT`）は、Bの価格kindではない。そのため、G会計では`Unknown compiled price kind`で拒否される（§2.2）。
4. **価格recordの扱い（人工価格だけで確認）**
   - Pauli価格の受理、辞書・law・SOURCE不一致の拒否を確認した。
   - nominal Nと十分Nは分離したまま。価格の欠測はUNKNOWNになる。
   - DIRECT、MODEL、FINITE_COHORTの区別を確認した。Pauliの決定論価格は拒否する。
   - 第1水準と第2水準が同じguard結果を使うことを確認した。
5. **Bの新decoder bridge**：interface仕様だけを作った（§4）。新しいIDは推測・生成していない。
6. **承認が必要な範囲**：§5と[有限価格検証計画](PRICE_VALIDATION_PLAN.md)§7。

## 1. Pauli physical-ID bridge（Cの正式planner）

### 1.1 呼出し

```python
from trottertracks import rpe_optimization_pipeline as I
req = I.plan(root, config, target=target, signal_source=assembly_source,
             implementation_sources={str(m): {'commit': ..., 'path': ...} for m in range(M + 1)})
# Pass pauli_dictionary=payload only for artificial fixtures; normally it is read from Git.
req['provider']            # 'A_PR_PAULI'
req['pending']['jobs'][m]  # A's job(): physical_context (canonical ID, dictionary, numerical recipe)
I.price_context(req, m)    # binding/profile/physical_candidate/structure/split/law
```

Bの受入view（`rpe_pauli_signal_intake.bind_packet`の`signal_packet`）からは、次の値をそのまま渡せる。
- `target=view['target']`
- `signal_source=view['source']`
- `implementation_sources=view['implementation_sources']`

### 1.2 検査（どれかが不一致なら`ValueError`）

| 対象 | 規則 |
|---|---|
| configuration | `B._validate`（V2のhashと全roundの生成規則）。routeは`PR_PAULI_TAIL`（辞書kindによる判定） |
| law混合 | `split.tail_dictionary == law.tail_dictionary` |
| 辞書のbyte列 | 固定SOURCE（`tail_dictionary.source`）の辞書JSONについて、`sort_keys`・compactでのSHA-256が`dictionary_sha256`と等しい（Aと同じ規約。Dの`canonical_sha256`とも同じbyte列） |
| 辞書identity | Aの`dictionary_identity(item)`（固定`714247619a`）が`split.tail_dictionary`と等しい（kind、SOURCE、prefix、mode mapping、係数の扱い、scalarの扱い、event space、P0残差のSOURCE） |
| H・λ・scalar | 辞書の`binding`がspecの`binding`と等しい（H、state、snapshot、profile hash）。`prefix == L_D`。`λ_P`が`split.lambda_R`とexactに等しい。`saved_DF_scalar == split.scalar`。`lambda_source`が辞書SOURCEに`selector=['lambda_P']`を加えたものと等しい |
| round別SOURCE | `{str(m)}`が全M＋1 roundを持ち、それぞれ40桁のcommitとpathを持つ |
| physical candidate | Aの`job`が返す候補の`id`がBのcanonical candidate IDと等しい。method=PR、system、formula、T・δ（exact）、prefix=L_D、K=finite_K、q=q_PF、R_outer、structureがB rowと等しい。`q_RPE=2^m`、`1≤prefix<rank` |
| 外部から渡された候補 | `pr_physical`を渡す場合はcanonical候補と完全一致が必要（旧形式IDは`no legacy-ID substitution`で拒否） |
| 入力の取り違え | `implementation_sources`・`pauli_dictionary`をPauli以外のconfigurationに渡すと拒否 |

### 1.3 出力と後段

- `pending.jobs`には、Aの`job`の出力（`B_candidate_id`、`B_identity`、`B_law_binding`、`physical_context`）に、`N_X/N_Y/C/G=None`と`status='PAULI_CANONICAL_PHYSICAL_BRIDGE_METADATA_ONLY'`を加えたものが入る。`science_calls=0`。
- `produce_signals`は拒否する。Pauliのmeanは、Aの公開producerからBのSOURCE束縛intakeを通して受け取る。
- `compare`の`provider_binding_check`：
  - `provider_binding.physical_context`が、Aの`job`のcontext全体（辞書identityとround別numerical recipeを含む）と等しい。
  - `numerical_source`が`mean.source`（raw carrier）と等しい。
  - `mean.implementation_source`と`provider_binding.implementation_source`が、`implementation_sources[str(m)]`と等しい。
- `price_context(req, m)`は、guardのsidecar、Dの価格request、第1/第2水準の会計で共通に使う。
  - 内容はcanonical候補、structure、`split`（辞書identityを含む）、`law`、`binding`、`profile`。
  - numerical recipeは価格の条件ではないので含めない。

## 2. 新Pauli guardとDのprice contextの整合

**経緯**
- `5ccc1dfc`の時点では、Dは別のID（`…_WITH_NATIVE_DF_HEAD`）を使い、routeを宣言せず、emitterのcommitを空にしていた。
- Dは`8b28dce0`で、Cの`cfc0fae6`の規約に合わせた（Dの[HANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2a2c3afd69d9013bc5f2e60973b2820ec8406b3/docs/research/d_pauli_pilot_full_pf_prices_20261114/HANDOFF.md)「Pauli pilotのSOURCE規約」）。
- 本件ではこの規約を変えず、physical IDと発行元の照合だけを強めた。

### 2.1 Dのsidecar（`8b28dce0`の`implementation_facts`・`price_records`）とCのguardの対応

| Dの出力 | Cのguardの要求 |
|---|---|
| `method='PR'`、`route='PR_PAULI_TAIL'` | configurationのroute（辞書kind）と一致。Pauliではrouteの明示が必須（DF系は省略可） |
| `native_policy='NATIVE_DF_HEAD_PLUS_PAULI_D2_TAIL'` | policyの`by_method.PR_PAULI_TAIL.native_policy`と一致 |
| `rte_rotation_lowering='PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF'` | `G.PAULI_D2`と一致、かつpolicyと一致。ZZ系との相互流用は拒否 |
| `rte_lowering_source=9c4006bf:src/trotterlib/h4_d2_pauli_control.py` | 指定commit・pathで、`_rotation`・`central_control_body`・`central_control_tokens`が`9c4006bf`とAST一致。同じcommitの`pauli_signal_adapter.append_pauli_event`も`9c4006bf`とAST一致。`central_control_body`と`append_pauli_event`の照合は本件で追加した |
| `native_source`＝実行commitの`pauli_tail_price_connection.py`、`wrapper_source=af284d85:ax2b_native_df_v5.py`、`scalar_source`＝辞書SOURCE＋`saved_DF_scalar` | commitとpathがあり、blobが存在する |
| `scalar_quantity=HAMILTONIAN_CONSTANT_PLUS_SAVED_DF_TAIL_IDENTITY_sR`、`ONCE_AT_TOTAL_T`、`resolved_scalar == expected_scalar` | policyの`scalar_quantity`と一致。総Tで一回 |
| `physical_context`＝binding、profile、`physical_candidate`、structure、split、law | `I.price_context(req, m)`と完全一致。**`physical_candidate`は、Cのbridgeのcanonical候補**（`id`＝Bのcanonical candidate ID、event_countを含む）。Dの`legacy_physical_candidate`（旧形式ID）は`Implementation Pauli canonical physical id`で拒否する |
| `profile_extension`・`price_profile`の`lowering_id`、`implementation_evidence`（固定JSONのassertion） | 従来どおり |

**Dへの依頼**（Dの「bridgeが新しい物理IDを定めた場合は`legacy_physical_candidate`を差し替える一点で接続できる」に対応）
- `scripts/run_pauli_ld1_m0_pilot.py`で`P.legacy_physical_candidate(...)`を使っている2か所を、`I.price_context(req, m)['physical_candidate']`に置き換える。ここで`req = I.plan(root, config, target=..., signal_source=..., implementation_sources=...)`。
- seedのkeyは、Dの方針どおりcanonical candidate IDのまま変えない。

### 2.2 価格kindの名前（C側の受入条件）

- Cの第1水準会計と、Bの`assess`・`cost`が受け付けるclassificationは、Bの`PRICE_KINDS`だけである：`DIRECT_COMPILED_DETERMINISTIC`、`DIRECT_COMPILED_FINITE_SAMPLE`、`MODEL_PREDICTION`。
- expectationは、`DETERMINISTIC`（決定論PF）、`QUALIFIED_EXPECTATION`、`FINITE_COHORT_ONLY`のいずれかで記録する。
- Dの`8b28dce0`のcallerの記録名は、guardは通るが、G会計では拒否される。
  - Full PF/Truncated p7：`DIRECT_DETERMINISTIC`／`EXACT_DETERMINISTIC_CIRCUIT`
  - Pauli pilot：`DIRECT_SINGLE_PATH_WIRING`／`NOT_A_LAW_MEAN`
- Full PF/Truncated p7をGに使う場合は、実行前にDの記録名を`DIRECT_COMPILED_DETERMINISTIC`／`DETERMINISTIC`にそろえる必要がある。Cで名前を読み替えることはしない（relabelになるため）。
- Pauli B0は配線確認なので、Gに入れないのが自然である。入れる場合は`DIRECT_COMPILED_FINITE_SAMPLE`／`FINITE_COHORT_ONLY`（tierは`FINITE_COHORT_SCENARIO`）とする。
- `price_model_validation.record`へは、Dの`fit_inputs`が`DIRECT_COMPILED_DETERMINISTIC`を明示して渡すので、model検査には影響しない。

### 2.3 Dと共有する規約

| 項目 | 内容 |
|---|---|
| 辞書 | Aのhash identity（`dictionary_sha256`、固定P0 SOURCE）。λは`split.lambda_R`の有理数（numerator/denominator）。Dの`round_request`はこの形を読む |
| event | Dの`seed`は既存のlong-time `trial_seed`（`af284d85`）に`row={'id': canonical candidate ID}`を渡す。各outer stepで`rte.sample_rte_events`がR個、計q·R個のeventを生成し、Kは有限 |
| SOURCE | Dの固定SOURCE `8b28dce0:src/trottertracks/pauli_tail_price_connection.py`（`f2a2c3af`と同じblob） |
| 照合 | `rpe_d_price_connection.pauli_requests(root, req)`。<br>・Dの`round_request`（固定blob）を、BのrowからCが作ったpacketに適用し、Cのphysical contextと照合する。<br>・Dの`LOWERING_ID`・`ROUTE`がCのguardと一致することも確認する。<br>・結果は価格`NOT_EVALUATED`の行で、期待するroute・lowering・native policy・scalar quantity・lowering SOURCEと`d_pauli_round`を付ける。<br>・旧DF adapter（`pending_requests`）は従来どおり`OUTSIDE_DOMAIN`を返す |

## 3. 追加・変更したテスト（人工の価格・辞書・N。実データはmetadataのみ）

| class | 件数 | 内容 |
|---|---|---|
| PauliRoute | 10 | route判定。Pauli D2とDF新ZZを同じpolicyで受理。DF用`PR`へのfallback拒否。Dのfactsの受理と、route欠落・不一致・emitter commit欠落の拒否。D2の`central_control_body`と`append_pauli_event`のAST不一致を拒否（差し替えblob）、D最新`f2a2c3af`での受理。**Dの実`price_records`**：canonical候補なら受理、`legacy_physical_candidate`なら拒否、wiring記録名はG会計で拒否。ZZ・Pauliの相互流用拒否。ZZ発行元の拒否。旧形式IDの価格を拒否。旧Full-H RZZの拒否 |
| PauliBridge | 6 | canonical ID・辞書・round別SOURCE。`_validate`の再現。`produce_signals`の拒否。旧形式IDの置換拒否。辞書hash・λ・scalar・λのSOURCE・lawの不一致を拒否。round SOURCE mapの欠落・不正を拒否。第2水準`compare`（guard 4 cell、SOURCE入替えと辞書なしcontextの拒否）。Dの`round_request`との一致と、λ不一致の拒否 |
| PauliPaperLevel | 3 | Pauli価格の受理（nominal N、十分N=None、G_paper exact、第2水準と同じguard結果）。FINITE_COHORT・未資格MODEL・決定論価格の拒否・欠測のUNKNOWN。辞書（split／law）・law binding・固定assertionの不一致を拒否 |
| RealPauliBridge | 3 | 実LD1/LD2の13 round×2：Aのcarrier recordとphysical context・ID・identityが完全一致。旧形式ID≠canonical ID。Bのintake viewの形でbinding検査に通る。Dの`round_request`と26行×2が一致。実round SOURCEの入替えを拒否 |
| PaperLevel | 9 | `cfc0fae6`の8件に加え、Dの記録名（`DIRECT_DETERMINISTIC`など）がG会計で拒否されることを確認 |
| ModelValidation | 3 | `cfc0fae6`から変更なし |

## 4. Bの新decoder canonical bridgeのinterface仕様（未実装）

**背景（`f3c84aa8` REPORT §5）**
- decoderは`spec.rmse_policy.error_contract.decoder`にある。新versionを採用すると、configuration ID、round candidate ID、identity、law_bindingが連鎖して変わる。
- **Pauliでは、physical IDがcanonical candidate IDそのものなので、回路が同じでもphysical IDが変わる**。DF tailの旧形式physical IDはdecoderに依存しない。
- Dのevent seedはcandidate IDから作る。Dの方針は「bridge後もseed keyは変えない（変えると別cohortになる）」である。そのため、既存cohortのseed keyは旧canonical IDのまま記録に残し、新IDで引き直したpathは別cohortとして扱う。

**Bが発行する記録（Cは新IDを生成しない）**

```text
kind: (Bが命名) 例: B_CANONICAL_DECODER_VERSION_BRIDGE
source: {commit, path, function}            # Bの生成関数
old_configuration, new_configuration: V2全体 # どちらもBが生成し、B._validateが通る
changed_spec_paths: [["rmse_policy","error_contract","decoder"], ...]  # 宣言した変更だけ
rounds: [{m, old_candidate_id, new_candidate_id, old_identity, new_identity,
          old_law_binding, new_law_binding}]  # 全round、1対1
claim: SAME_PHYSICAL_CIRCUIT_AND_LAW_DIFFERENT_DECODER
```

**Cの受入検査（Bの記録が公開された後に実装する）**
1. 両方のconfigurationが`B._validate`を通る。新IDはBの記録の値をそのまま使い、Cは生成しない。
2. specの差分が、宣言されたpathだけにある。そのpathはdecoder・error contractの範囲に限る。次は不変であること：binding、profile、formula、h、M、L_D、finite_K、split（辞書を含む）、law、kappa、event_mapping。
3. 全roundで次が等しいこと：T、actual_delta_pf、q_PF、R_outer、structure、mapping。加えて、旧・新の両configurationで`plan`したAの`job`のcontextが、`physical_candidate.id`を除いて等しいこと。
4. 価格の再束縛：
   - price record・sidecar・cohort keyを、identityの1対1対応だけで旧→新へ移す。値は変えない。
   - `rebound_from`（旧identity）とbridgeのSOURCEを付ける。
   - 既存の`implementation_evidence`は旧identityに束縛されている。次のどちらかが必要になる（ユーザーとBが決める）：
     - (a) Dが新identityで固定assertionを再発行する（再compileはしない）。
     - (b) Cのguardにbridgeを受け取る経路を追加する（期待identityを旧へ写像して照合する）。
5. signal meanは、旧identityのcarrierを同じ規則で再束縛する。numerical recipeとround SOURCEは変えない。N packetは新configurationで作り直す（`B.assess`はerror contractとspecの一致を要求する）。
6. **拒否する場合**：宣言されていないspecの変化、roundの欠落・多対一、physicalな差、Cで生成したID。

**担当**
- B：生成関数と記録。
- C：受入と再束縛。
- D：(a)を選んだ場合のassertion再発行。
- ユーザー：D1（新推定量の採用）と、(a)・(b)の選択。

## 5. 承認が必要な範囲（本件で追加・更新したもの）

| 項目 | 内容 | 前提 |
|---|---|---|
| A6 Pauli B0（更新） | LD1/m0、normal 1 path、X/Yで2 compile。Cの受入側（bridge、guard、Dとの整合）は準備済み | Dのcallerで`legacy_physical_candidate`を`price_context`の候補に置き換える（§2.1）。固定assertion。配線確認なのでGには入れない（入れる場合は§2.2の記録名で`FINITE_COHORT_SCENARIO`） |
| A1/A2の前提（追加） | Full PF/Truncated p7の価格をGに使う場合 | Dの記録名をBの価格kindにそろえる（§2.2） |
| A11 Bのguard再pin | Bのguard bindingを、本件のcommitの`price_implementation_guard.py`へ更新し、Bの照合テストを再実行する | Bのbranchの変更（Bの担当・指示） |
| A12 Pauli bridgeのB取込み | Bのintake viewの`C_plan_price_bridge='UNKNOWN…'`を、`I.plan(..., implementation_sources=view['implementation_sources'])`へ置き換える | A11 |
| A13 新decoderのbridge | §4。Bの記録→Cの受入→価格の再束縛 | D1の採用 |

価格の取得、Gの実計算、許容差の採択は、従来どおり[有限価格検証計画](PRICE_VALIDATION_PLAN.md)のA1〜A10で管理する。

## 6. 限界

- bridgeの照合は**metadataの一致**までである。Pauli meanの科学的な資格（sampler、state、MSE）は付与していない。
- guardのemitter照合は、producerがそのemitterを実際に有効化したことの証明ではない。証明はreceipt側で行う。
- Bの実intake moduleを、この環境で直接走らせてはいない。
  - Bのview形式（B HANDOFFのfield表）は、保存recordから組み立てて検査した。
  - B側で、`I.plan`と`I.compare`を実viewに通す確認を再実行する必要がある（A12）。
- Dの`round_request`は、S2・`signed_weights=[1.0]`・単一allocationのPR以外を拒否する（Dの仕様）。
- Dの実sidecar関数の確認では、架空の辞書SOURCEにblobがないため、`scalar_source`のlocatorだけを既存blobに差し替えた。それ以外はDの出力のままである。

**mandatory STOP**：本件の後、価格の取得、compile、G計算、Bのbranchへの適用へは自動で進まない。
