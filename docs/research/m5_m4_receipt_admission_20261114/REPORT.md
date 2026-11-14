# M5/m4保存receipt受入と決定論PF価格検証の準備

2026-11-14。技術実行のみ。**保存M5/m4 X/Yは受入PASS、再compile不要。** 許容差・比較policyは提案のまま未採択。実データfit・予測・Gは未実行。

## 1. SOURCEと保全

|対象|固定SOURCE|
|---|---|
|C guard・model検証|`c628e9e8447c11c984064b922cbce6c52dc72d48`|
|D metadata・record caller|`62ae1bd56718287de5d958cf0ed6411f5b7395c9`|
|D HANDOFF/NEXT_EXECUTION|`7e0d51ea86c773ededd694178ec0fb732ea5b6c1`|
|B受入基盤|`eecbcdaed88b626b00858c7e94b15f598258d5ec`|
|原価格carrier|`155529e20a8fac39e00786c40b57830b71203c4f`|
|D既存receipt照合catalog|`e7dca5f7301c8633b17a59b6a7161331303aef2d`|
|本受入bundle・固定assertion|`3368805575d459d948e3afe7b1e54b4814f64129`|

関連C branchが他worktreeで使用中なので、そのworktreeへ触れず、既存C worktree内でC628基点の`codex/m5-m4-receipt-admission-20261114`を使用した。branch全体のmerge、旧SOURCE/raw/STOPの変更はない。

## 2. 保存M5/m4の受入

対象：H4 DF10、Full PF p10、`4th(m5_best)`、T=1.6、q1、actual Δ=16h。physical ID **`0cd1527c80f7b9841a83`**。原receiptは`EXPLORATION/index0`で、両軸を個別に保持する。

|軸|保存compiled全RZ（copy）|分類 / expectation|第1水準の価格tier|
|---|---:|---|---|
|X|85960|`DIRECT_COMPILED_DETERMINISTIC` / `DETERMINISTIC`|`DETERMINISTIC_DIRECT`|
|Y|85960|同上|同上|

価格の新しい平均・合計・比率は算出していない。原trial、published carrierの各軸mean、既存catalogの価格が一致することを確認しただけである。cohort名を新D cohortへ変更せず、原SOURCE・trial locator/SHAを記録に残した。

**照合した科学・実装条件**（機械可読詳細：[ADMISSION.json](ADMISSION.json)）：

- H `c1a6c428…`、state `9bcc0b0e…`、snapshot `12219b7d…`。保存tol-only DF10、Linear H4 1 Å/STO-3G。carrier provenanceの入力bindingと現在configurationの該当全fieldが一致する。binding内の歴史的`T=.8`は入力ラベルで、対象回路時間はT=1.6。
- 原preparation hash `bb736dc3158aa52f205ae0c23e4be374ae4e5edfa614d3cbceda04dcd5a28cfd`。全10 DF fragment＋one-bodyの11 block。元fragment順、M5 signed weights、symmetric-directional controlを保全。
- scalar **c=2.29310124732**。原trialの値と、同一H/snapshotに束縛された既存manifestの`scalar.constant`が一致する。Full-H RTE用の`extracted_identity`や`total`はFull PFに流用しない。元composeはpiece内scalar=0、総Tに一度だけ`p(-cT)`。
- 元native SOURCE `27ebabdd2943a4de787ec8b976c38e06fda5a497`と現D native `af284d85…`のnative/signal/M5-native/M5-contract **4ファイルがbyte一致**。compose/native_context/compile_cached/cost_trialと、provider `c31ef424…`由来のdeterministic builder/full wrapperの該当関数も一致する。recipe名・compiler hashだけによる判定ではない。
- 測定付きX/Y full Hadamard wrapper、state preparationなしP0。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、compiler `1239304c…`。保存7個の作用/compiled-vector gateは元tolerance内、phase alignmentなし。gateを新しく評価したのではない。

原trial bytesは[ORIGINAL_TRIAL.json](ORIGINAL_TRIAL.json)。現在Dの`price_bindings`を再利用した[PRICES.json](PRICES.json)と[SIDECARS.json](SIDECARS.json)は元receiptのC側bindingであり、新しい取得receiptではない。原native名称から現`H46_LONG_TIME_NATIVE_COMPOSE`への対応は上記SOURCE一致に基づく。

固定bundleのassertionを実際に読ませたC628 guardの結果は[GUARD_CHECK.json](GUARD_CHECK.json)：両軸とも`IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`。比較用policyの代わりに、明示した**技術互換性probeのみ**を用いた。B eecの正式guardは同じC628をpinしており、Bの分類名とも適合する。正式comparison policyは採択していない。

## 3. 較正・holdout入力

[CELL_PLAN.json](CELL_PLAN.json)はDの既存`plans`を実行したmetadata。Full p10とTruncated p7 B0I、3公式、m4/5/6、X/Yの36行である。保存2行以外の34行は新規価格未取得のまま。p8、B0/B0I追加2 wrapper、Pauli、Full-H、旧T2再実行を加えない。

|公式 / actual幅|m4較正q|m5較正q|m6 holdout q|
|---|---:|---:|---:|
|S2 / 16h/7|7|14|28|
|Y4 / 16h/3|3|6|12|
|M5 / 16h|1|2|4|

**12個の独立したmodel**：方式2×公式3×軸2。featuresは`one=1,q=q_PF`、`C=a+bq`。`price_model_validation.record/fit/validate`、D `fit_inputs`をそのまま使う。各modelはm4/m5の2点のみでexactに解く。m6、他方式、他公式、別actual幅、X/Yをpoolしない。`fit_inputs`に渡す前に`CELL_PLAN.model_group`とphysical IDを照合する（検証器のdomain key単体には公式・幅が含まれないため、呼出側の分離が必要）。

[SAVED_CALIBRATION_INPUTS.json](SAVED_CALIBRATION_INPUTS.json)は保存2価格の入力整形のみ。m5がないので実modelのfitはしていない。具体的な取得・凍結順は[HANDOFF.md](HANDOFF.md)。

## 4. 許容差と実行可否

[POLICY_PROPOSAL.md](POLICY_PROPOSAL.md)の**H-b固定2%、観測値分母**を推奨する。未採択であり、2%は長qでの保証やGの許容誤差を証明する値ではない。失敗後の5%への切替・再fitは禁止。

Cの受入・D/B分類・scalar/quantity接続はREADY。ただしD62の`execute()`は較正とholdoutを一括取得する。Cの既存`full_pf_latest_price_guard_20261114/NEXT_PRICE_VALIDATION.md`にある**較正→fit/freeze→独立holdout取得**をそのまま満たす段階選択口はない。Dの34件を無条件に実行可能とは判定しない。Dに必要な変更は取得リストの段階分離で、科学的回路条件・34件上限の変更ではない。CはD SOURCEを変更していない。

## 5. 検証と終了

- 新規人工テスト7 PASS：receiptの非変更・欠測/不一致拒否、34新規＋2保存のmetadata、両軸guard、分類、12人工model、観測値分母・不合格時再fit不可。
- 既存C guard/paper/model回帰48 PASS。コマンドと対象は[TESTS.md](TESTS.md)。作者自身の技術確認であり、独立科学レビューではない。
- 実compile/sampling、保存S/Cの新平均・数値再解析、実fit・予測・G・N、policy採択は0件。既存C1係数・資格、旧T2、各STOPは変更なし。

残るUNKNOWNは未取得34価格、実affine精度、m7〜12移送誤差、全round同時誤差・共有systematic、正式比較の精度である。次の科学実行にはpolicy/許容差の事前採択、Dの段階分離SOURCE、取得＋C fit/freeze/holdout検証の明示承認が必要。公開・remote照合後 **mandatory STOP**。
