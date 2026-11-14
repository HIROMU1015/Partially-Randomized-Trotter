# 決定論PF Stage 1：12価格モデルの較正・係数とm6予測の凍結

2026-11-15 JST。C branch `codex/m5-m4-receipt-admission-20261114`を継続。**12/12 groupを個別に較正し、係数とm6予測を凍結した。入力保留0、独立holdout未評価。**

利用者が明示許可した範囲は、D Stage 1の24較正価格による12 fit、係数凍結、12件のm6予測固定、Dへの引渡しである。実行SOURCEは **`79c951b38bf9d8f1c80f28b2e7f82846d85c67e5`**。既存Cのexact `price_model_validation.fit/predict`を無変更で使用した。

## 正本成果物

- [run01/FROZEN_MODELS.json](run01/FROZEN_MODELS.json)：12 model_group ID、元24入力locator、SOURCE/cohort/scalar/物理条件、exact係数、m6 physical ID・軸・q・exact予測。Dの既存frozen_models readerの形式。
- [run01/FROZEN_SUMMARY.csv](run01/FROZEN_SUMMARY.csv)：12モデルの閲覧用一覧。比較数値はJSONのnumerator/denominatorを正本とする。
- [run01/INPUT_AND_EXECUTION_CHECKS.json](run01/INPUT_AND_EXECUTION_CHECKS.json)：受入12 group、保存4＋新規20、12 fit＋12予測、SOURCE、compile/holdout開封0。
- [PLAN.json](PLAN.json)：固定入力・モデル・比較policy・重複preflight。
- [TESTS.txt](TESTS.txt)：新規7＋既存15＝**22 test method PASS、skip 0**。fitは人工価格のみ。実24価格はmetadata照合だけ。
- `HANDOFF_D.md`：凍結結果commitとpath、Dの読取・判定条件。独立holdout取得・判定はDの別作業。

## 固定入力と重複照合

入口は **`b190bd815c886446291fac8b281c3c7df7343372:docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json`** と同HANDOFF_C。各groupの`C_inputs_source`を開き、`saved_records[*].record`と`new_records`だけを較正に使用した。indexに複製されたrecordとの完全対応と、元price/sidecar/既存guardの各軸の対応を確認した。

| route | new record/raw | bound sidecar/guard | 保存M5/m4 |
|---|---|---|---|
| FULL_PF | `1078d23621566728521d8056fead17118244be42`、`docs/research/d_full_pf_shortlist_prices_20261114/calibration/run01/` | `a393a65cc2f00cdad596e11b9910bae0394e2407`、同directory | 元record/price `3368805575d459d948e3afe7b1e54b4814f64129`、bound `2fa88065318f0f2ab13937ef610828200b35874c` |
| TRUNCATED_PF | `a04d40fe4d8f4bcf8083e4feebdf1535b5e7f5cd`、`docs/research/d_truncated_p7_t3_prices_20261114/calibration/run01/` | `2ff582779ee36741407cac2d27e0f7b243088a0d`、同directory | 元record/price `43c15bfa34a22f313e55b824223b200332b7f23f`、bound `2b94539784995790c8ff1ba7dee578137bfc6034` |

保存4件は元`EXPLORATION`のまま。原価格SOURCE `155529e20a8fac39e00786c40b57830b71203c4f`、元trial path/hash、個別X/Y receipt、元cohortを保持した。新cohortへの再発行・平均化・再compileはしていない。

共通指示§10.5に基づき、Cの固定HEAD `0b4b2d941`の関連凍結結果path、公開C/fit/freeze/Stage1関連ref、Dの固定index（fit/freezeはnull）、同Linuxユーザーの関連Python job名を確認した。同入力・式・比較規則の既凍結成果や重複jobは確認範囲にない。旧C1/H6等の別domain係数と人工fitを今回の保存結果として数えない。他担当private worktree/checkpoint・holdout値は読まない。実12 fitを再試行していない。

## 物理条件・価格quantity

Linear H4、1.00 Å、STO-3G、保存tol-only DF10、保存H/state/snapshot。Full PFはp10・一体項を含む11 block、Truncated PFはp7 B0I・元fragment順。FullのscalarはHamiltonian定数`2.29310124732`、p7はB0I tail identityを含む`2.293146549384713`。元scalarの量・SOURCE・総Tで一回の適用を保全した。

候補条件はS2・実幅16h/7、Yoshida4・16h/3、M5・16h、h=.1。較正m4/m5、T=1.6/3.2。予測m6はT=6.4。実幅とqは表示近似で作り直さず、固定CELL_PLANと既存C metadata plannerの物理条件に照合した。

価格quantityは**測定付きfull controlled Hadamard wrapperのcompiled全RZ数、P=0**。symmetric-directional control、Qiskit1.3.0、basis `rz/sx/x/cx`、opt1、compiler seed17、native `H46_LONG_TIME_NATIVE_COMPOSE`、RTE lowering対象外。価格のidentity/law/profile、X/Y、scalar、元SOURCE、H/state/snapshot・split/fragment順を既存C price_contextと照合し、D/Cの受入済みguard結果を再利用した。

## 凍結結果

`C_g(q) = beta_one,g + beta_q,g * q`。固定featuresは順に`one,q`。各group自身のm4/m5の2点だけを、`fractions.Fraction`による既存`EXACT_SQUARE_SOLVE`で解いた。X/Yの値が等しくても別model_group、入力record、係数、予測として保持した。

| route | 公式・実幅 | axis | q m4/m5 | 観測RZ m4/m5 | beta_one | beta_q | m6 q | 固定予測RZ |
|---|---|---|---|---|---:|---:|---:|---:|
| Full | S2・16h/7 | X | 7/14 | 54844/109381 | 307 | 7791 | 28 | 218455 |
| Full | S2・16h/7 | Y | 7/14 | 54844/109381 | 307 | 7791 | 28 | 218455 |
| Full | Y4・16h/3 | X | 3/6 | 70912/141517 | 307 | 23535 | 12 | 282727 |
| Full | Y4・16h/3 | Y | 3/6 | 70912/141517 | 307 | 23535 | 12 | 282727 |
| Full | M5・16h | X | 1/2 | 85960/171613 | 307 | 85653 | 4 | 342919 |
| Full | M5・16h | Y | 1/2 | 85960/171613 | 307 | 85653 | 4 | 342919 |
| p7 B0I | S2・16h/7 | X | 7/14 | 38898/77489 | 307 | 5513 | 28 | 154671 |
| p7 B0I | S2・16h/7 | Y | 7/14 | 38898/77489 | 307 | 5513 | 28 | 154671 |
| p7 B0I | Y4・16h/3 | X | 3/6 | 49924/99541 | 307 | 16539 | 12 | 198775 |
| p7 B0I | Y4・16h/3 | Y | 3/6 | 49924/99541 | 307 | 16539 | 12 | 198775 |
| p7 B0I | M5・16h | X | 1/2 | 60950/121593 | 307 | 60643 | 4 | 242879 |
| p7 B0I | M5・16h | Y | 1/2 | 60950/121593 | 307 | 60643 | 4 | 242879 |

上記は凍結artifactの転記で、新たな比較・順位ではない。全係数・予測は分母1のexact値。丸め表示を比較に使わない。calibration_residualsは全てexact0だが、2点2係数の補間であり独立検証の証拠ではない。intercept307を一般的なwrapper固定費と認定していない。

## m6独立holdoutの規則と状態

比較規則は **`91365dcf0610362a96aab101fbe0ec0e78f58af1:scripts/pf_price_stages.py:COMPARISON_POLICY`** をそのまま保存した。現在の固定policyはH-b・観測値基準1/50であり、過去C資料の「未採択2%案」は当時の履歴として保全した。

`abs(C_pred − C_obs) / C_obs <= 1/50` を12 model_groupごとに独立判定し、campaign成功は12件すべて個別PASSの場合だけ。X/Yやroute・公式をpoolしない。m6価格取得・開封・判定は今回0件で、`campaign_pass=null`。

全モデルは **CALIBRATED_UNVALIDATED / qualified_for_candidate=false / extrapolation_scope=null**。観測m6を見てのrefit・特徴量変更・閾値緩和は禁止。m7〜m12、別compiler/lowering/幅/routeの資格はない。係数の識別可能性と独立予測精度は別問題であり、未知のsystematic/covarianceを0としない。

## ロードマップへの影響・残る不足

決定論PF価格の課題は「Stage 1価格取得済み・未fit」から、**12価格モデル較正済み・係数と12件のm6予測を凍結・独立holdout未評価**へ進んだ。入力の未解決は0件。次の依存は、Dの別作業による同物理条件m6 X/Y・12wrapperの取得と、凍結済み予測/規則での判定である。今回の許可をDの実行許可へ拡張していない。

全RPE価格への長q外挿、signal/risk/十分N、全G、winnerは未評価。Full-H/Pauli fit、PR価格モデル、信号評価を実行していない。ロードマップ本体は統合担当が更新する。

原C候補対応表`0b4b2d941`、旧較正、保存4receipt、D Stage1原本、SOURCE/input/raw/grant/ledger/marker/KEEP/旧STOPは不変。他担当branch/worktreeを変更していない。作者自身の技術確認であり独立科学レビューではない。

**mandatory STOP：凍結結果とD HANDOFFの公開・remote照合で終了。m6取得・開封・判定、自動refit、他round予測、MSE/N/G、winnerへ進まない。**
