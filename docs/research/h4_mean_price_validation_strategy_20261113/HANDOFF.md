# B → D/C/A：平均compiled価格の検証・最終Gへの入力

2026-11-13 JST。[REPORT](REPORT.md)／[PRICE_VALIDATION_PLAN](PRICE_VALIDATION_PLAN.md)／[有限scope決定表](FINITE_SCIENTIFIC_PLAN.md)。**価格精度・confidence・方法・11menuの正式採択はしていない。新科学実行0、G/winner=null。**

## D/Cへ依頼する情報（次の実行を指示するものではない）

| 項目 | 必要な内容・その理由 |
|---|---|
| 推定対象 | ideal P又は具体的sampler QのE[C]、同candidate/schedule/axis、compiler・lowering ID、full wrapper/P0。旧価格と新価格の混在を防ぐ |
| 方法 | paired direct / 解析補正 / 凍結parameter model。残差はC−Sか、別の定義か。Sの二重加算を防ぐ |
| model設計 | p、features、rank/校正domain、係数SOURCE、feature期待値の取得法、全域model discrepancy根拠。Bで新model/kernelを作らない |
| 役割 | P1/development、P2 pilot、calibration、独立holdout、normal/forced strata、cohort/index/event digest/seed。replay・X/Yを標本数へ水増ししない |
| domain | per-round q/R/K/配分・preparation、最大m、被覆regimeと未評価セル。m0/m6だけでm12までqualifiedにしない |
| 数値・sampling | S/feature enclosure、mean-estimator CI又は共分散、range/varianceの根拠、n_path、confidence/同時被覆family、停止規則 |
| rare/law | stratum確率・条件付きmean/上界、TVの単位（event/path）・移送根拠。未観測rareの寄与を0にしない |
| 実行可能性 | 特にm12のfull-wrapper実測/保存が可能なSOURCEと負荷見積り。m6 timingの比例外挿を保証として使わない |

C9d95の構造packetは既存正本として参照し、値/数値区間を再計算しない。D39dddのP2はfresh n4×m0/m6のpilotであり、既に方法設計に使われたdataを後から最終独立holdoutにはしない。新旧loweringは`CONTROLLED_ZZ_CX_CRZ_CX_V1`と`CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424`として別contextを維持する。

## 精度要求は現在は条件付き

Nが未確定でも、セル別に|μ_j−c_j|≤b_j又は同時価格区間[L_j,U_j]を示せる。後で適格Nに対し`|ΔG|≤ΣN_jb_j`。同時にb_j≤r c_jが成立すれば、任意の非負Nで予測Gに対する相対誤差≤rとなる。**そのr、α、domainの採択は未了**。Dの2%/95%や1.2%×E[S]のMC予算をそのまま正式policyへ写さない。

A/Bが必要N又は根拠付き集合𝒩を供給した後、候補間で区別したい資源差に合わせて価格精度を絞る。現在のnominal NやS3観測を正式重みにしない。Bは実N×C/Gを計算していない。

## 既存schemaへの接続

| 既存部分 | 返す/維持する内容 |
|---|---|
| `V2MeanPriceLookup`・既存price records | `quantity=FULL_WRAPPER_MEAN_COMPILED_PRICE`、identity/law_binding/source/axis/cohort、mean.rz_count、model domain/qualification、uncertainty。構造quantityやpath値を全law平均として返さない |
| `compiled_residual` | paired_residuals/cohort_summaryのfinite標本統計。combine/b_price_recordは自動資格付与しない。複数D modelを重ねない |
| `uncertainty` | 各boundのtarget law/domain/confidence/source・stat/model/rare/lawの包含範囲、未知項null。path covarianceとmean estimator covarianceを別記する |
| `joint_uncertainty` | 既存2arm APIのordered labels、basis=`mean_price_estimator`、source、依存完備のcovariance。区間経路はsupplied simultaneous_price_boundsとbounds_cover_stat_model_and_lawの実証が必要 |
| V2 `assess/cost/select`・`evidence.plan` | 候補資格/不足とexact rational点会計を再利用。4方式のtotal CIは自動生成されないため、供給区間を伝播する必要な接続だけを将来追加する |

最後のflagを単にtrueにして未資格項を消さない。現在はnew mean-C recordを登録しておらず、C構造/D pilotをQUALIFIED_EXPECTATIONへ昇格していない。price profileへlowering_idを明示する将来の最小接続は前回menuのHANDOFFを継続する。巨大Nの点会計は既存整数/有理数を維持し、float covarianceの出力をexact資源値の代わりにしない。

## 最小の次の科学案

直接法なら必要roundの各n_jをrange/precision/confidenceから固定する。解析補正又はp parameter modelなら、校正rankを満たし凍結した上で**m12＋内部challenge1round**を独立実測する案を優先する。必要compile数は両軸で`2Σ n_path`。nとp/誤差根拠が未確定なので固定compile数を捏造しない。

2roundが通過しても未測定11roundの誤差根拠がなければ全13資格はない。全域解析根拠を受けるか、必要セルだけ直接測るか、domainを限定する。全13round Gは未被覆セルが残る間null。m7/8/9の逐次compileやDのP2継続をBから開始させない。

**次：D/C**が上記入力を供給、**A/B**がN/claimを整理、**ユーザー**が比較上必要な差・価格精度/confidence/domain・有限科学scopeを採択して明示許可する。今回は文書・人工回帰のみで、公開remote照合後mandatory STOP。
