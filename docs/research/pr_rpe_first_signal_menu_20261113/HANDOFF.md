# PR-LD1/LD2初回menu → B・価格担当

2026-11-13 JST。**承認待ち・科学実行0**。[REPORT](REPORT.md)／[MENU](MENU.json)／[有限案](SCIENCE_PLAN.md)／[全round CSV](SCHEDULE.csv)。二候補は最終採択/winnerではない。

## 正本と比較条件

B SOURCE `437ba4acdb82708b6fc98e3cd119a2a734f7cf64`のfixed-A metadata adapterで生成したV2 configurationをそのまま渡す。controllerのblobは旧3216と同一だが、Bのshot設計は437ba4acdのwhole-shot family/claim区別を採用する。Bの新N solverが完成済みとは報告しない。

- [PR-LD1](CONFIGURATION_LD1.json)：`a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`。
- [PR-LD2](CONFIGURATION_LD2.json)：`1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`。

全roundにB `candidate/identity/law_binding`と元`physical_candidate`を保存。元exactκ/h/M・S2/Δ.1/K2・H4 DF10 H/state/snapshot・prior/decoder/εE/予算・compiler/control/wrapper/P0は共通。prefix依存λ/scalarと派生R・allocationはconfig別。qRPEとqPFはこのmenuで等しくても別fieldとして保持する。

旧S0のcanonical IDは元SOURCEのまま保全。今回のLD1は同physical recipeを新構成recipe/descriptor出典付きspecで生成した新ID。旧packetのNやscience資格を新IDへ貼らない。

## Aが承認後に供給するもの

数値SOURCE `fefd548ecd56f60d72031b3d385f6f5e6ee95d06`のproviderで、全roundのraw/corrected/exact-tail/reference・logB/B/attenuation・bias分解・allowance/数値gate・state/reference SOURCEを保存する。新reference0、旧mean再解析0。取得前の数値fieldはnull/未評価のまま。

保存13reference pairは同H/state/Tに一致するものだけを使用。原子Hamiltonianからの新DF、eigensolve、SCF、state生成はしない。λ/scalarも保存descriptorをcopyし、one-bodyはPRのdeterministic headに含める。Full-H pure RTEやTruncated PFのsignalと混ぜない。

## Bの受入先

最新B HANDOFFのcanonical `means`形式へ、各new recordの`B_identity/B_law_binding`をそのままcopyする。`mean.raw`はX=Re/Y=Imの未補正signal、`mean.corrected`は診断だけ。normalizationはproducer供給のfinite logBで、Bはμ/Bを再評価しない。B²の二重乗算をしない。

```python
# 科学実行後のrecordからSOURCE付きでcopyする形。今回この数値変換は未実行。
mean = dict(m=record['m'], identity=record['B_identity'], law_binding=record['B_law_binding'],
    source=numerical_carrier_commit_path_selector,
    raw=record['signal']['raw'], corrected=record['signal']['corrected'],
    normalization=dict(kind='FINITE_LOG_B_SUPPLIED_BY_PRODUCER',
        log_B=record['signal']['log_B'], source=record['source']),
    bias=producer_bias_and_sources, uncertainty=producer_numerical_and_unknown_errors,
    target=same_target_descriptor, qualification='UNQUALIFIED')
```

targetは元保存DF-H/同sector energy、shift0、保存数値state条件付き。B targetの`quantity='ORIGINAL_DF_H_TARGET_ENERGY'`はDF前の元分子Hamiltonian保証を意味しない。保存solver proxyの厳密ground資格や新simulationへの採択をAが自動的に行わない。truth proxyの出典はMENU.targetへ保持し、prior/branch oracleに使わない。

このnew producerは旧固定A packet用`mean_from_a`へ無理に通さず、上記canonical recordとの正当なSOURCE付きbridgeを用いる。実データのB `prepare/simulate/summarize`、whole-shot familyの値・loss/claim・MC精度・N資格は今回実行/採択しない。

## 価格担当へ渡すもの

同じconfig/全round CSVとlaw、physical candidate、prefix・fragment順、S2、q/R/K・allocation、scalar一回、axis X/Y、full measured Hadamard wrapper・compiler/control/profile/P0を使う。matrix powerは古典評価の圧縮で、回路event数やcompiled価格を減らすものではない。

LD1のm12 R8160とLD2のR2559は価格検証の必要条件を示すmetadataで、既存価格domainの採用/外挿許可ではない。one-shot価格・総event数・m12だけで順位を決めず、担当側がdirect/model/未評価と適用不確かさを返す。価格計算・compileはAの今回scopeに含めない。

## 次の判断

全round数値inputが揃った後、Bの同一target/claimで有効な全shot配分とC coverageが得られた範囲でのみ全Gを比較する。二構成で未資格なら優劣を保留し、INCONCLUSIVEやdeferred prefixを残す。次のprefix/κ/幅/K/formula menuは全RPEのN×C寄与を見て提案し、今回の二候補を探索の終端にしない。

**ユーザー**がMENUと有限signal scopeを承認するまでAはSTOP。承認後は**A**が最大26候補のsignal、**B**がshot/risk、**価格担当**が平均compiled価格をそれぞれ既存経路で扱う。
