# A → B・価格担当：全RPEを目的とするPR探索

2026-11-13 JST。実科学・保存数値再解析・新候補最適化0。[REPORT](REPORT.md)／[SOURCE_MAP](SOURCE_MAP.json)／[有限科学計画](SCIENCE_PLAN.md)／[MOCK入出力](ARTIFICIAL_IO.json)。B正本は公開branch確認済み`3216d27a69ac8532d2a22e78f5f94491d202d238`。新しいB共通controllerやC価格modelをAで作っていない。

## 1. 比較単位と候補identity

比較単位は一つの全configuration θ。共通H/state/snapshot・original target（全保存DF-H）、h、RMSE target/誤差予算・loss/prior/decoder、価格profile/P0を揃え、全m/aのN×Cを比較する。Track Aの有限Tのwinner/Paretoやm12価格だけを転用しない。

configurationはBの`generate(spec, event_mapper=...)`又は元A fixed-grid adapterで生成する。B V2 `config.id/row.candidate.id/identity/law_binding`をそのまま採用し、別のcanonical IDをAで計算しない。元physical IDは`h46_long_time_signal.cid`由来の別IDとしてSOURCEと共に保持する。

| field群 | 必須情報 |
|---|---|
| physical input | H/state/snapshot/rank10/sector(2,2)/generation order/weighted-Hermitian completion、input/SOURCE、targetとstate資格 |
| split/preparation | L_D、one-body deterministic、deterministic fragment list、tail fragment list、full-Fock scalar/λ定義・SOURCE、preparation hash又は未取得null |
| whole schedule | M/h、全round T/q_RPE/qPF/actualδ、formula IDとsigned weights、共通κ kind/value exact/definition/mapper SOURCE、整数R/allocation、K・law、初期例外の有無 |
| physical row | candidate ID・原recipe、m/T/qPF/R/K/prefix、scalar(c+s_tail)一回、event数とgrouping、負時間・適用順 |
| numerical provenance | direct-head expm、finite-K polynomial/raw・corrected独立、exact-tail対照、precision/gates・empirical allowance、reference source・欠測と資格 |
| price context | X/Y axis、full measured Hadamard wrapper、compiler/control/profile、P0、units/cohort、model domainと校正/検証SOURCE、law・finite cohort/expectationの区別 |

H/state/Tが同じreferenceはL/R/K/κに依存しないが、そのSOURCE・独立pairの資格を保全する。tail λ/scalar/preparationはprefixに依存する。signed-piece allocationやq/R groupingをevent数だけで置換しない。one-bodyをrandomizeしたFull-H RTE、Full PF、B0/B0IはPRとは別の比較方式。

## 2. Aのmetadata-only API

```python
from trottertracks.pr_rpe_signal_plan import plan

# Bが生成・検証したconfiguration。物理候補対応は既存APIで提供する。
pending = plan(
    config, physical_by_round,
    b_validate=B._validate, b_exact_scalar=B.a.exact_scalar,
    numerical_recipe={
        "source": fixed_new_signal_source,
        "deterministic_PF": "DIRECT_EXPM",
        "finite_RTE": source_bound_existing_paired_Taylor_recipe,
        "sampling": False,
    },
    saved_inventory=metadata_only_inventory,
)
```

`plan`はB nativeのuniform L/formula/Kと全M+1roundを要求する。source-bound A fixed-gridでもuniform H4は対応可能だが、H6早期prefix例外は本helperの対象外で既存adapterを用いる。exact時刻とphysical binary64が異なる場合はSOURCE/grid不一致として拒否し、丸め直して合わせない。PR-Y4は現B正本が拒否する。

出力jobは`B_configuration_id/B_candidate_id/B_identity/B_law_binding`、`physical_context`、m/T/q_RPE/qPF、保存出典、欠測field、状態を持つ。science/provider calls=0、N_X/Y/C/G/scientific_PASS_FAIL/global_RMSEはnull。実行runner・N solverではない。

保存inventoryは`physical_context`と`source/status/signal_fields`だけを渡す。matching contextは物理recipe・structure・binding/profile/split/law・numerical recipeの**全一致**。保存値は読まず、`SAVED_MEAN_METADATA_NEEDS_ACCEPTANCE`までしか返さない。違うSOURCEの重複保存証拠はcallerが選ぶ。`NUMERICAL_FAILURE`はno-retry、field不足はno-fallback、候補無しはpermission-required。旧N/G recordをmeanとして渡さない。

Mやκが異なって同physical rowになる場合、raw meanの再利用可能性と、新B identity/whole-scheduleのN・RMSE資格を分ける。numerical backend変更後の旧meanは自動hitにしない。旧spectral meanを再利用したい場合は、それ自体の受入/allowance bridgeをSOURCE付きで別に評価する必要があり、今回実行していない。

## 3. 科学実行後のsignal packet

| output | 意味・制約 |
|---|---|
| raw | ideal finite-K lawの実際の±1観測mean。X: Re(raw)、Y: Im(raw)、p(+)= (1+coordinate)/2。outcome conventionは既存Bのまま |
| corrected / exact_tail / reference | normalization補正後mean、同PF exact-tail内部対照、全保存DF-H/reference。互いを勝手に代入しない |
| B/logB/attenuation | whole interrogationのfinite式。N policyで一回だけ考慮し、Cへ追加乗算しない |
| bias decomposition | corrected−reference、exact_tail−reference、corrected−exact_tailのsignal空間値。energy bias/RMSEではない |
| numeric evidence | exact-tail norm/unitarity、raw/corrected独立closure、normalizer感度、empirical u/logB margin、power境界観測の範囲、例外型/残差/未開始/取消 |
| scientific qualifications | 保存state/ground、finite-K移送、signed law、prior/alias/decoder、actual sampler/law、global boundの未資格/nullを保持 |

N packet化は既存A/Bのpolicy・translation関数で行う。raw signalはscalarであり、サンプル数やtrajectoryが既に取得されたことを意味しない。旧高確率型HoeffdingとRMSE nominalを混ぜず、nominal Nを十分Nとして渡さない。

## 4. Bに必要な接続判断

- **PR標準Y4**：Track Aでは既存だが、B3216のgenerateはPF以外にY4を許可していない。実menuに含めると決めた場合のみ、Bが最小formula対応・人工検査を行う。AでM5へ偽装しない。
- **K横断**：B selectはfinite_K/κ定義/mapperを同一方式内で固定する。K別stratumは現APIで扱えるが、横断最終比較にはBで明示menu仕様が必要。Aの探索器で代用しない。
- **round別独立R/分割**：既存R-selectionは高確率policyの固定予算下の仕組み。共通κ RMSE scheduleへ勝手に移さない。まず全round一つのL/formula/Δ/Kと共通κを比較し、任意round別のL/formula/Kを新変数として追加しない。
- **grid/M/予算**：B native ceil-gridと既存A fixed-gridは別定義。h/M/κ/λのexact表現と初期実幅を保持。M/h/予算やpolicyを変えた場合、元Nや誤差資格を維持したままIDだけ変えない。

Bは全round N/RMSE・loss/decoderと`cost/select`を担当する。二時点やm12の局所診断だけでall-round RMSEは資格されない。finite-K lawのraw分布を用いるdecoderがroundを結ぶため、共通κの違うroundを単純に混ぜない。

## 5. 価格担当への依頼仕様（今回取得しない）

候補の全round物理tupleとB identity/law、preparation、q/R/各piece time/count/K、wrapper/compiler/axis/cohortを受け、既存C interfaceのDIRECT/MODEL/欠測/外挿と資格を返す。L_D・formula・K・負時間・R変更は価格domainを変えるので、旧単一T単価やTrack Aの価格を自動移植しない。

`h46_long_time_native.compose/sample_requests/compile_cached`はnative recipe・fingerprint再利用先であり、そのlauncher/旧grantを再実行しない。価格モデルのgroupingが変わる場合は担当側で新descriptor・適用証拠を示す。meanのmatrix powerを量子回路のq/R縮約や無料実装とみなさない。normalizationと古典価格標本数をC×Nへ重複乗算せず、full-law期待値とfinite-cohort scenarioを分離する。

## 6. 次の担当・停止

**ユーザー**が最初の2全scheduleの固定menu・共通κ・grid/誤差契約と有限science scopeを選ぶ。**A**が直接head PF＋既存finite meanの必要最小限adapterを実装・人工テストし、承認後のみ全round signalへ進む。**B**が同RMSEのN/decoder/資格と全G比較を行い、**価格担当**が対応するC domainを返す。今回の新7＋旧10の17技術test PASSは科学資格ではない。公開・remote照合後mandatory STOP。
