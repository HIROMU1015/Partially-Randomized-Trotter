# H4 PR-Pauli：κを変えた構成の信号取得とpacket公開

2026-11-16 JST、B2（Claude Code）。確認は作成者自身によるもので、独立レビューではない。承認は利用者の指示（comment(B1114)）による。

## 範囲

- 対象は、保存基準構成があるPR-Pauli（L_D=2・3・5）。条件はS2、Δ_PF=h、K=2、κ倍率∈{2/3, 1, 2}。
- 構成は、B2の最適化器がAのP1 B API（view `39a1fc7df`）で生成した。
  - 保存構成（κ×1）は完全に再現した。
  - 6構成をcommitした（`31d1bcc0b` の `configurations/`）。
- 価格：D2のPriceRules（MODEL、SOURCE `c1f95e08`）が、6構成すべてで26軸を返した（MODEL_REFERENCE 26）。不足軸はない。
- 信号は、Aの共通入口（SOURCE `93a5ef3b9`）で取得した。このbranchは、Aの `715af7788` から分岐している。入口の2 fileは、`93a5ef3b9` とbyte一致する。

## 結果

- **§10.5の確認**：この6構成IDの実信号を、他branchで探した。見つかったのはAの計画（`715af7788` のPLANS.json）だけで、実平均値はなかった。
- **計画（`--plan`）**：要求roundは、生成時に「変わった」と判定したroundと一致した。対応区分はすべて `SUPPORTED_VALIDATED_FOR_PAULI`。
- **取得**（`--execute --approve-new-means`）
  - SOURCE `31d1bcc0b`、新規平均値は計51件。
  - 各runの所要は約6秒。statusは `ENTRY_CARRIERS_COMPLETE_NOT_RPE_QUALIFIED`。
  - 取得したrunは raw `35b37ff33` に保存した。
- **組立て**（`--assemble`、`1537dfdf9`）
  - 9 packetを作った。内訳は、新構成6件と、κ×1の3件（保存記録の同一再利用）。
  - Aの組立て時に行うBの取込み条件の照合は、9件すべてPASSした。
- **Bの通常の取込み**：Bの `accept_saved(handoff_source=…)`（SOURCE `bd42dd1e9`）とtotal binding（`T.create`・`T.bind_signal`）を、9件すべてで通した。statusは `SOURCE_BOUND_LD{p}_ALL_13_MEANS_NOT_RPE_QUALIFIED`。
  - κ×1のLD3・LD5では、信号入力の平均値が、保存経路の平均値と一致した。
  - total configuration IDも、保存経路と一致した。

| packet | 新規 | 再利用 | 新規round | R(m12) | log B(m12) | \|corr−ref\|(m12) |
|---|---:|---:|---|---:|---:|---:|
| LD2 κ×2/3 | 13 | 0 | m0–m12 | 2804 | 0.0617 | 0.05473 |
| LD2 κ×1 | 0 | 13 | — | 4205 | 0.0411 | 0.05473 |
| LD2 κ×2 | 13 | 0 | m0–m12 | 8410 | 0.0206 | 0.05473 |
| LD3 κ×2/3 | 10 | 3 | m3–m12 | 440 | 0.0616 | 0.05465 |
| LD3 κ×1 | 0 | 13 | — | 660 | 0.0411 | 0.05465 |
| LD3 κ×2 | 11 | 2 | m2–m12 | 1319 | 0.0206 | 0.05465 |
| LD5 κ×2/3 | 1 | 12 | m12 | 3 | 0.0518 | 0.05476 |
| LD5 κ×1 | 0 | 13 | — | 4 | 0.0388 | 0.05476 |
| LD5 κ×2 | 3 | 10 | m10–m12 | 8 | 0.0194 | 0.05476 |

- 新規51件を含む全117行が、局所gateを通った（`LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED`）。
- κを2倍にすると、Rが2倍になり、減衰の log B はほぼ半分になる。|corr−ref| は、S2の決定論的誤差が支配するため、κによらず同じ値になる。全行は [SIGNAL_SUMMARY.csv](SIGNAL_SUMMARY.csv) にある。
- 範囲外として、新しいpath sampling・build・compile・価格計算・N・shot配分は、いずれも0件。

## 場所

| 内容 | 場所 |
|---|---|
| 構成 | `configurations/`（`INDEX.json` に、変種と基構成の対応がある） |
| 計画 | `plans/` |
| 取得run | `runs/<構成>/run01/`（raw `35b37ff33`） |
| log | `logs/` |
| packet | `packets/<構成>/MEAN_HANDOFF.json`・`ROUND_SUMMARY.csv`（`1537dfdf9`） |

Q3と、shot配分の探索・費用比較は、B2の最適化器branch（`codex/b2-rpe-method-optimizer-20261116`）の `docs/research/rpe_method_optimizer_h4_20261116/kappa_round01/` にある。
