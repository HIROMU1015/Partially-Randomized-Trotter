# PR-Pauli LD3：PF達成費用予算での有限shot配分評価

FULL_PF__B_PRICEが適合。条件付き参照費用は541936295.9549784 RZ（exact 625936421828/1155）。今回生成した4列内の最安適合点。 p7基準との差T−395,962,678は145973617.9549784 RZ、Full PF基準との差T−559,457,386は-17521090.04502165 RZ。

Full PF基準以下の適合点は得られた。p7基準以下の今回の2列には適合点がなかった。

新規Q3は4件。停止理由：`FIRST_FEASIBLE_IN_COST_ORDER`。追加信号、sampling、MC、compile、価格再fit、retryはいずれも0件。mandatory STOP。

## 固定4列と結果

H4／1.00 Å／STO-3G／tol-only DF10／sector(2,2)、PR-Pauli LD3。元ID dd5b881d…、total ID 6757998a…。保存H/state/snapshot、LD3辞書・law・共通κ・全13round・raw/normalization/scalarは不変。モデル参照価格は全26軸で適用されている。費用はP=0のfull controlled Hadamard wrapperの参照価格による条件付き算術であり、E[C]や正式Gではない。

| 順序・配分 | 条件付き費用 RZ（表示） | 予算残額 RZ（表示） | MSE下界 Ha² | MSE上界 Ha² | 判定 |
|---|---:|---:|---:|---:|---|
| P7__A_SHAPE | 389727554.4935065 | 6235123.506493506 | 0.00006611885493084976645618564180 | 0.00006611885607488526580922411385 | MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD |
| P7__B_PRICE | 395206798.2406926 | 755879.7593073593 | 0.000007598001948178956360376901640 | 0.000007598001948178956477278767110 | MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD |
| FULL_PF__A_SHAPE | 510918151.0476190 | 48539234.95238095 | 0.000005443474515472940520533416755 | 0.000005443474581780158234733415300 | MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD |
| FULL_PF__B_PRICE | 541936295.9549784 | 17521090.04502165 | 0.000002529235675649838530158704039 | 0.000002529235675649838987499630579 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |

全4列を評価し、UNKNOWN・INCONCLUSIVE・未評価は0件。上界<=exact τのみ適合、下界>τはモデル不適合、区間がまたげばINCONCLUSIVE、gate不成立はUNKNOWN。安い保留列：[]。不適合の結論は評価した各配分に限り、MSEのshot単調性やLD3全体の不適合を主張しない。

### X/Yの具体列・exact会計

- `P7__A_SHAPE`：X = `[21, 19, 18, 17, 15, 14, 12, 11, 10, 8, 7, 5, 3]`、Y = `[21, 19, 18, 17, 15, 14, 12, 11, 10, 8, 7, 5, 3]`。exact費用 `30009021696/77`、予算残額 `480104510/77` RZ。
- `P7__B_PRICE`：X = `[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 15, 4, 1]`、Y = `[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 15, 4, 1]`。exact費用 `456463851968/1155`、予算残額 `873041122/1155` RZ。
- `FULL_PF__A_SHAPE`：X = `[26, 24, 23, 21, 19, 18, 16, 14, 12, 10, 8, 7, 4]`、Y = `[26, 24, 23, 21, 19, 18, 16, 14, 12, 10, 8, 7, 4]`。exact費用 `10729281172/21`、予算残額 `1019323934/21` RZ。
- `FULL_PF__B_PRICE`：X = `[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2]`、Y = `[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2]`。exact費用 `625936421828/1155`、予算残額 `20236859002/1155` RZ。

保存VP（新規0call）のモデルMSE区間は[6.206210455307360129958626682E-7, 6.206210455307820120226411551E-7] Ha²で適合。保存VP費用は173557305044/165 RZ、p7基準との差T−395,962,678は655899776.8121212 RZ、Full PF基準との差T−559,457,386は492405068.8121212 RZ。元のSOURCE・結果・旧STOPは保全した。

VPを各軸上限として、各予算で既存A_SHAPE（重みVP）とB_PRICE（重みVP/C_ref）の正の有理数breakpoint規則を各1本適用した。4列とも別vectorで、SOURCE固定前に生成・順序・exact費用・残額を保存した。個別roundのriskを独立最適化せず、各列を全13round一組でQ3評価した。結果による配分変更はしていない。

## SOURCE・資格・保全

科学実行SOURCE `a844ac25f88c5cdf39e9e1837d85b6d4341774ff`。元価格SOURCE f25a05f8a92aab43cfbe30fc80b974054156aac1のHANDOFF_BC.json、信号・評価契約・total view・保存VPは2de33a4f4。具体的な入力・26軸積・生成元は[INPUTS](INPUTS.json)、重複照合・finite scopeは[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。有理数生成器e094b2d60のpure関数、修正済みLD3 metadata/observer、既存generic workerと高速DPを再利用した。作成者自身の人工検査であり独立レビューではない。

原API/partials/RESULT/RESOURCEはworker自身で保存し、親monitorから独立。科学結果snapshot `8afe9d59dcba9fc639c1968c1794d694ad24e3e0` は再実行せず保全した。各callのclosure/mass/residual/bounds/moments/scopeを既存validatorで検査し、runtime失敗はUNKNOWNで停止する。CPU・RSS・wall・子OS終了code/signalと、数値判定・保存状態をSUMMARY/原記録で区別した。全体wallは約395.622秒。各callのworker保存は完了し、記録されたworker/子process計381件の終了codeは全0。

Z1+T1 total decoder、保存solver energy −2.1663874486347634 Haへの全事象二乗loss、保存ideal finite-K rawを条件とするIID、EXACT_AFFINE_OF_BINARY64_RAW、exact τ=(0.00159360010199040 Ha)²を維持。binary64 allowanceは厳密certificateではない。物理ground資格、実sampler law移送、十分N/global RMSE/G_qualifiedは未認定。AのδEは加算していない。

全26軸はMODEL_REFERENCE_PRICE_NOT_E_C、qualified_for_candidate=false、CIなしを保持。較正q8/R660から各roundへの適用はAのモデルの範囲・未検証外挿を引き継ぐ。Q3適合は価格資格を与えない。指定PF基準も保存固定配分での条件付き達成費用であり、方式を最適化した場合の優劣・大域最適・正式winnerへ一般化しない。旧VP・旧予算・旧STOPは変更していない。

[C_COMPARISON_JOIN](C_COMPARISON_JOIN.json)にconfiguration・allocation・Q3原API・26軸価格・exact総額を結合した。ロードマップ本体は編集せず、影響は[HANDOFF](HANDOFF.md)。
