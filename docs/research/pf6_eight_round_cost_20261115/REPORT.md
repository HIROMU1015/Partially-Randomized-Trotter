# 決定論PF6：固定shot配分の既知8round費用と逆転条件

2026-11-15 JST。H4/linear 1.00 Å/STO-3G/tol-only DF10、sector(2,2)、保存H/state/snapshot、h=0.1/M12。Full PFとB0I p7各3公式の**保存直接価格96軸**を、BのQ3評価済み固定配分に結合した。今回の承認範囲は保存値の軸別積・部分和・差・逆転条件だけ。新Q3、shot探索、価格予測、compile、全13round正式Gは行っていない。

## 既知8roundの結果

各軸の配分は `N_X=N_Y=[59,55,51,47,43,39,35,31,27,23,19,15,11]`。X/Yを個別に積算した。以下の各round列は、そのroundの `N_X C_X + N_Y C_Y`。単位は**固定shot配分での総compiled RZ数**。state preparationはP=0、価格は測定を含むfull controlled Hadamard wrapper。

F=Full PF（一体項＋DF fragment0..9）、T=B0I p7（一体項＋fragment0..6、tail identity scalarを保持）。S2=2次・要求幅16h/7、Y4=標準Yoshida4次・16h/3、M5=Morales M5・16h。実q・実幅は各保存physical候補のまま。短roundで同じ幅や価格になっても別receiptを使った。

| 候補 | m0 | m1 | m2 | m3 | m4 | m5 | m6 | m9 | K_i（8round合計） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T-S2 | 686,760 | 640,200 | 1,155,966 | 2,101,746 | 3,345,228 | 6,044,142 | 10,826,970 | 56,820,074 | **81,621,086** |
| T-Y4 | 1,987,828 | 1,853,060 | 1,718,292 | 3,138,190 | 4,293,464 | 7,764,198 | 13,914,250 | 73,050,346 | **107,719,628** |
| F-S2 | 954,620 | 889,900 | 1,619,046 | 2,955,266 | 4,716,584 | 8,531,718 | 15,291,850 | 80,292,586 | **115,251,570** |
| T-M5 | 7,192,100 | 6,704,500 | 6,216,900 | 5,729,300 | 5,241,700 | 9,484,254 | 17,001,530 | 89,280,618 | **146,850,902** |
| F-Y4 | 2,791,408 | 2,604,800 | 2,415,360 | 4,422,982 | 6,098,432 | 11,038,326 | 19,790,890 | 103,944,682 | **153,106,880** |
| F-M5 | 10,097,024 | 9,416,000 | 8,763,840 | 8,077,232 | 7,392,560 | 13,385,814 | 24,004,330 | 126,095,338 | **207,232,138** |

この表の順は**既知部分だけの順位**。未取得U={7,8,10,11,12}は全6候補で残っており、全13roundの順位・winnerではない。96軸に欠測・重複・不一致は0件。残り60軸は入力表にC=nullとして保持し、0やモデル値で補完していない。

## 15組の同額・逆転境界

全13roundの形式的な固定配分費用を `K_i+U_i` とし、未知部分を次のまま残す。

```text
U_i = 31(C_i,7,X + C_i,7,Y)
    + 27(C_i,8,X + C_i,8,Y)
    + 19(C_i,10,X + C_i,10,Y)
    + 15(C_i,11,X + C_i,11,Y)
    + 11(C_i,12,X + C_i,12,Y)
```

下表はiを既知部分が安い側、jを高い側に揃え、差の小さい順に並べた。Δ=K_j−K_iはすべて整数。各行に共通して、

- `U_i−U_j < Δ`：全13roundでもiがjより安い。
- `U_i−U_j = Δ`：全13roundで同額。
- `U_i−U_j > Δ`：**逆転し、iがjより高くなる**。

| i（既知部分が安い） | j（既知部分が高い） | Δ=K_j−K_i（未取得5round加重価格差の境界） |
|---|---|---:|
| T-M5 | F-Y4 | 6,255,978 |
| T-Y4 | F-S2 | 7,531,942 |
| T-S2 | T-Y4 | 26,098,542 |
| F-S2 | T-M5 | 31,599,332 |
| T-S2 | F-S2 | 33,630,484 |
| F-S2 | F-Y4 | 37,855,310 |
| T-Y4 | T-M5 | 39,131,274 |
| T-Y4 | F-Y4 | 45,387,252 |
| F-Y4 | F-M5 | 54,125,258 |
| T-M5 | F-M5 | 60,381,236 |
| T-S2 | T-M5 | 65,229,816 |
| T-S2 | F-Y4 | 71,485,794 |
| F-S2 | F-M5 | 91,980,568 |
| T-Y4 | F-M5 | 99,512,510 |
| T-S2 | F-M5 | 125,611,052 |

例えばT-M5とF-Y4では、`U_T-M5−U_F-Y4=6,255,978`が同額、これより大きければT-M5が高くなる。T-Y4とF-S2では同じ向きで7,531,942が境界。大きな既知差がある対も、未知差の上限・符号を仮定していないので、全体順序を確定できない。Kiが安い方式を正式winnerとする判定は行わない。

## 入力同一性・再利用・資格

実際の有限集計SOURCEは **732c703b0ebfd6ca4fd32195fa0d60dd04298c3e**、[analyze.py](analyze.py)と[INPUTS.json](INPUTS.json)を積算前commitへ固定した。計画時・実行前の重複照合と適用経路は[PREFLIGHT](PREFLIGHT.md)。Bの既存研究系列 `codex/rpe-total-dp-acceleration-20261114` を継続し、D/C branchは書換えず固定Git blobのみ参照した。

| 正本 | 固定SOURCE・入口 |
|---|---|
| B固定配分・候補／total join | `8dba096dc39788b8948af9eafe394f03d386c520`、`docs/research/pf_shortlist_q3_fixed_results_20261115/{INPUTS,C_RESULT_JOIN}.json`、HANDOFF |
| C physical alignment | `0b4b2d941cd669aaa45231e1236bc2d3158a9d12`、`docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json` |
| m0 | `c95353a44b4ce99ed86e949e408d522d6ad7e7fe`、`d_pf_m0_direct_prices_20261115/run01/AXIS_PRICES.json`（12軸） |
| m1/m2 | `6e110c083ec2db2530cdda3f7b5c84d2a7d3f31c`、`d_pf_m1_m2_direct_prices_20261115/run01/AXIS_PRICES.json`（24軸） |
| m3 | `3b1301738507d612694bf865d904b9376e038f69`、`pf_m3_saved_price_admission_20261115/ADMISSION.json`（12軸） |
| m4/m5 | `b190bd815c886446291fac8b281c3c7df7343372`、`d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json`→元C_INPUTS・receipt（24軸） |
| m6 | `a55813373b84b9128cbd10c6812c0d26143a4280`、`d_pf_stage2_holdout_20261115/run01/RESULT.json`→各軸receipt（12軸） |
| m9 | `0a35101914c6833244b6e0bad7bc2767a49b5a47`、`d_pf_m9_holdout_20261115/run01/RESULT.json`→各軸receipt（12軸） |

価格入口の相対pathはすべて`docs/research/`以下。各行の元commit/path/selector、元生成SOURCE・canonical候補・physical ID、cohort、sidecar/implementation proof、B original/total candidate mappingはINPUTSと[AXIS_PRODUCTS.json](AXIS_PRODUCTS.json)へ保存した。旧価格IDをtotal decoder用IDで上書きしていない。

Cの既存implementation guard（c628e9e8）をB wrapperから実際に呼び、当該96軸がcheckedに入ることを確認。H/state/snapshot・split/retained Hamiltonian・law・signed formula/q/actual幅・scalar一回会計・測定軸・元native・compiler/lowering/full-wrapper/P0が元証拠に対応した。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric_directional、native=H46_LONG_TIME_NATIVE_COMPOSE、RTE非該当を保持する。

m3の原cohort null、M5/m4保存再利用のEXPLORATION、新直接価格のroute別cohortをそれぞれ保持。決定論的回路の直接価格を使い、cohort標本数やnormalizationをshot/価格へ二重乗算しない。m6/m9ではobserved_RZと元receiptを使用し、凍結予測・2%モデル判定を価格の代用にしていない。未取得roundのaffine価格や誤差を計算していない。

**固定Q3の部分保存状態は不変**：科学SOURCE313ed1950の保存componentから6件の条件付きモデル上界が閾値適合、原API return・CPU/peak RAM欠測、driver運用UNKNOWN。RECOVERED_RESULTを完全APIとして扱わず、再評価していない。配分はUSER_SUPPLIED controlであり、十分N・最適Nでも原論文nominal Nでもない。ground/target包含・実sampler移送・数値certificateの未資格も変更していない。

## 検証・成果物・STOP

人工5test PASS：軸別積と和、逆転符号と同額、既知差の順序反転、欠測/重複/null/非整数拒否、巨大整数、元metadataの保全を確認した。[TESTS](TESTS.txt)。整数はPython任意精度intのまま処理し、binary64への変換はない。96軸の小さな算術であり、追加の科学workerや独立AI agentを起動していない。

公開後、CSV改行だけをCRLFからLFへ統一し、writerにも同じ指定を追加した。各CSVの解析済みセルが結果commit `0a6f6e12fa8f847c3d2a758668848b7a62692ff2` と完全一致することを確認した。数値解析SOURCEは上記732c703bのままで、積算・Q3・価格取得の再実行はない。

- [AXIS_PRODUCTS.json](AXIS_PRODUCTS.json)／[CSV](AXIS_PRODUCTS.csv)：元receiptまで追える96軸のN・C・積。
- [PARTIAL_SUMS.json](PARTIAL_SUMS.json)／[CSV](PARTIAL_SUMS.csv)：6候補のround別X/Y・round小計・Ki・既知部分順位。
- [PAIRWISE_CONDITIONS.json](PAIRWISE_CONDITIONS.json)／[CSV](PAIRWISE_CONDITIONS.csv)：15組の整数境界。JSONに不等号・同額・未知10項のweightを明記。
- [INPUTS.json](INPUTS.json)：元参照96軸＋未取得60軸、6候補のallocation・SOURCE・部分保存状態。issue0件。
- [HANDOFF](HANDOFF.md)：D/Cへの具体的境界とロードマップ差分。

`G/G_qualified/sufficient_N/winner=null`。既知部分の最小値を全13round正式Gのwinnerへ昇格しない。旧結果・raw・STOP、D/Cの進行中作業は変更しない。結果commit/push・remote照合後**mandatory STOP**。Q3再実行・追加価格取得・shot探索・全13roundGへ自動進行しない。
