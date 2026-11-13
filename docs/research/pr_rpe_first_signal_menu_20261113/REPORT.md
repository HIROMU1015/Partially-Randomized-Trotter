# 初回PR全RPE signal menu：L_D=1対2の承認待ち提案

2026-11-13 JST。**推奨は保存H4 DF10のPR-LD1とPR-LD2の二構成、各m0…12。prefixだけを外側設計変数として変更し、元exactκ・S2・ΔPF=.1・K2・h/M・target・誤差契約を保持する。** このmenuは科学実行の承認対象を具体化した提案であり、正式採択や最適候補の認定ではない。新signal/normalizer/N、保存科学値の再解析、MC、compile/C/Gは全て0。ここで一度STOPする。

[MENU.json](MENU.json)／[全26round schedule](SCHEDULE.csv)／[固定科学実行案](SCIENCE_PLAN.md)／[B・価格HANDOFF](HANDOFF.md)。[PREPARE_MENU.py](PREPARE_MENU.py)はBの既存metadata生成/validatorのみを用い、NumPy/SciPy/Qiskitをloadしない。[確認記録](METADATA_CHECKS.txt)。実行入口は既存`pr_rpe_signal_provider.SignalBatch`のままで、新optimizerや科学runnerは追加していない。

## 1. 選択理由と保留候補

| 候補 | 今回の位置付け・根拠 |
|---|---|
| PR-LD1 | 既存S0/S1＋S2全13roundと同じphysical scheduleの対照。最適と仮定しない |
| **PR-LD2** | 保存descriptorを持つ隣接prefix。DF fragmentを一つだけtailからheadへ移し、分割変更を最小にできる。新λ fit・新state・formula変更を要さない。性能順位を使わず、局所的な設計変更に対する全roundの応答を調べる |
| L_D=3/5/7/9 | 保存descriptorがあり後続候補として保留。悪い候補として除外したのではない。初回から大きなsplit変更・全prefix網羅をしない |
| L_D=4/6/8 | この保存contextには対応λ/scalar descriptorがない。今回新norm計算を追加しない |
| L_D=0/rank10 | Full-H pure RTE/Full PFの別比較方式。PRのprefix変更だけの初回menuへ混ぜない |
| M5・標準Y4・K変更・κ変更 | 初回は変更しない。PR標準Y4はB正本も未対応。K/κ/幅等はN×Cへの寄与を見た後の別有限menu |

同じκでもprefixごとのλ_Rが異なるためR_outerは変わる。**独立にRを探索した比較ではなく、共通paperκ規則から派生するscheduleを含めたsplit比較**である。headを増やせば1shot価格が増え、tail変更でbias・normalization・Nも変わり得るが、方向や総費用の優劣は今回予測しない。

## 2. 固定configuration・SOURCE

| 項目 | 承認対象の固定案 |
|---|---|
| 系/target | Linear H4/1.00 Å/STO-3G/保存tol-only1e−8 DF10、sector(2,2)/dimension36。全保存DF-Hの同sector ground energyを目標とし、保存数値stateを条件付き使用。元分子Hの厳密精度・ground同定は未資格 |
| PR | 一体のone-body＋DF prefixをdeterministic head、残りDF tailをideal finite-K RTE。S2、ΔPF=.1、K2、fragment順・scalar一回・law・P0を保持 |
| RPE | h=.1（元binary64のexact表現を保持）、M12、13round、energy shift0、元prior/decoder |
| RMSE予算 | εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski、finite_B_diagnostic。値は元S0からcopy。今回N生成・予算成立認定をしない |
| exactκ | 下記の元S0有理数。表示近似24.305274577756407をschedule生成に使わない |
| numerical SOURCE | **fefd548ecd56f60d72031b3d385f6f5e6ee95d06**：`pr_rpe_signal_provider`、direct head expm＋既存paired Taylor＋独立raw/corrected＋power/cache |
| B正本 | **437ba4acdb82708b6fc98e3cd119a2a734f7cf64**。controllerは前3216と同blob。Bのfixed-A adapter/identityを使用し、generic ceil-gridへ切替えない |
| A schedule | **510fbacd01a2d855202de4c045d8ba1972e5ff3d**のexact fixed-grid/commonκ処理。今回使用する既存schedule/mapper関数本文が同じことを静的確認 |
| input/tail | INDEX **7a8bc7f132e166390c943e6b32c92cc6abe1baf4**、H4 context SHA256 `8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f`、`prefixes.1/2` |
| reference/誤差契約 | **bba150834ca30bdb186fbfd58c55a8d51177b855**の`PHYSICAL_INPUTS.json`。既存13reference pairを両構成で共有。既存m3/m4のnorm欠測は旧証拠のまま保持 |
| S0条件の正本 | **7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75**：`run01/CONFIGURATION.json`。保存13meanは**169d3316dd008eb98dd9be56fe4ecefc5c1a7d60** |

```text
κ_paper =
13112614441153360648422410673503886836581176429267589536059228160
/
539496659426909122096437389293855970805348475396548904610992201

h_exact = ΔPF_exact = 3602879701896397/36028797018963968
```

全保存H/state/snapshotのhash・compiler/control/profile・prior/decoder・誤差予算・λ exact値を各[CONFIGURATION_LD1](CONFIGURATION_LD1.json)/[CONFIGURATION_LD2](CONFIGURATION_LD2.json)へ保存した。sourceの役割は入力由来と新構成recipeを区別し、元D1で新prefixが検証済みと見せない。bindingの物理hashは維持する。

| 提案 | B canonical configuration ID | 保存descriptorのコピー（再計算なし） |
|---|---|---|
| PR-LD1 | `a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d` | λ_R=2.8629570945298686、s_tail=.4177175120626939 |
| PR-LD2 | `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae` | λ_R=1.6030956277929231、s_tail=.2223759094107018 |

これらはBが新SOURCE付きspecから生成した**提案ID**。旧S0 V2 IDを新IDへ付け替えたものではない。LD1は全13physical candidate/IDが旧正本と完全一致することをmetadataで確認し、旧結果のSOURCE/数値は変更していない。

## 3. 全roundの整数schedule（metadata）

| m | T | qPF=qRPE | LD1 R_outer | LD2 R_outer |
|---:|---:|---:|---:|---:|
| 0 | .1 | 1 | 2 | 1 |
| 1 | .2 | 2 | 4 | 2 |
| 2 | .4 | 4 | 8 | 3 |
| 3 | .8 | 8 | 16 | 5 |
| 4 | 1.6 | 16 | 32 | 10 |
| 5 | 3.2 | 32 | 64 | 20 |
| 6 | 6.4 | 64 | 128 | 40 |
| 7 | 12.8 | 128 | 255 | 80 |
| 8 | 25.6 | 256 | 510 | 160 |
| 9 | 51.2 | 512 | 1020 | 320 |
| 10 | 102.4 | 1024 | 2040 | 640 |
| 11 | 204.8 | 2048 | 4080 | 1280 |
| 12 | 409.6 | 4096 | 8160 | 2559 |

ここではh=Δのため数値が一致するだけで、qRPEとqPFは別field。S2はpiece1つ・allocation=[R_outer]、whole events=qPF×R_outer。小roundの最小R=1も元整数化規則を維持する。Rの値やevent数はcompiled RZ数ではない。

## 4. 保存再利用と承認を求める新計算

保存H/state/snapshot、prefix1/2のλ/scalar、13full-H独立reference pair、元formula/allocation/law、既存コード・数値gateを再利用する。新reference/action/SCF/DF/state生成は0。

既存全13meanとLD1のphysical identityは一致し、Track A archiveにもLD1のm3/m4がある。LD2は今回照合したbba全13とTrack A archiveにphysical一致meanがない。既存値は旧spectral又は逐次actionと混在した数値SOURCEで、**fefdのdirect head＋独立normalized-power backendの値として自動受入できない**。今回は新しい受入/allowance bridgeによる保存値再解析を含めず、両構成を同backendで評価する最大26候補案を推奨する。新規数値leaf・normalizerの重複はplan内にない。

これは旧baselineが無効という意味ではない。新13候補＋旧13meanのSOURCE付き受入を選ぶ別案はあり得るが、新backendとの比較資格・必要な保存値診断を別途明示する必要がある。今回その案を無断実行せず、旧meanをcorrected/B等で再構成もしない。

| 科学処理 | 提案上限 |
|---|---:|
| round-candidate | **26（2構成×13）** |
| raw/corrected/exact-tail出力 | **78 complex（各候補3）** |
| S2 finite normalizer | **26 piece評価** |
| 保存reference pair | 13組共有、元数値gateの確認を実行scopeに含める |
| 新reference action | **0** |
| 旧保存mean/Nの再解析 | **0** |
| 新N/decoder MC/compile/C/G | **0** |

## 5. 次の全RPE総コスト判断へ進む基準

まず全roundのSOURCE・数値gate・出力が揃うかを判定する。未評価や失敗を0埋めせず、local成功を全RMSE資格へ昇格しない。bias/logB/attenuationはBの統計評価・後続menuの評価順に使う入力であり、単独の厳密pruningやwinner判定には使わない。

次に**B**が同target/decoder/claim/lossでwhole-shot配分を評価し、**価格担当**が同canonical/physical scheduleの平均compiled Cとdomain/不確かさを供給する。適格な全round NとCが揃った範囲でのみΣN E[C]を比較する。Nの点値が小さい、1shotが安い、m12が安いという理由だけで候補を確定しない。

後続は二候補を探索の出発点とし、全scheduleのbias・Nへの感度とhead/tail価格の寄与を見て、別prefix又はκ/幅/formula/Kの**一つの追加自由度**を次の有限menuとして提案する。未選択prefixはdeferredのまま残す。二点でPR探索を終了せず、実G比較が未資格なら順位を保留する。

## 6. 今回のSTOPと承認事項

今回完了したのはSOURCE調査・既存metadata経路による2canonical configuration/26roundの具体化・再利用inventory・有限計画のみ。実分子配列をloadせず、NumPy/SciPy/Qiskitもloadしなかった。productionコードと旧結果は無変更。

**ユーザーに求める承認は一件**：[SCIENCE_PLAN](SCIENCE_PLAN.md)のPR-LD1/LD2、元exactκと外側条件、signal SOURCE fefd、最大26候補/78complex/normalizer26・新reference0・保存mean再解析0の有限signal実行。N/MC/価格/G・候補追加・retryは含めない。現段階でSTOPし、その明示承認後にだけCodex Aが実行する。
