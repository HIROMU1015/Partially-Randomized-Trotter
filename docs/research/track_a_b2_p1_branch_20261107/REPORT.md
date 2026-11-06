# Track A B2 P1：保存raw平均による全18round branch/energy復元

2026-11-07 JST / SERVER 2（GPUA100-gleap02）/ Codex主エージェント。

**`P1_CONDITIONAL_BRANCH_ENERGY_PASS`。保存raw平均から全18roundのnearest-branchを一意に復元し、最終energy `−2.1663821124672245 Ha`を得た。復元後の元H参照との差は`5.336167536285075e-6 Ha`で、P1目標`.00015936001019904 Ha`以内。** 既知energy・元H理想信号をdecoderに渡さず、branchの事後修正もしていない。これは保存された理想平均を使う条件付き数値correctnessであり、実量子測定の95%成功保証、全RPE費用改善ではない。**検証完了後mandatory STOP。**

## 固定SOURCE・入力と来歴

実行SOURCE **[`71cb4e10c4df30fa5fdbf5793a4ff9eeefab0936`][SOURCE]** を人工検査後・実行前にcommit公開し、remote/local treeと4変更ファイルのbyte一致を確認した。科学実行は1回、1.223秒。

- [`result.json`](../../../artifacts/track_a_b2_p1_branch/2026-11-07/run01/result.json)：固定commit/path/blob/SHA256、input context、alias、decoder全記録、事後correctness、SOURCE・環境・未保証範囲。
- [`rounds.csv`](../../../artifacts/track_a_b2_p1_branch/2026-11-07/run01/rounds.csv)：全18roundの入力raw位相、wrapped/復元位相、branch、best/runner-up距離・gap、energy、事後誤差。
- [実行前PLAN](PLAN.md)。原SOURCE・保存信号・前段STOP・private入力は不変更。

| 保存信号 | 固定結果commit・実行SOURCE・使用範囲 |
|---|---|
| 全18点の結合結果 | `d7a55ac181de71cd2b2c49644810373cafd0a450`、`artifacts/track_a_b2_p1_remaining_rounds/2026-11-07/run01/all_18_rounds.json` |
| 新規16点の原結果 | 同commitの同run `result.json`、実行SOURCE `071cb00955379b0bf4da149bd9793d4125968cad`、m1..16 |
| endpointの原結果 | `ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5`、`artifacts/track_a_b2_p1_two_round/2026-11-07/run01/result.json`、実行SOURCE `5ef55f923e73e61bada383d0cfe98283e7984e9d`、m0/17 |

結合結果の各rowを、それぞれの原結果と全fieldで照合し、来歴付加fieldを除いて一致した。両実行SOURCEとその保存SOURCE SHA256、科学条件・元H/DF/state/sector/order、m/Q/T、元の条件付きPASSを確認。異なるSOURCEを同一と偽装せず、各branch rowにsignal実行SOURCEと結果commit/pathを保持した。

固定contextはlinear H4 / 1.00 Å / STO-3G、cutoff前H_phys、保存DF12、Nα=Nβ=2/sector36、保存元H基底state、prep費用除外。厳密基底状態は研究上の仮定である。B2はgeneration-prefix L_D3、partial-S2、r4/K2、δ=.02、m0..17/Q=2^m。P1のε_E=.00015936001019904 Ha、95%目標、β=.4、prior[−128,128] Ha、shift0、U=exp(−itH)/φ=−δE、6成分予算・均等α=.05/36・u=1e-8・独立照合1e-9を変更していない。[P1正本][P1]、[全18点の前段検証][SIGNALS]

保存metadataの主identityは、raw H_phys `bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2`、full state `1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db`（S0 array hash）、DF Hamiltonian `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。interleaved α/β、Qiskit LSBとOF sector hashも一致。これは**保存結果間の来歴照合**であり、private state/HDF5やoperatorを新たに読込・再構築した検証ではない。PF/RTE誤差・半径も再計算していない。

## 既存decoderへの接続

`src/trotterlib/rpe_four_round_phase_validation.py`の**`reconstruct_rpe_phase`を無変更で直接使用**した。decoder SOURCEと`rpe_resource_accounting.py`は固定`66a41ad9219f12377e7d325e272c1a90d66ab7e5`とbyte一致。新しいdecoder・独自のbranch選択は実装していない。

入力は各roundの`Arg(z_raw)`だけ。B/Aが正、BA≈1、`z_raw≈A z_finite`、rawとcorrectedの位相整合、保存raw phaseの符号規約を固定uで確認した。normalizationを二重適用せず、核scalar/identityの再加算・phase alignmentをしない。補正後信号は整合チェックにだけ使い、decoderへ渡していない。

既存`_vectorized_reconstruction`も同じ保存raw位相で実行し、scalar版と全roundのbranch index・最終phaseが一致した。曖昧性は元SOURCEどおりbest/runner-up距離の`np.isclose(rtol=0,atol=1e-12)`で判定し、新しい閾値を追加しなかった。この距離・gapの単位は**base phaseのrad**で、raw round位相のβと直接比較しない。

新規は薄いadapter、限定runner、人工test、PLANだけ。**人工8件PASS（2.43秒、新規6＋既存decoder2）**。位相の正負・wrap越え・raw減衰、priorだけによるalias、曖昧branch拒否、不正round列/非有限値拒否、事後oracleがdecoderを修正しないことを確認した。private読込・分子信号解析は人工テストで禁止した。

## 初期alias：PASS

P1の事前energy区間から、真のbase phase区間は[−2.56,2.56] rad。β=.4を加えた範囲[−2.96,2.96]はprincipal interval[−π,π)の内部で、端点の残余marginは**.18159265358979304 rad**。対応するprincipal energy区間は`(−π/δ,π/δ]`。shift0のままm0のbranch0を選べる既存条件に整合する。

m0のraw Argは`.043327642246236156 rad`で、このprior＋β範囲にある。P1が採択した既存係数和に基づくpriorだけでこの判定を行い、基底energyや元H理想位相を使用していない。元係数和の厳密interval証明を今回追加したという意味ではない。

## 全18roundの復元

各rowの全精度値はCSV/JSONに保存。`base_phase_estimate`は既存decoderのprincipal base phase、ここでいうunwrapped round phaseは**Q_m×base_phase_estimate**である。branch indexはdecoderの0..Q−1候補番号であり、独自のunwrap番号に置換していない。

| m | Q | raw Arg(rad) | 選択branch | energy推定(Ha) | 事後絶対誤差(Ha) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | .0433276422462 | 0 | −2.166382112311815 | 5.336323e-6 |
| 1 | 2 | .0866552844925 | 0 | −2.166382112311882 | 5.336323e-6 |
| 2 | 4 | .173310568985 | 0 | −2.166382112312193 | 5.336323e-6 |
| 3 | 8 | .346621137970 | 0 | −2.166382112313414 | 5.336321e-6 |
| 4 | 16 | .693242275942 | 0 | −2.166382112318255 | 5.336317e-6 |
| 5 | 32 | 1.386484551895 | 0 | −2.166382112336263 | 5.336298e-6 |
| 6 | 64 | 2.772969103860 | 0 | −2.166382112390752 | 5.336244e-6 |
| 7 | 128 | −.737247099274 | 1 | −2.166382112462961 | 5.336172e-6 |
| 8 | 256 | −1.474494198490 | 2 | −2.166382112474508 | 5.336160e-6 |
| 9 | 512 | −2.948988397052 | 4 | −2.166382112467447 | 5.336167e-6 |
| 10 | 1024 | .385208513108 | 7 | −2.166382112469023 | 5.336166e-6 |
| 11 | 2048 | .770417026144 | 14 | −2.166382112467247 | 5.336168e-6 |
| 12 | 4096 | 1.540834052295 | 28 | −2.166382112467336 | 5.336167e-6 |
| 13 | 8192 | 3.081668104606 | 56 | −2.166382112467424 | 5.336167e-6 |
| 14 | 16384 | −.119849098070 | 113 | −2.166382112467113 | 5.336168e-6 |
| 15 | 32768 | −.239698196092 | 226 | −2.166382112467180 | 5.336168e-6 |
| 16 | 65536 | −.479396392084 | 452 | −2.166382112467269 | 5.336167e-6 |
| 17 | 131072 | −.958792784259 | 904 | **−2.1663821124672245** | **5.336167536e-6** |

全18点の選択は一意で曖昧性なし。m17のbest距離は8.88178e-16、runner-up距離は4.79368996205e-5、gapは4.79368996196e-5 rad。これが最小gapで、既存1e-12の曖昧性条件には該当しない。m0には前round・runner-upがないため、そのgapは数値0で埋めずnullとした。

## Energy correctnessと頑健性の範囲

**全18round復元完了後に初めて**、固定[元H state成果 `89165cf…`][STATE]の公開結果を読み、`E_phys=−2.1663874486347607 Ha`を参照した。これは事後oracleであり、初期推定・branch選択・位相補正へ戻していない。既存`_oracle_branch`の事後診断とも全roundで一致し、選択branchを一つも修正していない。

最終base phaseは`.04332764224934449 rad`、unwrapped round phaseは`5679.040724906081 rad`。符号規約どおり`E_hat=−phi_hat/δ`を適用した。最終energy誤差`5.336167536285075e-6 Ha`はP1目標内で、nominal horizonの`β/(Q_max δ)=.000152587890625 Ha`にも整合する。全roundのenergy誤差は、それぞれのnominal分解能内だった。

事後的な共通元H targetへのraw位相誤差は全roundでβ=.4以内。最大はm17の`.013988443020542992 rad`で、前段で保存されたPF位相誤差が支配的な信号分解と整合する。ここでPF/RTE誤差を再simulation・再fitしたわけではない。

既存のstrict branch条件`β<π/3`は成立し、そのmarginは`.6471975511965976 rad`。初期aliasと、**全roundが同じ元H target位相からβ以内にあるという十分事象**の範囲では、既存nearest-branchの条件に整合する。新しいperturbation sampling・確率評価・独自の安定性定理は追加していない。観測raw平均を中心とする任意の測定摂動に対する頑健性は`UNASSESSED`である。

| 判定対象 | 結果 |
|---|---|
| 保存平均によるnearest-branchの数値correctness | **PASS**、一意・scalar/vectorized・事後oracle一致 |
| 固定P1・保存平均でのenergy精度 | **条件付きPASS** |
| 共通真位相から各round≤βという既存十分条件 | 条件付き成立範囲を確認。実測でその事象が起こる確率は未評価 |
| 実量子測定を含む95%成功保証 | **NOT_CERTIFIED** |
| shot・価格・総RPE費用・強PF比較 | **NOT_EVALUATED** |

## 次の担当・承認範囲とSTOP

**ユーザー**：次に固定P1・保存18roundの実半径を使う条件付きshot会計へ進むか判断し、進めるなら誤差disk・数値allowanceの証拠水準と、既存IID/bounded-outcomeモデルによるshot算出の範囲を別途明示承認する。**Codex**は承認後にだけ既存実半径会計へ接続する。実測95%保証、価格MC、強PF適格性・価格、総費用比較を同時に採択したとは扱わない。

今回、private state/HDF5・積分/MOの読込、Hamiltonian/state計算、PF/RTE信号生成、sampling、回路build/compile、shot、価格・総費用、強PF評価、詳細科学レビューは0件。Python3.11.0rc1/NumPy1.26.4、GPU不使用。観測CPU affinity64、開始時空きRAM約388 GiB・load16.27/12.53/6.34、BLAS thread変数未設定で、資源cap・共有設定や他プロセスの変更はしていない。

SOURCE・公開JSON/CSV・REPORTを専用branchへ公開し、remote SHA/tree/全文一致を確認して**mandatory STOP**。旧SOURCE/raw/result/grant/marker/STOP、M1終了提案・M2保留とprivate入力を維持し、後続計算へ自動進行しない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/71cb4e10c4df30fa5fdbf5793a4ff9eeefab0936
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7bc952772e451f27c1654dab6803192f2b204c9f/docs/research/track_a_b2_signal_conditions_20261106/REPORT.md
[SIGNALS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d7a55ac181de71cd2b2c49644810373cafd0a450/docs/research/track_a_b2_p1_remaining_rounds_20261107/REPORT.md
[STATE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89165cf70958e7e4676d0580e3acb7faf588772c/docs/research/b2_hphys_ground_state_20261106/REPORT.md
