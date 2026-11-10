# H6 DF19：係数prior・D1一括評価の実行準備

2026-11-10 JST。**H4 D1完了実装の再利用とH6拡張を準備し、人工12件PASS。実H6の新prior／phase／radius／N／部分Gは未実行。**

今回の添付指示書§9は「実H6の新energy prior、phase/radius/N、部分Gの数値評価は、ユーザーによる明示的な科学実行承認後に実施する」としている。そのため現在の許可範囲では技術準備・人工test・計画を完了し、実H6入力への新しい科学適用は行っていない。**公開照合後mandatory STOP。** priorの数値をユーザーに手入力してもらう必要はなく、次の実行許可後に保存係数から構成できる。

## 再利用と変更点

- [H4 D1固定完了版][h4] `ffe8a71da98bec330ce1af109a9606f14e1fc598`の`h4_df10_rpe_d1.py`と既存7件人工testをファイル単位で取り込んだ。H4 branch/worktreeへの書込み、branch統合・履歴変更は0。
- [H4 scalar helper](../../../src/trottertracks/h4_df10_rpe_d1.py)の`coefficient_prior`、exact-rational Gershgorin／signed-square計算は同じ関数をそのまま利用する。H4の固定2点入口を保持し、既にsystem/time bindingされたpointを処理する`analyze_bound_point`を分離した。phase/radius/disk/Hoeffdingの式はH4 D1のものを再利用する。
- [H6 D1 adapter](../../../src/trottertracks/h6_df19_rpe_d1.py)は12 mode／rank19／sector400、固定4候補、T=.8/1.6への対応だけを追加する。H6の`configuration`はh=.8→.1だけを試し、両方不成立ならSTOP。H4の「さらに小さいhを探索する」分岐はH6で呼ばない。
- [明示実行CLI](../../../scripts/run_h6_df19_rpe_d1.py)は保存snapshot・2receipt・固定projectionを照合し、係数・occupation indicesだけを読む。state vector、full Fock Hamiltonian、trial state vector、eigensolveを作らない。`--help`だけを確認済みで、実入力を渡した実行は0。
- per-axis uはprojectionのscalarから推測せず、元`155529e2`の固定endpoint行からcopyする。旧finite-time Nやαaxis=.025は使わない。探索n9と確認n33は別出力とし、PF／PR各々の保存2round部分費用を既存会計へ接続する。価格modelの調査・再fitは行わない。

## priorの構成方法と科学条件

保存表現 `H=c+dΓ(h)+Σ λ_l dΓ(G_l)^2` を使用する。H4関数が行う処理は、Hermitian係数行列のGershgorin区間→6粒子の演算子区間→signed λごとのsquare区間の加算である。上界には保存sectorの先頭occupationという固定determinantの変分式

`<dΓ(G)^2>=(Σoccupied G_pp)^2+Σ(p occupied,q empty)|G_qp|^2`

を使う。H6ではNα=Nβ=3を照合する。determinantや区間は保存E0に合わせて選ばず、prior関数はE0/state energyを引数に取らない。binary64係数を正確な有理数として扱い、採用ground区間のfloat端点を外向きに丸める。包含の対象は**保存Hermitian DFモデル**であり、元分子H、completion前のH、数値stateの厳密ground同定へ保証を広げない。

| 項目 | 固定条件／実行後の判定方法 |
|---|---|
| target | Linear H6／1Å／STO-3G／保存tol-only DF19 ground。保存completion・signed weights・fragment順・scalar・sector・H/state bindingを保持。元HへのDF energy reserveを追加しない |
| energy/failure/state | 既存CA=1.59360010199040e−3 Ha、各方式αtotal=.05、保存数値state、P=0。`ground_state_certified=False`を維持 |
| h | 保存係数priorで`h max(abs(Emin),abs(Emax))+.4<π`を確認。.8が成立すれば採用、失敗なら.1、両方失敗ならphase/N/費用を評価せずSTOP。energy shift=0 |
| β・horizon・α | βtotal=.4/stat=.2。選んだhで既存`required_rpe_round_horizon`を使い、全round/両軸一様`.05/[2(M+1)]`。旧13/18roundやH4のh/Mを流用しない |
| disk | H4と同じ保存bias＋元軸uのbox disk、referenceと保存solver位相のoffset、T×保存residual、log B margin／raw closureを使う条件付き数値pilot。residualだけで真groundを証明しない |
| ideal/actual | idealは宣言したfresh IID ±1／exact nativeの条件付きN。actual implementation/law誤差は未知のまま、actual Nとactual費用はnull。数値uや未観測eventをrigorous boundや誤差0へ格上げしない |
| 部分費用 | Nの条件成立時だけ、同candidate/compiler/whole-wrapperの保存Cを使用。主RZと副資源を別表示し、n/Bを再乗算しない。included/missing rounds・partial-onlyを保持。全CI/covariance／未知幅はnull |

**現在、上表の実H6 prior・alias margin・採用h・M・α・4候補N・部分Gの新しい数値は全て未算定。** この資料はH4の数値をH6結果として表示しない。

## 承認後の一campaign

1. 固定[H6保存4点][prepared]と元receipt／snapshotを読み取り照合する。T=.8はFull19 `76601611270ed9f0677d`とPR5/R32/K2 `2606efdab20e3b8522e5`、T1.6はFull19 `a4f5ffc0f320644dadfa`とPR7/R16/K2 `2d2af6ca830dff14a695`。全てm5_best/q_PF1。別候補へ変更しない。
2. 係数priorを一回構成し、h=.8→.1の順でalias条件を判定する。両方不成立なら理由とpriorを保存して終了する。成立時だけhorizon／一様αを決定する。
3. 固定4点をそれぞれ一回評価する。h=.8ならm0/1、.1ならm3/4。q_PFとq_RPEを別記し、state/reference・normalization・bias/u・disk・radius・βdet・Nの資格を保存する。保存E0はreference診断だけに使い、priorやdecoderのbranch選択には使わない。
4. 成立したideal Nと固定直接価格でPFの部分費用、PR探索n9・PR独立確認n33の部分費用を別々に出す。条件不成立／未資格はN又は費用をnullにする。全law期待価格保証や全RPEの勝者は認定しない。
5. SOURCE/input、prior導出、alias/配置、4行の結果、3方式/cohort別部分費用表、未評価round、資格とSTOPをfresh resultへ保存し、GitHub公開・remote SHA/bytes照合後mandatory STOP。

CLIは次の形であり、**このコマンドの実入力実行には§9の科学実行許可が必要**。秘密情報・承認原文・grant等の入力は要求しない。

```bash
python -B scripts/run_h6_df19_rpe_d1.py \
  --snapshot <saved-H6-snapshot.npz> \
  --snapshot-receipt <original-snapshot-receipt.json> \
  --df-receipt <original-DF-receipt.json> \
  --saved-points docs/research/h6_df19_rpe_preparation_20261110/SAVED_POINTS.json \
  --output <fresh-result.json>
```

## 人工検証と保全

- [H4既存7件PASS](h4_df10_ARTIFICIAL_TEST_RESULTS.txt)：exact係数prior、複素cross term／signed square、Hermitian/sector、horizon／α、保存scalar glue、ideal/actualの区別。元test本文をそのまま再利用した。
- [H6差分5件PASS](h6_df19_ARTIFICIAL_TEST_RESULTS.txt)／[test source](../../../tests/check_h6_df19_rpe_d1.py)：同じprior関数への委譲、h=.8/.1の両失敗でSTOP、12mode/19fragment/400sectorのtoy係数loaderがstate object配列を読まないこと、2時刻のscalar処理、PF/PR・価格cohort・欠測roundを分離する部分会計。

testは人工係数・人工signal/価格/energy精度のみ。保存実H6係数によるprior、新H6 phase/radius/N/部分G、H4科学結果の再計算は0。既存H6専用venvを無変更で利用し、依存installや共有設定・他process操作は0。H4 prior関数と元test、既存H6 projectionの保全を確認した。人工PASSは研究性能の証拠ではない。

同じH6準備branchを`21d4a098abd67e83ddf451626e5080b04a819b7f`から継続する。過去のREPORT・raw・結果・SOURCE・grant/ledger/marker/KEEP/STOPは書き換えない。今回の新資料は数値結果のstatusを更新しない。

次の担当：**ユーザー**がこの固定campaignの科学実行を明示許可する。**Codex**は許可後に上記を一括実行し、途中の細かな再承認は求めない。D2/D3、追加signal、長q価格、compile、詳細科学レビューには進まない。

[h4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ffe8a71da98bec330ce1af109a9606f14e1fc598/docs/research/h4_df10_rpe_d1_20261110/REPORT.md
[prepared]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/99ad77ef949b70192918a8700b092dd33c17af8c/docs/research/h6_df19_rpe_preparation_20261110/REPORT.md
