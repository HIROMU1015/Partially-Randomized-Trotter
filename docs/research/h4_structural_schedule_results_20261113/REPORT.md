# H4 PR全13round：K0検証と平均構造RZ数（2026-11-13）

**残11roundのK0は全てPASSし、その各kernelを保存してからXY二値を評価した。新22値と、再実行せずcopyしたm0/m6の既存四値により、固定configurationの全13round・26軸のE[S]が揃った。** これはideal finite-K lawの平均構造RZ数であり、平均compiled価格E[C]、N×C/G、全RPE成功保証ではない。

正本：[26軸coverage](run01/COVERAGE.json)／[数値CSV](STRUCTURAL_MEANS.csv)／[共通metadata](run01/SHARED_METADATA.json)／[11law](run01/LAWS.json)／[保存四値](run01/SAVED_VALUES.json)／[CHECKS](CHECKS.json)／[STOP](run01/STOP.json)／[HANDOFF](HANDOFF.md)。各roundのK0/K1・kernel配列・状態辞書は`run01/m{m}/`に保存した。

## 固定条件と実行SOURCE

Linear H4/1.00 Å/STO-3G、保存tol-only DF10、保存H/state/snapshot、S2/L_D1/ΔPF=.1/finite-K2/h=.1/M12/P0、元fragment/basis/qubit順・scalar一回、symmetric-directional control・X/Y測定付きfull Hadamard wrapperを保持した。数値stateのground仮定は条件付きのままで、新たなground証明を付与しない。

| 項目 | 固定SOURCE・値 |
|---|---|
| 今回の実行SOURCE | **`de2427d4a31d6b4b39d2c8eb8b63e3c972ffb89e`**。コード改変・再起動なし、一括run01を1回実行 |
| 実行計画の公開 | `3a12c962f75d530a30884a47280c53fb72a02318`の[SCIENCE_PLAN](../h4_structural_schedule_preparation_20261113/SCIENCE_PLAN.json) |
| 既存m0/m6の正本 | `83d219cc68c25540507a5539ebd2da94c29d06b3:docs/research/h4_structural_kernel_20261113/run02/`。値・kernel・STOPを不変保持 |
| preparation | `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、324非identity component・9 basis |
| Hamiltonian | `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0` |
| state | `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b` |
| snapshot | `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96` |
| 元Accounting/native | `8457c443554af5fadaa3ba36f97f6316ca3c5050`。元lowering・literal比較・Z順序を保持 |
| 元RTE provider | `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。元component確率と有限K2 grammar |
| B公開受入基盤 | `3216d27a69ac8532d2a22e78f5f94491d202d238`、公開intake/source chainを参照 |
| B configuration | `030f3f548a35d98a88e62e92a35f132e0243a293:docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json` |
| B JSON SHA256 | `22a1bd1af8704006568c545eb8bcbce5bd476202ad59aaeaf79f50850815f943` |
| configuration ID | `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911` |

共通κは元S0のexact文字列を保持した。R255以降を2冪へ丸めず、表示κからscheduleを再生成していない。

```text
13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201
```

[preflight](PREFLIGHT.json)で固定SOURCE、B JSON、元入力hash、旧四値/kernel/STOP、数値契約を確認した。元snapshot/receiptと既存Contextで同じpreparationを1回復元し、保存hash・scalar・全component/basisを照合した。SCF/DF/Hamiltonian/stateは新生成していない。実行したのはCの構造価格scopeだけで、Dの残差・compile計画は実行していない。

## 全13roundの結果

quantity=`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`、expectation=`IDEAL_FINITE_K_SOURCE_LAW`。単位は測定1回当たり、元controlled PR・scalar・Hadamard wrapper・measurementを含む構造RZ数。X=cosine、Y=sineを別recordとして保持した。**今回の固定SOURCEでは全roundでX/Yのcentral値・数値区間が一致した**ため、表は両軸を一行に表示する。一般のcompiled価格や他条件での軸一致は主張しない。

| m | qPF | R_outer | E[S_X] = E[S_Y] | 両軸の数値区間 | 出典 |
|---|---:|---:|---:|---|---|
| 0 | 1 | 2 | 1743.6682557187337 | [1743.6682557186837, 1743.6682557187817] | 既存copy |
| 1 | 2 | 4 | 4061.9122038827045 | [4061.9122038824644, 4061.9122038829473] | 今回 |
| 2 | 4 | 8 | 11181.82399250839 | [11181.823992506812, 11181.823992510015] | 今回 |
| 3 | 8 | 16 | 35356.298237484414 | [35356.29823747044, 35356.298237499286] | 今回 |
| 4 | 16 | 32 | 123444.10221303834 | [123444.10221287991, 123444.10221321041] | 今回 |
| 5 | 32 | 64 | 458575.18667195557 | [458575.1866698496, 458575.18667426653] | 今回 |
| 6 | 64 | 128 | 1764659.275564426 | [1764659.275537014, 1764659.2755917928] | 既存copy |
| 7 | 128 | 255 | 6893622.590919902 | [6893622.590456948, 6893622.591432492] | 今回 |
| 8 | 256 | 510 | 27298207.30935696 | [27298207.302130263, 27298207.317372765] | 今回 |
| 9 | 512 | 1020 | 108641024.20905557 | [108641024.0948432, 108641024.3358428] | 今回 |
| 10 | 1024 | 2040 | 433461247.85987735 | [433461246.0437929, 433461249.8769487] | 今回 |
| 11 | 2048 | 4080 | 1731640054.5676427 | [1731640025.5995228, 1731640086.7473123] | 今回 |
| 12 | 4096 | 8160 | 6922151105.606054 | [6922150642.8322315, 6922151619.742445] | 今回 |

Tは各B rowの値（.1×2^m）、physical candidate・全identity/lawは[INPUT_METADATA](run01/INPUT_METADATA.json)とcoverageに保持する。元binding内のT=.8は参照metadataであり、今回の回路Tとして使わない。m0/m6の四値は元K1 recordと完全一致でcopyし、`m0/`・`m6/`という新実行directoryは存在しない。

## K0：11lawを独立に検証

R2/R128の旧PASSを他Rへ移植せず、今回R4/8/16/32/64/255/510/1020/2040/4080/8160の11lawを元APIで構成し、毎roundで全9wireのK0を検査した。

- 共通native分岐parity **21,474件**、元event metadata照合 **17,820件**。元angle/phase・products→rotation順・完全to_dict表現を維持した。
- phase/control barrierを元πの逐次加算0〜24,480回と3control入力で確認し、**73,443件**一致した。phase flagは証明済みの合法非identity K2 zero/positive同値類だけに使い、Z角の非ゼロflag化やmod 2πは行わない。
- 全99 wire-lawでlocal integer reward/正確なpending Z、閉包、entry/forward・close/reverse・outer出口、scalar一回・XY suffix/terminalの対応を確認した。state数は全lawでsystem wire0〜6が各56、wire7が48、controlが22となり、各reset由来enclosure内だった。
- 全99件でunit mass・外向き区間・longdouble probability powerを通過。最大mass defectはm12の **1.7141843500212417e−12**、固定tolerance **1e−10**以内。確率の事後再正規化・閾値緩和はしていない。
- 元event_probability式との因数分解相対差の観測最大は **2.369063583674636e−16**、固定γ16=1.7763568394002536e−15以内だった。samplerとideal法則の距離を資格したものではない。

各K0の正本は[m1](run01/m1/K0.json)、[m2](run01/m2/K0.json)、[m3](run01/m3/K0.json)、[m4](run01/m4/K0.json)、[m5](run01/m5/K0.json)、[m7](run01/m7/K0.json)、[m8](run01/m8/K0.json)、[m9](run01/m9/K0.json)、[m10](run01/m10/K0.json)、[m11](run01/m11/K0.json)、[m12](run01/m12/K0.json)。共通の分岐/phase件数を各K0へ出典付きで記録しており、round間で独立検査件数として重複加算しない。

各roundのK0 PASSとkernel/状態辞書を保存した後に、そのroundのK1を評価した。K0.json内`K1_started=false`と共通metadataの`physical_K0_for_each_R_still_required=true`は、それぞれK1前・law別K0前のsnapshotである。最終statusは各K1・全coverage・STOPを正本とする。全合法continuationへの対応は、元native alphabet/分岐のparity、全componentを含む閉graph、決定論boundary mapからの帰納であり、全trajectoryの列挙一致ではない。

## 数値精度・計算量・未資格事項

元[NUMERICAL_CONTRACT](../h4_structural_kernel_20261113/NUMERICAL_CONTRACT.md)と[NUMERICAL_ADDENDUM](../h4_structural_schedule_preparation_20261113/NUMERICAL_ADDENDUM.md)を固定したまま適用した。native Zは元順binary64・literal zero・hexで保持し、local rewardは整数完全一致。shared active basis/phaseをwire marginalへ残し、期待値の線形性でrewardを足す。wire間独立性やCIの独立合成を仮定しない。

centralは元nominal probabilityを補正せず使用した。理想weight比とIEEE/BLAS丸めモデルに基づくP/gの外向き区間を合成/二乗法で伝播し、同じγ16式のqR回分を固定精度interval二乗法で囲んだ。新しい角度丸め・近似law・mass修復を導入していない。表の区間は構造数と確率伝播の**数値enclosure**で、confidence interval、実samplerとの距離、compiler差、共有covariance、全RPE保証を含まない。後半roundで絶対区間が広がることをそのまま保存し、価格精度の採択基準を勝手に設定しない。

m12は33,423,360event位置に対応するが、全event列・324³の高次tupleは生成していない。局所有限alphabetとmatrix二乗法でP^R・outer合成を評価し、kernel/誤差因子はlog反復、phase確認はO(maxR)時間・定数追加cacheで実行した。保存kernelは11archive・693配列・99状態辞書。実compiler回路の巨大化を解決した結果ではない。

[資源記録](RESOURCES.txt)：同一preparationを1回復元し9worker poolを再利用、専用processのBLAS/OMP1、wall **4分38.00秒**、user 1336.49秒、最大RSS **580,560 KiB**、exit0。このrunの観測値であり、全worker RSS合計や他条件の速度保証ではない。共有設定・他ユーザーのprocessは変更していない。

実行途中、読取status commandに対する自動承認レビューが利用上限で完了せず、監視の一部が中断した。既存workerは停止・再起動せず、処理はexit0で完了した。ユーザーの継続指示後に既存sessionと保存STOPを照合した。これは科学的な失敗/再試行ではなく、追加の科学計算も起動していない。

## 検証と成果保全

準備段階の新8＋回帰18＝**26技術test PASS記録を再利用**し、コードは改変していない。実物理の証拠は今回の11 K0/99 wire-lawであり、人工testを代用していない。実行後のCHECKSでは、26recordの全identity/axis/law、旧四値完全copy、11 NPZ hash・693配列shape/finite/区間包含、K0保存→K1保存順、旧SOURCE/数値契約/STOPの不変を確認した。E[S]をもう一度計算して検査したものではなく、独立科学レビューでもない。

旧m0/m6結果、失敗したkernel run01、C1/C2、connected-cluster、S0a run01失敗/run02成功、SOURCE/input/raw/cohort/grant/ledger/marker/KEEP/STOPを保持した。今回の一括実行は1回、scientific restart/retry0。新trajectory sampling、full-path S、新回路build/compile/transpile、compiled C、C−Sの解析/fit、N/RMSE/G、H6、別preparation、8次PFは全て0。

## B/Dへの接続と次の作業

Bへ渡すのは構造quantityの26recordとSOURCE/interval。`FULL_WRAPPER_MEAN_COMPILED_PRICE`へ供給せず、E[C]とGはnull、C1 MODEL falseを維持する。normalization Bを価格へ乗算しない。

D固定[HANDOFF_C](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af6f27d7a6467bb355686421acd067544edcbbd2/docs/research/d_compiled_residual_c_minus_s_20261113/HANDOFF_C.md)が要求する同identity/axis/full-wrapper、同pathのC/S、law一致・未資格事項の分離を保持する。本Sは元C1 AccountingのSであり、実compiler loweringと同じnative構造量だと新たに認定したものではない。Dが指摘したcontrolled ZZの展開差を理由に、今回S′への変更や残差fitを行っていない。

E[C]=E[S]+E[C−S]には、同じlawの適格な残差と不確かさが依然必要。ideal E[S]とPCG64/CDF有限cohortのDを無条件に加えて全law平均compiled価格とはしない。旧C1 affine/local/paired残差を重ねず、どれを使うかはD側の科学計画で判断する。

次は**Codex B**が公開結果を同canonicalの構造diagnosticとして受け入れる。**Codex D**はユーザーが許可した範囲で同SOURCE/law/quantityの残差・compiled価格検証を担当し、**ユーザー**がその有限scope・必要精度・必要ならS′方針を判断する。**Codex C**の今回範囲は完了。追加scienceやGへ自動進行せず、commit/push・remote SHA/tree/主要成果物照合後**mandatory STOP**。
