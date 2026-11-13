# H4物理kernel run02：K0 PASS・K1平均構造RZ四値（2026-11-13）

**固定H4の1 preparation・R2/R128についてK0をPASSし、そのkernelを保存した後、承認範囲のm0/m6×X/YのE[S]四値を完了した。** 全event列・324³のorder2 tuple列挙、trajectory sampling、full-path S計数、回路build/compile、compiled C、fit、N/RMSE/Gは実施していない。

正本：[K0](K0.json)／[状態辞書](KERNEL_STATES.json)／[kernel配列](KERNELS.npz)／[K1四値](K1.json)／[CHECKS](CHECKS.json)／[STOP](STOP.json)。出力quantityは`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`、期待値は`IDEAL_FINITE_K_SOURCE_LAW`である。**平均compiled価格E[C]ではない。** 本結果は作者自身のSOURCE/parity/数値検査で、独立科学レビューではない。

## 固定対象と実行来歴

Linear H4/1.00 Å/STO-3G、保存tol-only DF10、保存H/state/snapshot、S2/L_D1/ΔPF=.1/finite-K2/h=.1/M12/P0、symmetric-directional controlとX/Y測定付きfull Hadamard wrapperを保持した。元fragment/basis/qubit順、scalar一回、元binary64演算順を変更していない。保存数値stateのground仮定は条件付きのままで、本価格状態検査が厳密ground証明を付与するものではない。

| 項目 | 正本 |
|---|---|
| 今回実行SOURCE | **`19b95d8b3c88063d2dbae4e5bb47455b52b00903`**。実装改変なし、run02起動1回 |
| 再開計画 | `7aca17fe84c801a78113fdf8cabb910480ac3b96`の[RUN02_PLAN](../RUN02_PLAN.json) |
| 前段実物理診断/縮約 | `a637ed1b813dc3952339cb05f36beb79168ace23`、診断SOURCE `eeccdf18ee839ccddb824afbc5c8d0aa8b52b7b6`、縮約SOURCE `2b3b0ee7ffea92de25a2c0c549d64f258c34a611` |
| preparation | `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、324非identity component/9 basis |
| Hamiltonian | `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0` |
| state | `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b` |
| snapshot | `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96` |
| 元Accounting/native counter | `8457c443554af5fadaa3ba36f97f6316ca3c5050`、今回bytes一致 |
| 元provider | `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、rte/df_partial_s2/df_rte_tailを照合 |
| B configuration | `030f3f548a35d98a88e62e92a35f132e0243a293:docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json` |
| B正本JSON SHA-256 | `22a1bd1af8704006568c545eb8bcbce5bd476202ad59aaeaf79f50850815f943` |
| configuration ID | `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911` |

B `f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8`の受入索引から正しいconfigurationへ辿った。指定最新`3216d27a69ac8532d2a22e78f5f94491d202d238`でもこの受入JSONがbyte一致することをpreflightで確認した。SOURCE locator以外の科学関数body・数値契約は初回SOURCE `0f7d65d5f20de26c449b33c8da9d110188245808`から変更されていない。元run01のUNKNOWN/STOP/logは不変である。

共通κは元S0のexact文字列をそのまま保持し、近似値からscheduleを再生成していない。

```text
13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201
```

保存snapshot/receiptと元C1/S0a Contextで同じpreparation descriptorを1回復元した。新SCF/DF/Hamiltonian/state生成は0。preflightの入力hash・元SOURCE・B chain・scalar/支持集合の固定metadataは[記録](../PREFLIGHT_RUN02.json)を参照する。

## K0：実物理kernelとparity

前段の有限状態十分条件はSOURCE上の構成的論証だった。今回は、その条件に従う**実物理のtransition/reward kernel、閉包、境界、確率検査を構成してPASSした**。実全event列を走らせたparityとは区別する。

状態は共有active basis（完全operations/inverse・元to_dict表現を区別）、order2-seen flag、wire別の正確なpending Z hexである。Noneと数値0を区別し、Z加算は元binary64逐次順、zeroはliteral判定とする。元C1が実際に使うto_dictのtuple/list・matrix_sha256=Noneを含む表現を保持し、head metadataへ付け替えない。非ゼロflagやmod 2πでZを縮約していない。

| 検査 | 結果 |
|---|---|
| 元native basis/role分岐 | **3,978件完全一致**。324 component×3 application種×2law×2phase＋全90 basis切替 |
| 元event metadata | **3,240件一致**。全rotation component、order0/order2、正負product4template、2law。angle/phase/application順・to_dict表現を検査 |
| wire局所reward/出口Z | 有限graphで使用する全局所表を元Accounting._nativeと完全照合。整数rewardとpending Z hexが一致 |
| phase/control境界 | **1,155件一致**。πの元順加算0〜384回×3 control入力と、元reverse-head reset条件を確認 |
| 閉包・状態数 | 両lawともwire0〜6は各56、wire7は48、control wire8は22。各reset由来enclosure以下 |
| close/reverse・outer出口 | 全状態で履歴に依存しない同じ正確な出口Zと元queued head basisへ到達 |
| probability mass | 最大欠損`3.019806626980426e−14`、固定tolerance`1e−10`以下。事後再正規化なし |
| 数値区間/longdouble | 各law/wireのP^Rでunit massを区間に含み、全18件のlongdouble確率伝播が区間内 |

局所parityの件数は[K0.wires](K0.json)に記録した。同一wireでR2からR128へcacheを共有するため、`exact_local_table_parities`は累積値であり、18行を独立件数として足し合わせない。

全合法continuationに対する対応は、元SOURCEの有限alphabet分岐一致、全componentを含む閉state graph、決定論boundary mapからの帰納である。324³の全tupleや全trajectoryを列挙した証拠ではない。shared basis/phaseを各wire marginalへ残し、期待rewardの線形性を使用する。wireの統計的独立性や共分散の独立合成を仮定していない。

order2はphase-positive→product→product→rotationへ因数分解した。これは元のjoint component lawに対する数学的な期待kernelの合成で、sampling順や回路のproducts→rotation順を変えたものではない。元event_probability式との相対差の観測最大は`2.2130915275249095e−16`で、固定γ16=`1.7763568394002536e−15`以内だった。実PCG64/CDF samplerとideal lawの距離を資格したものではない。

## K1：平均構造RZ数四値

K0 PASS・KERNELS.npz・状態辞書を保存した後、既存MeanKernelの合成/二乗法を使って四値だけ評価した。K0.jsonの`K1_started=false`はK1開始前に保存したsnapshotの値であり、最終実行statusはK1.json/STOP.jsonである。

| Round / axis | Physical candidate | qPF / R | E[S] central | 数値区間 |
|---|---|---|---:|---|
| m0 / X(cosine) | `53e00d8b07d1547d19a7` | 1 / 2 | **1743.6682557187337** | [1743.6682557186837, 1743.6682557187817] |
| m0 / Y(sine) | 同上 | 1 / 2 | **1743.6682557187337** | [1743.6682557186837, 1743.6682557187817] |
| m6 / X(cosine) | `2a8c6718220eec59d419` | 64 / 128 | **1764659.275564426** | [1764659.275537014, 1764659.2755917928] |
| m6 / Y(sine) | 同上 | 64 / 128 | **1764659.275564426** | [1764659.275537014, 1764659.2755917928] |

Tはm0=.1、m6=6.4。四値を別々に評価し、この固定SOURCEではX/Yが同じSになった。他のcandidate/compilerでの軸価格一致は推論しない。単位は測定1回当たり、controlled PR＋scalar＋Hadamard wrapper＋measurementを含む構造RZ数。状態準備費用P=0である。

**boundary/terminalの記録**：各wireのfirst prefix/steady prefixとaxis suffix＋terminalの整数rewardはK0.jsonとKERNEL_STATES.jsonに保存した。close＋reverse-headの状態依存rewardはKERNELS.npzの各`wire*_R*_terminal`配列に保存し、tail kernelの出力状態へ一度だけ適用した。firstとsteadyのentryを区別し、検証済みの固定outer出口を明示してq反復を合成する。scalar −cTとX/Y wrapperは最後のsuffixへ一回だけ入れる。途中でcounterをfinish/resetする操作はない。別のboundary期待値を新たに再集計せず、使用したmap/rewardそのものを正本として渡す。

## 数値精度と資格の限界

[固定NUMERICAL_CONTRACT](../NUMERICAL_CONTRACT.md)を変更せず適用した。Z/native演算のparityは整数・hex完全一致。確率centralは元component/有限orderのbinary64値を使い、観測massに合わせた補正は行わない。数学的ideal weight比をFractionから方向丸めし、非負P/gのdot・加算・二乗合成誤差をIEEE γ_(2d)とunderflow項で囲んだ。元event_probabilityの因数分解丸め差もqR回分の有理数累乗で区間へ含めている。

この区間は**固定SOURCEの構造数と確率伝播に対する数値enclosure**。通常のIEEE/BLAS丸めモデルと今回検証した状態対応を前提とし、形式検証済みの量子実装保証や統計的confidence intervalではない。実samplerとの距離、compiler C−S、長q/Rへの別domain移送、全law compiled価格の不確かさ、共有covariance/total CI、全RPE成功は含めない。

kernel/source/identity/法則の適用範囲はこの1 preparation・R2/R128・q1/64・S2/savedδだけ。H6、他11round、他R/δ、別controlやPF公式へ自動適用しない。K1のE[S]を平均compiled Cや全費用Gとして使わない。C1 MODEL falseは維持する。

## テスト・実行資源・保全

新規科学SOURCE改変なし。前回固定の人工7＋metadata1＋closure回帰10の**18 method PASS記録を再利用**した。今回はlocatorの固定metadata test1件を再確認してPASS。実物理の証拠は今回のK0であり、人工testを物理PASSの代用にしていない。実行後の[CHECKS](CHECKS.json)では、保存kernelの126配列/18状態辞書・shape/finite/区間包含・SOURCE・旧STOPの不変を検査した。E[S]の再計算や独立な科学レビューはしていない。

9wireを独立taskとして9worker並列、専用process内BLAS/OMP1で実行。共有設定は変更していない。[time記録](../RUN02_RESOURCES.txt)ではwall 50.68秒、user 236.14秒、最大RSS 582,028 KiB、exit0。この実行の観測値で、全q/R・別環境の速度保証や全worker RSS合計ではない。

旧run01 UNKNOWN/STOP/log、固定SOURCE/数値契約、C1/C2/connected-cluster、S0a run01失敗/run02成功、input/raw/cohort/grant/ledger/marker/KEEP/STOP、A/B worktree/process/cacheは保全した。今回の科学runはrun02一回で、失敗retry・scope拡張なし。snapshot/元descriptor以外のHamiltonian/state再生成、trajectory sampling/full-path S/build/compile/fit/compiled C/N/RMSE/Gは0。

## 次の価格評価に不足する証拠・STOP

E[C]=E[S]+E[C−S]の後半は未取得。この二候補と同じlaw・trajectory・compiler/full wrapperに対応するpaired C/S、残差の平均・分散/系統差と、採用するなら校正/独立holdoutが必要である。C1 q8/16・R4/8の係数を今回R2/R128へ無条件に外挿しない。未知rare寄与・実sampler距離・共有covariance/total CIを0にしない。

前回のdirect pilot（2候補×normal2path×XY、最大8 wrapper）は有限development参照の別案として保持し、今回は実行しなかった。次に選ぶ場合は、同一物理law/SOURCE・新seed/cohort・必要標本数/価格精度・STOPを先に固定し、sampling/compile/paired S/C等の新しい科学実行範囲を明示許可する必要がある。n2は全law平均の精度保証ではない。

成果をcommit/pushしremote SHA/tree/主要成果物を照合した後、**mandatory STOP**。次は**ユーザー**がC−S/compiled価格検証の目的・有限scope・必要精度を判断する。**Codex C**が適合する保存C/Sを優先して再利用し、不足する限定検証のSOURCE/計画を具体化する。**Codex B**は今回四値を構造価格の診断metadataとして参照し、compiled CとGは欠測に保つ。新科学計算へ自動進行しない。
