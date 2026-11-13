# H4 PR全13round：平均構造RZ評価経路の技術準備（2026-11-13）

**固定B V2全13roundのmetadataを接続し、m0/m6をcopy専用にしたうえで、残11roundのK0→PASS時K1を一括処理する経路を整備した。新8＋既存18＝26技術test PASS。** 今回は準備だけで、残11roundの実物理K0・新E[S]は0。既存m0/m6の科学再実行も0である。

正本：[全13round metadata coverage](prepared/COVERAGE.json)／[有限科学計画・未実行](SCIENCE_PLAN.json)／[数値・再検証条件](NUMERICAL_ADDENDUM.md)／[HANDOFF](HANDOFF.md)／[新8test](TESTS.txt)／[kernel回帰8](REGRESSION_KERNEL.txt)／[closure回帰10](REGRESSION_CLOSURE.txt)。SOURCE **`de2427d4a31d6b4b39d2c8eb8b63e3c972ffb89e`**。作者自身の技術検査で、独立科学レビューや新lawの科学PASSではない。

## 固定SOURCEと全roundのidentity

対象はLinear H4/1 Å/STO-3G/保存tol-only DF10、元H/state/snapshot、S2/L_D1/ΔPF=.1/finite-K2/h=.1/M12/P0、元fragment/basis/qubit順・scalar一回、symmetric-directional control・測定付きfull Hadamard wrapper。preparation hashは`b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、元324非identity component・9 basisを保持する。数値stateの条件付きground仮定を厳密保証へ昇格しない。

| 役割 | 固定SOURCE・扱い |
|---|---|
| 既存科学結果 | `83d219cc68c25540507a5539ebd2da94c29d06b3:docs/research/h4_structural_kernel_20261113/run02/`。K0/K1・kernel/状態辞書/STOPを不変保存 |
| 元科学実行SOURCE | `19b95d8b3c88063d2dbae4e5bb47455b52b00903`。旧runの再実行なし |
| B最新版確認 | `git ls-remote`でB branchは`3216d27a69ac8532d2a22e78f5f94491d202d238`と確認。作業中worktreeは参照しない |
| B canonical configuration | `030f3f548a35d98a88e62e92a35f132e0243a293:docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json` |
| B JSON SHA256 | `22a1bd1af8704006568c545eb8bcbce5bd476202ad59aaeaf79f50850815f943`。B公開intakeのsource_filesと照合 |
| canonical ID | `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911` |
| 元演算/確率 | 元C1 Accounting、closure module、元provider RTE API、既存MeanKernel/Band/状態閉包・parity処理を再利用 |

共通κは元S0のexact文字列をそのままcopyする。表示近似・Rの丸めから再生成しない。B固定schedule/全identity/lawを読み取り、B controllerやN側を再実装していない。

```text
13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201
```

| m | T | qPF | R_outer | physical candidate | 今回の科学状態 |
|---|---:|---:|---:|---|---|
| 0 | .1 | 1 | 2 | `53e00d8b07d1547d19a7` | 保存K0/K1をcopy、再実行なし |
| 1 | .2 | 2 | 4 | `c00e00dc1e9c414fb7d9` | 未評価 |
| 2 | .4 | 4 | 8 | `f2fcd66b1ef9ae2da5df` | 未評価 |
| 3 | .8 | 8 | 16 | `045843b2429bece6f29b` | 未評価 |
| 4 | 1.6 | 16 | 32 | `3df0cfd3356c358aeda8` | 未評価 |
| 5 | 3.2 | 32 | 64 | `57d641a831d76360b570` | 未評価 |
| 6 | 6.4 | 64 | 128 | `2a8c6718220eec59d419` | 保存K0/K1をcopy、再実行なし |
| 7 | 12.8 | 128 | 255 | `b42287268e5249dfb4d1` | 未評価 |
| 8 | 25.6 | 256 | 510 | `224aef0a626068b43184` | 未評価 |
| 9 | 51.2 | 512 | 1020 | `06a78c7f9856f373336f` | 未評価 |
| 10 | 102.4 | 1024 | 2040 | `b872ad1347a35c8f4315` | 未評価 |
| 11 | 204.8 | 2048 | 4080 | `d48210cc27c599e268c2` | 未評価 |
| 12 | 409.6 | 4096 | 8160 | `b55c521fdce6a448dbcb` | 未評価 |

coverage JSONには全B candidate ID、physical recipe、identity、law_binding、scalar/allocation/profile等の元metadataを保持した。m0/m6の4recordは保存K1と同じ内容をcopyするだけで、new kernel入力から除外する。他22軸のmean S・数値区間、11lawの確率/角度はnull。保存R2/R128のPASSを他Rへ付けていない。

## 何を再利用し、何を再検証するか

同じpreparationならcomponent support・basis/native operations・head/switch/close/resetのSOURCE構造は再利用できる。旧m0/m6のkernelは**そのlawの固定成果**で、別RのP/gへ無条件に移さない。

各Rで元make_rte_configから新しいevent確率・rotation angleを作り、元Accountingのlocal reward/正確なpending Z、active basis/phase、閉包、entry/close/outer、mass/longdouble/intervalを再検査する必要がある。q/Tは反復とscalar/XY terminalに入るので、実Tを明示してsuffixを構成する。phaseは元π逐次和/control barrierを新最大Rまで確認し、途中tailの再初期化やmoduloを導入しない。

詳細な共通/依存条件と数値契約の適用は[NUMERICAL_ADDENDUM](NUMERICAL_ADDENDUM.md)に整理した。既存の小状態論証はreset等の構成に利用するが、新lawのlocal parity・有限state graph・確率精度がPASSするまでE[S]を評価しない。新しいRでstate数が旧56/48/22と同じと仮定しない。

## 実装差分と計算量

[h4_structural_schedule.py](../../../src/trottertracks/h4_structural_schedule.py)は固定B reader、保存四値copy、11lawの遅延入力構成、既存K0/K1を使うround順batchだけを追加した。準備reader/default CLIは物理Context・law API・kernel・meanを呼ばない。

旧[h4_structural_kernel.py](../../../src/trottertracks/h4_structural_kernel.py)の追加は、実Tの入力hook、K1の確率誤差式評価hook、完成wireだけも保存できるkey列挙への変更に限定した。元Accounting/MeanKernel/closure/S0aやB controller/providerを変更せず、C−S model・fitを実装していない。

旧2点用suffixの「R2以外はT6.4」という条件を新roundへ流用しない。新adapterは各B rowのTを必ず供給し、旧callerのfallbackは保持する。人工testで旧defaultと新T/scalar/wrapperの対応を検査した。

また、元K1のγ16に対する(1±γ16)^(qR)という**同じ数式**を、外向き固定精度intervalの二乗法で評価できるようにした。巨大Fractionの分子/分母bit長がqRに比例する問題を回避する。central・γ・法則・Z順序・mass toleranceは不変で、式の外包に必要な追加丸め分だけ保守的になる。旧callerは従来のFraction経路を保持し、m0/m6成果を再計算しない。

kernelはfinite component/tableとmatrix合成・二乗法を使い、全event列/324³ tupleを生成しない。dが検証済みenclosure内ならwire/law当たり概ねO(dL×native長＋d³logR)、outer/誤差因子はO(logq＋log(qR))。phase検査だけはO(maxR)時間だがcacheをstepごとに解放し、qRのevent保持にはしない。matrix memoryはO(d²)、11lawの固定local metadataと保存kernel分は別に必要である。

準備したm12経路もこの構造を使う。実m12のstate数・CPU/RSS/時間・数値区間幅・科学的な成立性は未検証で、速度向上の実測値を主張しない。

## 人工・回帰テスト

**新8＋kernel回帰8＋closure回帰10＝26 method PASS。** 主な確認は、13identity/lawとexactκ/R、保存m0/m6 record不変、確率/角度のR依存を扱うmock、元nativeを使う人工新Rの閉包/terminal、K0失敗時のK1/次round非起動、K0とkernel保存後だけK1を呼ぶ順序、compiled quantity拒否、partial kernel保全である。

大きなq/Rは、人工1state kernelと指数だけを使って処理経路を検査した。γ式の新外包は小指数の厳密Fractionと大指数の100桁Decimal参照を含み、二乗合成のoperation数がlog指数以内であることを確認した。実H4の新law/新Sをtestに紛れ込ませていない。

新testの初回はmock fixtureのstructure field不足で1件errorとなり、人工入力を補って修正した。科学処理は未開始。最終PASS logを保存し、元kernelの物理結果は再実行していない。

## 残11roundの一括科学計画（今回は未実行）

[SCIENCE_PLAN](SCIENCE_PLAN.json)に各roundの元candidate/law参照、reuse、新しいK0/K1、精度/不確かさ・STOP・fresh output・固定SOURCEを記載した。提案scopeは**同preparation復元1回、11law、11roundのK0、PASSしたroundごとのXY二値＝最大22新E[S]、既存四値copy**。m0/m6再実行、sampling/full-path S/build/compile/C−S fit/compiled C/N/RMSE/G/H6は0である。

共通SOURCE/law/native/phase検査を済ませ、m1,2,3,4,5,7,8,9,10,11,12の順に処理する。一つの9wire poolを再利用し、各roundでK0 PASSとkernel/状態辞書を保存してからK1へ進む。共通条件又はK0がUNKNOWN/FAILならその結果・完成wire・それまでの完成roundを保全し、当該K1と次roundを開始しない。近似/閾値変更/別law/自動retry/direct compileへ移らない。K1でprecision/quantityが破綻した場合もSTOPする。

この一括scopeをユーザーが明示許可すれば、候補ごとの承認やrunner作成を繰り返す必要はない。今回はdefaultのmetadata出力だけを実行し、scientific --executeは使用していない。計画内のexecution_now_authorized=falseは今回未実行であることの記述であり、承認証明書/新grantを要求する仕組みではない。

## B/Dへの引渡し・保全・STOP

E[S]のquantityは`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`。全roundで得られてもE[C]ではなく、Bのcompiled価格recordへ自動登録しない。normalization Bを価格へ乗算しない。実sampler対ideal law距離、compiler残差、rare/長q systematic、共有covariance/total CI、全RPE成功は未資格を保持する。

Codex DがE[C−S]を担当する。Cは同candidate/configuration/law・full-wrapper/axis/P0に対応する構造metadataを渡すだけで、残差の新解析・fit・compiled価格を重複しない。有限cohort C−Sとideal-law E[S]を無条件に足して全law compiled価格とすることも行わない。

既存C branchを継続。旧run01失敗/STOP、run02 K0/K1/kernel、C1/C2/connected-cluster/S0a、SOURCE/input/raw/cohort/grant/ledger/marker/KEEP/STOPを保全した。A/B/Dのworktree/process/cacheへ介入していない。今回の新科学処理は0。

成果をcommit/pushしremote SHA/tree/主要成果物を照合後**mandatory STOP**。次は**ユーザー**が残11round一括scopeを承認するか判断する。**Codex C**は明示許可後に同じ固定SOURCEでK0→PASS時K1だけを実施する。**Codex B**はmetadata/欠測を参照し、**Codex D**は別に許可された残差scopeを担当する。今回新科学計算へ自動進行しない。
