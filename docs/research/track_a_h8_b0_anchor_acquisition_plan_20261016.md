# Track A H8：新条件B0 anchor取得の最小計画

**提案はS2の4構成・8実測軸。各軸で3構成をfit用、残る1構成を独立検証用に事前固定する。今回は計画・契約・metadata guard・人工検証だけで、新規compile、実データfit、残差、再会計は0。旧B1のSTOPは解除していない。**

branch：`track-a-h8-b0-anchor-readiness-20261016`。起点：`b09f408873cad94a371261386f7f3bf9cc0baa91`。
[公開計画・無効grant・証拠](../../artifacts/resource_applicability/track_a_h8_b0_anchor_readiness/2026-10-16/README.md)。
[直前B1の来歴判定](track_a_h8_phase_b1_anchor_readiness_20261016.md)を維持し、新しい取得scopeを分離した。

## 最小構成と選択理由

|用途|prefix|q|PF|実測する軸|wave|
|---|---:|---:|---|---|---:|
|TRAIN|14|1|S2|cosine / sine|1|
|TRAIN|21|1|S2|cosine / sine|1|
|TRAIN|14|8|S2|cosine / sine|2|
|HOLDOUT|21|8|S2|cosine / sine|2|

S2候補式 `a+b*q+d*prefix*q` の3係数には、各軸で少なくとも3つの非共線条件が必要である。3構成・6軸だけではfitが飽和し、独立した予測精度の検証点がない。検証条件を1構成加えると4構成・8軸となり、今回要求した「各軸3点fit＋独立1点」の最小件数を満たす。旧16軸の全再compileは要求しない。

TRAINの設計行列は列 `(1,q,p*q)`、行 `(1,1,14),(1,1,21),(1,8,112)`。整数座標だけから求めた行列式は−49で非ゼロ。これは設計の技術確認であり、保存RZを使ったrank/fit/残差の計算ではない。prefix14/21、q1/8の四隅を使い、大きい側p21/q8を未使用の検証点として残す。q2/4や新prefixを含む全格子での精度保証は、この最小構成からは得られない。

8軸はすべて独立した実測receiptを必要とする。cosine/sineのRZが過去に同値でも片軸を転用しない。TRAIN/HOLDOUTは新測定値を見る前に固定し、同じ取得Phaseで両者を保存するが、fit用と検証用の出力を分ける。HOLDOUTを後からTRAINへ入れたり、結果に応じて別候補へ置き換えたりしない。

## SOURCE・profile・preparation・比較条件

対象はlinear H8、隣接1 Å、STO-3G、16 modes、Nα=Nβ=4、sector4900。保存tol-only DF rank28、tol1e-8、cutoff0、signed generation order、weighted Hermitian projection。T=.8、今回のqは1/8、δ=.8/.1。5精度・経験的allowance factor1/10、full-H reference未回転、厳密bias保証未確立という既存条件は変えない。

native providerは `c31ef424175c02379eafebe8fc6c7548c1d9aabe` の265ファイル。今回Git原本と現在diskのbytes/SHAを照合した。compiler hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/couplingなし。目標runtimeは完全profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`、34,273 files。今回読んだのは保存inventoryであり、runtime全ファイルのfresh照合ではない。実行前・実行終了時の全byte照合が必要で、差異があればSTOPする。

保存snapshot・snapshot/DF/projection receipt・preparation export・preparation identity receipt・full-H referenceの7入力をhash照合した。配列を開かず、実H8 Gate/operatorを生成していない。旧B0 costのpreparationを推定して使うのではなく、保存exportのdiscard:p14/p21操作列を新しい取得の入力として固定する。両rootのblock記述は保存full p28の対応prefixと一致する。

|prefix|保存rootのpreparation hash|
|---|---|
|14|`993f3b7cd9390daa64e2fc9b6584bc1797a783e27001e427abed097eddf803a6`|
|21|`b3baf8d6fe14d3e0e637e7b0b0152a4f2b2c0ecddc4702e98e2dbfd02651ab47`|

これらはB0+I既存実測の観測preparation hashにも結合済みである。新B0では保存操作列を復元し、新SCF/DF/固有分解/Gaussian basis生成を行わない。復元したlive操作列を再記述し、qubit順序・basis hash・ordered block記述・original constantが固定metadataと一致することをcompile前に確認する。保存root hash、ordered block SHA、新view recipe SHAを区別し、保存hashを単にruntimeの「観測hash」として書き写すことを証拠にしない。

B0 scalarはoriginal constantを一回だけ付け、tail identityを加えない。symmetric-directional control、measured full Hadamard wrapper、state preparation費用除外を維持する。保存B0 signal/independent-forward/full-H referenceは4構成分をそのまま転記し、新signal・bias・shotを再計算しない。新nativeの左右branch・X/Y control probeを保存独立signalと照合し、非finiteまたは誤差1e-8以上ならSTOPする。control probeは今回取得に必要なcorrectness検証で、PR samplingや新signal探索とは別の限定callとして登録する。

各cost taskに、新実行SOURCE、完全profile SHA、入力hash、root/view/basis記述、control/scalar/native policy、axis、compile前後wrapper fingerprint、compiler hash、実測RZ、exit code、wait4 reapを直接結合する。今回欠けていたcost用来歴を、signal側receiptから後付け推定する構成にはしない。

## 並列化・call上限・資源案

最大4 compile worker、各1 thread。2構成のcontrol検証をすべて通した後、その2構成×2軸を同じwaveで実行する。親に持つnative構成は最大2つとし、終了・回収後に次waveへ進む。snapshotと保存full preparationの読込/復元を共有し、q間では2つのprefix viewを再利用する。B0+I用の旧grant/whitelistを変更して起動することはない。

|操作|新規取得Phaseの上限|
|---|---:|
|snapshot load / 保存preparation decode|各1|
|prefix view|2|
|native evolution build|4|
|measured wrapper / probe wrapper copy|各8|
|control probe|16|
|compile|8|
|fit / cost residual / 再会計 / S4 / B0I compile|すべて0|
|新signal / full-H reference / SCF / DF / state / PR / law / tail scalar|すべて0|

|資源|上限案|
|---|---|
|process・core|driver1＋worker最大4＋observer1、空きphysical core6個、各thread1|
|AS / RSS|driver各64/32 GiB、各worker64/32 GiB、observer512/256 MiB|
|aggregate RSS / admission reserve|160.25 GiB / 16 GiB。実効空きRAMが合計176.25 GiB未満なら起動しない|
|累積CPU / phase wall|24 CPU時間 / 12時間。予測所要時間ではなくSTOP上限|
|observer|CPU1800秒、1秒間隔、観測gap5秒超でSTOP|
|stage / output / logs|8 GiB / 256 MiB / 128 MiB|
|fresh空きdisk / inode|16 GiB / 40,000以上|
|instruction|native4,000,000、transpiled10,000,000|

AS合計は実RAMの測定値ではない。上限案は旧H8の成功済みper-role条件を参考にした保守的な予約条件で、8軸の実測peakやspeedupの保証ではない。未知のhost/cgroup quota・容量、専用core不足、profile不一致はgateで拒否し、他jobの停止・affinity変更・venv修正で補わない。CPU数の自動縮小/拡大や資源上限の実行中変更を行わない。

## 承認境界・STOP

1. 今回：計画・固定候補・契約・人工テスト・無効grant案を準備してSTOP。科学実行0。
2. 新B0取得：別途明示承認に加え、新adapter/launcherの実装・人工接続検証・SOURCE/input/provider/runtime seal、新scope grant、空の専用outputと未使用one-shot lock、30秒以内のfresh資源gateが必要。今回のmetadata checkerは科学launcherではなく、実行adapter/launcherは未seal。
3. 取得中：basis/scalar/control/fingerprint/compile/exit/reap/資源/receipt/予算の不一致で新規投入を停止し、進行中のown PID/start/UIDが一致するworkerだけを終了・回収する。同じwaveの最大4軸はすでに進行している可能性がある。partial outputと消費済み予約を残し、refund/reset/retry/resume/追加compileをしない。モデル予測は未計算なので、取得中にモデル2%判定を行うという契約にはしない。
4. 取得完了：8軸すべて正常でもmandatory STOP。B1へ自動移行しない。
5. B1：独立承認後、新B0のTRAIN3点を各軸でfitし、HOLDOUT p21/q8はfitに入れない。各軸 `abs(predicted-measured)/measured < 1/50` をPASSとし、2%以上なら即STOP。軸平均で救済せず、再fit・再分割・追加点を行わない。B0+Iの16軸は既存来歴結合済み結果を再利用し、旧B0の16軸やH6係数を混ぜない。B0/B0I全4層の成立が必要。
6. B2：B1 PASSと係数/予測seal後、別承認。既定のB0 p15/q1 S4・B0I p21/q8 S4の最大4軸、compile並列1、各軸2%以上即STOPを維持する。C再会計も独立承認である。

旧B1の「旧保存anchorだけではSTOP」は改変しない。将来B1が新測定を受け入れるには、この新取得台帳とTRAIN/HOLDOUTを明示的に固定した新契約が必要である。旧guardを緩めて新結果を通すことはしない。独立検証1条件は最小の反証機会を与えるが、全prefix/q/S4へのモデル保証や厳密bias/PR母平均保証を与えない。

## 技術検証・公開・保全

[stdlib計画/receipt guard](../../src/trotterlib/h8_b0_anchor_plan.py)、[metadata checker](../../scripts/check_h8_b0_anchor_plan.py)、[人工tests](../../tests/test_h8_b0_anchor_plan.py) を追加した。10 unittest methodsがPASS。設計の非共線性、8軸ちょうど、HOLDOUTの固定、無断fit/後続Phase、来歴欠測、wrapper変化、control境界、exit/reap、旧scope、予算逸脱を人工fixtureで検査した。実native復元・transpile・モデルfitを検証したものではなく、primary agent自身によるlocal技術検証である。

公開するのは固定計画、無効grant案、保存control値の必要最小転記、SOURCE/input hash、人工検証、保全receipt、報告/索引だけ。private path・配列・runtime・有効grant・ledgerは公開しない。旧673保護ファイル、113旧worktree、旧V2未追跡scriptを前後照合する。V1・V2 NO_GO・H5/H7および既存SOURCE/raw/grant/ledgerを保全し、公開後remote SHA/tree確認でmandatory STOPする。
