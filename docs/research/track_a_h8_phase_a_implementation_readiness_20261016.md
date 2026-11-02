# Track A H8 Phase A：signal adapter・保存操作列復元・並列制御の技術準備

**追加152候補のPhase A実装と限定人工検証を完了した。実H8のsignal・bias・shot・適格性は未評価で、新規科学計算・数値再解析・compile・有効grant発行は0。次は公開SOURCEへ契約を固定した後、利用者によるPhase Aの明示的な実行承認とfresh gateが必要である。**

branchは`track-a-h8-strengthened-deterministic-readiness-20261016`、起点は`36756714fee37a7a6e1188bb4d765292d877ea18`。
[前回の候補・全Phase計画](track_a_h8_strengthened_deterministic_readiness_20261016.md)を変更せず、今回そのPhase Aを実装した。
[固定plan・SOURCE/input結合・人工検証・保全監査](../../artifacts/resource_applicability/track_a_h8_phase_a_implementation/2026-10-16/README.md)。

## 今回できたこと

|実装|役割と制限|
|---|---|
|`h8_strengthened_contract.py`|旧32と追加152候補の分離、既存PF iteratorによる時間台帳、Phase Aだけの予算・資源。実行権限を持たない|
|`h8_strengthened_export.py`|保存full p28と旧discard p7/14/21の全block記述を照合。permitの後だけ保存Gateを復元し、新prefix viewを作る|
|`h8_strengthened_signal.py`|H8 spin/CSR/forward経路と旧allowanceを接続。保存full-H参照・p21 scalarを再利用し、B0からB0Iを補正する|
|`h8_strengthened_runtime.py`|SOURCE・input・runtime・grant・30秒fresh gate、不可逆なtask予算予約、出力制限|
|`run_h8_strengthened_phase_a.py`|defaultはplan表示だけ。明示的な承認済み引数・sealed契約が揃った場合だけ科学adapterをimportする|
|`launch_h8_strengthened_phase_a.py`|V1監視実装をH8用に分離。role別RSS/AS、aggregate CPU/RSS、PSI/OOM、storage、owned cleanupとwait4を監視|
|`prepare_h8_strengthened_phase_a.py`|既存receiptを読んで新しい未承認private契約を作る。分子配列・実Gateは復元しない|

旧V1 pool・scalar bridge・provider・V2 runnerは変更していない。新H8 supervisorはV1実装を参考にした別ファイルで、旧実行SOURCEを書き換えていない。subagentは使用していない。

## 科学的条件と処理量

linear H8、1.00 Å、STO-3G、保存tol-only DF28、tol1e-8/cutoff0、signed generation order、weighted Hermitian projection。16 modes、Nα=Nβ=4、sector4900、T=.8、q1/2/4/8・δ=.8〜.1。追加S2はp8〜20から既存p14を除く96候補、追加S4はp15〜21の56候補。いずれもB0・B0Iの両方を含む。旧32候補の再signal/compileは登録しない。新PR候補・sampling/law・新T/q16・新SCF/DF/stateは含めない。

新B0 signal76と対応するB0I overlay76、13 prefix task。full snapshotは1回、保存full preparationの復元は最大1回、prefixごとのtruncated referenceは最大13 pairs。q・S2/S4で参照とprefix viewを共有する。p21のscalarは保存receiptを再利用し、他の12 prefixだけでtail scalarを得る。full-H参照の新exponentialは0。

primitive検証は22 block task、全550 block/time pairs×3 probe＝1,650 probe actions。S4負時間、merged ordinary時間、unmerged directional時間と逆符号を含む。最大28時刻/blockで既存spin cache上限32以内。過去のprimitive数値は旧profile/basisまで同一と証明できた再利用receiptがないため、自動で今回のPASSへ移送しない。この限定primitive検証も将来のPhase A承認scopeに含み、今回は実H8では実行していない。

各primitive taskのreference matvec予約は最大100万、各prefix taskは最大200万、計4,800万でPhase上限5,000万以内。1 exponential actionの上限は10万、deterministic actionsは1 cell当たり10万。予約はfork前にfsyncし、完了時に実績を照合する。未使用分を返却して追加taskへ使わず、失敗・unknownでも予約を保持する。次waveは前waveのnative exit0・全回収・結果受理後だけ開始する。最大16 worker、各thread1、nested parallelismなし。

## 保存証拠をどう結合したか

native providerは`c31ef424175c02379eafebe8fc6c7548c1d9aabe`の265ファイル、現profileは`622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。providerをGit/diskで照合し、既存runtime34,273ファイルを読取専用でSHA照合した。実行前・終了時にも同じclosureを検査する。Qiskit1.3.0・旧compiler hashは保持するが、Phase Aはtranspile・費用測定を呼ばない。旧hostのbinary同一性を新たに認定しない。

入力は保存snapshot・snapshot/DF/projection receipt、H8 export、full-H参照、p21 scalar、preparation identity receipt、font metadata、元referenceのlineage manifestとworker terminalの11ファイル。旧inputのbytes/SHAを照合し、private絶対pathは公開しない。

- 既存理論結果commit`7e416705…`のH8 exportを使う。metadata照合でfull p28の先頭操作列・diagonal・scalarが旧discard p7/14/21と一致した。runtime復元時は各Gateの実際のmatrix hash・角度・qubit順序を再記述して比較し、違えばSTOP。新prefixには新しいview hashを付け、旧preparation hashを流用しない。新たなDF/Gaussian preparationは作らない。
- H8の既存spin-sector検証では小さいone-spin eigensystemを使う。これは承認後の独立primitive作用の処理であり、保存Gaussian操作列や分子preparationの再生成と区別する。H6用dense4900参照や全Fock unitaryは作らない。
- full-H参照は旧resource-map完了lineageにhash結合された`input_reference.json`を再利用する。元worker SOURCEは`cb8a56c6b2a772827275f9200b02638030f33f72`、terminal COMPLETE、後続保存manifestがそのreceiptを引き継いでいる。H/state identity、3 production/CSR probe差とsplit-time signal差を確認して旧reference allowanceを保持する。
- B0Iのscalarはfull-Fock tail identity、元constantに一回だけ加える定義を保持する。B0 signalとexact-truncated signalだけを回転し、full-H referenceは不変。p21は既存scalarの正確な保存有理数・誤差receiptを再利用する。

binary64、safety10、floor1e-12、roundoff64、u/headroom≤.01、factor1/10、各軸alpha=.025、epsilon/sqrt(2)のshot割当を維持する。signal/軸分解と数値allowanceは従来どおり経験的であり、厳密biasや母平均保証ではない。欠けた来歴・分解を0として補わない。

## 人工検証と残る実証の限界

primary agent自身の28 unittestがPASSした。保存block順序・basis hash・nonfinite・不正Gateの拒否、permit前callback/import禁止、改変grant・別scope・retry・stale gate・未seal拒否、S4負時間と固定schedule、不可逆予算、p21再利用、role別メモリ、PSI/OOM、default CLIを確認した。Linuxの小さい人工childでpoolの成功/失敗とobserverの成功/強制STOPを実行し、native exit・reap・cleanupを確認した。他jobへsignalを送っていない。

固定科学venvの2 qubit人工fixtureでは8項目を確認した。保存角度からのGate復元、独立な小行列Hamiltonianのexpmとnative作用の正負時間一致、merged S4と独立な3段S2合成の一致、scalar once・full-H reference不回転・signed分解closure、改竄拒否を含む。分子入力、H8 state、実H8 Gateの復元、transpileは使っていない。

これらはlocalな限定技術検査で、CI・別担当の独立レビュー・実H8の数値PASSではない。実H8での復元matrix同一性、primitive一致、signal/allowance/eligibility、実測RAM・wall・speedupは承認後に初めて確認する。不一致ならその証拠を保持してSTOPし、自動retryしない。

## 実行契約・gate・STOP

資源は前回案のまま：driver AS64/RSS32 GiB、worker最大16・各AS64/RSS16 GiB、observer AS/RSS512 MiB、aggregate RSS288.5 GiB＋admission reserve32 GiB、aggregate CPU192h、worker/driver CPU48h、wall48h。output2 GiB、stage8 GiB、log各128 MiB、起動時空きdisk16 GiB・inode40,000、最大20,000 entries。これらはSTOP上限で、時間予測・RAM予約・速度測定ではない。

未承認private configは入力・SOURCE closure・profileに結合し、公開commit確定後にそのSHAをSOURCEへsealする。grantは承認false・CPU[]・approval textなし、fresh gateも未発行で、有効grantではない。configの技術的sealは実行承認と別である。

明示的承認後に、新grantのSOURCE/config/plan SHA、固定16 worker＋driver＋observerの18物理core、5秒idle、quota/cgroup、RAM/PSI/OOM、旧667保護ファイル・111他worktree、input/runtime/profile、exclusive output/lockをfresh確認する。quotaの実効性を確認できなければ起動しない。full runtime照合が30秒を超えても古いgateを使わず、driverの照合完了後にobserverが保護状態と資源・割当CPUを再観測してgateを更新する。

監視は1秒、gap5秒超でSTOP。role/aggregate RSS・CPU・AS、thread/affinity、出力・storage、host full PSI≥1%・cgroup full PSI正、OOM増加、input/SOURCE/profile改変、primitive/reference予算、非finite/sector leakage/経路差、native exit不明/非0、回収失敗はSTOP。cleanupはPID/start/UIDが一致するown childだけを対象とする。旧消費をrefund/resetせず、一度きり・retry/resumeなし。

Phase Aはsignal・bias・適格性だけで完了STOP。費用model固定B1、最大4軸compile B2、費用再会計Cは別承認のまま。V2のNO_GO終了、V1、H5/H7、旧SOURCE・結果・grant・ledgerを保持する。

**公開可能なcode・人工fixture・軽量receipt・報告だけをcommit/pushし、remote SHA/tree確認後mandatory STOP。今回の継続指示によって科学計算が承認されたとは扱わない。**
