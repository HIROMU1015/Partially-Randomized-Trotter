# Track A H8 Phase B1：anchor照合と費用model固定の準備

**B1全体は `STOP_MISSING_REQUIRED_ANCHOR_PROVENANCE`。旧B0のruntime全体とcost用preparation/basisの同一性が未確立のため、数値fitを起動しない。B0+Iの保存16軸は現在のprofile・保存preparationへ結合できたが、B0の代替として使用しない。**

branch：`track-a-h8-phase-b1-readiness-20261016`。起点はPhase A結果commit `01309059636f2432e29b663bcd2eb0e400f5ae0a`。
[機械可読台帳・契約・証拠](../../artifacts/resource_applicability/track_a_h8_phase_b1_readiness/2026-10-16/README.md)。

## 範囲と固定条件

linear H8、隣接間隔1 Å、STO-3G、16 modes、Nα=Nβ=4、sector4900。保存tol-only DF rank28、tol=1e-8、cutoff=0、signed generation order、weighted Hermitian projection。T=.8、q=1/2/4/8、δ=.8/.4/.2/.1。Phase Aの追加152候補（S2 96、S4 56）の保存結果を再利用する。full-H referenceは保存された未回転のもの、allowanceは経験的factor1/10、厳密bias保証は未確立のまま保持する。

今回実行したのはJSON/CSV/ソースの読取、既存RZ値の転記・一致照合、SHA照合、保存適格性のboolによる選択、人工テストだけである。分子配列を開かず、preparationを復元せず、design rank・係数・残差・予測費用・総費用を計算していない。新compile/signal/PR sampling/SCF/DF/state生成、実行可能grantの発行は0。技術チェックはprimary agent自身によるlocal検証であり、独立レビュー・CI・外部再現ではない。

## anchorの照合結果

|層|保存実測軸数|来歴結合済み軸数|判定|
|---|---:|---:|---|
|B0 / cosine|8|0|runtime全体・cost用basis未結合|
|B0 / sine|8|0|同上|
|B0+I / cosine|8|8|保存profile/preparationとのmetadata結合済み。数値fit未実施|
|B0+I / sine|8|8|同上|

各層はp14/21 × q1/2/4/8のS2。p7は旧screenで不適格・未compileであり、ゼロ費用やanchorにしない。full p28やPRは別構成であり、欠けたB0 anchorへ混ぜない。

旧B0の結果は `681ec23c93fdd1090d325908f4b155a79115afdf` のwrapper/group台帳と、8 cost taskの原cost・wrapper・terminal・native reap receiptを照合した。q1の4軸の実行SOURCEは `cb8a56c6b2a772827275f9200b02638030f33f72`、q2/4/8の12軸は `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。全軸で保存RZ、fingerprint、compiler hash、task、正常終了・reapが一致した。

旧実行のfresh receiptにはprofile `4e32bb9b…`、native files318件、subset digest `830f07d2…` の証拠がある。証拠が全くないわけではない。一方で `full_34273_inventory_repeated=false`、`compiler_output_bit_equivalence_verified=false` が明示されている。現在の完全profile `622ce4a8…` への遡及結合は行わない。原cost portは各worker内で `_prepare_discard` を実行しているが、そのcost taskのpreparation hash/basis操作記述を保存していない。signal側のpreparationや同じSOURCE/inputだけでは、cost用basisの実行時同一性を証明できない。

B0+Iは結果commit `5f79257f432bc7e0317f122090f62fec379e2a1b`、実行SOURCE `54ed98c5fa5b7190f77255271f4e3bec8db15630`。公開軸表・scalar/preparation・correctness receiptと、16原cost receiptのRZ/fingerprint/compile前後fingerprint/exit0を照合した。実行契約は完全profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`（34,273 files）を固定している。p14/p21の観測preparation hashは、既存理論成果commit `7e41670565d9284da9aded94063be591fdcaa3dd` のH8 discard rootと一致した。保存exportのblock記述もfull p28の各prefixと一致し、Phase Aの保存lineage receiptへ結合できる。これは歴史的receiptの照合であり、今回fresh runtime再inventoryを行ったという主張ではない。

compiler hashは全32軸で `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/couplingなし、symmetric-directional control、measured Hadamard full wrapper、state preparation費用除外を維持する。B0はoriginal constant一回、B0+Iはoriginal＋full-Fock tail identityを一回。異なるscalar policyを同じ層へpoolしない。

今回の証拠inventoryは固定Git/disk source・公開成果物・限定原receiptの82件。調査範囲に必要な証拠がないことを記録したものであり、全storageに証拠が存在しないとの断定ではない。広域回収や新compileで欠測を自動補完しない。

## 費用modelと実行契約の準備

S2は層ごとに `a+b*q+d*prefix*q`、S4は条件付き仮説 `a+3*b*q+3*d*prefix*q` を保持する。係数はnull。H6の係数やfit結果は使用しない。事前契約の「各層に少なくとも3つの非共線anchor」は変更しない。今回のguardは来歴と件数の必要条件だけを検査し、非共線性・rank自体は評価しない。

native providerの固定ソースでは、S4は負時間を含む3段S2を非mergeで反復し、scalarを最後に一回、wrapperを外側に一回付ける。これはS4仮説の構造的前提だけを支える。transpilerの最適化後にも費用が加法的になることは未検証で、B2が必要である。

将来のB1数値fit案は、来歴が成立した全4層で保存整数RZの有理数最小二乗を個別に評価し、rank不足・非正予測・保存anchorとの各軸相対誤差2%以上ならSTOPする。別の形への自動変更、H6係数の移送、層のpool、B0+Iのみへの自動scope縮小はしない。この追加診断も未承認契約案であり、今回実行していない。

`execution_contract_DRAFT_v1.json` と `grant_DRAFT_INVALID_v1.json` はapproved=false、SOURCE=null、CPU[]、command=nullで、起動機能・grant issuerを持たない。実fit runnerとone-shot runtime sealは未完成である。来歴gateがSTOPしているため、実fit runnerの実装・sealを終えて計算可能と見せることはしない。まず不足証拠または独立したscopeについて利用者が判断できる状態にした。

B1資源案はprocess1/thread1、AS512 MiB、RSS256 MiB、累積CPU240秒、wall600秒、output64 MiB。数値fitの別承認に加え、新SOURCE/input/profile seal・新grant・専用空きcore・30秒以内のfresh memory/disk/quota/PSI/OOM/保全gateが必要である。承認だけで今回の来歴不足を解消した扱いにはしない。

## B2候補の事前選択と後続境界

保存済みS4 56候補の適格性を再計算せず読み取り、Phase A以前に固定した規則を適用した。

|順序|候補|軸|予測RZ/model SHA|
|---|---|---|---|
|1–2|B0 p15/q1 S4|cosine → sine|未計算/null|
|3–4|B0+I p21/q8 S4|cosine → sine|未計算/null|

B0は適格S4の(prefix,q)辞書順最小、B0+Iは最大。これはmetadata選択であり、費用順位による選択ではない。B2は最大4軸、compile並列数1、各軸 `abs(predicted-measured)/measured >= 1/50` で次軸を起動せず即STOP。平均による救済、refit、追加compile、retryはしない。B1が成立し、具体的予測値/model SHAをsealした後でも、B2には別途明示承認が必要である。Cの5精度再会計も独立承認であり、今回は行わない。

## 人工検証・保全・次の判断

[guard](../../src/trotterlib/h8_phase_b1_contract.py)、[checker](../../scripts/check_h8_phase_b1_readiness.py)、[人工tests](../../tests/test_h8_phase_b1_readiness.py) を追加した。8 unittest methodsがPASS。欠けた来歴、旧profile、重複/欠測、別control、未測定値、部分familyへのfallback、B2選択、2%ちょうどの不合格、measured分母、無効grant/Phase境界/H6係数移送を検査した。人工値だけを使用し、実データfit runnerは実装していない。

旧673保護ファイル（grant/ledger/carry/one-shot）、112旧worktreeのHEAD/status、旧V2未追跡scriptを前後照合する。SOURCE・raw・消費済みgrant・ledger・V1/V2/H5/H7を変更せず、新worktreeの公開可能成果物だけをcommit/pushする。

**次の判断は、旧B0の2つの来歴不足を追加証拠で解消できるか、または別scopeを採用するかである。条件を緩めてmodelを作らない。今回の準備を公開しremote SHA/treeを確認後、mandatory STOPする。B1数値fit、B2 compile、C再会計はいずれも未承認・未実行。**
