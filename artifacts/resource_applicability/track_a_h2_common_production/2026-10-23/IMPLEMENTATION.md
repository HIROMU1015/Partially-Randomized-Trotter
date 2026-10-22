# production実装の境界とprovenance

control planeは `h2_common_production_contract.py`、native/会計backendは `h2_common_production_backend.py`、instrumented診断は `h2_common_production_diagnostics.py`、CLIは `run_h2_common_production.py`。旧mock module/CLIは不変である。

## 接続

- acquisition 4ec26921…の `h46_new2_native.prepare` をprefix3/1へ使用。Gaussian callbackのrotation bytes、gate/parameter.hex/wire列を記録し、finallyで元callbackへ戻す。元providerファイルは変更しない。
- `h46_m5_native.c.core.schedule` と `h46_m5_native.compose` を使用。provider c31ef424…にはh46_m5_plan自体がなく、別providerの同名moduleへ解決させない。
- composeにはFullのrequests=[]、scalar=ham.constant、PRにはK4/outer32 request、scalar=ham.constant+tail_identityを渡す。各pieceのscalar=0と最終scalar-onceは固定composeの仕様を保持する。
- PR requestは固定make_requestsとfactory/cutoff関数を接続。任意分岐だけが実行する。Full経路はrequest factoryを呼ばない。条件付きイベントを費用cohortへ混ぜない。
- wrapper、compiler cost API、fingerprint、simulatorはprovider c31ef424…に固定。compile中だけprocess-localにtranspileをphase guardへ接続し、finallyで戻す。native fingerprintとcompiler settings hashを照合する。

## 診断とallowance

primaryは固定coreのspectral algebraとfinite Taylor polynomialを使用し、両pathのcorrected/raw列の全実行stageを観測する。independentは固定H2 SOURCEのascending matrix Taylor（degree1..5）をinstrumentする。保存SOURCEの無観測版とbitwise一致を主張しない。Hamiltonian、signed時間、PF係数、有限RTE法の意味は保持する新比較系列である。

workは「各vectorに対するdeterministic propagator実適用」と「tailの実matrix-vector適用」で数える。batched2列なら2として加算する。primary tailの基底入出力も実matvecとして数える。独立pathはTaylor termごとのtail matvecを数える。scheduleのstage proxyから代入しない。traceにはprimaryの基底入出力/diagonal、独立pathのterm/累積sum、raw除算、piece、最終scalarを含む。rawの既知attenuationはlogで保持し、normをcorrected単位で記録する。ここでの全stageは実装が実行するstate段階であり、厳密forward-error boundではない。

path signal/vector差、log_B差、corrected-signal closureはstrict <1e-9の診断gateを通す。共通schemaのreference、unit、coverage、work、trace/native/input結合を検証して旧providerとGit blob一致のempirical_allowanceへ渡す。係数64、安全係数10、floor1e-12、経験的規則。raw-vector closure・endpoint最大norm・未取得reference errorを代用しない。

会計は固定 `u_aware_shots`、ε=.0005、α=.025、ordinary signed bias、log_B+marginを使う。u/headroom≤.01をguardにする提案。保存paired RZのsample平均と新shotの積からGを作る。eligibilityまたは整数shotが未確定ならGはNoneとし、順位Cを判定しない。PASS時にも出力はconditional empirical comparisonで、causal attribution/科学GOは発行しない。

## 実行権限と封印

外部USER_EXPLICIT approvalはcontract digestと3政策、grantはapproval digest・contract digest・run_root・完全予算に結合する。CLIに承認/採択を作る機能はない。未使用rootをexclusiveに作り、最初の科学import前にCONSUMEDをfsyncし、以後同rootの再利用を拒否する。各entry前にexclusive ledger fileへ予約を記録し、失敗・中断でも枠を戻さない。

import sealは通常importのoriginと直接SourceFileLoaderの両方を検査する。封印source bytesをcompileして使用し、未封印pyc cacheを実行しない。全てprocess-localに設定しfinallyで戻す。既にloadedなmoduleもclosure検査対象。SOURCE root内の使用fileは固定Git commitと一致することを要求する。runner/acquisition/providerの役割を分け、一般runtimeのsealと混同しない。

preflightはSOURCE/input/runtimeのbyte・schema・identity検査であり、数値kernelをimportしない。科学kernelを呼ぶentryは別のexecuteのみ。approvalやgrantのschemaは外部の明示承認を記録する方式であり、ファイル内容だけからユーザー意思を暗号学的に認証する仕組みではない。今回実grantは作成していない。

出力は新run/results、ledger、CONSUMED、completion/STOPに分離する。partial出力はSTOP_PARTIAL_NOT_PASS。終了時には全budget一致と使用closure再検査を要求する。再試行、job自動拡大、他worktreeのHEAD/status監視、PSIだけを理由にしたworker killを行う処理はない。共有環境の設定・他processを変更しない。

## 未確認の範囲

本番数値pathのH2作用等価性、Gaussian全操作の数値再構成、compiled-vector gate、instrumented診断の数値妥当性、新policyでの費用順位は未実行。72テストはmock/AST/control-planeの確認である。runtime closure、PR C witness、科学政策/実行承認のBLOCKERを解消せず実行可能とは報告しない。
