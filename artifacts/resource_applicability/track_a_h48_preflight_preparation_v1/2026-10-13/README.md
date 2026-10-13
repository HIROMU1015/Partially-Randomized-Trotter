# H4 native anchor補完・H8入力事前検証：起動準備入口

現在のSOURCE `d5900669c9e58215b19c7addfc5760ff63a2ed00`。監視writer/elapsed callbackの接続を起動前に訂正し、[19限定検査](local_tests_trace_fixed.json)に合格。計算call消費0、旧source・入力・compiler・予算条件不変。[新H4認可](h4-signal-auth-02/manifest.json)、[新H8入力認可](h8-input-auth-02/manifest.json)、[SOURCE closure](source_closure_trace_fixed.json)。旧auth01は未起動の履歴として保持する。以下のb6d4/18件は前段記録。

2026-10-13 JST。利用者の「準備できたら開始してよい」と対象「1と3」により、H4比較補完とH8小規模事前検証を認可。H8本格比較は認可しない。

SOURCE `b6d4bdd7f5e388dba6a386c94a0c8cfee8b105c1`、[closure](source_closure.json)、[18限定scalar/mock](local_tests.json)。旧257 runtime・科学library・既存H4 state/Hamiltonianは不変。既存private venvを使い、install/upgrade/共有設定変更0。SOURCEはGitHub取得byteを照合済み。

## 対象と順序

1. H4_SIGNAL：linearH4/1Å/STO-3G/既存DF rank12/8modes/電子2+2/sector36/T.8、fullPF S2/S4×q1/2/4/8とprefix3 PR(q,r)=(1,4),(2,4),(4,4),(8,2)各K2/K4の16 signal。ε=.05/.01/.005/.001、H6と同じ経験的u/shot規則。入力再生成なし。
2. H4_COST：signal完了と適格を受理後、その固定panelのpaired measured wrapperを新seed/namespaceへ生成。最大264 group、primary528compile＋ordinary2anchor4compile＝532、PR256trajectory/960occurrence/72control probes。各workerのcompile/trajectory/occurrence上限をtask別2/4・0/1・qへ狭め、全task和が532/256/960を超えないことを検査した。原稿段階の528はprimary枠で、追加感度4を明示して訂正する。
3. H8_INPUT：linearH8/1Å/STO-3G/16modes/電子4+4/sector4900、RHF/conv_tol1e-9/max_cycle50、tol-onlyDF1e-8・cutoff0・生成順・strict Hermitian gate、HF初期eigsh(k1,SA,tol1e-12,ncv40,maxiter1000)の入力/DF/state feasibility。新integral1/DF1/solver matvec<=10000、trajectory/compile0。raw DFをHermitian gate前に保存し、失敗時にもSCF/rawを保全する。H6 weighted projectionと同一政策として認定せず、必要になってもfallbackしない。matrix-free信号/新16-mode compilerのfeasibilityは次の技術段階に残る。本格H8平均費用比較の契約は未確定。

## 源・grant・資源

[H4 signal plan/auth](h4-signal-auth-01/manifest.json)、[H8 input plan/auth](h8-input-auth-01/manifest.json)。phaseごとの別grant/新output/one-shot lock、SOURCE/profile/input/plan/carry binding。実H4 cost grantは完了signal receiptを読むまで生成しない。旧grantを再利用せず、自動retry/本格H8の自動進行なし。

H4既存M1 actual compile12128/random6208は履歴として保持し、他historical actualはnull。稼働production H4は別lineageの予算を保持し、今回変更しない。H6 known1716/842＋予約20/10も保持。新H4532/256は別scopeの追加、失敗call返却・budget reset0。H8入力はcompile/trajectoryを加えない。

H4 cost最大8worker、CPU20/21/24/25/26/27/28/29、co22/23、内部thread1。signalとH8 inputは1worker CPU20。workerAS32/RSS9GiB、coAS8/hard32/RSS2、aggregateRSS76/jobRAM96、余白2GiB。各phaseoutput2GiB/個別writer32MiB、log2MiB/journal32MiB、wall capなし。全値はphase grantへ固定した。own-run affinity/AS/core0/thread1だけを適用する。

fresh host248.25/effective96GiB、runtime host168.25/effective16GiB、host full PSI>=5%、cgroup full PSI>0、OOM増分、AS/RSS/5秒超・欠測でown-run STOP。filesystem最低10GiB/inode100000とquota KNOWNを再照合する。独立coordinator監視を既存H6 supervisorから再利用し、退出時の所有確認・ESRCH cleanup・wait/reap/FD閉鎖を維持する。資源値はkernel/cgroup予約保証ではない。

## 高速化と確認範囲

H4 costはFingerprintReuseで同じcanonical bytesを再利用し、独立groupを最大8workerへ分配する。構成・seedは完了順に依存させない。旧H6数値/native/compile関数のglobalsをcopyしたadapterでloader/modelのみ結合し、元moduleや他processをpatchしない。16構成のH4内部・common-control anchor証拠であり、旧rank12とH6 tol-only19を同じDF契約のsize scalingとして扱わない。

18 testsはtask別budget和、source/input/grant/CPU/geometry変更拒否、path escape、既存module globalsの不変、4900次元の人工solver callback前budget等を検査。実H8、実新H4 panel、compiler出力・母平均bound・科学的winner・CIのPASSではない。実行時はadopted profileのnative runtime subsetとBogoliubov sourceをfresh SHA照合し、旧34,273-file環境campaignを再実行しない。

起動結果は[launch receipt](launch_receipt.json)へ追加する。原稿/共通比較の[前段資料](../../track_a_h6_manuscript_h8_plan_v1/2026-10-13/README.md)とは別の実行準備である。
