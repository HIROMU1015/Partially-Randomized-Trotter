# H4 Pauli central PF cosine：Arb包含評価

利用者は2026-10-25にPF cosine 1件の実H4 Arb包含評価を明示承認した。
基点 `5c69e1339955c77273b356df30e94ec3c57458b1`、専用branch
`codex/non-df-h4-central-arb-pf-cosine-20261025`。

本SOURCE固定時点では、実H4状態作用0、科学grant発行/消費0。
production接続・人工36テスト・保存QPYの構造検証を完了し、次にSOURCE/runtime/入力/契約を固定する。
結果・終了監査・公開状態は計算後に追記する。

## 固定科学範囲

linear H4、隣接1.00 Å、STO-3G、Pauli-JW非identity184項/prefix27、S2、T=.8/q8/r4/K4。
system8＋ancilla wire8、保存D2 PF cosine compiled QPY、既存raw ray・PF C-target区間を使用する。
新compile/sampling/Hamiltonian/state生成なし。C SOURCE e80d7510880f8db993b777c627bb8e95b05a070d、
D2 SOURCE 74515d8eb4f073c40fc783681a3775dbca89e9b1。

## 再利用とproduction差分

`h4_arb_bridge.py`のnative作用・exact center＋global L2半径・最終Z/ν包含を共通のprivate kernelへ分離した。
演算式は準備SOURCEと同じ。人工入口の小規模fixture guardは保持し、実H4では新adapter/controllerが固定入力・新grant・非返還call予約を確認する。
global/relative phase、標準rz/sx/x/cx、CXの向き、little endian、末尾measure、binary64値、rawνを変更しない。
入力state/出力stateを再正規化せず、出力norm除算やclippingを使わない。
PF C-targetは元C resultとR0保存区間をexact照合し、区間全体を使う。midpointをcertificateへ格上げしない。

既存Cの`real/complex_ball/phase/dyadic/endpoints/array_hash`、`CallBudget`、durable `save`、native witnessを再利用。
native witnessは待機・subreaper・fsyncの本体を変えずscope labelだけ今回のworkerへ変更する。
run01内の新しいgrant/CONSUMED/claim/ledger/native記録/STOPだけを作成する。
workerが照合するsealは実際に使うcode/runtime/入力に限定し、他worktreeのHEAD/statusや一時PSIを停止条件にしない。

## 人工検証

最終構成は新production11＋既存Arb10＋QPY6＋sine9＝36 PASS（修正前35件の記録も保全）。
早期停止（幅目標未達でも判定後は終了）、straddle/nonfiniteの有限昇精度、128bit後INCONCLUSIVE、
contract改変拒否、direct interval差とbudget3分岐、共通kernel一致、非返還call、grant再消費/STOP拒否、
人工processのnative exit/reapとscopeを確認した。
新たな実H4のStatevector作用やPR/C-target生成は行っていない。

## 精度・終了条件

64→96→128bit、各精度のcompiled状態作用最大1回、初回最大3状態作用、各精度の局所残差scalar縮約最大1回。
保存ρ予算は `10107730069649565297/3815672118158673332666368`（約2.6490038338323596e-6）。
`ρ_local = B_saved abs(m_compiled − m_C)`、B_saved=1。
上端≤予算または下端>予算ならその時点で終了する。straddle/nonfiniteの場合だけ次精度へ進む。
幅1e-8は参考目標であり、判定後の高精度化の理由にしない。128bit後はINCONCLUSIVEでSTOP。
失敗callも保持し、技術回復が必要なら新SOURCE/未使用grantで成功済み結果を再利用する。
正式certificate受理/LOCAL_BUDGET_MET更新、PR/R2/N_Gへ自動移行しない。

## 運用と保全

最新2026-10-23 policyの§8.4/§10/§10.1/§10.2を適用する。提供ファイルに§10.3はないため仮定しない。
CPU/RAM/worker数の任意上限なし。1科学対象の順序依存walkは逐次とし、phase cacheとexact permutationで重複を削減する。
共有設定・他ユーザーprocessの操作、AI subagent利用は0。旧証拠412 SHAと旧manifest186行を保全する。
局所数値包含と正式certificate受理、今回native終了と元H4のnative UNKNOWNは別に扱う。

[承認・入力・人工・SOURCE/grant・結果・監査](../artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/README.md)。
計算後は結果を保存しcommit/push/remote確認してmandatory STOPする。

## grant発行前の技術回復

最初のproduction SOURCE bbb29c868e7464e729253fe6881d341c55be96b7では、旧C runtimeの
システムlibz.so.1.3 SHAと現在値の相違を検出して、grant発行前にprepareを停止した。
実H4状態作用0・grant発行/消費0、run01未作成。変更主体や時点はUNKNOWNであり、共有ファイルは変更しない。
pre_grant_runtime_diagnosis.jsonとpre_grant_STOP.jsonに旧SOURCEと失敗を保存する。
修正ではFLINT数値packageの旧版一致を維持し、現在のsystem runtime差を明示記録して新SOURCE/新sealへ固定する。
runtimeに関する回帰を追加し11件PASS。既存の旧SOURCE/runtime/STOPは書き換えず、新しい未使用grantだけを発行する。
科学的条件・許可call数・精度scheduleは変更しない。
