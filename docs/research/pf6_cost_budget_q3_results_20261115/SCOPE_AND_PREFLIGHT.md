# PF6・費用予算3/4：実行前照合と固定範囲

2026-11-15 JST。ユーザーが7f27e18fのREPORT §3を採択し、具体列生成・保存価格の条件付き費用算術・不足最大12件のQ3を許可した範囲。別幅・新signal・compile・MC・結果依存の配分変更・retryは含めない。

## §10.5差分と再利用

- 開始HEAD/固定計画7f27e18f06dde12127c85be898777b685d29bacc、指定B branchを継続。remote同SHA、cleanを確認した。
- 関連Git履歴で現6候補の保存Q3は8dba096dcのN0各1本のみ。7f27以後にこのbranchの別配分Q3成果はなく、参照可能な同ユーザーの関連scientific runner processも未同定。確認した範囲外のjobまで不存在とはしない。
- B2 remote a7eceecbaまでの差分はPauli LD2のモデル参照価格会計等で、今回のPF別配分Q3とは同等でない。使用価格は指定の1002f1dfeaの156軸から変更しない。
- 元6 Q3の採択条件・N0・候補の対応を再利用し、基準点のMSE・費用と部分保存状態をINPUTS.baselineへ転記。元API return/資源欠測を埋めるための再実行は0件。
- 生成した12整数列は同候補N0ともA/B相互とも異なる。受入可能な同条件保存結果は確認範囲でなし。Pauli/Full-H・旧partial・旧H_eff・別幅は物理候補又は契約が違うため今回のQ3として再利用しない。

## 入力・価格・生成

A packet正本1adb7e65a9、数値fb9a1878、carrier06a15ee4、B INPUTS8dba096dcを直接再利用。既存PF metadataとtotal bridgeで全13round・元/total ID・law/normalization/scalar/targetを照合した。新H/state/reference/mean/normalizationは計算しない。

B2 1002f1dfeaのAXIS_TERMS156行を元configuration・total configuration・m/axis・physical ID・元/total candidate IDに照合。96直接軸、60条件付き閉包軸の分類を保持し、別候補への移植・価格予測・新compileは行わない。元receipt/SOURCEへ戻る固定selectorを各新N×C行に保存する。

β=3/4、A_SHAPEのw=N0、B_PRICEのw=N0/C、1≤N≤N0、exact整数/Fraction・最大の予算内breakpointという採択規則だけを使った。Full PF予算629,020,185 RZ、p7予算445,166,514 RZ。[具体12列・費用・未使用予算](ALLOCATIONS.md)、正本[INPUTS](INPUTS.json)。価格・生成規則をriskを見て変更しない。

## 科学条件・count-only・停止

保存deterministic PF raw/B=1・IID二項・EXACT_AFFINE_OF_BINARY64_RAW、Z1＋T1 total decoder、明示DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、元prior/alias/shiftを維持。τはεE=0.00159360010199040のexact二乗。δE/δE²は加算せず、旧Minkowskiは別参照。nominal/十分N・物理保証へ昇格しない。

scopeはρ1、outcomes4225、verification55043173、非終端frontier8192、transitions141559036／call。各具体列でCartesian count包含を確認し、すべて固定範囲内。実行中の新count格子に対するbinary64閉包・質量・残余loss・MSE/momentsの検査は各callで行う。旧ρ閉包PASSを今回へ流用しない。zero HOLD/tie argminを保持する。

有効上界≤τは適合、下界>τはモデル不適合、跨ぐ場合INCONCLUSIVE、必要gate失敗はUNKNOWN。重大異常では未開始callを止め、開始済みjobを保全。retry・範囲拡張・別列・別幅を自動投入しない。

## 再利用する実行経路・資源

新callerはPF metadata、T.create→bind_signal→evaluate、kernel710aa69b、accelerator9e294778をそのまま使用。旧部分DP/MC/cost/selectの正式入口は起動しない。条件付きT算術は科学N資格とは別fieldに置く。

run_pr_df_fullh_total_q3_fixed.pyのworker自身によるAPI_RETURN/partials/RESULT/RESOURCE_FINAL/PERSISTED保存、run_full_h_kappa2_scale3o2_q3_fixed.pyの協調的segment/OS exit観測を再利用。今回の実際のinstalled worker/supervisor入口を人工fixtureで検査し、親monitor例外でも開始済みworkerが保存することを確認。新6＋回帰11＝17 test method PASS（TESTS.txt）。人工結果をH4の科学結果とは扱わない。

開始前観測は64 CPU、MemAvailable約497,990 MiB。検証済み各8 DP worker・batch8を再利用し、親7並列をCPU affinity/(子8＋親1)から選んだ。CPU/RAM/workerの恒常的な制限ではなく、共有設定・他jobは変更しない。段階logと親観測で進捗を確認し、faulthandler等の非同期frame読取は追加しない。

caller・INPUTS・本preflightと人工検証をcommit/pushしてから専用run01を一回だけ作る。終了又は重大異常でmandatory STOP。旧SOURCE/結果/STOP・他担当worktree・ロードマップ本体は保全する。
