# p7/S2・幅h：粗幅達成費用予算での固定2配分

2026-11-16 JST。ユーザーの今回指示に基づき、具体列・保存価格算術・不足分最大2 Q3だけを実施する。開始HEADはbec032fc26093177f529754060df06d5375944b4、既存B branchを継続する。

## §10.5とSOURCE

- 元configuration e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc、total view b8ceb77562c4648f2ae51d7beebdc4069edb0f273d316dfcaee500d1d33015fb、binding 8e1e602dfc3969251499d516ad03f4a9fccfca907db7d1ee84c9e51a971d94db、13round packetと元physical IDをbec032fc2のINPUTSから再利用。5fb2c32e…の幅h固有schedule・raw・normalization・scalar・targetを変更しない。
- 価格はecf45508a63ddf75a4806b47179f5f5102fa90e0の `docs/research/d_pf_p7_s2_h_comparison_prices_20261115/run01/COMPARISON_PRICES.json`。全26軸についてconfiguration/physical ID、元canonical候補ID、m/axis、q/T/実幅binary64を信号側へ照合。直接10軸・外挿16軸、元receipt/sidecar、外挿誤差上界nullを保全する。C guardやcompilerを新規起動しない。
- 既存bec032fc2幅hの列は末尾13,4、今回A/Bとは一致しない。b4296b110の6候補12配分・8dba096dcの元6配分は別要求幅/physical scheduleで、今回のQ3結果へ代用しない。readiness 999304918の幅hはmetadata接続で、保存risk評価ではない。
- 関連all-ref Q3/価格・配分のGit履歴、同幅の結果ディレクトリ履歴、参照可能な同ユーザーのQ3 processを限定照合。今回2列の同configuration・decoder・target・law・lossに合う保存結果/進行中jobは確認範囲で未同定。738df226のLD2再配分や4ef702742の候補別rational配分は別物理候補である。今回2規則も異なる列なので、新規予定2件・保存Q3再利用0件（既存結果は比較参照のみ）。信号・価格は全て保存再利用。

## 実行前に固定した算術・列

[INPUTS](INPUTS.json)と[ALLOCATIONS](ALLOCATIONS.md)が具体列・軸別単価/積・SOURCE・実費用・残額の正本。

予算435,481,090 RZは粗幅16h/7・B_PRICEの保存達成額。上限は元N0=[59,55,51,47,43,39,35,31,27,23,19,15,11]で、直近適合列の末尾13,4を上限に使わない。既存 `run_pf6_cost_budget_q3_fixed.generate` の整数breakpointをbyte変更せず再利用。N=max(1,min(N0,floor(λw)))、Aはw=N0、Bはw=N0/C_h。各chainで予算内最大のbreakpointを選び、次breakpoint価格が予算超過することも保存する。

| 規則 | X/Y各軸、m0〜12 | モデル費用 RZ | 予算残額 RZ |
|---|---|---:|---:|
| A_SHAPE | [21,19,18,17,15,14,12,11,9,8,6,5,3] | 421,323,290 | 14,157,800 |
| B_PRICE | [59,55,51,47,43,39,35,31,27,16,7,2,1] | 434,203,864 | 1,277,226 |

## Q3契約・有限範囲

bec032fc2のZ1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic PF rawのideal IID二項、EXACT_AFFINE_OF_BINARY64_RAW、exact τ=(0.00159360010199040 Ha)²を維持。元Minkowskiは別参照、AδEを足さない。

ρ=1、outcomes上限4,225、verification上限55,043,173、非終端frontier8,192、transition上限141,559,036を維持する。

| 具体列 | round outcomes上限 | verification組上限 | transition組上限 |
|---|---:|---:|---:|
| A_SHAPE | 484 | 686,678 | 14,620,196 |
| B_PRICE | 3,600 | 38,006,708 | 109,711,888 |

count-onlyはscope内。各新count格子のbinary64閉包・質量・principal guard・残余loss・MSE上下界/momentsを1call内で新たに検査し、旧列のPASSは流用しない。有効上界≤τでモデル適合、下界>τでモデル不適合、閾値をまたぐとINCONCLUSIVE、gate不成立はUNKNOWN。

## 再利用経路・技術検査・STOP

新callerは具体入力の接続のみ。既存PF metadata/total binding、Q3 kernel、`F.installed_workers`→worker保存→`M.supervise`、高速DP/8 child・batch8、協調stage/segment・子終了観測を再利用する。現環境のCPU affinityから親2件を並列実行できることを確認。CPU/RAMのOS制限は追加しない。既存の人工monitor故障時の保存検証・実結果保存を持つ同じ経路であり、非同期frame dumpや新monitorは追加しない。

今回の3人工/metadata検査は、具体列/caps/費用/直接外挿区分・risk未呼出し、異なるSOURCE/physical/origin拒否、同列2規則の1call化を確認。初回にcanonicalのexact有理数T/δと価格表のbinary64表現を直接比較する不整合を検出し、既存exact_scalarで値・binary64 hexを照合する最小修正を行った。科学入力を変更せず、修正後3件PASS。実H4 riskはこのテストでは起動しない。

caller・INPUTS・検査をcommit/pushして固定後、新run01へ各1回だけ実行する。重大異常では未開始の投入を止め、開始済みworkerの原API/partials/資源を保全する。無断retry・列変更・scope拡張・別幅・新signal・fit・build/compileなし。十分N・global RMSE・G_qualified・方式winnerはnullを維持し、結果公開・remote照合後mandatory STOP。
