# p7/S2・要求幅0.1：同一配分Q3の固定1件

2026-11-15 JST。開始HEAD b4296b110b129d226dbef2743c14f725a27c3bf6、既存B branchを継続。ユーザーの明示指示による最大1件の精度比較であり、価格取得・価格予測・費用計算・別配分・retryは含めない。

## §10.5照合と保存入力

- 現6候補Q3、PF6費用予算12件、旧baseline受入/readiness、関連Git履歴・公開B ref・参照可能な同ユーザーのQ3 runnerを確認。e6754564…の今回配分・total decoder・同target/law/lossによる受入可能な保存Q3又は進行中同等jobは、確認範囲で未同定。PR-Pauliの新配分計画は別物理候補であり同等結果ではない。
- 幅0.1の13round packetは5fb2c32e39e63267ea1748861269c362a93a1b86のdocs/research/truncated_fullh_signal_results_20261114/run01/B0I_P7_S2_D01/PLAN.jsonとB_SIGNAL_INPUT.json。元signal producerは5a172629db4d32a4230f8bb10a871b42807b1cd5、numerical SOURCEは5cc7c07e112ce97205948465c94441cba0042416。旧readiness 999304918のlocatorを再利用し、signalを再生成しない。
- 元configuration e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc、total b8ceb77562c4648f2ae51d7beebdc4069edb0f273d316dfcaee500d1d33015fb、binding 8e1e602dfc3969251499d516ad03f4a9fccfca907db7d1ee84c9e51a971d94dbを照合。
- 比較先はb4296b110のp7/S2・16h/7・B_PRICE原API。旧評価は再実行しない。元configuration.specは要求幅delta_pf以外が完全一致し、H/state/snapshot・profile・S2/p7・h/M・law・rmse policy・prior/decoder等を維持する。評価target objectも一致する。

## 固定した入力と範囲

N_X=N_Y=[59,55,51,47,43,39,35,31,27,23,19,13,4]、全13round。要求幅0.1の固有scheduleはq_PF=1,2,…,4096、実幅は各round0.1。元scalar・normalization・raw/target/qualificationは変更しない。

Z1＋T1 total decoder、保存DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic raw・ideal IID二項・EXACT_AFFINE_OF_BINARY64_RAW、exact τ=(0.00159360010199040 Ha)²を比較先から継承。元Minkowskiは別参照、δEを再加算しない。原ground未認証・数値allowance未認証・十分N/global RMSE/G未資格を保全する。

count-only上限はoutcomes3600、verification38526308、transitions116519440。維持するscopeはρ1、outcomes4225、verification55043173、非終端frontier8192、transitions141559036。新signalの閉包・質量・残余・上下界gateは1call内で新たに検査し、比較先のPASSを流用しない。

幅0.1の条件付き費用はnull。435,481,090 RZは比較先16h/7だけの保存値であり、今回へ付け替えない。

## 技術確認・実行経路

既存PF metadata/total bridge、高速kernel/8 DP worker、worker自身の原API・partials・資源保存、協調的segmentとOS exit観測を再利用する。新callerは固定1件の入力接続だけで、新decoder・risk kernelを実装していない。

新人工/metadata検査3件がPASS：幅固有scheduleと配分・価格不使用、別configuration拒否、区間差/余裕の符号。初回toyで表示専用exact helperが整数を受けない問題を検出し、既存exact_scalarへ統一して修正済み（TESTS.txtに経緯を保存）。これは科学実行前の技術修正で、実H4 risk/MCは0件。親monitor例外時の原API/資源保存は、直前のPF6 callerで検証済みの同じinstalled worker/supervisor経路を無変更再利用する。

caller・INPUTS・検査をcommit/pushしてSOURCEを固定後、新run01へ最大1回だけ実行する。gate不成立はUNKNOWNとして部分成果を保存。重大異常・1件終了後はmandatory STOPで、追跡範囲拡張・別列・価格・新signal・無断retryを行わない。
