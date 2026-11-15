# PR-Pauli LD3・LD2共通VPでのtotal/Q3限定評価

LD3のtotal viewを作成し、新規Q3 **1件**で採択済みモデル閾値に適合した。LD2は保存VP結果を引用し、再評価していない。数値gateのUNKNOWNではない。原API・partials・資源記録をworker自身で保存し、親processも終了code 0。mandatory STOP。

## 同じVPでの結果

対象はH4／1.00 Å／STO-3G／tol-only DF10、sector(2,2)、保存H/state/snapshot。PR-Pauli LD2とLD3はそれぞれ固有の辞書・共通κ schedule・理想finite-K rawを使用した。全13 roundのX/Y各軸は共通のUSER_SUPPLIED VP：

`[48,45,42,39,36,33,29,26,23,19,16,13,9]`

| 構成 | MSE下界 Ha² | MSE上界 Ha² | 判定・取得 |
|---|---:|---:|---|
| LD2 | 6.207498269405979019554497794E−7 | 6.207498269406694956690683824E−7 | モデル適合・保存結果引用 |
| LD3 | 6.206210455307360129958626682E−7 | 6.206210455307820120226411551E−7 | モデル適合・新1call |

表示は28桁。判定は表示値ではなくexact有理数を使用。閾値は十進εE=0.00159360010199040 Haのexact二乗、τ=0.00000253956128506381328204169216 Ha²。

LD2引用：[50be68b80 RESULT_LD2.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50be68b803a90fd893a1499ebce74e033225144d/docs/research/pauli_q3_three_quarter_20261115/run01/RESULT_LD2.json)。LD3原結果：[c79612386 API_RETURN.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7961238679f7dc7dd358baa924ec1e3579a7121/docs/research/pauli_ld3_vp_q3_results_20261116/run01/PAULI_LD3_VP/API_RETURN.json)。exact上下界と診断は[SUMMARY.json](SUMMARY.json)。

両者とも、このVPと採択済みrawモデル・proxy・lossの範囲での適合である。この表だけから費用の優劣、十分N、最適配分、物理的global RMSE保証を認定しない。価格計算は0件。

## SOURCE・total view

科学実行SOURCEは **7733089b85d013bad314e108d539b3afe0271635**。callerと[INPUTS.json](INPUTS.json)・全viewを実行前にcommit/pushし、remote SHA一致を確認した。[SCOPE_AND_PREFLIGHT.md](SCOPE_AND_PREFLIGHT.md)に§10.5の調査範囲、重複なしの根拠、有限範囲を記録した。

- 元LD3 ID：`dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847`。
- total ID：`6757998a50bec243e1cd14d1b968457c673d383c47cb95c6b52e8b0ed7e1a0eb`。
- binding ID：`122900d26d9412a569496e8ef98ff425614eac9885e43164bab400991980a858`。
- 元signal：cdbce383fのMEAN_HANDOFF、原carrier 26bac4f32、数値SOURCE 8c8ce542e。configは421258d40、辞書は50640b74c。

A指定の`b_intake_dictionary_identity`を8c8ce542eからpure関数としてそのまま読み、既存P1の`job`を使用した。辞書hash・bit順・残余locator・各round出所・scalar一回適用を既存`bind_packet`で照合した。旧LD1/LD2用の辞書helperだけでは拒否されることも確認した。

`rpe_pauli_signal_intake.accept_saved(..., handoff_source=LD3_HANDOFF)`を固定SOURCEのみに拡張し、total bindingは評価時にも元assemblyを再照合する。Cの固定planner本体は変更せず、依存helperの接続でLD3辞書SOURCEだけをAの指定helperへ渡す。raw・normalization・元target・law・qualificationは保全した。新canonical IDはdecoder-only viewであり、価格receiptの付替えは行っていない。

## 数値gate・残余・zero/tie

Z1＋T1 total decoder、直接モデルMSE、保存full DF10 solver energy −2.1663874486347634 Haへの全事象二乗lossを使用した。保存ideal finite-K rawを条件とするshots／X／Y／round間IIDとEXACT_AFFINE_OF_BINARY64_RAWである。Pauliの既存f54b41a2e経路の契約・options・target・acceleratorをLD2 VPから変更せず継承した。kernelは710aa69bの固定bytes。

- ρ=1の有限binary64閉包：PASS、検査912,226組。
- DP遷移53,006,168組、非終端の最大保持8,192状態。終端15,296状態は非終端capの対象外。
- 残余mass：約2.39283814443372024E−21。残余loss上界：約4.59990267784869119E−20 Ha²。捨てずに上界へ計上。
- any-zero確率：約2.01221640155426880E−8。zero保持・tie argminのtotal規則で全事象を扱う。追跡内tieがあるroundはm2/m4/m7/m10。各roundのzero/tie質量は原DPとSUMMARYに保存し、これをそのroundのMSE寄与とは呼ばない。
- bias²診断：約9.44625569902577544E−9 Ha²。Var診断：約[6.11174789831710238E−7, 6.11174789831756237E−7] Ha²。exact momentsとMSE区間の整合を検査。
- 質量保存・残余loss・上下界・有効上界のmin・scope・momentsの保存後検査：PASS。

transition allowanceは1E−12 radの宣言されたbinary64工学的allowanceであり、厳密な数値certificateではない。保存targetのground認証false、proxy採択前の元flag、実sampler law移送未認証は保持した。AのδEは加算していない。`sufficient_N/global_RMSE_bound/G/G_qualified/winner`はnull。

## 実行・技術検査

既存generic Q3 worker、worker-owned保存、8 worker高速DPを再利用した。workerのwall約159.685秒、worker CPU約153.886秒、子CPU約104.336秒。worker peak RSS 625,280 KiB、完了した単一子processのpeak最大535,888 KiB。親monitorの観測process tree peakは3,985,832 KiBであり、RSSの加算観測・samplingの範囲を持つ。詳細はRESOURCE_FINALとSUPERVISION。

観測hookの別module参照により、run01では詳細SEGMENTSと最上位workerの個別OS終了code wrapperが適用されていなかった。104 DP子processの終了codeは全0、最上位workerは成功futureと保存済みreturn、caller processは終了code 0を保全した。個別OS終了code未記録を補完・推定しない。round開始・伝播終了、主要gate、原API・CPU/RSS保存は取得できており、数値kernelへの影響はない。

この観測接続だけを完了後に修正した。実際の入口に対する人工検査が修正前にFAILし、修正後は人工supervisorから詳細ログと終了codeを保存できた。実H4の再実行は行っていない。科学実行の再現には7733089b8を使い、公開HEADの観測修正と区別する。[TESTS.txt](TESTS.txt)：最終25技術・回帰case PASS。

新signal・path sampling・MC・compile・価格・追加配分・Q3 retryはいずれも0件。旧結果・旧STOPを変更せず、限定1件完了で停止した。
