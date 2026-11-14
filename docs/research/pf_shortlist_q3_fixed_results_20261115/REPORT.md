# PF shortlist6候補の固定Q3評価：保存kernel結果とdriver欠測

2026-11-15 JST。H4/1.00 Å/STO-3G/tol-only DF10/sector(2,2)、保存state、h0.1/M12のFull PF・Truncated B0I p7各3公式を評価した。

**科学worker6件は各1回完了し、保存DP端点は全件採択閾値以下。一方、親driverの監視不具合で最終API returnとCPU/peak RAM記録を保存できなかった。実行全体は部分保存・運用UNKNOWNとして保全し、再実行していない。** [DRIVER_ERROR](run01/DRIVER_ERROR.json)／[COMPLETION](run01/COMPLETION.json)。

## 件数・条件付き判定

予定最大6／実行6／科学worker完了6。保存componentからのモデル判定はPASS6、不適合0、INCONCLUSIVE0、数値risk UNKNOWN0。これとは別に、**driver監視・最終return保存はUNKNOWN**。完全なend-to-end実行PASSとは表示しない。

全6構成、両軸`[59,55,51,47,43,39,35,31,27,23,19,15,11]`。USER_SUPPLIED paper-shaped controlで、十分N・最適Nではない。各allocation IDはlabel＋`_DET_PAPER_SHAPED_CONTROL_01`。[INPUTS](INPUTS.json)を実行前に固定した。

Z1＋T1 total decoder、zero HOLD/tie argmin、保存DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、保存binary64 rawに条件付くIID二項測定。τは十進εE=`0.00159360010199040`のexact二乗（表示約2.539561285063813282041692e−6 Ha²）。旧Minkowskiは別参照、δE/δE²の二重加算なし。

| 構成 | MSE下界 Ha²（表示） | MSE上界 Ha²（表示） |
|---|---:|---:|
| Full PF S2・16h/7 | 8.255048976770948750408424e−7 | 8.255048976770948750408802e−7 |
| Full PF Yoshida4・16h/3 | 7.969265386751240788457598e−7 | 7.969265386751240788458870e−7 |
| Full PF Morales M5・16h | 4.625148099114778355837148e−7 | 4.625148099114778355838201e−7 |
| B0I p7 S2・16h/7 | 8.356033913794202152062946e−7 | 8.356033913794202152063318e−7 |
| B0I p7 Yoshida4・16h/3 | 7.882628817223086740102446e−7 | 7.882628817223086740103736e−7 |
| B0I p7 Morales M5・16h | 4.616158368748513116040767e−7 | 4.616158368748513116041819e−7 |

全行`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。表示値は25桁の近似であり、判定と端点保存はexact有理数。[SUMMARY](SUMMARY.json)から候補別の**元`PARTIAL_TOTAL_DP.json`**へ戻る。小さい端点幅は数値certificateを意味せず、raw/transition allowanceは未認証。

## driver不具合と、再計算しない結果保全

固定科学SOURCE **313ed1950d3e02cbb0c392bcd147bc40aafcf176**のdriverで、`pending`（Future→候補名）のkeyをresource directory名として扱い、最初の監視で`PosixPath / Future`のTypeErrorが発生した。ProcessPoolの終了処理は開始済み6 workerの完了を待ったが、親のRESULT保存loopへ戻らなかった。科学worker自体の数値gate失敗ではない。

全候補のSTAGESに`SCIENTIFIC_CALL_START`が1回、`EVALUATION_COMPLETE`と閾値適合statusが保存済み。科学callは各worker開始から約0.72〜0.91秒で全件開始しており、2秒timeout後の最初の監視例外より前だった。例外後に新しい科学callを投入していない。coarse・lemma・total DP・進捗も全件保存済みで、worker終了を確認した。原`RESULT.json`は0件、完全なAPI returnは復元できない。原科学ファイルは変更せず、**RECOVERED_RESULT.json**を別名で保存した。これを元returnと偽装しない。

復元処理は保存済みDP端点・momentsの転記、全保存上界のminがDPであることのexact比較、既存`q3_summary` helperによる許可された閾値判定、SOURCE/identity metadata照合だけ。元stageの判定とも一致した。`T.evaluate/Q.evaluate/partial evaluate/binomial/count table/DP/MC/cost/select`を呼出し禁止にして実施し、**新科学call0、retry0**。Cへ渡すものも復元索引であり、未保存の原API記録や十分N packetではない。

現callerには`pending.values()`を参照する修正と、その不具合を再現する人工回帰を追加した。修正後SOURCEで科学再実行はしていない。実際に用いた313ed1950と過去の科学結果はGit上で保全した。

## bias²・Varと同公式pair

| 構成 | bias²区間上端 Ha²（表示） | Var区間上端 Ha²（表示） |
|---|---:|---:|
| Full S2 | 4.271132493062610e−7 | 3.983916483708339e−7 |
| p7 S2 | 4.381347002018467e−7 | 3.974686911775735e−7 |
| Full Yoshida4 | 3.304859771916961e−7 | 4.664405614834280e−7 |
| p7 Yoshida4 | 3.215421865385554e−7 | 4.667206951837533e−7 |
| Full M5 | 1.153765347940451e−9 | 4.613610445635374e−7 |
| p7 M5 | 6.831691253800202e−10 | 4.609326677494713e−7 |

区間両端・mean/second moment/biasは元DPとSUMMARYに保存。bias²＋Varの区間とMSE区間の整合もexactで確認した。同じshot policyの下で、S2 pairはp7のMSEがFullより大きく、Yoshida4/M5 pairはp7のMSEがFullより小さかった。これは当該保存raw・scalar・保持Hamiltonian・scheduleを含む条件付き比較であり、公式だけの効果、切断の普遍的優位、資源winnerではない。

この6vectorはいずれもモデル閾値を満たすため、「この配分ではモデル上界が大きすぎて判定できない」という不足は生じなかった。より少ないshotで成立するか、cost-aware配分が有利かは未評価。bias/Varの診断は次の有限配分案の材料になるが、riskの厳密単調性・未評価vectorへの補間・最小Nは主張しない。今回の結果を見て新vectorを生成・探索していない。

## 数値gate・zero/tie・残余

全候補でρ1の有限binary64閉包が`VERIFIED_FINITE_BINARY64_SUPPORT`、検査2,558,439組。DP遷移87,623,455組／構成、非終端frontier8192以下で固定範囲内。終端状態は約17,300〜17,500で、非終端frontier上限とは別。count-only preflight3600／38582272／117985808は[事前照合](SCOPE_AND_PREFLIGHT.md)のとおり。

保存first-unsafe13分割＋all-good=1、各roundのgood＋unsafe=1をexactで照合。既存kernel/backendの各round質量保存gateを通過し、保持loss＋残余loss上界がDP上界に一致。全候補でDPがcoarse/lemmaより小さい有効上界であり、上界同士を加算していない。各first-unsafeの確率とlemma capは元coarse/lemmaに全件保存。unsafeは誤branchと同一ではない。

今回のany-zero確率は全候補exact0。追跡した各roundのzero HOLD/tie argmin massも全て0だが、残余履歴のtie率まで0と認定していない。旧partial規則の未解決診断は各候補`[0,残余mass]`。total decoderは全事象に出力を定義するため、未定義eventのlossを捨てていない。

残余massは約9.99e−31〜4.10e−30、残余loss上界は約3.72e−29〜1.29e−28 Ha²。principal guard失敗として残余へ会計したmassは約6.34e−37〜6.44e−37。正確な端点・per-round残余はSUMMARYとraw。数値allowanceは既存per-transition1e−12 radという未認証engineering allowanceのまま。

## SOURCE・資源・テスト

- B基盤50be68b803、kernel710aa69b、accelerator9e294778、C c628e9e8を無変更で再利用。caller/INPUTS/範囲は科学実行前commit313ed1950。
- A完成packet carrier06a15ee45、統合数値fb9a1878、REPORT1adb7e65。元値のproducer SOURCE、configuration/physical/total IDの対応はINPUTS・[C_RESULT_JOIN](C_RESULT_JOIN.json)・A PACKET_INDEXを参照。signal再計算、再packet化なし。Pauli control/3/4・旧partial DPの再実行なし。
- 6構成並列、各8 worker・batch8。各workerの保存済み数値完了stageは開始から約145.46〜148.97秒。**正確なCPU・peak RAM、全driver wall、最終returnは欠測/UNKNOWN**。別jobの性能を転用しない。実行前の空き資源と実行中に読取観測した空きメモリは十分だったが、これをjobのpeak測定値として扱わない。
- 実行前の人工5＋baseline7＝12 test PASS。監視不具合はその範囲で検出できなかった。修正後はFuture-key取り違え回帰を追加し、新caller6＋baseline7＝**13個のtest method PASS**。[TESTS](TESTS.txt)。主agentの検証であり、別担当の独立レビューではない。

## 資格・引渡し・ロードマップ

条件付きモデル閾値適合と、運用の部分保存・科学的資格を分離する。ground/target厳密包含/state誤差/raw認証/実sampler移送/transition certificateは未資格。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。正式N資格producer、価格受入、G/順位を実行していない。

[HANDOFF](HANDOFF.md)とC_RESULT_JOINは、元configuration・保存packet・physical ID・新total ID・round/axis・allocationを区別する。価格receiptを付け替えず、Cの既存bridge/guardを通す。現時点の復元recordを無条件の十分Nとして受け入れない。

統合担当への影響：6 shortlistの「total Q3未評価」は、**指定controlで6件の保存componentモデル上界が閾値適合**へ更新できる。同時にdriverの最終保存・資源記録欠測を記す必要がある。必要N最小化・適格価格/G・物理保証は未完了。数値再評価を要求する前に今回保存kernel証拠を再利用し、CPU/RAM記録を埋める目的だけの無断再計算をしない。ロードマップ本体は変更していない。

結果公開・remote照合後**mandatory STOP**。追加shot探索・他方式MSE・価格・G・winnerへ進まない。監視修正を理由に既に消費した6科学callを再実行しない。
