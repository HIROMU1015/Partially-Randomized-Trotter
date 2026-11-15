# Full-H κ2 m0受入・固定1.5倍配分の会計準備

**D1の公開m0結果は所定gateでSTOPしており、CはX/Yとも参照単価を保留する。全13 round総額T、m0部分額、T内割合、PF二基準との差はnullのまま。** 保存24軸の再予測や実総額の計算は行わない。

技術SOURCE `5e619de429ddfc69a3a64da24a766b97db2b546a` を固定後、[D1_MAPPING.json](D1_MAPPING.json) から保存記録の受入経路を一回実行し、[run01/RESULT.json](run01/RESULT.json) と [AXES.csv](run01/AXES.csv) へ保存した。これは失敗したD1取得の再実行ではなく、公開されたSTOPとaxis別receiptのmetadata照合である。

## 固定条件と根拠

H4直線1.00Å/STO-3G/保存tol-only DF10、Full-H pure RTE（one-bodyを含む全非scalar、headなし）、finiteK2・κ2・P0。m0はFULL_H_2a9a68a11d08af234a8c、T=.1/q1/R9。symmetric-directional、new ZZ CONTROLLED_ZZ_CX_CRZ_CX_V1、測定付きfull controlled Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を維持する。

- 価格規則：4a1cb0d7364fece475437749129fdce740fd8fc4 の full_h_m0_price_rule_20261115/PROPOSAL.json。単一normal pathの経験的参照であり母期待価格/CIではない。
- D1 producer：7f85f20e38af4df1a649ac239e57bfe76d0899ee。公開結果：b25d7da6dedebc4cd22e8d6f00e6de75d7072292 の d_full_h_m0_reference_20261115/。HANDOFF_C成功packetはなく、STOP_DETAIL.json、OBSERVED_COMPILES.json、run01/PRICES.jsonと各元compile recordを用いる。
- cohort D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1、index0、seed15612141344395026302。X/Yは同じ1 pathであり独立2標本ではない。
- preparation hash 92325333cfe8c5e6c5f9bf69cdfc0ea6538f8eaa66327a3820d390dc2e8aa100、scalar −0.33147781341681615を総Tへ一回。D1保存contextのH/state/snapshot、compiler/control/wrapper/P0、component順の検査を再利用する。

## 両軸の保留理由

D1はnormal draw1回・compile2件を完了した。公開OBSERVED_COMPILESではX/Y各1062 RZだが、両方のper_shot_RZはnullで、分類はOBSERVED_COMPILE_ONLY_NOT_ACCEPTED_REFERENCE。軸別fingerprint・測定tail・元receipt bytesは別々に保存されている。**1062を受入単価として扱わず、96倍もしない。**

保存pathはorder0のみで、中央rotationは一量子ビットだけ、two-qubit ZZ applications=0。元P1 pilotの要求 parity.replaced_controlled_zz > 0 が満たされず、acquire_pathの複合gateでSTOPした。これは新ZZ操作のcoverage条件の問題であり、価格1062や保存7数値gateのPASSだけから所定gate成立へ読み替えない。

run01/NATIVE_TRIAL/result.jsonの7数値gateはD1でPASS済み。一方、複合gate例外時に個別parity/phase/XYの返却値が保存されていない。Cは未保存の検査を推測せず、gate変更、path再抽選、再compileを行わない。旧STOPはそのまま保持する。

## 会計の準備・再利用

[account_full_h_reference_scenario.py](../../../scripts/account_full_h_reference_scenario.py) は既存Git evidence readerとF1のexact有理数表現を再利用する。モデルfit/predict、sampling、回路構築、compiler、Q3は呼ばない。

保存F1 a1aa38df46c92e2489095891f6082a5be1a277f5 のm1〜m12・24軸を転記し、各physical ID/axisをB_joinと照合する。m0の旧負値とF1のT_ref=nullは変更しない。固定shotはX/Yそれぞれ[96,90,84,78,71,65,57,51,45,38,32,26,18]。

人工価格で、同じ入口の欠測停止と正確な重み付き和・割合・符号を検証する。実データ経路では両軸保留のため加算処理の前にnullを返す。将来の適合するm0受入に際しては、実際の公開schema/判定SOURCEに合わせた最小reader接続が必要であり、現STOP専用readerを成功packetの受入と称さない。集計許可は今回の1構成・1配分について継続しているが、固定gate変更や取得retryの許可とは別である。

比較正本はb4296b110b129d226dbef2743c14f725a27c3bf6 の pf6_cost_budget_q3_results_20261115/RESULTS.csv。Full PF S2=615304502、B0I p7 S2=435481090 RZ。保存allocation/sourceを保持し、旧PF基準を混ぜない。差の規約はT−T_PF（正ならFull-Hの方が高い）。現在は両差ともnull。

## §10.5・性能・保全

C既存branchと固定4a1提案の重複調査を再利用し、その後の関連公開ref/履歴、今回の出力path、同名runnerの自分のprocess名を差分確認した。同等の完成集計・実行中C jobは確認範囲で0。D1の新しいb25d7da6d STOP公開を再利用し、未知archive/他ホストの網羅や他担当private領域の確認は行わない。

新規処理は小さい保存JSON/CSV/compile receiptの読取と人工テストだけ。24モデルquoteは再予測せず、event列/QPYを展開しない。共有環境・他担当branch/worktreeは変更しない。新しい並列化framework、性能向上の未実測主張はない。

旧F1/D7、原receipt、旧STOP、資格、Q3を保全する。正式E[C]/CI/G/winnerは未評価。ロードマップ本体は更新せず、HANDOFFの差分を統合担当へ渡す。結果公開後mandatory STOP。

## 実施結果・検査

| 区分 | 結果 |
|---|---|
| m0経験的参照単価の受入 | 0/2、保留2/2 |
| 保存compile観測への接続 | X/Y各1、同一normal path1本（価格資格は保留） |
| F1 m1〜m12保存model引用 | 24/24軸、再予測0 |
| T / m0部分額 / m0割合 / 2基準との差 | すべてnull |
| Cのsampling / build / compile / fit / Q3 / 新予測 | すべて0 |

結果statusは`HELD_NO_TOTAL`。24有効保存model軸があってもm0のnullを0や1062で補完せず、実データ加算は起動していない。集計元を固定したSOURCE・D1各locator・PF比較正本はRESULT.jsonに保持する。

[TESTS.txt](TESTS.txt)：新規6＋D7受入回帰6＋F1人工回帰6＝18 test PASS。正の会計・exact割合・差の符号、片軸欠測、両軸producer STOP、scalar/SOURCE/axis不一致、成功packetをSTOP用readerへ誤投入しないことを確認した。人工価格のテストであり、実Full-Hの価格資格や独立科学レビューではない。

再開上の不足は、D1の固定coverage条件と元pathの合法性・保存されなかった個別実装gateの対応を明確にすること。現在のSTOPを上書きせず、保存pathでの限定再検査等が必要ならD1側で対象・SOURCE・許可範囲を具体化する。新pathやcompileをC側から自動要求・投入しない。

本公開はユーザー指定の「両軸受入できた場合だけ集計」の不成立分岐を完了したもの。全13roundの数値総額・PF順位は未判定、mandatory STOP。
