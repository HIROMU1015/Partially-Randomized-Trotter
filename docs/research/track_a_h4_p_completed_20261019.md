# Track A H4：固定PF対照に対するPR独立確認Pの完了

状態：`COMPLETE_MANDATORY_STOP`。主比較10/10・診断比較4/4で、事前登録した費用差の点推定方向を再現した。
これは固定nの新確認標本における結果であり、母平均優位や厳密biasの保証ではない。

## 固定範囲と承認

今回の利用者指示「計算を開始してよい　中断された場合は修正して再計算して」に基づきPを実行した。
技術中断に対する修正・別来歴での再計算も承認されたが、今回のPは中断0・再試行0で完了した。
科学的な非再現を理由とする再試行、候補再選択、n変更、後続stageへの自動進行は許可していない。

H4 linear/隣接1.00 Å/STO-3G/legacy固定DF12/8 system qubits＋ancilla、T=.8、P=0。
B2二次partial PF、元finite-RTE law・quantum weight・scalar・fragment順序を保持した。
L_D=2/3、q∈{1,2,4,8}、δ=T/q∈{.8,.4,.2,.1}、r=2/4、K=2/4の事前登録9候補に限定する。
通常signed/factor1、ε=.05/.01/.005/.001/.0005の保存shotと完成DのPF対照を使った。
native symmetric-directional、Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling指定なし。

P科学SOURCE：`73ec1964501f14390d037a378c3149110e273635`。
D固定結果commit：`4e8ad1936363266d1b55bb689c36846e6c3dfcdb`。
S_STATS固定結果commit：`5e18773db3ad18c591b3c0fdc83cc1b5f1a29cfc`。
P実行spec digest：`b80ed8b5c2a303c7113296ef88e1347bd372f8ec135211a644725e0d3044cc94`。
P科学result SHA：`4f79637089d2c61f5e85edbc887cfafd28d8df0624df3b04c9ecad58f6504043`。

## 実行と並列化

新4185 trajectory/native evolution、8370 full-wrapper compile、9770 occurrence、37662 RTE eventを完了した。
9候補のreplica0でcontrol36 probeと独立event oracle9件を全てPASSした後、残りの軌道を実行した。
新endpoint会計は1 pass。旧探索n8/n32を混合せず、候補・固定n・14比較・PF・shotは変更していない。
fresh resource admissionで61 workerを選択し、native driver wallは1167.727秒。
各workerは1 thread、prefix2/3のpreparation cache、cos/sinのevolution共有、fingerprint再利用を用いた。
workerごとに2 prefix、合計122件のpreparation再構成予約を別receiptで保存した。
SCF/DF/state/reference/signalの新規solve、shot再計算、bootstrap、新n設計は0。
OS/cgroup/affinity/共有venv/他jobの変更は0。PSI観測だけによる停止は行わなかった。

## 固定nと観測層

|候補|固定n|S0_ONLY|S2_SINGLE|S2_MULTI|S4_PRESENT|
|---|---:|---:|---:|---:|---:|
|B2_p2_q4_2nd_r4_K2|901|856|44|1|0|
|B2_p2_q8_2nd_r4_K2|115|114|1|0|0|
|B2_p2_q8_2nd_r4_K4|115|112|3|0|0|
|B2_p3_q1_2nd_r2_K2|277|268|9|0|0|
|B2_p3_q1_2nd_r4_K2|2119|2077|42|0|0|
|B2_p3_q2_2nd_r4_K2|459|452|7|0|0|
|B2_p3_q4_2nd_r4_K4|145|145|0|0|0|
|B2_p3_q8_2nd_r2_K2|26|26|0|0|0|
|B2_p3_q8_2nd_r2_K4|28|28|0|0|0|

層は得られた全固定標本を事後分類した件数。層別の再抽出・条件付き再sampling・除外は行っていない。
S4_PRESENTは全候補で0。K4候補でもS4の費用は今回観測されておらず、未観測tailの不確実性は残る。
標本SEや点順位の再現を、S4を含む母平均の保証とはしない。

## 事前登録した14比較

Gは保存shot×実測RZのcos/sin和。mean・SD・paired SE・同一候補内のε間共分散は新標本だけから、runnerの1回の会計で求めた。
下表は保存済みexact分数の表示用丸めであり、機械可読結果はexact分数を保持する。

|役割|候補|ε|n|PR mean G|paired SE|PF G|方向判定|
|---|---|---:|---:|---:|---:|---:|---|
|主|B2_p3_q1_2nd_r4_K2|0.05|2119|70632973|124170|86045388|REPRODUCED|
|主|B2_p3_q1_2nd_r2_K2|0.05|277|73346137|159525|86045388|REPRODUCED|
|主|B2_p3_q2_2nd_r4_K2|0.01|459|3.4542009e+09|9.85849e+06|6523645968|REPRODUCED|
|主|B2_p2_q4_2nd_r4_K2|0.01|901|4.9279016e+09|8.02605e+06|6523645968|REPRODUCED|
|主|B2_p3_q4_2nd_r4_K4|0.005|145|2.1504365e+10|7.22754e+07|42602298697|REPRODUCED|
|主|B2_p2_q4_2nd_r4_K2|0.005|901|2.1356317e+10|3.47829e+07|42602298697|REPRODUCED|
|主|B2_p3_q8_2nd_r2_K2|0.001|26|1.0322572e+12|2.75644e+09|1835103650289|REPRODUCED|
|主|B2_p2_q8_2nd_r4_K2|0.001|115|9.8878586e+11|3.37845e+09|1835103650289|REPRODUCED|
|主|B2_p3_q8_2nd_r2_K2|0.0005|26|5.3237058e+12|1.42159e+10|11241349628381|REPRODUCED|
|主|B2_p2_q8_2nd_r4_K2|0.0005|115|5.1021843e+12|1.7433e+10|11241349628381|REPRODUCED|
|診断|B2_p3_q8_2nd_r2_K4|0.001|28|1.0311731e+12|2.14262e+09|1835103650289|REPRODUCED|
|診断|B2_p3_q8_2nd_r2_K4|0.0005|28|5.318139e+12|1.10503e+10|11241349628381|REPRODUCED|
|診断|B2_p2_q8_2nd_r4_K4|0.001|115|9.8869747e+11|3.53494e+09|1835103650289|REPRODUCED|
|診断|B2_p2_q8_2nd_r4_K4|0.0005|115|5.1017436e+12|1.82405e+10|11241349628381|REPRODUCED|

判定は事前固定したsign(PF G−PR mean G)と保存探索の方向との一致。14件の期待方向はすべて+1。
S_STATSのfamily alpha/powerは標本数設計の入力であり、この点比較を多重比較補正済み有意差検定や母平均保証と読み替えない。
同一候補の複数εは同じn軌道を使い、保存結果に共分散行列を記録した。候補間共分散は推定しない。

## 終了・保全監査

native driver exit0、親によるwait/reap、全61 workerのexit0と回収、全固定n・両軸・台帳一致を確認した。
6 SOURCE file/42入力/924 kernel-provider file/34273 runtime fileの実行前・実行後byte照合を通過した。
以前の377記録と技術sealの982記録・各copyは不変。今回の21072記録を新しいSHA storeへ保全した。
seed planと4185 master/11559 slot一覧は技術sealから変更していない。旧系列照合receiptの衝突0を再確認した。
公開projectionからseed値とRTE event列を除き、private原本・台帳・receiptは保存した。
終了監査は作者自身によるmetadata照合であり、別reviewerによる独立レビューではない。

## 保存先と次の作業

- [科学集計結果](../../artifacts/resource_applicability/track_a_h4_p_completed/2026-10-19/confirmation_result.json)
- [軌道ごとの実測cost projection（seed非公開）](../../artifacts/resource_applicability/track_a_h4_p_completed/2026-10-19/paired_costs_projection.json)
- [control結果](../../artifacts/resource_applicability/track_a_h4_p_completed/2026-10-19/controls.json)
- [承認済み実行契約・fresh資源の公開projection](../../artifacts/resource_applicability/track_a_h4_p_completed/2026-10-19/execution_contract.json)
- [終了監査](../../artifacts/resource_applicability/track_a_h4_p_completed/2026-10-19/completion_audit.json)
- [事前固定した技術契約とseed設計](track_a_h4_p_technical_seal_20261019.md)

Codex：今回の承認範囲を完了しmandatory STOP。後続科学計算は自動開始しない。
ユーザー：結果と限定を確認し、後続の研究判断・追加実験の要否を決める。
remote pushは行っていない。
