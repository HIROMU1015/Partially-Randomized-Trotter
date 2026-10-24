# H4/H6 標準Yoshida4：修正後の独立確認完了

**H4/H6とも、独立PR標本で標準Yoshida4/q4がS2/q8より低費用となる方向を再現した。G比（Y4/S2）はH4=0.864959895、H6=0.862762409。事前固定した2比較familyのBonferroni正規近似区間も、両系で差G_S2−G_Y4の正側にある。** 保存Fullでは両系ともS2/q8が低費用であり、異なるqを選んだ固定構成でのFull/PRの方向逆転を、新しい独立PR標本も支持する。

これは固定されたsignal・reference・empirical allowance・analytic shot政策に条件付けた回路費用標本の確認である。次数だけの因果効果、母平均優位、希少tail、厳密biasや全simulation誤差の保証ではない。旧strict COMPAREはBLOCKEDを維持し、mandatory STOPした。

## 独立標本の主結果

linear H4/H6、隣接1 Å、STO-3G、tol-only DF1e-8、rank10/19、T=.8、ε_signal=.001、α_axis=.025、P=0、usual signed bias。PRはprefix3/5、K4・ABS_TIME_PER_OUTER32。symmetric-directional control、状態準備を除くfull cosine/sine Hadamard wrapper、compiled RZを使用した。

| 系 | PR公式・q | 独立n | 平均C_cos / C_sin | 平均G | paired SE(G) |
|---|---|---:|---:|---:|---:|
| H4 | S2/q8（保存標本） | 25 | 51413.640 / 51413.640 | 2.195108496032e12 | 2.380446064e10 |
| H4 | 標準Y4/q4（新標本） | 45 | 49901.711 / 49901.711 | 1.898680813959e12 | 1.232423568e10 |
| H6 | S2/q8（保存標本） | 23 | 182086.565 / 182086.565 | 9.177565844525e12 | 7.279669810e10 |
| H6 | 標準Y4/q4（新標本） | 32 | 186172.469 / 186172.469 | 7.918058821083e12 | 3.328643611e10 |

上表は丸め表示。[4群のCSV](independent_groups_4.csv)と[完全な結果JSON](results.json)に候補ID、固定N、軸別平均、covariance、exact整数/Fraction momentを保存した。S2は旧ε=.0005のGを転記せず、保存trajectoryのC_cos/C_sinを今回のε=.001のNで新しく重み付けした。cosine/sine covarianceを含めた分散恒等式は4群とも整合した。新S2 sampling/compileは0。

| 系 | 差G_S2−G_Y4 | 差のSE | Bonferroni正規近似区間 | G_Y4/G_S2 | 判定 |
|---|---:|---:|---|---:|---|
| H4 | 2.964276821e11 | 2.680558023e10 | [2.363455814e11, 3.565097827e11] | 0.864959895 | REPRODUCED_APPROXIMATE |
| H6 | 1.259507023e12 | 8.004589985e10 | [1.080091925e12, 1.438922122e12] | 0.862762409 | REPRODUCED_APPROXIMATE |

[2比較のCSV](independent_comparisons_2.csv)。familyはH4/H6の2比較、α_family=.05、両側Bonferroniの正規近似を事前固定した。差SEは公式間の独立標本設計を仮定するhypot、各公式内はpaired covarianceを保持した。点方向と近似区間の判定を分離し、区間が0をまたぐ場合のINCONCLUSIVEを許容したが、今回は両系とも区間が正側だった。比の区間やp値は計算していない。正規近似区間に厳密coverageや母平均順位のcertificateはない。

## 保存探索・Fullとの対応

以下は`eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23`のε=.001 endpointを引用したものであり、新たなFull compileや会計ではない。

| 系 | 標本・方法 | S2/q8のG | 標準Y4/q4のG | 保存費用方向 |
|---|---|---:|---:|---|
| H4 | Full、決定論的 | 2668093076979 | 3341048717564 | S2低費用 |
| H4 | PR探索（n48/n33） | 2188164571311.0 | 1885209985424.3333 | Y4低費用 |
| H6 | Full、決定論的 | 15636885592900 | 16951728941568 | S2低費用 |
| H6 | PR探索（n33/n33） | 9289932763968.213 | 7949791328110.182 | Y4低費用 |

[保存8 endpointの転記](quoted_full_and_exploration_8.csv)。Fullのprefixはrank全体の10/19、PRは3/5。探索標本と独立標本はpoolしていない。新しい独立PR比較は、両系で探索時と同じY4低費用方向を示した。Fullの決定論的保存費用へPRの標本SEを付与していない。

元の同一q対照ではq4はFull/PRともY4、q8はともS2だった。今回もこの区別を維持し、観測した逆転を次数だけの効果や全候補での最適性へ拡張しない。H6では新独立標本のY4の平均単価はS2より高いが、固定Nで重み付けしたGはY4が低い。

## 固定契約と修正内容

| 項目 | 値 |
|---|---|
| 修正後の科学SOURCE | `a2a53698a72da7280d217adc7931938121ed732d` |
| exact契約SHA256 | `4750841471e8f98a358db64e2cee8c616a7888408ef1a4bdbd6f1c6d98ab0f74` |
| 入力manifest SHA256 | `1409340dbe84ca07a4bdcfc73400699ce0cfb6ec14559d4febec98498336c128` |
| 再実行承認receipt SHA256 | `91cff59529df3c35361f3d0cba3e48c95c5fc960b8993c2bf8c65459699d4fb6` |
| 再利用する数値実装SOURCE | `c028f1c2885afdba65ef8c62c075738aadd7323d` |
| provider SOURCE | `217c971dd537b1cc7fbc13742bc0f6c263ff7f45` |
| 保存独立S2結果commit | `3eca46148070ba4d4b0f5815ef45943407cc959e` |

利用者の「修正を行って再実行して　意図しないところで中断された場合は修正を行って再実行してよい」に基づき、新worktree・private領域・SOURCE・契約・grantで実行した。再実行許可は技術的中断の修復に適用し、期待方向に合わせた候補変更・追加標本へは使っていない。個々のgrantは一回消費、返金・再利用なし。

修正は、遅延初期化された具体的runtimeを保持してworker/cache guardから参照する点と、旧停止証拠の保全binding。元sampling・native・Gaussian・compiler・paired統計の数値関数を変更していない。準備処理の変数名衝突もgrant発行前に修正した（科学call0）。実際のrebind/get_runtime、fork poolのinitializer、workerとcache参照を通す人工回帰検査を含む25件が予定Python3.11.0rc1でPASSした。

Qiskit1.3.0、NumPy1.26.4、basis rz/sx/x/cx、opt1、seed_transpiler17、compiler fingerprint `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`を維持した。新SOURCE/inputのmetadata-only preflightを経て、今回の利用者指示を新契約へbindingした。[契約の公開可能部分](execution_contract.json)。

## 科学call・終了・保全監査

| 範囲 | 完了数 |
|---|---:|
| 新Y4 trajectory / compile軸 | 77 / 154 |
| request sampling / RTE event | 924 / 9856 |
| native evolution / partial-native build / wrapper生成 | 77 / 924 / 154 |
| trajectory statevector action / oracle action | 462 / 77 |
| native task台帳（TASKを含む） | 2926 |
| 親context台帳 | 18 |
| 会計台帳（bootstrap1/load1/family定数1/paired統計4/方向比較2） | 9 |
| 修正後実行の全台帳entry | 2953 |

全77 trajectoryの8数値gate、両軸のCT01（誤差strictly <1e-7）、その他のnative gate（<1e-9）を受け入れた。位相合わせは行っていない。実際のpartial12/wrapper2生成を軌道ごとに確認した。statevector462はlow/high/uncompiled/compiled actionの内部合計であり、追加の別予算ではない。

H4は17 worker、native wall329.232秒、H6は27 worker、898.210秒で完了。両native launcher・controllerはexit0、全44 workerもexit0で回収済み。修正後の科学実行は一回で、予定外の科学中断・追加科学retryは0。旧SOURCEの停止runは別履歴であり、その親context18 entryと消費済み2 grantを保持した。

新rawは専用private領域のlossless archive462件に保全し、原byte SHAとarchive SHA、SOURCE・event・compiler・費用receiptの対応を検証した。sealed入力569 fileを照合し、完了出力6698 fileのprivate保全snapshotを作成。旧停止runの64 fileもSHA一致した。旧SOURCE・元結果・旧grant/ledger/marker/STOPに変更なし。[機械可読の終了監査](end_audit.json)。

資源は実際の空きCPU/RAM・affinity/quota・disk/inodeを観測して並列実行し、旧worker数を恒常capとして適用していない。host PSIや無関係なworktree更新による停止、他process操作、共有venv・OS・schedulerの変更は0。private raw・seed・authority・内部絶対pathは公開しない。

## 証拠の限界と終了

H4のS2/Y4は全標本がS0、H6の保存S2もS0のみ、新Y4はS0が31・S2_singleが1だった。S2_multi/S4など未観測の費用・分散・tail上界はnullであり、0と補完していない。保存pilotに基づくn45/32は母分散上界・power保証ではない。

旧独立S2は、受入済みpair CSVと固定成功監査の来歴を条件付きで再利用した。原raw全体や実使用registryを今回再回収・replayしたわけではなく、歴史上の全seed非衝突も未証明。新namespaceは取得可能な旧seed集合から除外した。Pythonの旧runtime byte同一性を追加認定せず、固定native/compiler仕様と実行した各軌道の数値gateを証拠範囲とする。

**次の担当はユーザー：固定構成・固定shot政策に条件付けた独立回路費用の方向再現として、研究上の採用範囲を判断する。Codexによる追加計算はなし。** 詳細科学レビュー・最終evidence freeze・旧strict COMPARE解除・別分子系への拡張を自動開始しない。[mandatory STOP](STOP.json)。
