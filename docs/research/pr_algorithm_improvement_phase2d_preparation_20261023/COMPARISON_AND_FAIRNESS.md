# 4arm・selector・既知全列挙の公平性

共通物理taskと静的候補値は[入力仕様](FIXTURE_AND_INPUT_IDENTITY_DRAFT.md)を正とする。

| arm | F0辞書 | F1辞書 | 用途 |
|---|---|---|---|
| SS | frame | frame | 全frameの強いbaseline |
| SP | frame | physical Pauli | mixed候補 |
| PS | physical Pauli | frame | mixed候補 |
| PP | physical Pauli | physical Pauli | 全Pauliの強いbaseline |

S/Pで同じfragment全体を表す。P fragment間は同一wordを加算してからlambdaを計算、Sは元区別を維持。全armへ同じ係数代数、精度、recipe provider、cache、compiler、state prep、readout、scalar、D/R split、q/r/Kを与える。個別Pauliが粒子数を変えることは許すが、中間sector投影で消さない。

## 1. 二つの選択対照

Phase 2C selectorをそのまま採用候補にする。SSとPPの二端点からfragment ID 0,1の順に一回だけ反転を検討し、whole candidate価格の区間が厳密に改善するときだけ更新する。両端点を候補に残し、最後はSSを基準に残った候補をstable bit順で比較する。overlap／tieは現候補を維持。追加pass、未知のvariance、actual compiler cost oracleを入れない。

もう一方は**同じ4候補の既知の全列挙探索**。SS,SP,PS,PPを一度ずつ価格付けし、同じ区間比較・tie規則を用いる。小系の価格上の最良値を確認する対照であり新算法ではない。selectorは高々6評価request、unique候補は高々4。同じcacheを共有したunique価格付けは4件まで。冷cache構築の時間を両者へ隠さず、冷／warm双方を分ける。

主選択scoreはPhase 2Cの `J_RZ=N_X Phi_X,RZ + N_Y Phi_Y,RZ`。Phiは固定literal planで、state prep/readoutと融合済みDの固定部分を含める。RZ以外の座標を暗黙に換算しない。F1のselector診断でもN候補規則・eta/alphaはF2と同じだが、F1のq/K/lambda/Bで計算し直す。これをF1の量子shot実行や資源GOと呼ばない。価格区間providerが未確定なら `UNPRICED_FALLBACK` とし、優位性を採択しない。

## 2. 全armへ与える既存最適化

同source basisの連続run共有、singleton support-restricted completionと長run full basis、Pauli側局所basisの同一key共有、非制御basis compute/uncomputeと中央だけcontrol、phase集約、D境界同一half-step融合を全armへ与える。identityをrun境界にするliteral modelも共通。actual builderで合法に得られる追加相殺をbaselineから禁止しない。compilerの正当な局所簡約は全armで同版同条件にする。relative Gaussian等を追加で使うなら適用providerを全armで同時固定してから承認を受ける。

noncommuting operatorをbasis順にsortしない。arbitrary PauliをDF用Z/ZZへ偽装しない。dense UnitaryGate、free exact ground state、arm専用軌道回転、DF rank変更、r/K探索、sampling分布の独立最適化を追加しない。回路単価を良く見せるためstate prep/readoutを除外しない。

起点のbuilderが参照版より古いことをbaselineを弱くする理由にしない。[SOURCE接続案](SOURCE_IMPLEMENTATION_PLAN.md)の共通policy portを四armともに用意する必要がある。portが不完全なら承認準備未完了である。

## 3. Cacheと費用の境界

物理入力・SOURCE closure・typed IR・全角度・phase・mapping・precision・compiler設定でcache keyを作る。arm名だけを消して比較した場合でも、canonical **full IR全体が完全一致**しない限りcompiled回路を共有しない。角度やq/Kの異なる回路を同じcacheとして扱わない。科学case recordはarmごとに残す。計測を有利にする片側だけの事前warmは禁止。

dictionary生成、全体collect、law生成、score、compiler、cache初期化／reuse、peak memory、保存bytesはそれぞれcold/warmを記録する。既知全列挙は同じ候補mapとcacheを利用できる。classical費を無料oracleにせず、根拠のないRZ換算もしない。

## 4. 解釈

mixedがactual費で勝つことは成功の必要条件ではない。SS/PP最良、同点、literalとactualの順位逆転、Nによる相殺、古典負担が大きい、qualification失敗は全て報告対象。全armの最終IRが同一で意味のある構成差がなければ、その結果でF1 STOPしF2を自動実行しない。

Phase 2C-R1の一案の方法差不成立をRQ-01/03全体の終了へ拡張しない。本段階のPASSやmixedの勝利は独立新規性を確立しない。既知の四候補探索でも同じ選択を得る場合はその説明を残す。
