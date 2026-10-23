# F2条件付き契約案：actual回路費用と統計負担

`DRAFT_NOT_AUTHORIZATION`。F1完了・mandatory STOP・ユーザー/GPT確認後に、**別の明示実行承認**が必要。F1成功でも自動実行しない。比較の位置付けは既知法のPR応用である。

## 1. 再固定する対象

F1と同じphysical fixture／state／sector／scalar／D/R split、全4arm SS/SP/PS/PP。ただしF2は `T=.2,q=1,delta=.2,r=2,h=.1,K=0`。B、law、角度、N、finite target、信号、費用はこの条件で再構成する。F1/K2の値を再利用しない。order0 childの二回でbasis共有とsingletonの費用交換を見る限定実験で、高次数・分子一般・全RPEへの性能結論ではない。

SOURCE／input／environment／compilerをF2承認時に再freezeする。F1後にSOURCEを変える必要があれば差分と再検証範囲を承認へ返す。新しいr/K探索やsampling則最適化を加えない。

## 2. 全有限trajectoryとcall

arm aのcanonical component数L_aに対してordered `(child0,child1)` は **L_a²** 通り。重みp_i p_jで全件扱う。両軸X/Yで `Q=2 sum_a L_a²` logical full-wrapper requests、これに共通のprep/readout-only診断X/Y2件を登録し、logical build/compile requestsは **Q+2**。Lは未確定で、正確な整数件数はまだない。binding後にarm別L、L²、Q、出力件数、保存bytesを契約へ記載してから承認を受ける。実務的でなくても少数MCへ無断変更しない。

cacheは同SOURCE/input/env/compiler／full typed IR／全角度・phase・mapping一致に限る。unique compile misses Uは事前IR binding後に解決、`U<=Q+2`。logical requestと実build/compile attemptを両方記録。各unique miss一回、各arm requestの参照recordは残す。failed compileもattemptを消費し自動retryしない。古い科学run／cache contractを無断再利用しない。

各requestは **prep X0X1、ancilla H、actual controlled trajectory、X/Y readout、ancilla terminal measure** を含む。Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／topology-free、その他設定は[入力仕様](FIXTURE_AND_INPUT_IDENTITY_DRAFT.md)どおり。compiler seed／optimization levelをarm別に変えない。

## 3. 正しさの再資格

compile後の同じfull wrapperからterminal measurementだけ除き、layoutを追跡し、QC.global_phaseを含めて独立reference operator／信号を照合する。追加のevolution-only compileで良い費用へ置換しない。compile前／後は各Q+2 logical recordを検査する。F2 finite corrected mean4armを独立P1 targetへ照合、physical exact signalを共通fixtureで二精度評価しbiasを別報告。parameterが変わるためF1 PASSだけでは省略しない。

F2のreference casesは共通exact physical signal1、finite target1、full wrapper `Q+2`。各々90／128digitsの2評価を事前登録する。finite mean比較4arm、physical bias診断1件、probability mass4arm、shot qualification4arm。実際のquantum measurement shotの抽選は0件。

## 4. 統計とbias

候補 `eta_stat=.025, alpha=.05`、各axis `N=ceil(4 B_tot²/eta_stat² * log(4/alpha))`、F2は `B_tot=B²`。全armで同じ正当なbounded-IID規則を使う。lambda／finite B／Btot／N_X/Y／shot範囲／law誤差を保存。scalarやIを除外してBだけを有利にしない。N式を実際のquantum試行結果やenergy精度と表示しない。

候補bias allowance .025はfinite meanから真signalへの別予算。小系90/128digitsの差と `|z_K-z_exact|` は `NUMERICAL_DIAGNOSTIC_ONLY`。厳密certificateとは呼ばない。次の資格を分ける。

| 資格 | 比較へ付けられる表示 |
|---|---|
| law・circuit・IID/range・finite-bit allowanceが適格 | finite-target条件付き同統計精度の資源比較 |
| 上記に加え、採択したphysical bias budgetの正当なbound/enclosureと実装誤差合成が適格 | 採択した根拠の範囲でphysical signal同精度比較 |
| 小系bias診断だけがbudget内 | 数値診断付きfinite-target比較。厳密な真signal保証なし |
| 統計資格、mean、controlled phaseのいずれか不成立 | `INELIGIBLE_SAME_ACCURACY_COMPARISON`、STOP |

physical同精度を主結論にする契約なら、bias資格が欠けた時点でその主比較は不適格。finite-targetまでに制限する場合は**実行前**にその表示範囲を承認する。結果を見て比較対象をすり替えない。bias／数値条件を満たせなくてもr/K/etaを変更して救済しない。

## 5. 費用の報告単位

各axis／armの全列挙重み付き期待値としてRZ count、CX count、total depth、RZ depth、CX depth、circuit size、measurement countを報告。total depthはterminal measurement込み、quantum-only depthも別欄。RZ/CX depthは同じgate filter定義を全armに適用する。prep/readoutはfull wrapperに含む。共通診断2件はprep/readout-onlyの参考で、compiler fusionが非加法的なのでそのcountをfullから差し引いて「正確な内訳」としない。

主actual座標 `C_RZ=N_X E[RZ_X]+N_Y E[RZ_Y]`、CXも同様に別報告。1-shotで有利でもN加重で相殺され得る。T、RZ、CX、depthを同一単位に換算しない。native T／routing／hardware fidelityを追加しない。finite enumerationなのでtrajectory MC標準誤差はないが、確率／precision／score interval由来の不確実性は残る。

辞書生成・collect、law生成、selector、既知四候補価格探索、cache cold/warm、compileを別timing／peak memory／保存bytesで記録する。科学callは各phase一回、timingの反復benchmarkを勝手に足さない。秒やbytesを根拠なくRZへ換算しない。

## 6. 事前の判断分類

primary比較座標はN加重actual RZ、CX/depthは併記してtradeoffを残す。非重複数値区間の厳密な改善を有利とし、arbitraryな性能倍率を要求しない。完全同点／区間重複は未確定扱い。結果タグは次のうち適用するものを複数残せる。

1. `MIXED_ACTUAL_RZ_BEST`：SP/PSが有利。既知四候補探索でも得られるか併記。
2. `ALL_PAULI_BEST`：PPが最良。有効な結果。
3. `ALL_FRAME_BEST`：SSが最良。有効な結果。
4. `LITERAL_ONLY_ADVANTAGE`：literal順位の利得がactualに残らない。
5. `SHOT_WEIGHT_CANCELS_GAIN`：1-shotの利得がNで消える。
6. `CLASSICAL_OVERHEAD_UNRESOLVED_OR_HIGH`：古典費が大きい／償却回数不明。事前の実用workloadや閾値が無いなら定量値と償却条件を出し、恣意的に「過大」と断定しない。
7. `INELIGIBLE_SAME_ACCURACY_COMPARISON`：bias/finite mean/control/stat資格不成立。
8. `TIE_OR_NUMERICALLY_UNRESOLVED`：小系で順位確定不能。

mixed最良でも `INDEPENDENT_METHOD_DELTA_UNESTABLISHED` は解除しない。既知のcost-aware有限探索で同じ生成・選択を得る場合はその説明を残す。

## 7. 実行資源・出力・STOP

独立compile missesをprocess分離して並列化、BLAS/OpenMP/Rayon/Qiskit内側並列とのoversubscriptionを避ける。immutable inputを共有、cacheは専用領域、workerごとatomic output。CPU/GPU/RAM/worker/wall-timeに恣意的な上限なし。実空き資源・集約memoryで構成を決め、共有venv/cache/OS/scheduler/quotaや他jobを変更しない。科学call上限・output/log制限と資源上限は別である。

compile IR／counts／layout／provenance／case weight／qualification／aggregate／STOP reportを保存。全event matrixを一括保持せずstreamする。complex128 matrix一枚は16*d² bytes、d16/32の小系referenceに限定。Lとgate数のbinding後にoutput容量を正確化する。

失敗、SOURCE/input違反、call逸脱、無断条件変更、原本保全違反は即STOP、自動retryなし。成功・失敗にかかわらずmandatory STOP。molecular移送、K2以上、RPE全体、性能GO／新規性GO／投稿GOへ自動移行しない。
