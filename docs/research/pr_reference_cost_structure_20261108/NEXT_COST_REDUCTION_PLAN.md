# 新A：次のコスト削減判断に必要な限定検証

2026-11-08。**この文書は新しい科学実行の許可ではない。D0の準備までを完了し、実行前で停止する。**
[総括](REPORT.md)／[固定コスト構造](COST_STRUCTURE.md)。

主資源は同じ成功条件下の全18round総native RZ。今回のN・law・回路・compiler・物理条件を変えずに診断する。
他Trackへの依頼・担当変更・進捗管理・実験指示は含めない。

## 1. 改善候補の選別：静的調査の結論

| 対象 | 残る因果経路と既存根拠 | 純増分・トレードオフ・今回の判定 |
|---|---|---|
| native vacuum位相の算術／十分保証 | 保存phase_boundはnaive和と正規化の誤差を別に課金し、raw disk→ρ/β→ceil Nへ接続。m17だけ626対630の差があり、共通basis/D費にも掛かる | **条件付きの調査候補**。新しい算術方式で達成可能な誤差・N・総RZは未検証。CT guard改善とは別。まずD0で既存費用差と必要な利益を明らかにする。回路RZ自体を減らさなくても、将来Nが減れば全資源の実行回数へ作用 |
| D／RTE basis共有、inverse取消 | 既存D融合、same-basis run共有、compiler cancellationが含まれる | 同じ処理を追加候補にしない。異register/cutを跨ぐ融合、source inverse→compiled actual adjointへの置換は新資格を要する。現在、純追加効果の証拠なし |
| joint isometry／diagonal parity／中央control | 他条件のTrack Cに既存実績。新AのB′中央ZZは4-CX/2-RZ済み | 既存処理の再実装・他条件からの利益移送はしない。この固定pipelineへの純増分はUNKNOWN。SOURCE/phase/domain/誤差の新接続が必要で、今回の最小検証にしない |
| CSWAP別分解・gadget変更 | 保存CSWAPの128 CX/shot、追加RZ、2 cascadeのdepth負担が明示される。旧レビューに7-CX提案もある | CXを主に狙う既知の別実装候補。Compiler A後のRZ差・成功資格・総利益は未取得。主指標RZで有利な根拠がないため、今回価格取得案を作らない |
| h/r/α／PF／lawの再最適化 | 既存他条件に結果あり、統計配分と単価を同時に変える | 今回の固定条件外。旧最適化を新A独自機構として再提案しない |
| 保存word選択／guard frontier／新合成 | D1/D2等に保存poolの負結果・T増の記録あり | native RZを減らす操作ではない。総Tは角度／precision／word数を含む別資格。保留を解除しない |

新しいRZ削減機構の正式採択、新規性認定、予想削減率の順位付けはしない。第一段階は改善policyの探索ではなく、**どの費用が不足分を生んでいるかの固定データ診断**である。

## 2. D0：保存値だけのRZ差分分解（次の最小科学scope案）

### 固定対象・入力

全18round×X/Yの36セル。両armは固定B′／Referenceのまま、Nも保存値のまま。実入力はcommit **654b8501c1348cd0694daeefd19391df6b928354**の次の3 blobだけ。

| 保存入力 | Git blob | 使用フィールド |
|---|---|---|
| [B′ TOTALS_18ROUND.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json) | 454a0652a9f201e40db713856063f1c55aaaa6d7 | rowsのm/q/r/bits/N、price_per_axisのRZ上下界、exact総Fraction、保存外向き総計 |
| [transfer STAGE4_RPE.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json) | 0003b6ca217eea3df208045ebcec499c5f0c8519 | rows.arms.reference.shots_per_axis、m/q/r/bits。旧14-CX価格を新B′価格へ代入しない |
| [V3 ANGLE_KEY_MULTIPLICITIES.json.xz](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_rotation_v3/2026-11-07/ANGLE_KEY_MULTIPLICITIES.json.xz) | 90f586be170ee6607e82c918609b1ccf618456dc | rowsのarm/axis/N/同じschedule、partsの9 roleと保存Fraction。tokenを再分類せず数を合計 |

これらは科学入力の識別で、承認証明ではない。SOURCE/word/原本の再取得や新しい多数のmanifestは作らない。公開snapshotと元の科学SOURCEは各原報告／SOURCE.jsonの対応を保持する。

### 準備したSOURCEと再利用理由

[diagnose_pr_reference_cost_structure.py](../../../scripts/diagnose_pr_reference_cost_structure.py)を用いる。固定3 blobの読取、Fraction算術、既存roleとpriceの照合、fresh JSON出力だけの薄いadapterである。旧のprice_bounds、expected_parts、totals_and_stopを呼ぶとcommon.pkl、確率再算定、科学出力書込み等へ入るため、保存した結果を直接利用する。

保存V3の9 role：
D_unfused、D_boundary_fusion_difference、RTE_basis_frames、RTE_central_rotation、RTE_products、RTE_event_sign_phase、scalar_or_vacuum_correction、CSWAP、XY_wrapper。

新しいpolicy・basis・価格provider・sampler・量子builder・統計最適化器は実装していない。コードimport時の科学データ読取はなく、CLIを呼んだときだけ固定入力を読む。今回CLIは未実行。

### 算術と出力

各セルでN_B、N_R、保存単価区間I_B、I_Rを保持する。D0は次をexact Fractionで新しく評価するため、科学数値再解析として別途許可が必要である。

1. N_B I_B、N_R I_Rと、Reference−B′の直接差の区間。
2. 基準B：価格差N_B(μ_R−μ_B)、shot差(N_R−N_B)μ_R。
3. 基準R：価格差N_R(μ_R−μ_B)、shot差(N_R−N_B)μ_B。
4. 同じ式を保存9 roleへ適用。D境界はsigned会計補正、Reference product-Z下界は旧規約どおり全除去。空の保存roleは既知の不在として扱うが、role自体の欠損は拒否し0補完しない。
5. roundごとの両軸合計と全18round合計。shot差項がm17に局在すること、前17roundとm17の費用を評価できる形で保存する。新しい削減policy・counterfactual Nは作らない。

区間の4端点組において同じμ_B/μ_Rを両項へ使い、各恒等式を検算する。項ごとの上下界は相関するため、その独立極値の和を実現可能な差と呼ばない。直接差区間を別に正本として持つ。実価格や最適性・価格信頼区間を推定しない。

出力はfreshな一つのRESULTS.json。36セル、role別分解、18roundの寄与、全体総計、coupled intervalの注意、完了STOP状態を含める。RZの小数表示や寄与順位を必要とする場合は、その後の別policy探索にせず、同じ許可scopeでexact区間から外向きに表示することを明示する。準備CLIは順位・百分率を出さず、重なった区間を強制的に順位付けしない。

### 検算と科学call範囲

- m0..17のcoverageとX/Y、B′／Reference／既存registeredのV3行coverageを確認。実会計はB′／Referenceだけ。
- q/r/bits/Nの一致、9 roleの存在、重複・逆区間・負のwhole価格を拒否。
- role価格の合計が保存各軸RZ上下界とexact一致。
- B′総FractionとReference exact下界の一致、保存外向き総RZ端点による包含。
- 価格差／shot差の両基準について端点ごとの恒等式。融合補正を負の物理gateと誤解しない。
- **一つの固定campaign**で保存値分解・結果検算のみ。新しいRZ/CX/depth/T価格取得、sampling、phase_bound実行、N再算定、RPE再計算、回路build/compile、guard、合成は0。
- CPU/RAM/workerに恒常的上限を新設しない。保存Fractionの小さい処理は大きな並列科学jobを要さず、重複読取・計算を避ける。共有設定や他jobに変更しない。

人工算術4件PASSは[テストSOURCE](../../../tests/test_pr_reference_cost_structure_arithmetic.py)。これは実入力を使う上記検算が完了した証拠ではない。

### 実行方法（別途承認後のみ）

    python3 -B scripts/diagnose_pr_reference_cost_structure.py \
      --output artifacts/pr_reference_cost_structure/2026-11-08/D0_run01/RESULTS.json

指定は新しい独立checkout/outputで行い、旧成果物を上書きしない。実行に使う準備SOURCEのcommitを固定し、base654b8501の3入力を保つ。D0完了後は結果・SOURCEと検算を保存・公開しmandatory STOP。今回このcommandを実行していない。

### 判定・STOP

D0は資源改善のPASS/FAILを採択しない。「単価の節約をshot差が相殺した」「単価側にも増加が残る」「区間では判定不能」など、得た区間が支持する範囲のみ報告する。最大要因を無理に一つ選ばない。

入力／schedule／role／総計の不一致はSTOP。単価差もR側に不利、または候補を支持する追加余地がない場合も負結果として完了する。比率・丸め・価格を都合よく変えて救済しない。正常完了でも**native位相候補の数値診断や実装、他候補、旧STOP解除へ自動進行しない**。

## 3. Native位相候補：D0後に判断する条件付き案

[既存phase_bound](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/src/trottertracks/pr_reference_transfer_error.py)はSOURCE gate列のnaive減算とQiskit global_phaseのnormalization差を課金している。[pilot.vacuum_phase](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/src/trottertracks/pr_reference_interference_pilot.py)と[保存算術証拠](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE3_ALL_SUPPORT.json)を再利用できる。E0、gap、groundの古典記述を補正位相へ与えず、signed event／D／basisと実scalarから計算する規約を保持する。

候補は二つを混ぜずに考える。

- **同じSOURCEの証明だけを鋭くする場合：** 回路のvacuum P角を変えない。追加構造・全support境界が旧上界より鋭くなる根拠が必要。観測phase errorや期待event数だけではworst-case保証にならない。
- **位相生成の算術方式を変える場合：** 安定な和・analytic local phaseの集約等は既知数値技術の利用候補。元のrelative phaseとの一様誤差、period/remainder、actual inverse、finite-bit supportを新たに囲む必要がある。回路・算術SOURCEが変わるため、今回の固定条件を自動変更しない。

いずれも現時点で有効と認定せず、方式・finite domain・固定target／call範囲・native誤差規約を固める静的設計が先である。B′にも共通のscalar／算術改善が適用できる場合は同機会を与え、baselineだけ旧い保証へ固定しない。

Nを減らせる十分条件は、候補のρ_new、β_newについて保存式に戻し、
ρ_new² sin²β_new ≥ 4 log(1440)/N_goal を保証すること。N_goalや達成boundは今回採択・計算していない。例えば保存B′626は参照水準にはなるが、ReferenceにそのNを無料で代入してはいけない。

利益の因果経路は「追加位相disk減→半径／統計余裕改善→整数N減→全roleを含む総RZ実行回数減」。Nがceil境界を跨がなければRZの利益はない。phase証明を改善しただけで総Tの利益とは言えず、後者には全support word policy／domain／synthesis保証・実価格が別途必要。m7約1.87%を使った予測はしない。

この候補の数値phase評価・新SOURCE実装・native correctness・N再計算・総RZ再会計はD0範囲外。D0後に必要性が支持され、ユーザーが固定scopeへ明示許可した場合のみ検討する。m17 strict CT、D3、新guard、新word、43-key全面資格を同時に再開しない。

## 4. 重複確認と担当境界

[654b8501横断調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/REPORT.md)・[比較表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/COST_IMPROVEMENT_MATRIX.md)を再利用した。他研究の新しい進捗照会や全branch収集は繰り返していない。

| 既存項目 | 今回の扱い |
|---|---|
| controlled-RZZ／basis共有／inverse取消／D融合 | 新Aの採用済み構成。追加利益へ二重計上しない |
| joint isometry／diagonal fusion／h-r-α共同選択 | 他条件で実施済み。新A固定条件の純増分はUNKNOWNで、未実施とは断定しない |
| Reference／CSWAP／selector・cut等 | 既知SE型・既存研究あり。今回新gadgetや他Track計画を作らない |
| word coverage／strict guard/T frontier | 新Aで保存済み。native RZ差を解くために再合成しない |
| native vacuum位相・V6 | B′時点で方式変更は未実施と明記。現在の固定資料では、626/630と総RZ差を解消する新policyの資格証拠は未確認。一般の未実施・新規性の認定ではない |

今回の担当は新AのSERVER 2 Codexで、静的作業・準備と公開までを完了。**次担当はGPT／ユーザー**：まずD0保存値診断の要否・限定実行範囲を判断する。他Trackの担当や研究方針を変更する提案ではない。

**公開・remote SHA/内容照合後mandatory STOP。新しい科学数値診断、回路、RPE、合成へ進まない。**
