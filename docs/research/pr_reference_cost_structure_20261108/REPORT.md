# 新A Reference Interference：18roundコスト構造の静的分析

2026-11-08 JST／SERVER 2／主エージェントのみ。
基点：654b8501c1348cd0694daeefd19391df6b928354。保証系列の保存基点：7819602bdb7a5a0e6d4b43098144f681790f0651。
専用branch：codex/pr-reference-cost-structure-20261108。

**総RZ増加へつながる確認済みの経路は、強B′で小さくなった中央制御価格と、Referenceのm17だけに残る626対630 shots/axisの差である。共通basis費も追加shotごとに掛かる。ただし「何が最大の原因か」「単価差とshot差が何RZずつか」は、未実施の数値分解として残す。**

今回の主指標は固定成功条件下の全18round総native RZ。CX・native DAG depth・qubits・sizeを別々に保持し、最終目的の総Clifford+T T-countと区別する。新Aの静的調査・構造整理・限定診断の準備だけを実施した。他Trackへの作業指示・研究移管・進捗管理は行っていない。

## 1. 固定条件と既存の実績

Linear H4／1.00 Å／STO-3G／DF12、元H exact ground供給、準備費除外。Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law。Compiler A／Qiskit1.3／rz,sx,x,cx、B′4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum P・2 register・16 CSWAP/shot・8 resetを固定。D融合、basis共有、符号・scalar/event相対位相、X/Y wrapperを保つ。18roundのq/r/bits/N、energy1.5936e-4 Ha、failure.05、phase cap.4も変更していない。

| 既存結果 | B′／Reference比較の意味 |
|---|---|
| 期待総RZ | B′約3.571960961e12、Reference約[3.577340247,3.577340249]e12。増加率の保証下限0.15059756533579838% |
| 期待総CX | Reference削減率の保証下限4.681782236153505%。B′下界は全parity CXを捨てる保守的境界で、実行可能な価格と呼ばない |
| 期待総native DAG depth | Reference削減率の保証下限40.995952643300015%。exact期待depthや実機wall timeではない |
| 十分shots／qubits | X/Y計8598対8606、qubits9対17 |
| 全18round size／T／T-depth | 未資格／未取得。m7有限Tを代入しない |

[RZ/CX正本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_bprime_20261107/REPORT.md)・[有理価格／総計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json)・[depth](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_depth_v5_20261107/REPORT.md)・[inverse/fence補強](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_depth_v5_reinforcement_20261107/REPORT.md)。
固定保存Hermitian-H、理想native/reset/測定、独立finite bits、既存の算術モデル内の条件付き結果である。元の連続積分Hへの無条件包含・実機保証・全資源優越ではない。

## 2. 18round総RZ増加について分かったこと

### 2.1 保存証拠で確認できること

- B′は旧14-CX controlled-RZZの24 RZを2 RZへ変更済み。旧対照比のReference総RZ削減4.237674782178048%は、B′比較では消失した。これは[元報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_bprime_20261107/REPORT.md)の結果で、今回の新計算ではない。
- 保存Nはm0〜m16で両arm同じ、m17だけ626対630/axis。[保存行](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json)・[B′行](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json)を直接照合した。Referenceの位相生成diskを含めて算出した十分Nであり、同じ成功条件だから同じ整数Nとは限らない。
- SOURCEのbasis-run式とV3 role式は両armで同じRTE frame/inverse期待価格を使う。従って単価差では消えるこの共通費も、Nが異なるm17では消えない。これは記号的な会計恒等式であり、新しい費用額の推定ではない。
- ReferenceにはCSWAP、vacuum P、追加register/resetがあり、2 laneのcutでD境界共有の構造もB′と異なる。制御解除で減るRZ/CXと追加負担の両方を数える必要がある。
- r増加時はRTE eventとbasis-runの負担が変わる。qだけに比例させたり、r1のm7価格をr32へ移したりできない。

### 2.2 まだ判定していないこと

全18roundの差を「shot増加由来」「単価由来」に分けたRZ額、m17と前17roundの費用／差分の大小、role別総RZ寄与順位、同じNにそろえた仮想会計の符号は、新しい数値再解析が必要である。今回は積算・差引き・順位付けをしていない。

とくに「m17が全資源で最大」「増加は全部CSWAP」「shot差さえなくせばReferenceが総RZで勝つ」は未判定。SOURCEが示す正の増加経路と、定量的な最大要因を区別する。[式・保存入力・未判定表](COST_STRUCTURE.md)に分けて記載した。

## 3. m7有限T削減と全18round RZ増加は矛盾しない

[CT-4B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_ct4b_20261107/REPORT.md)ではm7/q128/r1の保存nonrare/rare各j0だけを、一様strict ε=1e-11で比較した。両arm199 shots/axisで、1-shot T+T†は約1.87%減。同じ有限例ではnative RZもReferenceの方が少ない。

これはcanonical law全体の期待価格ではない。全18roundの異なるq/r、m17のshot差、角度ごとの離散T価格を含めた結論でもない。RZ一つのT数は角度・word・precision・roleに依存する。有限例のROLE_T_COUNTSではD・中央回転の差がCSWAPの17,472 T/shotを上回るが、その内訳をm17へ外挿しない。

V3で確定任意角部分が減ったことも、全任意角・総T削減の資格ではない。D1/D2のguard/T診断と未資格roleも保持する。[保存D1/D2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_inverse_d_numeric_20261108/REPORT.md)。

## 4. 重複を除いた改善余地

**現時点で、固定N・固定回路の上に加える新たなRZ削減回路を、純粋な増分利益付きで推薦できる証拠はない。** D融合、basis共有、inverse cancellation、中央4-CX化、scalar/event集約は適用済み。共同isometry・対角fusion・h/r/α最適化は[横断整理](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/REPORT.md)の他条件の成果で、今回の新Aへ無条件移送しない。

条件付きで検討価値が残るのは、**native vacuum位相の算術上界と、m17の十分shot差の関係**である。[保存誤差報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_transfer_qualification_20261106/REPORT.md)は最終roundの追加位相上界0.00045377757276880644 radを記録し、naive和・phase正規化の寄与を分けている。これがraw disk／半径／統計余裕を経てNへ入るSOURCEも確認できる。

ただし、より良い算術／証明で630を下げられることも、それで総RZが逆転することも未検証。これはm17 strict Clifford+T guard改善とは別のnative候補であり、今回は実装・位相再評価・N変更をしない。共通joint global scalarを理由に、必要なcontrolled相対位相Pを削除する案でもない。

CSWAPの既知別分解などはCX候補になり得るが、固定Compiler A後の追加RZ利益と全費用の根拠が不足しているため、第一候補としての価格取得計画は作らない。既知gadgetや他研究の再実装も提案しない。

## 5. 必要最小限の実行準備

次の判断材料は、まず保存3入力だけのD0診断で取得できる。旧runnerの再実行・回路build・新価格取得・RPE再計算は不要。D0は改善policyの比較ではなく、固定された既存費用の分解である。

[準備script](../../../scripts/diagnose_pr_reference_cost_structure.py)は、TOTALS_18ROUND、STAGE4_RPE、V3 ANGLE_KEY_MULTIPLICITIESの固定Git blobだけを読む薄い算術adapter。既存SOURCEのprice/role定義を再実装せず、保存Fractionを積算する。旧accounting runnerは書込み・区間誤差/N再算定を伴うため今回の目的に適合しない。

[人工テスト](../../../tests/test_pr_reference_cost_structure_arithmetic.py)4件PASS。共通単価へのshot差課金、区間端点の相関、signed融合補正、欠損拒否を確認した。初期のgenerator消費不具合と、相関の出るanchorに関するテスト側の選択を修正した後の結果である。実科学入力へのrunner実行・新しい科学的PASSは0。旧テストの再実行もしていない。

[D0の固定入力・算術式・判定・STOP](NEXT_COST_REDUCTION_PLAN.md)を具体化した。今回、そのCLIを実行していない。人工テストは新Aの資源改善、実入力binding、RPE成功を証明しない。

## 6. 保全・終了

今回の追加は3文書と準備script・人工テストのみ。旧SOURCE/raw/result/grant/ledger/marker/STOP、回路、q/r/bits/N、PF/law/物理条件、compiler、他worktree・共有設定は変更しない。科学module import・QPY読込・simulation・compile・sampling・guard・合成・T価格評価・RPE再計算は0。

D3、basis全actual-adjoint化、composite guard、新word、m17 strict CT追加改善、43-key runtime全面資格、PF/law/H設計、18round再最適化は保留を維持する。正式科学レビュー・新規性採択も行わない。

**commit/push・remote SHA／tree／全追加ファイル本文照合後mandatory STOP。次担当はGPT／ユーザー。** D0の限定保存値再解析を行うか、その結果を見てnative位相候補をさらに設計するかを判断する。Codexによる追加科学処理は未開始。
