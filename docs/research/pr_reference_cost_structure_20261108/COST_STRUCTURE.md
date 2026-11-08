# 新A：固定18roundのコスト構造

2026-11-08。主指標は全18round総native RZ。
[総括](REPORT.md)／[次の限定診断](NEXT_COST_REDUCTION_PLAN.md)。
以下の保存数値は転記、式は固定SOURCEの読解または明記した会計恒等式である。新しい科学数値評価は行っていない。

## 1. 固定比較の定義

- Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給、状態準備費除外。
- Partial S2／LD3／h=.02／K2、ordinary diag(I,V)、canonical independent finite-bit law。
- Compiler A／Qiskit1.3／rz,sx,x,cx／全結合。B′4-CX中央controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP。
- D融合、basis共有、scalar/event相対位相、X/Y wrapper、q/r/bits/N、PF・alias・誤差diskを固定。
- εE=1.5936e-4 Ha、failure=.05、phase cap=.4 rad、各軸α=1/720。統計予算を新しく配分しない。

各arm sについて、G_s = Σ_(m=0..17,a=X/Y) N_s,m,a μ_s,m,a。
μは同じcanonical finite-bit lawによるfull no-prep interrogationの1-shot期待資源で、有限cohort平均・最適価格・実測shot費用とは別である。Referenceのμには安全側上下界を持つものがある。

RZ、CX、DAG depth、size、peak qubitは別資源。qubitにshotsを掛けて足さない。native RZをTへ単価一定で換算しない。

## 2. 保存round表と価格資格

[出典：TOTALS_18ROUND.rows](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json)と[STAGE4_RPE.rows](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json)。以下は保存フィールドの転記で、寄与の新しい積算・順位付けではない。Nは各軸共通の十分値。

| m | q | r | bits/draw | B′ N/axis | Reference N/axis |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 66 | 192 | 192 |
| 1 | 2 | 1 | 67 | 192 | 192 |
| 2 | 4 | 1 | 68 | 193 | 193 |
| 3 | 8 | 1 | 69 | 193 | 193 |
| 4 | 16 | 1 | 70 | 193 | 193 |
| 5 | 32 | 1 | 71 | 194 | 194 |
| 6 | 64 | 1 | 72 | 196 | 196 |
| 7 | 128 | 1 | 73 | 199 | 199 |
| 8 | 256 | 2 | 75 | 199 | 199 |
| 9 | 512 | 2 | 76 | 206 | 206 |
| 10 | 1024 | 4 | 78 | 206 | 206 |
| 11 | 2048 | 8 | 80 | 206 | 206 |
| 12 | 4096 | 8 | 81 | 221 | 221 |
| 13 | 8192 | 16 | 83 | 222 | 222 |
| 14 | 16384 | 16 | 84 | 256 | 256 |
| 15 | 32768 | 32 | 86 | 258 | 258 |
| 16 | 65536 | 32 | 87 | 347 | 347 |
| 17 | 131072 | 32 | 88 | 626 | 630 |

全18round RZ/CXは有限context・SOURCE多重度・正確なfloor-CDF有理確率による構造的会計。whole価格の有限cohortはq≤256/r≤2であり、r4/8/16/32のwhole compileを実施したことにはならない。短価格のaffine fitで長roundを埋めてもいない。

V5の全18round depthはmax-plus依存鎖と2 laneの期待max境界で別に資格化された。部品depthの単純和や平均長回路のdepthではない。全18round sizeは未資格のまま。[V5正本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_depth_v5/2026-11-07/DEPTH_18ROUND.json)。

保存B′/Reference結果のrowsにはN、価格の上下界、roleに必要な確率／多重度がある。一方、確認したTOTALS/V3/V5のschemaには全資源の共通寄与順位やshot/単価差の帰属欄はない。**最大寄与roundの判定・費用割合・前17round対m17の差分を今回生成しない。** SOURCEのq／qr増大だけを理由にm17支配を認定しない。

## 3. 単価とshot差の記号的分離

B=B′、R=Reference、δN=N_R−N_Bとする。同じm,aの厳密な単価について、どちらの恒等式も成立する。

    N_R μ_R − N_B μ_B
      = N_B(μ_R−μ_B) + δN μ_R        （Bのshot数を基準）
      = N_R(μ_R−μ_B) + δN μ_B        （Rのshot数を基準）

二つの「shot由来額」は基準が違うので一般に同じではない。片方を唯一の因果分解として扱わない。上下界を使う場合は同じμを共有する項の端点依存を保持する。異なる項の極値を独立の実現価格のように足して、厳密な差と呼ばない。

保存Nではm0〜m16のδNは0、m17だけ正。このため、shot差項の非零supportはm17に限る。ただし単価差項は全roundにあり、符号・大きさは別問題である。新しい重み付き和なしに総RZ増を丸ごとshot差へ帰属させない。

両armのRTE basis単価が共通F_m,aなら、そのroleの総差はΣ δN F_m,a。単価比較でframeが消えても、総費用でこの項が残る。これは「basis共有が未実装」という意味ではなく、既に共有済みの同じ回路を余分に繰り返す費用である。

### 保存総計が言っていること

[採用TOTALS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json)：

- B′期待総RZ：[3571960960634.802, 3571960960634.8027]。
- Reference期待総RZ：[3577340246876.2627, 3577340249201.294]。
- Reference下界がB′上界より大きく、増加保証下限5379286241.461207 RZ、0.15059756533579838%。
- CXはB′下界とReference上界で削減を示す。下界を実行可能baseline価格とみなさない。

B′のRZ exact fractionを外向き浮動端点で表示するための幅と、Reference product-Z cancellationを未特定として残す価格幅を区別する。既存区間は価格CIやMC SEではない。

## 4. SOURCE上のroleと価格の因果構造

p₂は実finite-bit order2確率、p_bはbasis bへの成分確率の和、sは回転成分のsupport数の期待値。いずれも旧SOURCEで算出・保存された量で、今回は再算定しない。

| Role | B′とReferenceの違い／RZとCXの作用 | 既存適用済み最適化・未判定 |
|---|---|---|
| D未融合＋D境界補正 | B′は非制御left・directional rightとcontrolled boundary。ReferenceはF†/Gの非制御D。対角controlには差があるがGaussian basisを全て消す構成ではない | 両側とも融合済み。q≥2のsource D呼出しはB側7q+1、R側7q+2、境界融合はq−1対q−2。q1は特別。これはnative価格差ではなく[保存SOURCE構造](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE1_STRUCTURE.json)。D roleの正味RZ差は再集計待ち |
| RTE basis／inverse | 同じbasis registry・run共有を使う。F actual SOURCE inverseによるrun反転もrun数は維持。単価の共通項 | basis共有・隣接inverse取消は適用済み。qΣ_b c_b[p_b+(r(1+2p₂)−1)p_b(1−p_b)]という旧期待式を再利用。c_bは実frame/inverse価格。独立な新利益として再提案しない |
| 中央rotation | B′はZ/ZZとも2 RZ/event、Referenceは1 RZ/event。CXはB′のZで2、ZZで4（相殺前）、ReferenceはZで0、ZZで2 | B′の4-CX化で旧ZZの22 RZ/event追加負担は既に除去。B側parity CXは相殺可能で、価格は安全な上下界。新しい「14→4」改善を計上しない |
| Event product | B′はcontrolled CZ、ReferenceはZ。product数はorder2とsupportに依存。Reference product-Zはsource取消しがあり得る | 保存RZ式はReference全product-Zを捨てる下界と残す上界を保持。観測取消率を長rへ外挿しない。CXのcontrolled product負担もB側にある |
| Event phase | B′にはorder2を含むoccurrenceのP。Referenceでは各lane scalar/global phaseとvacuum補正へ入る | B側期待P数q[1−(1−p₂)^r]。event符号・ordered productは変更しない。位相を省略して価格を減らすことは不可 |
| Scalar／vacuum P | B′は物理scalar P、Referenceはtrajectory vacuum P。保存モデルでは各1 RZ/shot、角度と生成算術は異なる | scalar集約済み。Reference Pは相対位相補正なので、joint global scalarの不変性を理由に捨てない。RZ個数が同じでも位相生成誤差はNへ影響する |
| CSWAP | B′なし、Reference16 CSWAP/shot。保存8-CX分解の128 CX/shotとRZ費を含む | 既存回路内の費用。V5 cascade span72を2 cascadeへ使い、局所depthを単純直列加算していない。別分解のRZ増分は未資格 |
| X/Y wrapper | 両側H×2、YのみSdg、測定1。RZ価格は保存contextで扱う | wrapper省略・X/Yの無条件同価格化はしない。CT4の有限Tで0でも全role無料ではない |
| Reference reset／register | 追加8qubitsとreset8/shot。これら自体はRZ gateではないがwidth／DAG／sizeに関わる | ideal resetとnative depth規約を維持。reset depth1は初期Hと並列、8層として加算しない。T／wall timeへの未知換算なし |

出典：[RZ期待式](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/src/trottertracks/pr_reference_transfer_accounting.py)・[B′差分／下界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/src/trottertracks/pr_reference_bprime_accounting.py)・[CX式](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/src/trottertracks/pr_reference_transfer_cost_bounds.py)・[V3 expected_parts/dparts](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/src/trottertracks/pr_reference_rotation_v3.py)。

### 既存式から読む1-shot RZ差

D_B(q)を融合済みB側DのRZ価格、D_R(q)をReferenceのD価格、C_swapをReference16 CSWAPのRZ価格とする。SOURCEの既存関係を読めば、Reference上界との差は次の形である。

    μ_R,upper − μ_B
      = [D_R(q)+C_swap−D_B(q)]
        − qr
        − 3 × E[product supportの総数]
        − q[1−(1−p₂)^r]

ここでE[product support総数]=2qr p₂ sは元の会計式。Reference下界は、この上界からさらに全product-Z RZを除く。式中の数値3は既存validate_reference_rz_lowerのliteral価格差であり、新たにgateを合成・価格評価した値ではない。

共通basis、wrapper、各1個のscalar/vacuum RZはこの単価差で相殺する。**総会計ではN差により再び費用差を持つ。** D＋CSWAPの項と中央／product／phaseの節約のどちらが勝つかを、今回の保存入力で新しく評価してはいない。有限m7での勝利を全roundへ移すことはできない。

## 5. CX／depth削減とRZ増加が両立する理由

CXはcentral control／productの削減により変わる。RZはbasis・D・中央angle・固定phase等を別の価格で数えるため、CXと比例しない。さらに、Referenceは追加registerの2 laneを並列化できるためdepthに利益があり得るが、parallel実行は総gate個数を半分にしない。V5はrandom lane costの期待max上界を使い、平均回路のdepthを使っていない。

Nを掛ける段階でも各資源の1-shot価格が異なるため、同じ626対630の増加が各資源へ及ぼす相対効果は異なる。従って全18round CX/depthが減りRZが増える既存不等式は整合する。これからsize・全T・FT space-timeまでの優越を推論しない。

## 6. Native誤差・phaseとshot差

[transferのphase_bound/qualify](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/src/trottertracks/pr_reference_transfer_error.py)・[error_and_shots](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/src/trottertracks/pr_reference_bprime_accounting.py)：

- 共通SOURCE／finite-RTE operator disk、event phase、finite-bit proposalをraw信号単位で一度だけ継承する。
- B′では旧compiler diskを新kernelのものへ置換。Referenceはnative joint誤差の2√2倍にvacuum位相生成diskを加える。旧compilerを再加算しない。
- 半径下界をdiskで下げ、PF phaseと偏角上界をphase capから引いて統計余裕βを得る。
- 保存十分式はε=ρ sin(β)/√2、N=ceil(2 log(1440)/ε²)。今回再評価しない。

保存m17のReference追加位相生成上界は0.00045377757276880644 rad。内訳はnaive和0.00045372585145842106、normalization差5.172131038537467e-08。元の[報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_transfer_qualification_20261106/REPORT.md)にある表示値を引用した。

同roundのReference raw半径下界0.5722335555306725、β下界0.38519464269240195 rad。B′は0.5726873297263416、0.38598699711850315 rad。[B′結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_bprime_20261107/REPORT.md)。この違いを「真の信号が悪化した測定値」とせず、使用する十分上界／下界と区別する。

位相上界の変更でNがどう変わるかは、ceil・他disk・予算も含む新しい科学診断が必要。真空phaseを全部無料／誤差0とした仮想値を、構成済みpolicyの成功保証にしない。

## 7. 証拠・SOURCE・未判定事項

閲覧snapshotは全て654b8501。元の実行SOURCEは各SOURCE記録に従い、今回の文書commitへ付け替えない。

| 入力／SOURCE | 確認できる内容／再利用 |
|---|---|
| [B′ SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_bprime/2026-11-07/SOURCE.json)、EXECUTED_ACCOUNTING_SOURCE.py | 実行時scriptと入力・実行順。RPE再会計やcompileを伴う旧runnerは再実行しない |
| [TOTALS_18ROUND](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json)、ROUNDS.tsv／ERROR_RPE | 各36セルのRZ/CX価格境界・N・raw radius／β・総計。B′shotとReference errorは別field |
| [STAGE4_RPE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json)／STAGE4_ROUNDS | Reference十分Nとraw条件。ここを旧14-CX価格baselineとB′の価格に混同しない |
| [V3保存role](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_rotation_v3/2026-11-07/ANGLE_KEY_MULTIPLICITIES.json.xz) | m/axis/arm、bits/N、9 roleのexact Fraction多重度・token。native総RZをrole別に集計するなら、保存済みpartsを使う。分類／T価格の再取得は不要 |
| [V5 DEPTH_18ROUND](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/artifacts/pr_reference_depth_v5/2026-11-07/DEPTH_18ROUND.json)／ROUND_DEPTHS | native DAGの上下界・lane penalty。CXRZ単価からdepthを再推定しない |
| [CT4B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_ct4b_20261107/REPORT.md)／ROLE_T_COUNTS | m7固定2trajectoryのwordごとのTとrole差。m17全supportの期待Tではない |
| [D1/D2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_reference_inverse_d_numeric_20261108/REPORT.md) | strict guard/T frontier再選択の負結果と費用増。native RZの値を改善したものではない |

未判定：総RZ差の数量的帰属、round／role順位、同じNでの価格差総和、native位相候補の達成可能bound／N／純利益、全18roundsize、全support期待T・総T。domain、central、emitter、shell、terminal等のCT未資格も不変。

この調査で保存価格／Nを読み取ることと、旧Fractionを新たに加算・再ランキングすることを分けた。後者は[NEXT_COST_REDUCTION_PLAN](NEXT_COST_REDUCTION_PLAN.md)のD0として未実行にしている。
