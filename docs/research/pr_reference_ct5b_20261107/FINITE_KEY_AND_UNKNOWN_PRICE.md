# CT-5B：有限key構造・既知価格・未資格項

2026-11-07 JST。固定q128/r1、K2、strict epsilon1e-11、compiler A。以下は保存行の照合と記号的条件整理であり、新key catalogue・word価格・極値・期待値を生成していない。[compiler条件C](STRUCTURAL_CERTIFICATES.md)を前提にする。

## 1. RTE中央：有限な4 positive keyで表せる

[rte._make_event/unsigned_rotation_angle](../../../src/trotterlib/rte.py)、[DF builderの中央作用](../../../src/trotterlib/df_rte_qiskit.py)、[V3 rotation](../../../src/trottertracks/pr_reference_rotation_v3.py)を照合した。

order o∈{0,2}について、既存SOURCEが計算するa_o=atan(tau/(o+1))を記号とし、component係数符号s∈{−1,+1}を含む実native full-angleをtheta=fl(2 s a_o)とする。tauとatanを今回再計算していない。B′は±theta/2のRZ2個、Referenceは±thetaのRZ1個。Z/ZZの違いはparity CXを変えるが、Cの下ではこの中央RZ価格を変えない。ReferenceのF†ではSOURCE角の符号が変わり得る。

採用wordをpositive keyからactual adjointで返す場合に限りt(−theta)=t(theta)。これはSOURCE inverseを別compileしたnative列の同一性とは異なる。固定r1の正常なbinary角の2倍/半分と符号はSOURCEの演算規則を保持し、隣接floatや理想角へ寄せない。

以下のhexと価格は既存[V2_INPUTS_CONSTANTS](../../../artifacts/pr_reference_bprime/2026-11-07/V2_INPUTS_CONSTANTS.json)および[CT-4B inventory](../../../artifacts/pr_reference_ct4/2026-11-07/KEY_REUSE_AND_SYNTHESIS.json)にある行の引用。再合成・再guard・新価格計算は0。

| order | 用途 | positive binary key | 保存e=-11 T+T† |
|---|---|---|---:|
| 0 | Ref full-angle | 0x1.7e3c19bcd3089p-6 | 118 |
| 0 | B′ half-angle | 0x1.7e3c19bcd3089p-7 | 112 |
| 2 | Ref full-angle | 0x1.fdaaba4d44069p-8 | 118 |
| 2 | B′ half-angle | 0x1.fdaaba4d44069p-9 | 116 |

従って1 eventの中央差を `d_o=2 t(half_o)−t(full_o)` と有限価格記号で表せる。d_0/d_2の数値、min/max、q倍総額は今回新しく評価しない。SOURCEの全registered labelに対するこの包含はcanonical73-bit lawのsupportを覆うsupersetとして使え、個々の符号/labelの到達確率を新しく認定するものではない。

## 2. Product：価格0と物理出現を分ける

order0ではproductなし、order2では2 product application。各supportはZまたはZZなので、1 eventあたりZ/CZ source gate数は高々4、q eventでは高々4q。これはSOURCEからの記号的個数境界で、新しいtrajectory極値計算ではない。

- ReferenceのZ保存templateはliteral RZ(float(pi))。
- B′のCZ保存templateはRZ(float(pi/2))とClifford SX/CXからなる。
- H/Sdg wrapperも同じ±float(pi/2) keyを使う。

保存4 contextのtemplate配列を照合し、[CT-4B inventory](../../../artifacts/pr_reference_ct4/2026-11-07/KEY_REUSE_AND_SYNTHESIS.json)の `0x1.921fb54442d18p+0`（pi/2）と `0x1.921fb54442d18p+1`（pi）はe=-11でT=0のstrict recordを持つ。負角はそのactual adjoint。

Cと同じrecordの採用を条件に、全supportでproduct由来Tは両armとも0、wrapper価格差も0となる。Referenceのsource product-Z取消しはRZ個数を変え得るが、T=0の寄与だけを変えるのでT差への未知残差にはしなくてよい。**gateや誤差が0という意味ではない。** 実binary角をexact Cliffordに昇格させたわけでもない。

## 3. B′ event位相：有限supersetに1つの同一精度欠損

固定SOURCEはevent符号位相を先に集め、productsを処理して負係数ごとにfloat(pi)を逐次加算する。r1/K2・identityのない保存tailでは、order0はk=0、order2は `k=1+#negative products`。従ってb_0=0、b_(k+1)=fl(b_k+float(pi))、k≤3のsupersetで覆える。b_kは位相値であり、order2確率p_2とは別の記号である。

以下は[CT-0 PHASE_SUPPORT](../../../artifacts/pr_reference_ct0/2026-11-07/PHASE_SUPPORT.json)のr1保存行と、CT-4B同一eの照合結果。

| 保存superset key | e=-11 record | 判定 |
|---|---|---|
| 0x0.0p+0 | CT-4BにT=0 record | 角0をSOURCEが省略する場合も同じ0寄与 |
| 0x1.921fb54442d18p+1（b1） | T=0 recordあり | strict採用可能な保存価格 |
| 0x1.921fb54442d18p+2（b2） | CT-0 mappingのrecordsは空、CT-4Bにもなし | **未取得。T=0としない** |
| 0x1.2d97c7f3321d2p+3（b3） | T=0 recordあり | strict採用可能な保存価格 |

この表は新しい到達key catalogueではなく、既存supersetの行を引用したもの。b2がcanonical supportで到達するかは、order2と正負product labelの非ゼロ73-bit cell等の条件に依存する。2保存trajectoryで見なかったことを非到達証明にしない。k*pi単一乗算やparityへ書き換えず、真の2piとbinary b2も同一視しない。

重要な区別：B′ event価格E_Bは差 `T_B−T_R` に**正符号**で入る。実行可能なwordがある場合、そのT価格は非負なので、差の安全な下界では `E_B≥0` と捨てられる。b2のT上界を得ることは、その下界のためには必須でない。一方、b2が到達するならB′を全supportで実行するにはword/strict保証が要り、非負性だけで取得欠損は埋まらない。

## 4. Frameと固定項のcoverage

保存CT-0 KEY_COVERAGEのm7/RTE_basis_framesにあるkeyを、保存CT-4Bの同一e inventoryへ直接照合した。全域coverageとはなっていない。例えば `0x1.08ba32d420f2cp+2` は、CT-0でNUMERIC_KEY_MISSING、CT-4Bにもない既存の保存keyである。

frameは両arm共通なので、共通価格記号の係数差は0となり、その未知価格の数値上界は差の証明に不要となり得る。ただし、**共通frameを実際に実装する有限cacheの完了とstrict guardは必要**。CT-4Bの299件を全m7の全keyと呼ばない。今回、欠損リストを新規生成/合成したり、追加Tを積算したりしていない。

D、固定scalar P、CSWAP、wrapperについては、[構造条件](STRUCTURAL_CERTIFICATES.md)の下で固定literal台帳を保存anchorへ対応させられる。D全support価格をsource固定だけから自動認定せず、context/record条件付きで使う。CSWAPの保存17,472、B′ scalarの保存118、wrapperの保存0はCT-4Bから引用し、再価格していない。

## 5. Vacuum：最後まで別の未知項

Referenceのvacuum Pは1shotに1個。保存F/Gの逐次vacuum_phase、global remainder、最後のfl(Fphase+Gphase)を保つ。q128/r1/m7の[CT-0保存境界](../../../artifacts/pr_reference_ct0/2026-11-07/PHASE_SUPPORT.json)は、実binary角の絶対値上界82.49247654978845 rad。保存のIEEE/remainderモデルに条件付きの引用であり、新しい境界評価ではない。

| 問い | 状態 |
|---|---|
| 位相/誤差域を固定SOURCEの有限recurrenceで表す | 既存資格を維持 |
| 到達する全binary角の集合 | 未列挙。上の区間全体と同一ではない |
| 全到達角でstrict wordを取得・有限終了するpolicy | CT-2/3Aの未資格を維持 |
| 実採用vacuum wordの一様T cap | 未資格 |
| 期待vacuum価格または中央差との結合境界 | 未資格 |
| CT-4Bの保存vacuum価格 | finite nonrare120、rare118だけ。上界や期待値ではない |

角度の有界性、supportの有限性、少数wordの最大値は、実policyの取得/停止/価格保証ではない。2pi正規化・W破棄・projective phase fit・新vacuum算法を導入しない。合成後に新controlを付けない。

## 6. 対応表と終了

| 部分 | 構造閉包 | 既存価格 | 残るもの |
|---|---|---|---|
| D/CSWAP/scalar/wrapper | C条件下で固定 | CT-4B anchor | compiler/record条件の維持 |
| RTE frame | 共通有限SOURCE pair | finite coverage不足 | 有限word取得。差の数値価格は取消し可能 |
| RTE中央 | 4 positive key | 同一eの保存価格あり | C・literal符号/half/full規約 |
| product | 固定Z/CZ template | 保存strict T=0 | SOURCE取消しとT=0を誤差0と混同しない |
| B′ event | 保存k≤3 superset | b2欠損 | 到達性/実行資格。差の下界なら非負性を利用可 |
| Ref vacuum | SOURCE recurrence・域のみ | 2 finite例だけ | 取得/停止と、全体差に必要なcap/期待境界 |

新規計算なしでの静的分類を完了した。新合成・再guard・数値極値・期待Tの計算には進まず、**mandatory STOP**。
