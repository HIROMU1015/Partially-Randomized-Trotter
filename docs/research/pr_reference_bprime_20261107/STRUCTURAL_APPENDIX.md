# V2補遺：有限context、相殺境界、算術定数

以下は固定SOURCEに対する著者の構造論証と実装検査の記録であり、形式証明・独立レビューを新たに実施した結果ではない。基準 `da9fceb944dae27f397e48734be30c6864e15f44` の旧REPORT・SOURCEを変更せず補足する。

## compilerの適用範囲

compiler Aは初段の固定InverseCancellation、UnrollCustomDefinitions、BasisTranslator、最終InverseCancellation(CX→X)をそれぞれ一度適用する。全結合、calibration/Targetなし、SessionEquivalenceLibraryに新規等価則を追加しない、同名異義gateなし、controlはlower前に存在することが前提である。数値的commutation、RZ角度cutoff、1qubit合成、反復最適化はない。

Qiskit 1.3.0の一次SOURCEを確認した。

- [UnrollCustomDefinitions](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/basis/unroll_custom_definitions.py)：targetまたはequivalence entryのあるgateを保持し、その他をdefinitionから再帰展開する。
- [basis_search.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/basis_search.rs)：equivalence graphをDijkstra探索し、source gate族を満たすまで規則を探索する。
- [compose_transforms.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/compose_transforms.rs)：(gate名,qubit数)とパラメータplaceholderに対し変換を合成する。
- [mod.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/mod.rs)：全結合・Targetなしではglobal instruction mapを作り、topological nodeごとにパラメータを束縛して適用する。
- [inverse_cancellation.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs)：gate名ごとのrunを収集し、同じordered qargsの隣接部分を偶奇で消去する。gateごとのrun収集は一度で、最終passはCXの後にXを処理し、CXへ戻らない。

固定context内の同名・同qubit数・同parameter構造のgateには同じ局所テンプレートが適用される。wire labelや全回路のqubit数によるrouting変化はない。任意のcompiler/versionへの不変性は主張しない。角度に対するCX数の不変性と、特定角度でのoperator誤差資格は別問題である。

## 常在族と任意族

測定・barrierを除くpost-unroll集合を列挙する。

| arm | 常在族 | 任意族 | context数 |
|---|---|---|---:|
| 登録14-CX | p, xx_plus_yy, rz, rzz, cx, h, crz | ccx（ZZ回転あり）, cz（productあり）, sdg（Y） | 8 |
| B′ | p, xx_plus_yy, rz, rzz, cx, h, crz | cz, sdg | 4 |
| reference | p, xx_plus_yy, rz, rzz, h, cswap | z（productあり）, sdg | 4 |

q=1でもB′の各occurrenceには少なくとも一つCRZまたはparity–CRZがある。q≥2のboundary blockにもCRZがある。Dには全4 blockと非零のbasis/対角族が固定して存在する。tailのidentity成分はなく、1-site Zと2-site ZZの両方を含む。q/r/order/符号はこれらの多重度を変えるが、族の種類を増やさない。旧SOURCEではCRZZがUnroll時にCCX/CRZへ分かれるが、B′にはこのCCXがない。

B′の全4 contextで、継承する局所loweringは旧対応contextとbyte構造一致した。24実角のB′ kernelは同じ4 CX・2 RZであった。128 whole compileのcontext包含はこの列挙の整合検査であり、列挙そのものの代わりではない。

## 旧fence補題と初段pass

chunkの露出CX prefixとは、そのCXが自分の全operand上でchunk先頭にあることをいう。最終CX消去pass直前、後続chunkにそのようなCXがなければ、同じordered qargsのCX runが前のchunkから入ることはできない。runは削除前に一度収集されるので、当該passのchunkを跨ぐCX削除は起きない。X処理で新たな隣接ができてもCXを再処理しない。従ってCX数を局所chunk価格から接続できる。RZは最終passの削除対象ではない。

この結論には初段SOURCE-level passがchunkの前提を破らない条件も必要である。旧baselineのframeはRZ/XX+YY、中央はCRZ/CRZZ/CZ/Pであり、初段の自己逆gateの対象ではない。Dのdirectional CXは対応D chunk内で処理され、chunk外とはframeで隔てられる。wrapperのH/Sdgもancilla上のP/CXで隔てられる。D chunk内の相殺は局所価格に既に含む。barrierなしのwhole回路への接続はこれらの条件によるもので、catalogueのbarrierそのものに依存しない。

旧のprefix検査と今回B′ catalogueのframe/D両側検査を区別する。今回の両側検査ではDと全frame/inverseの露出CX prefix/suffixが空であることを最終消去前に確認した。

## B′で新たに必要な価格境界

B′ kernelのnative列は、control c、system a,bについて

`CX(a,b), RZ_b(theta/2), CX(c,b), RZ_b(-theta/2), CX(c,b), CX(a,b)`。

thetaは旧RTE nodeから読み取った実角である。両端のparity CXは露出するため、旧の価格加法性をそのまま適用できない。初段source passでも同一supportのparity CX対が消える。異なるsupport、CRZ/CZ/Pを挟む境界も実例で確認した。

安全な下界では相殺集合の一致を要求しない。各kernelの中央CRZの2 CXは、最初と中間のRZにより同じqargsの隣接runから保護される。末尾のcontrol CXとparity CXはqargsが異なる。CZ kernelはtargetの非CX gateで始まる。D/frameの両側fenceによりparity CXとD/frame内部CXの跨ぎ相殺は起きない。従ってchunk外へ露出して削除され得る追加CXは、ZZ回転ごとの2 parity CXに限られる。

固定14-CX経路の加法価格をC14、ZZ回転数をZとすると、全pathで

`C14 − 12 Z ≤ C_B′ ≤ C14 − 10 Z`。

下界はparity CXをすべて無償として捨てた保守的な境界であり、実現可能な回路価格ではない。上界は残る全parity CXを数える。単一Z rotationは従来のCRZを保つ。全pathの相殺がreferenceと一致するという仮説は、今回の保証に使用しない。

RZは初段・最終passの対象ではない。B′の置換はZZ回転あたり24→2 RZなので `RZ_B′ = RZ14 −22Z`。contextごとの局所min/maxを用いる実装では、これを下界・上界の両方から引く。referenceではsourceのproduct-Z消去があり得るが、D/frame、rotation、CSWAP、P、wrapperのRZ数は残る。全product-ZのRZを捨てればreferenceの安全なRZ下界になる。最終CX/X消去はこれらRZを変えない。

finite-bit lawでp_ZZを成分ZZ確率、p₂をorder2確率、sを期待supportサイズとすると、parity消去前のB′期待CXは `D_B + frames + qr(2+2p_ZZ+2p₂s)`、安全な下界はここから `2qr p_ZZ` を引く。reference上界は `D_R + frames +2qr p_ZZ`。basis-run期待値は既存の厳密式 `Σ_b c_b[p_b+(r(1+2p₂)−1)p_b(1−p_b)]` をq倍する。有限bitのfloor-CDF有理確率を使用し、保存価格標本の平均を代入しない。

128軸で得た実価格、source/最終passの削除node ID・wire、上下界を保存する。有限cohortで相殺数が一致しても、その事実から全rの相殺集合一致を証明したとは扱わない。全round価格の根拠は上の安全な構造境界である。

## 336・49・24・20の確認

固定入力から各D blockのframe RZ数8、block0対角8、block1–3対角RZ/RZZ合計36を抽出した。frameと逆を含めるとblock0は24、他は52。片側n stepのD部分は `24(n+1)+2n(52+52+52)=336n+24`。tailのframe RZ最大8、eventあたりbasis-run最大3より `3×2×8+1=49`。RZ/RZZ subtraction数上界は従来通り `(336+49r)n+24`。

boundary-optimized laneの結果回路にはD更新が最大7n+1、RTE composeがn、scalar集約が1なので、global-phase更新は8n+2≤20n。RTE builder内部の位相設定n回は、従来の別項で評価する。temporary additive circuit内のDを、結果laneへ二重に加算しない。q=1の固定分割でも同じ20n上界で覆う。

構造式だけでなく、既存q4/r1/rare/j0のF/Gを生成し、global_phase setterを当該process内だけで観測してfinallyで復元した。n=2の各laneではRZ/RZZ数730≤794、constructorを除くphase更新18≤40だった。q=1の既存pathでも小端点を観測した。これは新しいround条件・長r価格取得ではない。旧ハードコード定数を増やしていない。

lambda_prepとlambda_tailのbinary64値は、ともに `0x1.2aa26bcd0dbf3p-1`。全保存r/orderでSOURCE由来の実角と旧区間評価への代入角がhex一致した。負号と2倍化はbinary scalingである。B′ kernelのnative誤差0はこの実角を保つCRZ解析恒等式に基づく。旧CRZZ区間・新B′の解析資格・数値correctnessを混同しない。

platform、Python build、glibc、Qiskit wheel、float情報は環境JSONに記録した。binary64最近接丸め・remainderの従来モデルを採用し、wheel全体やlibmを形式検証したとは主張しない。位相計算法、補償和、precisionは変更していない。
