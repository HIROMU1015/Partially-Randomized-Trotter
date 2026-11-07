# CT-2：全support価格への適用条件

基準 `6ac8e1aa3104d8563188612410c1a7533ec47072`、2026-11-07 JST。**静的SOURCE対応と条件式の整理のみ**。全trajectory列挙、新catalogue、合成、価格/shot再計算は行っていない。[理論と固定backendの差](STRICT_SYNTHESIS_BOUNDS.md)、[REPORT](REPORT.md)。

## A：全support phaseの定義

保存18roundのcanonical independent finite-bit lawが選ぶevent/application列をomegaとする。角度集合は、保存H4/DF12・Partial S2/LD3/h=.02/K2と、次の固定SOURCEが定義する**実binary64値の集合**である。

| SOURCE | 固定した意味 |
|---|---|
| [transfer.assemble](../../../src/trottertracks/pr_reference_transfer_qualification.py) | q1ではG=左half D＋scalar＋uncontrolled RTE、F=右half D。q≥2では保存の等分split・既存D境界融合の前/後半。splitを最適化しない |
| [pilot.vacuum_phase/reference_core](../../../src/trottertracks/pr_reference_interference_pilot.py) | 各laneの保存global_phaseを初期値にし、命令順にRZ/RZZ角の半分を減算。二つの結果を加え、Fのactual SOURCE inverseとGを別registerに置いたjointのancilla Pへ与える |
| [df_rte_qiskit](../../../src/trotterlib/df_rte_qiskit.py) | event符号を先に集め、products→rotationの順で処理。basisの共有と切替順、負productのscalar、rotationの符号を保持 |
| [transfer_error.phase_bound](../../../src/trottertracks/pr_reference_transfer_error.py) | naive加算・remainderと真のperiodとの差を、保存binary64/192-bit境界モデルで評価 |
| [compiler A](../../../src/trottertracks/track_c_compiler_a.py) | P(theta)→RZ(theta)は保存の実角を保つ。既にcontrolledなjoint全体の共通phase規約。新controlを付けない |

XXPlusYY/Z/P/idの真空作用を、RZ/RZZの逐次減算と混同しない。lane回路のglobal_phaseはQiskitの実remainder経路を通り、全phaseは解析的countだけで決まるとは限らない。最終thetaはF/Gの実順序に対応するfloat和であり、代表cohortの値に置き換えない。

CT-0の[既存phase結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f5f16fad64748236b0160284dc6c3017974b0132/artifacts/pr_reference_ct0/2026-11-07/PHASE_SUPPORT.json)にあるm17/q131072/r32の絶対値上界 **32751.887444214277 rad**は保存値の引用。`2*exact_prefix_upper + naive_accumulation_upper`でactual binary角を覆う。その保証は保存のround-to-nearest、overflow/underflowなし、固定remainder誤差モデルに条件付きで、今回再評価していない。この上端が実trajectoryで到達するとも主張しない。

有限alphabet・有限長・有限binary状態から位相域を表現できるというCT-0資格を維持する。ただし、有限性と有界性だけでは取得word・T上界が得られない。新しいremainder、2pi key同一視、vacuum位相生成法は導入しない。

## B′との対応

B′ event Pは、1 occurrenceについて実float(pi)を最大3r回、順次加算した有限supersetで扱える。CT-0のr別supersetをそのまま参照し、k*pi/parityへ置換しない。q occurrenceなら複数のevent Pがあり、Referenceのvacuum補正1個と同じ出現数にしない。supersetには非到達keyが含まれる可能性がある。

B′には、controlled Dの実half-angle・ancilla相対scalar、4-CX ZZ/CRZ中央、product CZ、wrapperもある。ReferenceにもD/frame/inverse・RTE中央/product・16 CSWAP・wrapperがある。いずれもcompiler後のliteral RZが合成target。未取得部分を無料にせず、sourceの制御付きthetaへ非制御RZ(theta)の価格を直接貼らない。

## 現資産を価格として使える範囲

| 区分 | 再利用できるもの | 残る条件 |
|---|---|---|
| 取得済みnumeric key | C2-B/CT-1のword、実T/T†、spec/hash、保存strict guard | 同一binary key・precision/guard規約・採用policy。CT-0の4 legacy recordのguard方式未資格は不変更 |
| missing numeric key | V3/CT-0のkey/role情報 | word価格は未取得。旧CT-0のmissing件数を、CT-1後の残数として再利用しない。今回は再照合・再集計していない |
| B′ finite phase superset | 保存CT-0の加算規則/key表 | 必要precisionの取得/一様上界。supersetの全key取得済みとはしない |
| Reference vacuum | 有限recurrence・保存全support角度境界、CT-1の2個の有限trajectory価格 | 全域で完了するword policyと明示価格上界が未資格 |
| 一様理論上界 | 厳密な前提が満たされれば、key全列挙に依存しない価格証明にできる | 現backendの実価格に移せる証明、又は別構成法の採択と実装資格 |

CT-1の268 word/epsilon1e-9/2 trajectoryの価格を、全supportや別epsilonの最大値にしない。既存guardの存在は対象外keyのwordの存在・取得保証にはならない。

## C：将来の会計式（未実行）

以下はCodexの条件整理。固定policyが、全到達角度にstrict errorを満たすwordを返し、class bごとに明示上界F_bを持つと仮定する。native後に新しいClifford+T融合/最適化を加えない場合、word T数は物理出現に加算できる。native側のpath依存fusion/cancellationは既存SOURCEと個数の資格に含める。

既知の採用word price c_kと、未知keyを覆う上界F_bを分けるなら、

\[
 \mathbb E[T_{a,m,x}]
 \le \sum_{k\in K}\mathbb E[n_{a,m,x,k}]c_k
       +\sum_{b\in U}\mathbb E[n_{a,m,x,b}]F_b.
\]

有限取得wordと別理論の上界をminで選ぶ場合にも、**実行するword policyがその選択を本当に構成すること**が必要。最小wordの存在上界を、既存backendの返却word価格に代用しない。理論式は単なる経験的native価格providerではない。

Reference vacuum Pは1shotにつき1個なので、全域で有効なF_vacが得られれば、この部分だけは

\[
 G^{\rm vac}_{T,\rm ref}\le\sum_{m,x}N_{m,x}F_{\rm vac}(\epsilon_{m,x})
\]

と接続できる。これはReference全回路の価格ではなく、他roleを落とさない。大きい角度でもstrictに全域を覆う証明なら有界角度keyを逐次列挙する必要はないが、backendの個別成功例ではこの前提を満たさない。

期待multiplicityは有理数として扱い、整数T価格のceil/gcd格子を期待Tへ直接移さない。線形会計には出現数とword価格の対応が必要で、独立性を勝手に仮定しない。一方、一様capが各出現をpointwiseに覆う場合は、そのcapで期待を上から抑えるだけなら新たな角度独立性は不要。

優越を保証するには、例えばReferenceの安全な全体上界がB′の安全な全体下界を下回る必要がある。**二つの上界の大小だけでは実費の優劣を証明しない。** B′の未取得価格を0埋めしない。一般的なworst-case/typical synthesis下界を、個々のB′角度の下界にもしない。LP下界は実行可能baseline価格ではない。有限候補/precision/価格取得・配分の機会も両armで揃える。

上界差がword policyや評価の保守性に由来する場合は、回路構造の利益と分ける。理論式が緩ければ比較が重なる可能性があり、今回は桁・価格・改善率を数値化していない。TをCX/native depth/qubitと無根拠に合算しない。

## D：precisionとRPEの条件

CT-0の **epsilon≤6.902168833913054e-16** は保存N維持に対する全18round共通の保守的十分境界の引用で、今回のprecision採択ではない。CT-1のepsilon1e-9はq1用で、長roundへ移さない。

全supportのnative位置について局所strict spectral上界epsilon_jを持ち、元global phase・W・actual adjoint・同じreset/測定を保つことを前提に、

`epsilon_CT <= sum_j epsilon_j`

`d_syn <= 2sqrt(2) epsilon_CT`

で旧raw diskへ一度だけ加える。worst-case個数または別途証明された平均誤差式を使い、期待gate数を全trajectory norm上界にしない。C2-BのBDB†重みをF†/Gへ転用しない。

`d_total=d_old+d_syn`、`rho_lower=rho0-d_total`、`beta_stat=.4-beta_PF-asin(d_total/rho0)`で、正半径/正統計幅・旧Nの十分条件・alias/branch条件を維持する必要がある。旧compiler/真空算術diskを二重計上せず、RTE normalization/rangeと固定sampling lawも保つ。これらを満たす**全key policyの実装資格**がない限り、CT-1有限pilotのPASSはcanonical law全体のRPE成功保証にならない。

この精度域でも、理論のetaとguardのFrobenius規約、厳密norm解、古典演算余裕、serializer/guard完了を対応付ける必要がある。backendのmpmath桁数設定だけでは全入力の価格/取得証明にならない。より緩いprecision・非一様配分・新N/alphaが必要なら、独立scopeのユーザー判断と承認へ戻す。今回はどれも計算・採択しない。

## E：次に必要な最小作業

1. **GPT／ユーザー**が保証区分を判断する。現backendを保つなら全域取得・最大分母/word長・数値判定の不足補題が先。別の既知構成法を使うなら、上界を保証するexact/数論/初期netの前提とpolicy変更を先に採択する。
2. **Codex（別途承認後）**が有限precisionと対象を事前固定し、数値上界を評価する。値の代入だけでも今回は許可されていない新価格評価なので、ここでは行わない。
3. 最初の実装確認は、再利用可能なCT-1の保存vacuum2 keyとB′側の保存event/scalar/control代表へ、同じ精度/guard/価格機会を与える有限pilotが候補。大角域を検査するなら範囲端近傍の人工local targetを別途固定し、実到達trajectoryの価格と区別する。何本成功しても、それだけで全support証明の代用にしない。
4. **全域の補題/policy資格が揃ってから**、V3/CT-0の有理multiplicityと全roleへ接続する。いきなり18round word取得・grid・最適化には進まない。

具体epsilon・新backend・新key集合・call回数は未採択。後続でのSTOP候補は、理論前提/strict phase/word規約不一致、全域での取得/guard保証欠損、上界と実wordの不対応、誤差/保存N条件不成立、公平性又は研究条件の変更が必要、承認範囲完了。今回も**固定backendの価格資格が不足した時点でmandatory STOP**し、追加計算は開始しない。
