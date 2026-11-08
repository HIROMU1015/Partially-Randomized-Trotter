# 新A：Frame inverse・deterministic Dの保証改善準備

2026-11-08 JST、SERVER 2。基準 **06895797e032bc236562da5ee541b33ce8eb5050**。Branch `codex/pr-reference-inverse-d-prep-20261108`。

**静的SOURCE／metadata照合、候補整理、D1/D2実行準備を完了しmandatory STOP。科学数値再解析・新guard・回路実行は0。** Inverseは109 key、Dは164 keyで、いずれも保存S2で現wordが非劣に残る。Forward43 keyの同時guard/T改善をそのまま移せない。最大basisのinverseは代替exact-literal recordが乏しく、現wordの保証改善／全体actual-adjoint化も別案として残る。

## 1. 固定条件と継承結果

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A／Qiskit1.3.0。既存B′4-CX／Reference F/G、actual SOURCE inverse、vacuum P、2 register／CSWAP、D融合、basis共有、scalar/event相対位相、X/Yを保持する。

M17 q131072／r32／bits88、N_B626／N_R630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを保持。[直近診断](../pr_reference_frame_impact_numeric_20261108/REPORT.md)から、G＝NTでforward43 keyの保存guard／局所Tが下がり、frame上界は0.0035060235→0.0019818615、非中央はB′0.0020730950／Reference0.0020716753、残余は両armで負という結果を引用する。新しく計算していない。

同じ保存最大basis `df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2`では、forward上界が1.2191451e-10→7.8499476e-13、inverseは1.5671920e-10のまま。DはB′約9.12306e-5／Reference約8.98122e-5で、Dだけでも旧joint予算を超える。これらは現在の保存wordと加法十分証明の結果であり、実誤差下界・RPE失敗・新規合成必須を示さない。

## 2. A：7 SOURCE basisとinverse側

[静的SOURCE](../../../scripts/audit_pr_reference_inverse_d_static.py)と[STATIC_BINDINGS.json.xz](../../../artifacts/pr_reference_inverse_d/2026-11-08/STATIC_BINDINGS.json.xz)に、7 ID、両arm各4 context、forward/inverse template、RZの順序位置・wire・signed hex、現word／guard／W／T、保存候補の元specを接続した。Full native列は元archive/memberを参照し、そのhashを記録した。命令順を再lowerしていない。

Basisの定義は[DFBasisRegistry](../../../src/trotterlib/df_rte_tail.py)の8 system qubitとordered local operation metadataを`ordered_local_basis_operations_v1`でhashしたもの。`df-basis-<hash>`を元registryのruntime operation列へ解決する。元定義は保存`common.pkl.xz`（unpacked SHA256 `68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`）とSOURCEで保全されている。今回はregistry再構築／gate materializationを行わず、この定義・原所在と既存compiled templateを対応付けた。

| SOURCE basis（省略suffix、完全IDはSTATIC_BINDINGS） | Inverseのdistinct literal数（各armのcontext union） |
|---|---:|
| `150788…` | 30 |
| `892597…` | 12 |
| `62644a…` | 30 |
| `52f015…` | 11 |
| `8b103a…` | 10 |
| `cb0c4f…` | 31 |
| `e6adc6…` | 15 |

全体unionは109 key。現wordはC0 68／N4 composite 41、40 keyにはexact-literal backend recordがない。Current composite certificateがないという意味ではない。保存非劣frontierで別のsequence hashが残るのは51 key、残り58 keyは1種類のword列である。Record provenance／別certificateとword列の種類を区別する。

最大basisのinverse31 keyはC0 5／composite26。25 keyにexact-literal backend recordがなく、frontierのword列は30 keyで1種類、1 keyで2種類。これはmetadataの件数で、新しいguard/T順位ではない。Inverseの現wordは109/109で非劣に残り、「同じ登録poolでguardを厳密に下げ、Tを増やさない」支配点は保存結果上ない。より高Tでguardを下げる点がどれか、weighted上界へどれだけ効くかは今回未比較。

SOURCEでは逆順＋gate.inverseを使い、event開始時はbasis_inverse、閉じる際はbasisを適用する。Compiler Aはjoint contextで独立にlowerする。Consumerは各native RZをそのliteralのwordへ置換するため、negative RZのcomplete actual adjointは保持しても、compiled basis全体が逆になるとは限らない。Forward G43 literalとinverse109 literalは別集合であり、Gを別角度へ流用しない。

[既存S3](../pr_reference_s3_s2_reuse_20261108/REPORT.md)の88 pairのNOT_PROVEDを引用し、再照合しなかった。NOT_PROVEDは不一致の証明でもeta>0の証明でもない。Compiler Aは既にcontrolledなjointのrelative phaseを保持し、一つの共通global phaseはstrictに約束しない。Local catalogueはnative global phaseを保存していないため、strict basis defectは未資格。後から新controlを加えたり、SOURCE phaseをnative phaseとして代用したりしない。

## 3. B：D構成・保存word・共通key

Dは融合後templateの既存式を保持する。

- B′：`left_all + right_all + (q-1)(left_no0 + right_no0 + boundary_control)`。
- Reference：`lane(q/2, ref) + lane(q/2, actual_SOURCE_inverse)`、`lane(n,a)=2 a(block0_half)+(n-1)a(block0_full)+2n Σ(j=1..3)a(blockj_half)`。

B′の右側はdirectional controlled liftで、逆順block、符号付きdiagonal、ancilla Pのrelative phaseを持つ。Referenceは非制御Dの二つのlaneとactual SOURCE inverse supplementを使う。この構成・境界・contextとword分布の差が別の上界式を生むのであり、単に同じ上界へ余分な係数を掛けたものではない。

Actual inverse supplementのSOURCE forward/inverse列・SOURCE global phaseは引用できるが、そのlocal native列にはnative global phaseがない。旧DAG span比較のPASSを、strict operator／phase又はClifford+T誤差の新保証として使わない。

D164 keyは全てC0の現wordで、全て保存非劣frontierに残る。複数の保存word列があるkeyもあるためprecision/T tradeoffは検討できるが、方向と寄与順位は新しい数値比較を要する。追加Tなしのguard改善を仮定しない。

Raw metadataの全recordを利用可能poolとは扱わない。関連raw一覧にはstatus=PASSでもguard methodが未文書化の3 recordがあり、旧S2はそれらをofferから除外している。今回も保存非劣IDだけを利用対象とし、この3件は採用候補に含めない。現wordのsequence hash／guard provenance、実際の非劣IDに対応するstrict仕様・methodを照合した。Guard/Tの大小比較は行っていない。

Inverse／forward／Dで共通するliteralが5つある。

`0x0.0p+0`, `0x1.2d97c7f3321d2p+2`, `0x1.2d97c7f3321d2p+3`, `0x1.921fb54442d18p+0`, `0x1.921fb54442d18p+1`。

同じliteralのwordを変えるなら全roleへ同じ変更を伝播させる必要がある。次のD1/D2は同じG forward43比較条件から独立に始め、同時に重ねない。Shared keyの他roleへの影響も出力し、一方のarm又はroleだけを都合よく固定するselectorを使わない。

## 4. 改善経路と準備SOURCE

候補は[INVERSE_D_IMPROVEMENT_OPTIONS](INVERSE_D_IMPROVEMENT_OPTIONS.md)で分離した。

- **現構成・保存record再選択**：wordが変わるだけでSOURCE／compilerを保持できる可能性。Inverse最大basisには代替が少なく、T増加と保証のtradeoffをD1で判別する。
- **現wordの保証を鋭くする**：compositeの保存Q＋Gを全wordのstrict評価で置き換える余地、又は現A/Cのetaを囲う案。新guard／operator評価が必要で未承認。新合成と同義ではない。
- **全体actual adjoint**：同じAを戻り側のA†として構成する別SOURCE。Logical eta=0の利点はあるが旧compiler disk／phase資格を自動移送しない。
- **Support／run構造**：既存basis共有・identity flush・event順序を状態遷移として上界化する案。現在の十分式より鋭い全support保証には、新しい閉包証明と有限検証が必要。
- **D**：まずD2で既存template・physical係数・key guardの寄与を分解し、精度改善と共役構造のどちらを狙うか判断する。未評価の寄与を0としない。

[D1/D2準備runner](../../../scripts/diagnose_pr_reference_inverse_d.py)は旧有限G/T/NT selector、weighted-template、coverage、frame witnessを関数単位で再利用する。D2寄与分解だけに必要なSOURCE係数を追加し、旧D式との一致を人工データで検査した。回路／providerを実装し直していない。

[人工test](../../../tests/test_pr_reference_inverse_d.py)は最終7件PASS。非数値metadata、SOURCE phaseとnative phaseの区別、現word非劣の場合のNT、shared key伝播、旧D算術、別context-maxの寄与を検査した。初回testで再利用namespaceへ自分のmainまで含めていた点を修正してからPASSしている。旧科学mainは一度も実行していない。

## 5. 実行境界・次担当

新しいguard/T比較・ランキング・weighted-template計算、D1/D2/D3、operator／eta、新strict guard、合成、回路build/compile/simulation、sampling、RPE／shot／全18round総Tは0。既存SOURCE／raw／results／grant／ledger／marker／STOPを変更していない。共通指示2026-10-25、適用AGENTS／PROJECT_MAPに従い、主agentのみで既知の関連pathを調査した。重い科学処理はなく、共有設定・他ユーザーへの変更や独自資源capはない。

**Mandatory STOP。次担当はGPT／ユーザー。** [NEXT_VALIDATION_PLAN](NEXT_VALIDATION_PLAN.md)のD1/D2を選んで承認するか、D3の資格不足を先に解消するかを判断する。D3の現word defect評価と新actual-adjoint構成は別案で、旧A1のtargetを無断変更していない。

G＝NT forwardは保存比較条件であり正式policyではない。B′ scalar域外、Reference vacuum／event domain、central、emitter、shell、terminal、RPE成功、全18round総T削減は未資格のまま残す。
