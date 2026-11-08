# 新A：D1・D2の独立した有限guard/T診断

2026-11-08 JST、SERVER 2。SOURCE基準 **e8cb9351704219297b6c46321fa9b063f14eeddd**、科学入力base **06895797e032bc236562da5ee541b33ce8eb5050**。Branch `codex/pr-reference-inverse-d-numeric-20261108`。[承認scope §1〜4](../pr_reference_inverse_d_20261108/NEXT_VALIDATION_PLAN.md)。

**D1：G／NTによるinverseの追加guard/T改善なし。D2：Gは36 keyでT増加を伴うD上界改善あり。どちらも全arm/axisで非中央残余は負。両scope完了、独立STOPを維持する。** D3、新guard、合成、RPE／総T評価へ進んでいない。

## 1. 固定条件・実行・正本

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A／Qiskit1.3.0を不変更。B′4-CX／Reference F/G、actual SOURCE inverse、vacuum P、2 register／CSWAP、D融合、basis共有、scalar/event相対位相、X/Yを保持した。

M17 q131072／r32／bits88、N_B626／N_R630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを保持した。必要shotsやRPE式は再計算していない。

共通baselineは保存G＝NT forward43 map＋その他は現word。D1とD2は各々このbaselineから独立に始め、一方の選択結果を他方に重ねていない。両armに同じliteral→record mapを与え、共通5 keyの変更を全roleへ反映する構成を維持した。

準備済み[runner](../../../scripts/diagnose_pr_reference_inverse_d.py)と[helper](../../../scripts/audit_pr_reference_inverse_d_static.py)をbyte変更せず、各scopeを1回、別processで並列実行した。旧S1／S2科学main、frontier、B1 fixture、S3の88 pair検査は再実行していない。

| 固定SOURCE | 実行前後SHA256 |
|---|---|
| diagnose_pr_reference_inverse_d.py | `b14220eb3b1b168b8e5405c02bc28af587c4d408e51bebde475860cbbc48ef3e` |
| audit_pr_reference_inverse_d_static.py | `ef2fda1e2f1c82240ec14922edb2bc3928efb2f22d41ac44a78266bc43a0ca34` |

正本はscope別のexact Fraction結果である。

| Scope | 入力・結果・検算・STOP |
|---|---|
| D1 | [INPUTS](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D1/INPUTS.json)、[RESULTS.json.xz](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D1/RESULTS.json.xz)、[CHECKS](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D1/CHECKS.json)、[STOP](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D1/STOP.json) |
| D2 | [INPUTS](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D2/INPUTS.json)、[RESULTS.json.xz](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D2/RESULTS.json.xz)、[CHECKS](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D2/CHECKS.json)、[STOP](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D2/STOP.json) |

Raw JSONは変更せずlocalに保全し、公開用にbyte可逆なxz copyを保存した。解凍後のraw byte一致、raw SHA256／サイズをCHECKSに記録した。D1 raw2,234,403 bytes、D2 raw9,075,584 bytesを全て保存しており、抜粋ではない。下表の小数は表示用の概数。

## 2. D1：Inverse109 keyの保存record診断

Gはguard→T→ID、TはT→guard→ID、NTは現T以下の候補＋現wordからG順で選ぶ既存有限規則を使った。保存非劣ID以外を候補に加えていない。

| 規則 | Key別guard/T | Frame上界（両arm） | B′非中央上界 | Reference非中央上界 |
|---|---|---:|---:|---:|
| 保存G forward baseline | 比較条件 | 0.001981861470688 | 0.002073094978589 | 0.002071675255914 |
| D1 G＝NT | 全109 keyでguard/T同値 | 0.001981861470688 | 0.002073094978589 | 0.002071675255914 |
| D1 T | 51 keyでT低下・guard上昇、58同値 | 0.08421840907821 | 0.08430964258611 | 0.08430822286343 |

NTはGと同一record mapとなったためaliasを記録した。Gのguard最小選択でもinverse上界は追加改善しなかった。これは**登録済み保存非劣pool内の負結果**で、同じwordの新しい保証やactual-adjoint構成まで不可能とする結果ではない。

G／NTの最大basisは保存baselineと同じ`df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2`。T優先では`df-basis-62644a8d0e0fcdf41b6beda2d77574185ca174831264dd7b3023a7c01375c92f`に最大が移り、frame上界が悪化した。Forward／inverseの別context-maxと全tieを結果へ残した。

D1ではD・products・CSWAPの上界は結果として変わらなかった。Shared keyをrole別に固定したためではなく、同じmapを全roleへ伝播させた結果である。全X/Y行の残余は負。G／NTの残余はB′−0.002063590308301／Reference−0.002036499406692、TはB′−0.08430013791582／Reference−0.08427304701421。

## 3. D2：D164 keyの保存record診断

| 規則 | Key別guard/T | B′ D上界 | Reference D上界 |
|---|---|---:|---:|
| 保存G forward baseline | 比較条件 | 9.123060223391e-5 | 8.981222210257e-5 |
| D2 G | 36 keyでguard低下・T増加、128同値 | 6.504315685215e-5 | 6.362476091781e-5 |
| D2 T | 148 keyでT低下・guard上昇、16同値 | 2.547830891593 | 2.547470187803 |
| D2 NT | 全164 keyでguard/T同値 | 9.123060223391e-5 | 8.981222210257e-5 |

Gでは保存record再選択によるD上界改善を確認したが、T増加を伴う。NTでは追加改善しなかった。T優先は安い局所価格と引き換えに保証を大きく緩めた。大きい値も既存の保守的加法上界であり、真のoperator誤差の測定値ではない。

| 規則 | B′非中央上界 | Reference非中央上界 | B′残余 | Reference残余 |
|---|---:|---:|---:|---:|
| D2 G | 0.002046907533207 | 0.002045487794729 | −0.002037402862919 | −0.002010311945507 |
| D2 T | 2.549812755969 | 2.549452050836 | −2.549803251299 | −2.549416874987 |
| D2 NT | 0.002073094978589 | 0.002071675255914 | −0.002063590308301 | −0.002036499406692 |

Frameは全D2規則で0.001981861470688のまま。D2 GでもD単独が保存joint cap（B′9.504670287679e-6／Reference3.517584922187e-5）を超え、非中央残余は両armの全X/Yで負である。RPE成功・実誤差下界・新合成必須は認定しない。

局所Tの例を以下に示す。保存wordの離散T+T†をnative出現数で加算したtemplate一つの値で、q係数やshotsを掛けた総価格ではない。

| Mapping／template／context | Baseline local T | D2 G local T |
|---|---:|---:|
| b／left_no0／`[false, false]` | 27662 | 29034 |
| b／boundary_control／`[false, false]` | 5346 | 5662 |
| ref／block:0:half／`reference_ccx1_rare0_Y0` | 4268 | 4584 |

Guard改善だけから全体T削減を推論しない。B′にも同じ選択機会を与え、異なる626／630 shotsを消去していない。

## 4. D2のkey寄与

各D templateのguard最大contextを全tie付きで保存し、分解にはtie辞書順最初を用いた。B′の5 template、Referenceのforward／actual SOURCE inverse各5 templateへ旧SOURCEの係数を掛けた寄与である。Key別寄与総和と旧式からのD上界はexact Fractionで一致した。

| Baselineで大きいkey | 証明上界への寄与（両armで同じ表示値） |
|---|---:|
| `0x1.b99b081f839a5p-11` | 1.982379818499e-6 |
| `0x1.48849316f0a43p+1` | 1.798011582806e-6 |
| `0x1.dbbad77194fedp+1` | 1.797889197641e-6 |
| `0x1.071e8b9c2fa62p+2` | 1.794295351410e-6 |
| `0x1.160253502656bp+1` | 1.794180094347e-6 |

D2 G後の上位3 keyは`0x1.b99b081f839a5p-11`、`0x1.dbbad77194fedp+1`、`0x1.071e8b9c2fa62p+2`で、寄与は上表のまま残る。両armを合わせた164 keyについて、各armに現れるkeyのbaseline／各規則の順位・template/context・物理多重度・guard・局所Tを正本に保存した。

これはtemplateごとに別のworst contextを許す**加法十分証明の分解**で、同時到達する一つのtrajectory、真の回路誤差、期待寄与ではない。Native literalをSOURCEのbasis／diagonal由来へ新しく再分類した結果でもない。寄与の大きさを理由に未承認のguard再評価や新合成へ進まない。

## 5. 検算・保全・STOP

[Post-run checker](../../../scripts/check_pr_reference_inverse_d_numeric.py)は旧frame witness検算をAST再利用し、SOURCE／固定入力hash、G baseline一致、保存offer／sequence hash／guard/T、local変化、parts和・残余、forward/inverse maxと全tieを検算した。D2にはSOURCE係数・選択context・key寄与総和・順位の検算も加えた。両scopeがEXACT_POSTRUN_CHECKS_PASS。診断runnerは再実行していない。これは著者Codexの検算で、独立科学レビューではない。

Scope別[環境D1](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D1/ENVIRONMENT.json)／[環境D2](../../../artifacts/pr_reference_inverse_d/2026-11-08/numeric/D2/ENVIRONMENT.json)に、GPUA100-gleap02、Python3.12.3／stdlib、CPU affinity64、実空きメモリ・load、SOURCE hashを保存した。実行は別processで並列、runner内部wallはD1約1.95秒、D2約2.46秒。CPU/RAMの独自cap、共有設定・package・他ユーザーへの変更はない。既存人工testや旧科学計算を不要に再実行していない。

旧SOURCE／raw／results／grant／ledger／marker／STOPを保全した。今回の新出力もD1/D2別directory・INPUTS／RESULTS／CHECKS／STOPで独立に保存した。新guard、Clifford+T合成、word生成／取得、回路build/compile/simulation、provider／consumer、sampling、D3／eta、RPE／shots／全18round総Tは0。Runtime policyは採択していない。

**D1・D2ともmandatory STOP。次担当はGPT／ユーザー。** Inverseの登録record再選択の負結果、DのT増加を伴う改善、なお残るinverse／D十分予算不足を踏まえ、次の保証改善が必要か判断する。D3、新strict guard、追加合成、RPE／総Tへ自動的に進まない。B′ scalar域外、Reference vacuum／event domain、central、emitter、shell、terminalと全RPE資格は未解決のまま保持する。
