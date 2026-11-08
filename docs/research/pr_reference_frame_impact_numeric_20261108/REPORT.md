# 新A：43 frame候補の有限guard/T診断

2026-11-08 JST、SERVER 2。SOURCE基準 **4ebbc3458e13fe0fa6a03f1e2b3bd82e6c3d686e**、科学入力base **340670387dbe58bdda5606fdf222408774be1932**。Branch `codex/pr-reference-frame-impact-numeric-20261108`。[固定scope §1〜5](../pr_reference_frame_impact_20261108/NEXT_VALIDATION_PLAN.md)。

**G／NTは保存S1のframe上界を改善し、43 key全ての保存guardと局所Tも下げた。ただしinverse・Dの課題が残り、両armの非中央残余は負のまま。T優先は安いが上界が悪化した。固定有限診断を完了しmandatory STOP。** Word policyの採択、RPE成功、全18round総T削減は認定しない。

正本は[RESULTS.json](../../../artifacts/pr_reference_frame_impact/2026-11-08/numeric/RESULTS.json)のexact Fraction、検算は[CHECKS.json](../../../artifacts/pr_reference_frame_impact/2026-11-08/numeric/CHECKS.json)。以下の小数は表示用の概数である。

## 1. 固定条件・SOURCE・実行

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A／Qiskit1.3.0を保持した。既存B′／Reference、D融合、basis共有、SOURCE inverse、relative phase、X/Yを変更していない。

M17 q131072／r32／bits88、B′626／Reference630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを保持した。ShotsやRPE式は再計算していない。

[準備済みrunner](../../../scripts/diagnose_pr_reference_frame_impact.py)を**byte変更なし・1回**実行した。実行前後SHA256は`6db4ba0ccb9ee82949d228321e7ab26bb13187d068207ced2b933797b6430aed`。旧S1 weighted-template算術とarm別式をAST再利用し、旧S1 `main/evaluate`、S2 frontier、B1 fixtureを再実行していない。旧SOURCE参照commit、保存入力hashと旧式来歴は[INPUTS.json](../../../artifacts/pr_reference_frame_impact/2026-11-08/numeric/INPUTS.json)に残した。

対象は固定43 literal、保存S2の非劣IDだけである。Gはguard→T→ID、TはT→guard→ID、NTは現T以下の候補＋現wordからGと同順序で選ぶ。同じrecord mapを両armへ適用した。NTはGと同一mapだったのでaliasを記録し、実評価は2 map。新word／precision／候補／選択規則を追加していない。

## 2. S1の最大basis・contextと43 keyの関係

現在のframe pair-maxを与えるのは、両armで

`df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2`

である。ここに43候補のうち26 literalが対応する。Forward／inverseそれぞれの最大contextは、B′の4保存context全て、Referenceの`reference_ccx1_rare0_Y0`／`rare0_Y1`／`rare1_Y0`／`rare1_Y1`全てでtieだった。Exact IDと全context名はRESULTSの`current_witness`にある。

S1式は`3qr × max_basis(forward_context_max + inverse_context_max)`のまま保持した。前後のmaxを同じcontextへ結び付ける別式や、期待値への置換はしていない。

| 最大basisのlogical template上界（両arm共通） | 現word | G／NT |
|---|---:|---:|
| Forward | 1.219145100270284e-10 | 7.849947560784244e-13 |
| Inverse | 1.567192038498129e-10 | 1.567192038498129e-10 |

G／NT後も同じbasisが最大で、未変更inverse項が主に残る。43 literalは全てforward側であり、inverse側へ自動的に同じwordを適用していない。残る4 SOURCE basis、全7 inverse、D／products／CSWAPも不変である。これにより「3 basisに存在する」という静的対応から一歩進み、**43候補が現S1の最大basisのforward上界を下げる**ことを、この保存有限条件で確認した。

## 3. Frame／非中央上界と保存joint予算

| 規則 | Frame上界（両arm共通） | B′非中央上界 | Reference非中央上界 |
|---|---:|---:|---:|
| 現word（保存値引用） | 0.003506023501946 | 0.003597257009846 | 0.003595837287171 |
| G＝NT | 0.001981861470688 | 0.002073094978589 | 0.002071675255914 |
| T | 85.33489793065 | 85.33498916416 | 85.33498774444 |

X/Yは今回の非中央式で同値。Wrapper／central／scalar等を同一視又は消去した結果ではない。Tの大きい数値は保守的な加法上界であり、真のoperator誤差がこの値であることを意味しない。固定S1式へ新しいcapや別保証を追加していない。

| 規則 | B′残余 | Reference残余 |
|---|---:|---:|
| G＝NT | -0.002063590308301 | -0.002036499406692 |
| T | -85.33497965949 | -85.33495256859 |

残余は保存joint予算（B′9.504670287679e-6／Reference3.517584922187e-5）から非中央上界を引いた値である。全4 arm/axis条件で負。DはB′9.123060223391e-5／Reference8.981222210257e-5のままで、Dだけでも保存予算を超えるという既存課題を保持する。

判定は**G／NTのframe十分上界改善あり、非中央の十分予算は未達**。実誤差下界、RPE失敗、既存recordでは不可能、新規合成必須という判定ではない。

## 4. 保存離散Tとのtradeoff

Gでは43/43 keyで保存guardが下がり、43/43 keyで保存Tが下がった。このmapがNTとも一致する。Tでは43/43 keyで保存Tが下がる一方、43/43 keyで保存guardが上がった。安い局所価格だけでは今回の十分保証を満たせない。

以下はB′の`[false, false]` context、forward template一つ当たりの保存価格によるT+T†和である。生成回路の新しい価格取得ではなく、保存wordの離散Tを固定native出現数で重み付けした結果。3qrやshotsを掛けた全RPE価格ではない。

| SOURCE basis（完全IDは[準備REPORT](../pr_reference_frame_impact_20261108/REPORT.md)） | 現word | G＝NT | T |
|---|---:|---:|---:|
| `8b103…` | 44580 | 2252 | 560 |
| `cb0c4…` | 145040 | 7264 | 3592 |
| `e6adc…` | 59616 | 3052 | 1508 |

全180保存template/context行（b78／ref82／inv20）、key別価格差、両armの値はRESULTSにある。24行が43-key置換に対応した。これは必須20／21／5 templateと保存contextに実在する組の数であり、存在しないtemplate/contextを0や別値で補った直積ではない。

Guard-max contextとT値を同じ確率的pathと仮定せず、期待価格や全round価格へ外挿していない。B′にも同じpool／選択機会を与え、異なる626／630 shotsを相殺していない。G／NTの局所優位を全18round総T削減へ読み替えない。

## 5. 検算・保全・STOP

[Post-run checker](../../../scripts/check_pr_reference_frame_impact_numeric.py)で、SOURCE／固定入力hash、baseline4行の不変、保存offer ID／hash／guard／T一致、key別差、context別guard/T差、forward/inverse別maxと全tie、3qrとの関係、部分和／予算残余、未変更非中央項をexact Fractionで検算しPASSした。診断runnerの再実行やfrontier再計算はしていない。著者Codexの検算で、別AIによる独立科学レビューではない。

[ENVIRONMENT.json](../../../artifacts/pr_reference_frame_impact/2026-11-08/numeric/ENVIRONMENT.json)にSERVER 2、Python3.12.3／stdlib、CPU affinity64と実空きメモリ・loadを保存した。短いFraction演算を1 process、runner内部wall約2.15秒で実行。独自CPU/RAM/worker cap、共有設定・他ユーザーへの変更はない。準備段階の人工8 testは固定commitでPASS済みで、今回不要な再実行をしていない。

旧SOURCE／raw／results／grant／ledger／marker／STOPは不変更。新規strict guard・合成・word生成／取得・provider／consumer実行・回路build/compile/simulation・A1 defect・sampling・RPE／shot／全18round総Tは0。43-key runtime policyを採用・適用していない。

**[Mandatory STOP](../../../artifacts/pr_reference_frame_impact/2026-11-08/numeric/STOP.json)。次担当はGPT／ユーザー。** G／NTの保存record再利用を将来のruntime資格へ進めるか、残るinverse／D保証を優先するかを判断する。Domain（B′ scalar／Reference vacuum／event到達性）、central、emitter、shell、terminalと旧diskへの接続は未資格・非zeroのまま。S3／A1、actual-adjoint別構成、追加合成、RPE、全18round総Tへは自動的に進まない。
