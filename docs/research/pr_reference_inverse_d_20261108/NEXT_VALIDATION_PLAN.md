# D1／D2／D3：次の有限科学検証

2026-11-08 JST。基準 **06895797e032bc236562da5ee541b33ce8eb5050**。本書は未実行の提案。今回の静的作業はmandatory STOPし、科学実行は別途明示承認まで開始しない。承認証明書・grant等は不要である。

## 1. 共通入力・SOURCE・固定比較条件

[REPORT](REPORT.md) §1の物理条件、compiler／law／構成、m17 q131072／r32／bits88、626／630 shots、alpha／failure／phase／energyを不変更。D1/D2は[準備runner](../../../scripts/diagnose_pr_reference_inverse_d.py)と[静的helper](../../../scripts/audit_pr_reference_inverse_d_static.py)を用い、baseのGit objectから入力を読む。新しいHamiltonian・angle・precision・wordは生成しない。

| 固定入力 | 目的 |
|---|---|
| `artifacts/pr_reference_frame_impact/2026-11-08/numeric/RESULTS.json` | G＝NT forward43のmap・4 aggregate行・180 local context行を引用。Forward選択を再実行しない |
| `artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json` | 保存375 frontierのうち対象keyのoffer／非劣IDを読む。再計算しない |
| `artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz` | 現C0／N4 composite、保存Q＋G・guard・T・provenance |
| `artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz` | SOURCE7 basis集合、旧pair診断、forward43 literal |
| `artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz` | 現C0の保存guard／T。Synthesis／guard実行なし |
| `artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz` | B′のSOURCE必須template／保存context |
| `artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz` | ReferenceのSOURCE必須template／保存context |
| `artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz` | Actual SOURCE inverse Dの5 template／保存context |

SOURCEは旧`diagnose_pr_reference_frame_impact.py`から`load_arithmetic`、`choose_offers`、`context_details`、`witnesses`を関数単位で再利用する。その下のS1 coverage／weighted-template式も元の固定SOURCEを使う。D2で追加するSOURCE係数は人工testで旧arm式と一致済み。旧科学main、S2 Pareto処理、B1 fixture、S3の88 pair判定は実行しない。

**共通baselineは保存G forward43＋その他は現word。** Gを正式policyとして採択するのではなく保存比較条件として使用する。D1とD2は同じbaselineから独立に始め、片方の選択結果を他方へ自動的に重ねない。両armは同じliteral→record mapを使う。

## 2. D1：Inverse保存recordの有限診断

### 固定対象・選択

TargetはSOURCE7 basisの`basis_inverse:<ID>`に現れる109 positive binary64 literalのunionで、STATIC_BINDINGSの`fixed_keysets.D1_inverse`に完全列挙した。Raw RZのsign／wire／orderを保持し、abs hexでrecordを参照する。G forward43との交差は0である。

Poolは各keyの保存`nondominated_ids`＋NTのcurrent fallbackだけ。G／T／NTの順序は前回と同じ。

- G：exact Fraction guard、integer T、record IDの昇順。
- T：integer T、exact Fraction guard、record IDの昇順。
- NT：現T以下の非劣候補＋現wordからGと同順序。

全109 keyでcurrentは保存非劣であり、NTは非増加Tで厳密guard改善を期待する探索ではなく比較controlでもある。同じmapはaliasにする。Record ID、guard方向、local T変化を出し、precision labelで順位を決めない。

### 評価するもの

Baseline180 contextのguard／local Tが保存G表と一致することを確認する。Aggregate4行は引用し再計算しない。選んだ最大3 mapについて旧weighted-template式で全必須templateを再評価し、forward/inverse別context-max、7 basis pair-max、非中央parts・保存cap残余を両arm各X/Yで出す。

InverseとD／forwardに共通する5 literalは同じmapで全roleへ伝播する。Shared keyの影響をparts変化とcontext表で分けて示す。Role別selectorでDやforwardを不自然に固定しない。G forward43の選択自体は変更しない。

保存Tはkeyごとの差とlocal template和まで。3qr／shotsによる総T移送、RPE再計算、runtime policy採択は含めない。Current又はcandidateのcompositeは保存Q＋G、backend wordは保存strict guardを使い、未評価emitter等を0にしない。

## 3. D2：D key寄与・保存recordの有限診断

### 固定対象・選択

Targetは旧S1式が参照するD templateに現れる164 literalのunion、`fixed_keysets.D2_D`である。B′は5 template、Reference forward／actual SOURCE inverseは各5 template。Standalone D basisのguardをもう一度足さない。

D1と同じG／T／NTを保存非劣poolへ適用し、最大3 mapを比較する。両armに同じpoolとmapを与える。新たなLP/MILP・全体T最適化・key追加はしない。

### 寄与と比較

基準G mapと各candidate mapについて、Dの`context × key出現数 × 保存guard`を評価する。Template-maxを与えるcontextを全tie付きで記録し、寄与表ではtieの辞書順最初を固定する。GuardのSOURCE係数は以下である。

| Arm | Template | 係数 |
|---|---|---:|
| B′ | left_all／right_all | 各1 |
| B′ | left_no0／right_no0／boundary_control | 各q−1 |
| Reference、各ref/inv mapping | block0_half | 2 |
| 同上 | block0_full | q/2−1 |
| 同上 | block1_half／block2_half／block3_half | 各q |

選ばれたtemplate-max contextのkey guard寄与へ係数を掛け、keyごとに集約したD上界分解を出す。Guard寄与の降順、tieはliteral順で全keyを並べる。合計が旧式からのD上界に一致することをexact Fractionで検算する。

これは別々のworst contextを許す**証明上界の分解**であり、同時到達可能なpath、真の誤差寄与、期待値ではない。Local Tへ同じq係数やshotsを掛けた価格は出さない。Dとframe共通keyの影響は全partsへ反映し、Dだけが変わると仮定しない。

## 4. D1/D2のSOURCE・出力・STOP

将来の実行コマンドは次の二つで、今回はどちらも実行していない。片方だけの承認でも独立に実行できる。

```bash
python3 -B scripts/diagnose_pr_reference_inverse_d.py D1
python3 -B scripts/diagnose_pr_reference_inverse_d.py D2
```

各々の新outputは`artifacts/pr_reference_inverse_d/2026-11-08/numeric/D1/`又は`D2/`。INPUTS.json／RESULTS.json／STOP.json、失敗時だけFAILURE.jsonを保存する。現在numeric directoryは存在しない。

INPUTSには実行前SOURCE hash、再利用式、固定Git input hash、target集合、3規則、保存G map／m17条件を記録する。RESULTSにはkey別record/hash／guard/T差、context表、forward/inverse witness、parts・残余、D2のみD寄与分解を保存する。

Missing record／guard／T／template、SOURCE・hash・saved baseline不一致、想定外domain条件はUNQUALIFIEDとしてSTOPし、0や便宜値で補わない。守る検算は現offer整合、保存非劣ID一致、G baseline context一致、SOURCE coverage、parts和・残余、D2寄与総和、実行前後SOURCE不変である。未資格項は`UNQUALIFIED_NOT_ZERO`、RPEは`UNQUALIFIED`、global Tは`NOT_EVALUATED`とする。

判断は「保存上界改善」「高Tとのtradeoff」「局所だけ改善／最大不変」「十分予算未達」「UNQUALIFIED」を分ける。実誤差下界、RPE失敗、最適性、追加合成必須、全18round改善を認定しない。

固定比較完了又は不一致でmandatory STOP。別precision、guard、key、optimizer、追加wordへ拡張しない。新規合成・strict guard・operator／eta・回路build/compile/simulation・provider／consumer実行・sampling・RPE／shot／全18round総Tはcall範囲外である。

## 5. D3：Defect／actual-adjointの別資格

**D3-current**は[旧A1](../pr_reference_s3_s2_reuse_20261108/NEXT_VALIDATION_PLAN.md#2-a1現wordの2-pairに対するinstruction-body-defect)の2 pair案を保持する。対象はlex-firstの`df-basis-8b103ae5327e20e021dcbb2eb64919b67de85b194d4d6c38b6093414f37e2fbe`、B′ `catalogue/rare0_Y0.json`／Reference `catalogue_boundary/reference_ccx1_rare0_Y0.json`、現C0＋85 word、8 qubit、192-bit／mpmath1.3.0のinstruction-body defect上界案である。G43を混ぜず、旧A1を無断で最大basis cb0c4へ変更しない。

現wordを保持した数値評価は新しい科学実行承認が必要で、今回は未実装・未実行。Native phaseが欠けるためbody etaとfull strict etaは異なる。厳密なmulti-qubit interval演算の実装・包含性testが必要で、既存numpy中点値や1q RZ用strict_guardをそのままfull eta証明にしない。

最大cb0c4を優先するD3 variantには別target固定の判断が必要。旧A1のbody診断を先に実施するか、欠けたphase／SOURCE→native関係を先に資格するかをGPT／ユーザーへ戻す。

**D3-adjoint**は別構成である。SOURCEのbasis挿入点、sharing／flush／D fusionの保持、AとA†のnamed word／W、既存compilerとの境界、旧diskの置換条件、全relative phase／terminalを先に固定する必要がある。Logical eta=0とemitter／shell誤差を分ける。現defect評価と同時にwordを選び直さず、G43 runtime化も混ぜない。これらの重要条件は未固定であり、架空のfixture条件を補わず、新しいscientific runnerを作っていない。

## 6. 次の研究判断

次担当はGPT／ユーザー。D1で既存record再選択の限界／高T tradeoffを確認するか、D2でD寄与を明らかにして構造保証とword精度を選ぶか、D3の資格設計を先行するかを判断する。

最大inverseの多くのkeyに別backend recordがないことから、**同じ保存composite全wordの新strict保証**も候補だが、D1/D2のcallに含めない。必要性が明らかになってからtarget／guard method／precision／STOPを別途固定する。新規合成を自動開始しない。

今回の静的branchではD1/D2/D3・新guard・回路・RPE・総Tを実行せずmandatory STOP。旧SOURCE／結果／STOPとG forwardの未採択状態、B′ scalar／Reference vacuum／event domain、central、emitter、shell、terminalを保全する。
