# 新A：非中央roleの構造保証・既存word再利用の静的調査

2026-11-08 JST、SERVER 2。基準 **800554fbb64c21e584bb29a1a8f52490ab4163e0**。Branch `codex/pr-reference-noncentral-reuse-static-20261108`。

**構造保証の条件整理、既存recordのliteral照合、人工テスト、最小数値検証の準備を完了しmandatory STOP。** 数値的な保証改善率・m17 RPE成功・総T削減は認定していない。

## 1. 固定する結果と研究条件

[V1](../pr_reference_post_v0_v1_budget_20261108/REPORT.md)では中央b37…65の116行・232局所比較が保存の加法cap条件を満たさず、非中央known loadはB′約6.974208e-2／Reference約6.973976e-2だった。保存joint予算約9.504670e-6／3.517585e-5を超えるという**上界会計の不足**を保持する。真の誤差の下界ではない。[V0](../pr_reference_post_n5b3_v0_20261108/REPORT.md)の同じb36 wordへの局所下界結論も保持した。これらの数値診断は再実行していない。

背景はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A。B′4-CX、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、既存D融合・basis共有・X/Y・relative phaseを固定する。M17 q131072/r32/bits88、B′626／Reference630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haを変更しない。

## 2. A：具体化した保証改善

### Template内のkey別guard和とSOURCE alphabet

[CT-0 count_assets／worst_counts](../../../src/trottertracks/pr_reference_ct0.py)は、local loweringのRZ数を先にcontext-max化し、role全体の最大guardを掛ける。これに対し、**同じ保存template内でkey別guardを先に足し、その後context-maxを取る**十分上界を具体化した。最大誤差keyが全ての物理位置へ現れる仮定を弱める方法で、誤差相殺を仮定しない。

さらに同`frames`はcatalog内の全`basis:`を候補にする。[catalogue SOURCE](../../../src/trottertracks/pr_reference_transfer_catalogue.py)はD0…D3のbasisも登録するが、RTE builderが使うのはRTE registryのdefinitionだけである。保存STAGE1_STRUCTUREの`source_basis_probabilities`のkeyは7つのdf-basis IDを持つ。**確率値を使わず、この有限SOURCE alphabetへmaxの範囲を限定する**候補を分けた。実際に上界がどれだけ変わるかは未評価。

両改善の非悪化性は[OPTIONS §1](IMPROVEMENT_OPTIONS.md#1-template別の重み付き上界静的に証明できる部分)に証明した。前提は、対象全templateのliteralとguard coverage、既存compiler条件C、保存contextの閉包、SOURCE alphabetの包含である。数値的改善が必ずあるという定理ではない。

### 共役blockとinverse defect

理想`B D B†`に対し、現在の前側wordをA、帰り側wordをCとする。`C=A†`を保証できればnear-identityなDによってbasis誤差係数を鋭くできる。しかし現在の[consumer](../../../src/trottertracks/pr_reference_common_word_consumer.py)は各native RZをliteral単位に置換し、basis対全体を保持するinterfaceではない。SOURCEのgate.inverse、negative RZのactual adjoint、近似basis全体のactual adjointは別の条件である。

この不足を隠さず、`eta=||C-A†||`を含む十分式と証明を[OPTIONS §2](IMPROVEMENT_OPTIONS.md#2-共役blockの条件付き保証とinverse-defect)へ記録した。現在のpipelineでeta=0又は小さいとは認定していない。C2の数値結果・Full Y4の係数をPartial S2へ移していない。

### SOURCE順序からの限界

RTE builderはproductsの後にrotationを置き、隣接同basisは既に共有し、identityでactive basisを閉じる。Productsを含むrunは小角rotationだけのrunとは限らず、全runへsmall-time係数を付けることはできない。異なるbasisの間にある逆符号key、D境界、2つのCSWAP列も、間の演算を無視して相殺しない。同一wordの系統誤差は反復で蓄積し得るため、期待値やsqrt反復数で全support保証を緩めない。

## 3. B：既存wordの利用可能性

今回行ったのは**exact literal／record ID／word hashのmetadata照合**である。保存guardとTは原字段を引用しただけで、大小比較・加算・Pareto評価・価格rankingをしていない。対象はV1保存D／frames key集合。固定inventoryに記載されたC0 299、旧strict 7603、CT-1 268の3 archiveを参照し、43 basic／85 certificateも照合した。科学runnerや旧grantを使用していない。

| Arm／role | 対象positive literal | 直接raw recordあり | 現wordと異なる、method文書付きstrict wordあり |
|---|---:|---:|---:|
| B′ D | 159 | 159 | 148 |
| Reference D | 160 | 160 | 144 |
| B′ frames | 216 | 176 | 158 |
| Reference frames | 216 | 176 | 158 |

「別word」はsequence hashが異なり、PASS・strict spec・保存guard methodが文書化されたrecordが存在することだけを意味する。精度・Tが現在より良いとは判定していない。同じwordに別precision labelがあるだけでは別wordに数えない。Guard method欠損recordは保管するが、適格候補に格上げしない。集計はrole別に重複するので総word数へ足さない。

Frameの40 keyにはこの3 raw archive内で直接対応がなかったが、既存composite certificateによる現wordは存在する。これを新規合成必須とは扱わない。43 basicによる現在の構成、別表現、直接guard、構造保証という分岐を残す。高多重度roleの全keyを対象にしており、新しいm17 key別物理多重度rankingはしていない。

[RECORD_METADATA.json.xz](../../../artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz)に各literalの現word、各archive/member、spec、word hash、保存guard／T／T†／W、provenance、43 basicへの対応を保存した。別recordがある場合は、まず既存wordの再利用可能性を数値診断し、再合成を既定にしない。

## 4. 実装準備・未解決事項

[静的audit SOURCE](../../../scripts/audit_pr_reference_noncentral_reuse.py)は3 archiveを並行読取し、保存metadataだけを照合した。[次のS1/S2 SOURCE](../../../scripts/diagnose_pr_reference_noncentral_saved.py)は未実行で、key重み付き上界と有限guard/T frontierを評価する入口を準備した。元のworst_countsへ誤差値をそのまま代入すると`scalar_and_wrapper`の定数`+1`までnormになるため、D／frames／products／CSWAPの式だけを取り出した。

[人工test](../../../tests/test_pr_reference_noncentral_reuse.py)は6件PASS。別specで同じwordを別候補にしないこと、異literal／provenance欠損の扱い、template重み付きmax、missing／負の物理数の拒否、SOURCE alphabet制限、離散guard/T frontierを人工値で確認した。Saved guard／Tへ数値helperを適用しておらず、科学性能の証拠ではない。

B′ scalarの実域外reject、Reference vacuumの包絡と実到達域、event安全supersetとcanonical到達性、中央・emitter・shell、terminal経路は残る。新provider、basis-pair実装、domain変更、strict guard、合成、compile、RPE再計算、T価格評価は行っていない。

## 5. RPE・全18round総Tへの接続とSTOP

新しい十分式で旧d_oldとsynthesis diskを正しいraw単位で一度だけ合算し、同じPF控除・alpha・保存Nを守る必要がある。M17ではB′626／Reference630を保持し、共通frameの1-shot価格を総shot差から消さない。全18roundは完成policyの全role／domain・各roundの成功条件と実離散価格が揃ってから比較する。

両armへ同じrecord pool・構造保証・word選択・取得機会を与える。現在のC0優先を変更する場合は別の有限overlayとして固定し、同じliteralへ共通wordを選ぶ。Pareto候補・LP下界を実行可能な強B′価格と扱わない。

既存SOURCE／raw／result／grant／ledger／marker／STOPと他Trackを不変更。正式共通指示2026-10-25、親／worktree AGENTSと研究資料の優先順位を適用。主agentのみで静的解析し、独立したSOURCE／archive読取は並行した。CPU/RAMの独自cap、共有設定・他ユーザーへの変更はない。

**Mandatory STOP。次担当：GPT／ユーザー。** [NEXT_VALIDATION_PLAN](NEXT_VALIDATION_PLAN.md)のS1/S2保存入力診断を承認するか、共役pairの資格を別途優先するか、保留するか判断する。新規合成の必要性、m17成功、全18round総T優越は未確定。正式な研究保証・新規性の採択を代行していない。
