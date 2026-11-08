# 新A：V1有限precision・role別joint予算の限定数値診断

2026-11-08 JST、SERVER 2。実行SOURCE／scope commit **e610d7336b9580be56b0499fbb7ba5f2204efbc0**、科学入力base **460726569c60ea4cf8f9839f2aff16f84157babc**。Branch `codex/pr-reference-post-v0-v1-budget-numeric-20261108`。

**承認された計画§2のA/Bを完了しmandatory STOP。** 中央4角のb=37…65を評価したが、保存guardを使う加法十分証明は旧局所capを満たさない。また、非中央の保存logical上界だけで両armのjoint予算を超える。**中央precisionの変更だけでは、今回の固定word／role-max会計でm17の十分保証は閉じない。** 実誤差・RPE失敗・新規合成の必然性を示す下界ではない。

## 1. 固定条件・SOURCE・実行範囲

[NEXT_VALIDATION_PLAN §2](../pr_reference_post_v0_precision_budget_20261108/NEXT_VALIDATION_PLAN.md)の固定入力と有限規則を実行した。[準備済みSOURCE](../../../scripts/diagnose_pr_reference_post_v0_precision.py)は変更せず、SHA256は`6e1a69a0882d6aafb8ab376093c1fc660b4408ad3429bd400099285442e8163f`でe610d7336…のbytesと一致する。

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0/compiler A。B′4-CX、Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event相対位相・X/Yを維持した。M17 q131072／r32／88 bits、B′626／Reference630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haは不変更。

旧b36行と[V0の4/16/2結果](../pr_reference_post_n5b3_v0_20261108/REPORT.md)は引用のみ。既存b36のQ／G／index、旧guard／fixture／科学runnerを再実行していない。中央4 literal以外の新keyや到達集合を生成していない。

## 2. A：中央4角の有限precision

Binary64分母から従来規則でb_max=65が決まり、新bは37…65の29個となった。4角×29の116 component行、X/Yの232局所cap診断を保存した。Indexは既存round_indexの変更しない関数bodyへ隔離global Bを渡し、SOURCE上232回の正負評価に対応する。Ascending binary構成・complete actual adjoint・saved sqrt2有理上端を保持した。

232比較すべてが`KNOWN_ADDITIVE_CERTIFICATE_EXHAUSTS_LOCAL_ALLOCATION`。`R=cap-Q-G_known`が非正であり、missing guardへ正の局所残余がある行はなかった。これは**選択した保存guardの加法上界**による診断で、V0-Aと異なり新wordの真の誤差下界ではない。

| 既存43だけで全選択指数を覆えた行 | b | Positive literal | 選択chunk数 |
|---|---:|---|---:|
| B′ | 37, 38 | 0x1.7e4088e024c5fp-12 | 各11 |
| Reference | 37 | 0x1.7e4088e024c5fp-11 | 11 |
| Reference | 37 | 0x1.fdab627b8d37ap-13 | 16 |

完全coverageは116中4行で、残る112行は固定43 poolにmissing指数を持つ。全上界はnullのまま保存した。Missingは全archiveに不存在という主張ではない。b=65では4角ともQ=0だが、次の既知guard和が残り、4角ともmissingがある。

| Arm | Positive literal | b65のG_known（概数） | 選択chunk数 |
|---|---|---:|---:|
| B′ | 0x1.7e4088e024c5fp-12 | 1.419697e-13 | 23 |
| B′ | 0x1.fdab627b8d37ap-14 | 1.939527e-13 | 33 |
| Reference | 0x1.7e4088e024c5fp-11 | 1.459913e-13 | 23 |
| Reference | 0x1.fdab627b8d37ap-13 | 2.026609e-13 | 33 |

保存局所capはB′約6.902169e-16、Reference約2.564898e-15。Qを0にできても、保存guard和をそのまま使う今回の証明はcapへ入らない。新basicを追加するだけで既存guard寄与まで減るわけではない。一方、guardは上界なので、これを理由に真のword誤差や別構成の不可能性を主張しない。b／word policyを正式採択せず、Tをchunk数から推測していない。

## 3. B：保存role上界・joint残項

固定C0 299 recordのstrict_error_upper、固定85 certificateのQ＋word_error_upperを用いた。Emitterは別残項。Dはunfused／boundary差／actual SOURCE inverse supplementのpositive unionを使い、signed price差を負のnormとして引かなかった。8 groupすべてで保存keyを対応できたが、m17への移送は既存条件C・SOURCE閉包付きである。

| Role group | B′ positive key数／logical max概数 | Reference positive key数／logical max概数 |
|---|---|---|
| D | 159／1.755762e-12 | 160／1.756304e-12 |
| frames | 216／5.302319e-12 | 216／5.302319e-12 |
| products | 1／4.329780e-17 | 1／8.659561e-17 |
| CSWAP | SOURCE不存在／0 | 2／9.847056e-13 |

保存の非負物理多重度Lを用いる`sum L_role * max_key e_role`は次となった。X/Yで今回評価したgroupの数は同じで、未評価scalar_and_wrapper側のY差を消したわけではない。

| Arm | X/Y共通のknown非中央load（概数） | 保存joint予算（概数） |
|---|---:|---:|
| B′ | 6.974207906e-2 | 9.504670288e-6 |
| Reference | 6.973975574e-2 | 3.517584922e-5 |

中央の既知項も加えた116の残差行は**全て負**。したがって中央だけを高精度化しても、現在の非中央wordとrole-max加法会計では十分保証は成立しない。Frames／Dのmaxが同時に到達するという下界ではなく、現在の証明の保守性又はword保証の改善が別に必要という限定結果である。

中央だけなら、例えばb65のknown loadはB′約1.626993e-6、Reference約8.500212e-7まで小さくなる。ただしmissingがありcomplete中央上界はnull。中央だけの正の差を全体PASSにしない。V0-BでReferenceに残った余地は、非中央を合わせる今回の会計では閉じなかった。V0-Aの「同じb36 wordで旧局所capは不可能」という下界結論も保持する。

## 4. 未資格事項と判断への影響

Event／scalar_and_wrapper／vacuum、emitter／shellを`UNQUALIFIED_NOT_ZERO`で残した。B′ scalarの実域外reject、Reference vacuumの保存包絡超過と実到達角の違い、event安全supersetとcanonical到達性は未解消。Reference独立event PのSOURCE数0を、vacuum誤差0の意味に使っていない。

既存43、C0、85を保全し、他の保存record再選択・direct joint guard・signed digit・domain拡張・新規合成を行っていない。Missing又は今回のguard上界不足は、選択した構成・証明へ追加対応が必要という意味であり、新しい合成が必須という証明ではない。現在の上界を鋭くできるか、別保存wordで対応できるかは未確定。

未評価roleを0にせず、full joint error／m17 RPE／全18round総Tは未資格。Raw disk・位相・半径・shot式は起動せず、旧diskも再加算していない。新T価格／T合計・精度最適化・改善率は未取得。強いB′にも同じ改善・取得機会を与え、N626／630の差を保持する。

## 5. 検算・SOURCE保全・STOP

[RESULTS](../../../artifacts/pr_reference_post_v0_v1_budget/2026-11-08/RESULTS.json)に全有限表とexact Fraction、missing、role key集合、残項を保存した。[SOURCE_INPUTS](../../../artifacts/pr_reference_post_v0_v1_budget/2026-11-08/SOURCE_INPUTS.json)は11保存入力／SOURCEの来歴とrunner hash、[CHECKS](../../../artifacts/pr_reference_post_v0_v1_budget/2026-11-08/CHECKS.json)は指定件数・端点・対応・未評価項保持の確認を保存する。保存入力bytes／hashと固定commitも一致した。

Runner内の正負対称、ties-to-even、index再構成、delta上界、guard非負、終端Q=0、116／232／116件、8 group／4 loadを検算した。出力後は同SOURCE一致・input hash・null／残項・STOPを確認しただけで、科学runnerを再実行していない。計画の人工7 testは前段の結果を再利用した。

[ENVIRONMENT](../../../artifacts/pr_reference_post_v0_v1_budget/2026-11-08/ENVIRONMENT.json)：GPUA100-gleap02、Python3.12.3／stdlib、64 CPU affinity、起動前MemAvailable約514 GB。小さい有理演算と保存metadata処理のため、準備済み単一processのまま約0.165秒で完了。CPU／RAMの独自cap、共有設定・他ユーザーへの変更はない。

新guard／合成／word生成／provider／consumer／build／compile／Operator／sampling／RPE／T評価は0。旧SOURCE／raw／result／grant／ledger／marker／STOPと他Trackを不変更。著者Codexの限定診断であり独立科学レビュー・新規性判定ではない。

**[Mandatory STOP](../../../artifacts/pr_reference_post_v0_v1_budget/2026-11-08/STOP.json)。次担当：GPT／ユーザー。** 中央precisionだけの追加取得より先に、非中央role-max／support保証／保存word選択のどれを改善対象とするか、又は保留するか判断する。別の科学実行、guard・合成・domain・policy変更・RPE／総T評価へ自動進行しない。
