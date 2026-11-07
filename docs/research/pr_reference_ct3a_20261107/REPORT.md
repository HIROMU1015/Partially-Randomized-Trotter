# 新A CT-3A：全support strict fallbackの限定調査

2026-11-07 JST、SERVER 2。固定参照commit **`6b0b6669bbf58ef22e789bfef134a1a6da9f2343`**、専用branch `codex/pr-reference-ct3a-20261107`。文書・SOURCEの静的確認と記号的な数学条件整理のみ。

## 結論

**今回確認した資産から、`READY_FOR_LIMITED_IMPLEMENTATION`を付与できるfallbackは確定しなかった。** strict SU(2)のSK型構成は`THEORETICAL_ONLY`。固定pygridsynthをそのまま全support policyとする資格は`UNQUALIFIED`、CT-1の保存有限keyのpolicyは`PARTIAL_POLICY`を維持する。研究全体のno-goや、fallbackの数学的不存在を意味しない。

CT-2から進めた点は、(i) Wを残すdeterminant-one生成集合への記号的な展開、(ii) 理想的なexact-real SK再帰に使える保守定数の導出、(iii) 保存RZ word群だけでは小半径の全SU(2)初期netにならない理由、(iv) backendが返らない場合も含む複合policyの停止・word選択条件である。**これらはCodexの導出・適合性整理であり、独立科学レビュー、実装資格、正式policy採択ではない。**

不足するのは、実wordと結び付いたstrict全域初期net、最大基底T数、有限精度再帰とguard完了の保証である。net生成や合成を行わずに、この不足を実測成功で代替しない。CT-3Bを既定計画として設定せず、mandatory STOPとする。

## 候補の比較

| 候補 | 全入力停止性 | strict誤差・実出力T上界 | 今回の判定 |
|---|---|---|---|
| 固定pygridsynth 2.0.0＋現在のwrapper | 外側探索の完了保証が未確認 | 取得後guardは再利用可。受理後長さ制限は取得保証ではない | 全supportは`UNQUALIFIED`、保存有限域は`PARTIAL_POLICY` |
| Selinger／Ross–Selinger型数論法 | 数学的な完全探索と固定backendを区別。固定kのnorm解・有限数値判定の前提が残る | 成功分母kと正規形を条件にT≤2k。所要kの一様保証は未充足 | 明示価格式は`THEORETICAL_ONLY` |
| strict SU(2) SK | 証明付き基底表と有限精度手順があれば有限深さ | 記号的にc=12の十分定数を導出。実netのeta0/t0と丸め保証が欠ける | `THEORETICAL_ONLY` |
| 全wordの完全探索という理論的対照 | 密性＋計算可能target＋厳密な余裕付き受理を、公平に探索すれば停止を示せる | 終了時Tは数えられるが、現在の資産から有用な事前一様F(epsilon)は未取得 | `THEORETICAL_ONLY`、実装候補として採択せず |

候補別の文献・SOURCE・不足条件は[FALLBACK_POLICY_CANDIDATES](FALLBACK_POLICY_CANDIDATES.md)、証明条件と実出力の上界は[POLICY_TERMINATION_AND_BOUNDS](POLICY_TERMINATION_AND_BOUNDS.md)。

## 固定条件と参照資料

Linear H4・1.00 Å・STO-3G・DF12、元Hの厳密groundを各shotで供給し準備費除外。Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、保存18round q/r。エネルギー精度1.5936e-4 Ha、全失敗確率.05、位相予算.4 rad。Qiskit 1.3.0・compiler A・rz/sx/x/cx・全結合。B′の4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar・event符号・X/Y wrapperは変更しない。

| 固定資料 | 静的に再利用した内容 |
|---|---|
| [CT-2 REPORT](../pr_reference_ct2_20261107/REPORT.md)、[理論/実装境界](../pr_reference_ct2_20261107/STRICT_SYNTHESIS_BOUNDS.md)、[全support条件](../pr_reference_ct2_20261107/FULL_SUPPORT_PRICE_APPLICABILITY.md) | backend7 SOURCEの既存照合、理論式、phase/誤差/価格の区別。既存hash照合を再実行しない |
| [CT-1：6ac8e1a](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/6ac8e1aa3104d8563188612410c1a7533ec47072/docs/research/pr_reference_ct1_20261107) | q1/r1・epsilon1e-9の有限取得とcorrectness。268保存wordの結果を全supportに拡張しない |
| [CT-0：f5f16fa](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/f5f16fad64748236b0160284dc6c3017974b0132/docs/research/pr_reference_ct0_20261107) | 全support実binary64 phaseの定義、共通一様precisionの保守的十分境界 |
| [V3：224440f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/224440f740051ed5794e8db87a57908fc02dd02f/docs/research/pr_reference_rotation_v3_20261107) | 回転分類、roleと物理出現多重度。全Tへの自動移送なし |
| [V5補強：a095a22](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/a095a225ceae383027b300e1f31ec5ccbb1522fc/docs/research/pr_reference_depth_v5_reinforcement_20261107) | native depthの保証域とSTOP。T-depthへ移さない |
| [C2-B pilot：7800494](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78004943ed5b718c12841e80f55fddf1d1644364/docs/research/track_c2_b_pilot_20261106/REPORT.md)、[強baseline：602ca4b](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/docs/research/track_c2_b_strong_baseline_20261106/REPORT.md) | strict word・guard・actual adjointの再利用条件。旧配分optimizerやBDB†重みを新Aへ無条件転用しない |

正式共通運用指示2026-10-25、親およびworktreeのAGENTS.md、PROJECT_MAP.mdに従う。重要なSOURCE・結果・STOPを変更せず、新規grant/承認証明等は作らない。

一次資料の限定照合はSelinger v2のTheorem22／Remark26／Algorithm23／Fact27／§8、Ross–Selinger v3の§7–8、Dawson–Nielsen v2のDefinition1／§3／§4.1–4.2。参照節を網羅的読了や全アルゴリズム監査とは扱わない。追加で既存venvのQiskit discrete_basis 4 SOURCEの対象箇所を静的確認し、公式1.3.0のsolovay_kitaev.pyも参照した。文献の主張と今回の導出は上記2文書で分離した。

## 新Aの全support・RPEへの接続

対象はcompiler A後の**literalな有限binary64 RZ**。厳密な角度有理数、signed angle、global W、actual adjointを保つ。2pi縮約・理想piへのsnap・新vacuum位相算法を導入しない。strictに全実角を覆うfallbackなら、保存の有限supportを列挙せず覆える可能性がある。ただし数学的存在だけでは実policyの資格にならない。

保存SOURCEの[reference_core](../../../src/trottertracks/pr_reference_interference_pilot.py)にはvacuum Pが1shotあたり1個ある。これはReferenceのD/frame/inverse・RTE・products・CSWAP・wrapperの合成価格を省略する理由にはならない。B′のcontrolled D・RTE中央・event/scalar・Gaussian frame/inverse・product・wrapperにも同じstrict precision・取得/選択機会を与える。

CT-0の保存N維持に対する全18round共通一様precision境界 **epsilon≤6.902168833913054e-16** は保存結果の引用で、今回の採択precisionではない。CT-1のq1用epsilon1e-9も移送しない。この小さいprecisionで必要なnet/丸め/guard条件が満たされる証拠はないため、fallback T上界の数値評価・shot再計算は行わない。

将来、全native位置のstrict operator上界epsilon_jを証明できれば、同じjoint operator・reset/測定の下で `epsilon_CT≤sum_j epsilon_j`、`d_syn≤2sqrt(2)epsilon_CT` を再利用できる。旧compiler/真空算術diskに一度だけ加え、半径・位相・保存N条件へ戻す。期待gate数をworst-case誤差個数に使わず、normalization・有限bit law・alias条件を維持する。

Tの一様上界が得られても、両armの上界の大小は実費の優越証明ではない。期待multiplicityとの対応と、比較相手の資格済み価格/下界が必要。未取得価格を0とせず、LP下界を実行可能baseline価格としない。既存native CX/depth結果、追加qubit、RZ増加をTへ無根拠に合算しない。今回T改善率・価格の大きさ・将来の優越は推測していない。

## 検証範囲とSTOP

実施したのは固定SOURCE/文書の読取、一次資料の該当節照合、記号的導出、4文書と最小索引差分の整合確認。**backend import/実行、word/key/net生成、合成、guardの新規実行、科学数値再解析、test、回路build/compile、simulation、価格取得、18round会計はすべて未実施。** 既存保存値の引用と新規計算を区別する。

初期net・有限精度保証の資格には今回範囲外の作業が必要であり、ユーザー指定STOP条件1–3および静的調査完了の条件5に到達した。**mandatory STOP**。旧SOURCE/raw/result/grant/ledger/marker/STOPを保全し、共有設定・他ユーザー・他Trackを操作していない。

次担当は**GPT／ユーザー**。今回のstrict生成集合/定数導出を科学的に確認し、net/有限精度保証を揃える価値があるか判断する。実装や新科学検証を選ぶ場合は入力・precision・証拠・STOPを別途固定して明示承認する。[CT3B_MINIMAL_SCOPE](CT3B_MINIMAL_SCOPE.md)は不足条件を記録したもので、承認済み実装計画ではない。
