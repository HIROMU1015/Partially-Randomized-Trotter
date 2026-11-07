# 新A CT-2：全support strict合成価格上界の構成可能性

2026-11-07 JST、SERVER 2。固定参照 `6ac8e1aa3104d8563188612410c1a7533ec47072`、独立branch `codex/pr-reference-ct2-20261107`。

**理論上の構成手順と条件付き価格式は確認したが、固定pygridsynth2.0.0／研究wrapperへの全support取得・価格保証の移送は `UNQUALIFIED`。静的調査を完了しmandatory STOPした。** 理論側は `THEORETICAL_ONLY`、既存CT-1の有限取得域は `PARTIAL_COVERAGE` と区別する。全supportに `CONSTRUCTIVE_BOUND_AVAILABLE` を付与していない。

これは「strict wordや安全な上界が数学的に存在しない」という結論でも、研究の科学的GO/NO-GOでもない。追加の補題・構成法/word policyの資格が必要という技術的状態である。

- [STRICT_SYNTHESIS_BOUNDS.md](STRICT_SYNTHESIS_BOUNDS.md)：一次文献の条件、明示式、fixed backendとの照合、phase/norm。
- [FULL_SUPPORT_PRICE_APPLICABILITY.md](FULL_SUPPORT_PRICE_APPLICABILITY.md)：全support位相、両armの価格・RPEへの接続、最小後続範囲。

## 固定条件と実施範囲

Linear H4、1.00 Å、STO-3G、DF12。元Hの厳密groundを各shotで供給、準備費除外。Partial S2、LD3、h=.02、finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、保存18round q/r。energy1.5936e-4 Ha、全失敗率.05、phase cap .4rad。Qiskit1.3.0/compiler A、rz/sx/x/cx、全結合。B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合/basis共有/scalar/event/XY、旧native/alias/RPE保証を不変更。

今回行ったのは文書/SOURCE/保存provenanceの読取、一次文献の限定照合、数学的条件整理と公開だけ。**新規合成、word/angle catalogue生成、guard再評価、科学数値計算・再解析、T価格の新評価、回路build/compile、simulation/sampling、precision/alpha最適化、新shot schedule、全18round期待T、T-depthは全て未実施。** 科学moduleをimportしていない。

変更は本書を含む新規3文書とPROJECT_MAP/docs索引の追記だけ。旧SOURCE/raw/result/grant/ledger/marker/STOP、旧概要・manifest・研究ノートを変更せず、新しい管理/承認/実行基盤も作っていない。STOPは本書に記録する。

## Stage A～Eの判断

| Stage | 確認事項 | 判断 |
|---|---|---|
| A | 固定SOURCEがbinary64演算順を含めてvacuum角を定義。既存の有界全support表現を再利用できる | key列挙不要の表現はあるが、価格証明ではない |
| B | strict RZの理論的構成とphase/norm条件。既存backendは候補を棄却しながら分母を増やす | 文献条件と固定実装の全入力取得を同一視できない |
| C | 一様価格capが**取得policy全域**に有効ならvacuum P1個へ適用可能 | 現固定backendに対する当該capの資格は不足 |
| D | CT-0の保存N用epsilon境界とraw disk規約を維持可能な条件を整理 | 新precision/shotを選ばず、全key実装資格待ち |
| E | 理論前提の補強又は別の既知構成法を含むpolicy判断が先 | GPT／ユーザーへ戻す。数値化/実装を自動開始しない |

## 決定的な区別

1. **correctnessと取得完了**：返却されたwordは既存strict guardで証明できるが、それは任意到達角で必ずwordが返るという証明ではない。
2. **受理上限と全support上界**：研究wrapperのword長20,000拒否は取得後の条件。全角度で受理に達する前件なしに、一様価格として会計へ入れない。cache shortcutにも同じ検査が自動で走るわけではない。
3. **理論の上界と実価格**：異なる構成法で短いwordが存在しても、現backendがそれを出力するとは限らない。期待/典型的なT則や漸近式はpointwiseな有限capではない。
4. **strict phase**：binary RZ、actual adjoint、Wを保持する。backend diagnosticの`error()`はstrict matrix certificateの代用にしない。これはSOURCE式の静的確認であり、CT-1で誤wordが返ったという観測ではない。
5. **上界と勝者**：vacuum部分の上界や両arm上界の大小だけでは総T優越を証明しない。B′未取得価格を0とせず、全role・公平なprecision/価格取得機会・適切な下界/実行可能価格が必要。

根拠と条件式は[境界文書](STRICT_SYNTHESIS_BOUNDS.md)、会計への含意は[適用文書](FULL_SUPPORT_PRICE_APPLICABILITY.md)。T価格・上界の桁・改善率を新たに数値化していない。

## 確認したSOURCE・保存資産

基準commitのCT-1 [REPORT](../pr_reference_ct1_20261107/REPORT.md) / [SCOPE](../pr_reference_ct1_20261107/SCOPE.md)、CT-0 `f5f16fad64748236b0160284dc6c3017974b0132`、strict静的調査 `b3ff9d1b4d356ae67e8f55bf1e2cfd0715814cbb`、V3 `224440f740051ed5794e8db87a57908fc02dd02f`、V5補強 `a095a225ceae383027b300e1f31ec5ccbb1522fc`、C2-B pilot `78004943ed5b718c12841e80f55fddf1d1644364` / strong baseline `602ca4b67b1037e221e910a2733444630c5add62` の関連pathを参照した。正式共通指示2026-10-25、親/worktree AGENTS、PROJECT_MAPに従う。

| 再利用資産 | 今回の静的確認・後続への再利用範囲 |
|---|---|
| [track_c_h4_synthesis](../../../src/trottertracks/track_c_h4_synthesis.py) | binary strict target、cache/spec、epsilon/4 backend headroom、exact word＋区間guard、長さ/失敗条件 |
| [track_c2_b_verify](../../../src/trottertracks/track_c2_b_verify.py) | written product逆順、W、負角actual adjoint。旧numeric suiteは未実行 |
| [track_c2_b_strong_baseline](../../../src/trottertracks/track_c2_b_strong_baseline.py) | 保存record採用規則、実離散価格と下界の区別。有理期待Tへ整数ceil/gcdを移さない |
| [CT-0](../../../src/trottertracks/pr_reference_ct0.py) | 全support phase境界、finite event superset、worst L、既存headroom/誤差規約。今回再解析せず保存値を引用 |
| [CT-1](../../../src/trottertracks/pr_reference_ct1.py) | 事前固定key、finite wordのguard/adjoint、whole normと状態診断の区別、raw disk/保存N条件 |
| [V3](../../../src/trottertracks/pr_reference_rotation_v3.py) | native angle/role、有理multiplicity、fusion差分、symbolic phaseの未資格区分 |
| [transfer](../../../src/trottertracks/pr_reference_transfer_qualification.py) / [error](../../../src/trottertracks/pr_reference_transfer_error.py) | F/G生成順、真空算術境界と全空間誤差、P passthrough |
| [B′ accounting](../../../src/trottertracks/pr_reference_bprime_accounting.py) | compiler diskの置換とraw単位の一度だけの加算、保存RPE/alias条件 |

CT-1成果物ではSOURCE/STOP/BUNDLESを読み、strict archiveの**1 recordのschema/provenanceだけ**を確認した。specは `0x1.1339836591c71p+1:10^-9:pygridsynth2.0.0:strict:seed17`、保存PASS/guard方式/sequence hashがあることを読んだ。全268 recordの再guard・価格再集計をしたわけではない。旧7,603件やcoverageの新しい集合演算も未実施。CT-0の4 legacy recordの未資格状態を保つ。

保存結果を引用する範囲では、CT-1のepsilon1e-9・268 word・21 local・8 whole・q1保存192/axisの予算条件PASSを維持する。この価格を全support平均/最大値にしない。CT-0の共通十分境界6.902168833913054e-16は最適precisionでも今回の採択値でもない。

## 一次資料と実装version

以下の**関連節だけ**を照合し、全関連文献の再レビューはしていない。

- [Selinger, arXiv:1212.6253v2](https://arxiv.org/pdf/1212.6253v2)：Theorem22、Algorithm23、Fact27、§8.2–8.3、Remark26。
- [Ross–Selinger, arXiv:1403.2975v3](https://arxiv.org/pdf/1403.2975v3)：§7–8のstrict問題、correctness/optimality/期待値の前提。
- [Dawson–Nielsen, quant-ph/0505030v2](https://arxiv.org/pdf/quant-ph/0505030v2)：Definition1、§3の再帰/基底表、§4.2の定数と有限精度の留保。
- [pygridsynth v2.0.0](https://github.com/quantum-programming/pygridsynth/tree/65f6dd06538675d63a239009574dea5c636668b1)：installed SOURCE7件を同tagとbyte照合。全dependency監査・全アルゴリズムの形式証明をしたとは称さない。

未確認なのは、全到達角・有限precisionでのnorm解/取得保証、数値探索とguardの全域完了証明、固定backendへのworst-case Tcapの適用、厳密SK定数/strict基底netの具体certificate、将来のcapの有用性の数値確認。これらを確認済みとして扱わない。

## 公開前の確認

25個の相対文書/SOURCEリンクと索引の原文prefix、変更対象が新規3文書＋索引2件だけであることを確認した。SOURCE・科学データのtestやrunnerは実行していない。

## 次に必要な判断とmandatory STOP

**GPT／ユーザー**が、現backendで不足する全域取得/価格補題を補強するか、別の既知構成法を含む証明可能なword policyを採択するか、有限cohortで止めるかを判断する。別法の採択は自動で行わない。科学的価値・新規性の判断も次担当に戻す。

採択後の**Codex（別途実行承認）**には、全域証明条件の固定、有限precision/target/call範囲の事前固定、最小の局所guard/phase/価格上界検証が必要。具体候補は[適用文書E](FULL_SUPPORT_PRICE_APPLICABILITY.md#e次に必要な最小作業)。word数・precision・backend・追加budget・shot自由度は今回固定していない。数値capを試してから条件を広げる流れにはしない。

**mandatory STOP実施。** 停止理由は、固定backendへ全support T上界を移す資格が不足し、有用な価格資格には今回の範囲外の証明補強・実装/数値検証が必要なこと、及び静的scope完了（指示§7の2/4/5）。T資源優越・世界的新規性のGO/NO-GOは決定しない。新規合成・別backend実装・18round会計・精度最適化へ進まない。

SERVER 2のaffinity64、空きRAM/pressureを読取観測した。処理は限定SOURCE/資料の読取と文書作成のみで、科学workerを起動せず恣意的な資源capを追加していない。共有設定・他ユーザーprocess/file、他Trackにも変更していない。
