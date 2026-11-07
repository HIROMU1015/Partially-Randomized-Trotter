# 新A CT-0：key coverage・全support位相・precisionの限定検証

2026-11-07 JST、SERVER 2。基準 `b3ff9d1b4d356ae67e8f55bf1e2cfd0715814cbb`、独立branch `codex/pr-reference-ct0-20261107`。

**保存numeric keyの一部は再利用候補であり、追加合成誤差の正のRPE予算も存在する。ただし、Referenceの全support vacuum位相に対するstrict word・価格上界は未資格。CT-0を完了しmandatory STOPした。** T費用・改善率は計算していない。科学的GO/NO-GO・新規性の判定ではない。

## 固定条件と実行範囲

Linear H4、1.00 Å、STO-3G、DF12、元Hの厳密groundを各shotで供給、準備費除外。Partial S2・LD3・h=.02・finite RTE K=2、ordinary `diag(I,V)`、canonical independent finite-bit law、保存18round q/r。energy精度1.5936e-4 Ha、全失敗率.05、phase cap .4 rad。Qiskit1.3.0／compiler A／native rz,sx,x,cx／全結合。B′4-CX controlled-RZZ、Reference F/G・actual adjoint・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar・event符号・X/Y wrapperを維持した。

評価対象は**compiler後の実binary64 RZ**、`diag(exp(-iθ/2),exp(iθ/2))`、strict `up_to_phase=False`。近いfloatをexact Clifford/Tへsnapせず、joint relative phaseとWを保全する。

承認範囲の保存key/record照合、位相規則の有限数値確認、残存RPE予算の再解析、人工9検査だけを行った。**新規合成・word生成・guard再計算・回路build/compile・sampling/simulation・T価格取得・precision最適化・新shot schedule・全18round総T・T-depthは全て0。** 旧SOURCE・結果・grant/ledger/marker/STOPは不変更。

## 結果

| Stage | 確認結果 | 限界 |
|---|---|---|
| A | 保存7,603件のspec/hash/PASS/保存guard≤epsilon/T field整合。V3のB′400 positive key中153、Reference384中154にrecordあり | precision未採択。角度欠損247／230、symbolic phaseを別記。4 legacy recordのguard方式metadataは未資格 |
| B | B′ event位相は最大3r回のπ反復加算で保守的に有限表現。Referenceは既存IEEE recurrenceと保存位相境界で全support角度域・誤差域を表現可能 | phase/error域は `FULL_SUPPORT_REPRESENTABLE`。全support strict word価格は `UNQUALIFIED`、Stage B総合も `UNQUALIFIED` |
| C | 全18round・両armで正の追加raw disk予算、保存整数Nを維持できる正のheadroomを確認 | word/価格/全key資格込みは `CONDITIONALLY_FEASIBLE`。precision値・配分を採択していない |

詳細は [KEY_COVERAGE.md](KEY_COVERAGE.md)、[PHASE_PRECISION_FEASIBILITY.md](PHASE_PRECISION_FEASIBILITY.md)。機械可読結果・SOURCE・入力hashは[成果物入口](../../../artifacts/pr_reference_ct0/2026-11-07/README.md)。

保存native CX/RZ/depthの優劣・shots8598/8606は変更していない。今回のprecision条件は旧Nに追加誤差を入れられる**十分条件**であり、合成済み・最適精度・最小N・FT全体優越の証明ではない。旧native削減率を総Tへ移さない。

## SOURCEと再利用

固定参照は、直前静的調査 `b3ff9d1...`、V3 `224440f740051ed5794e8db87a57908fc02dd02f`、B′ `43bfc5918703e8dcd020be4fcf96bb8dfd99f909`、V5補強 `a095a225ceae383027b300e1f31ec5ccbb1522fc`、C2-B pilot `78004943ed5b718c12841e80f55fddf1d1644364`、強baseline `602ca4b67b1037e221e910a2733444630c5add62`。正式共通指示2026-10-25、親/本worktree AGENTS、PROJECT_MAP、直前2文書と関連REPORTを確認した。

再利用した内容は次の通り。

- `track_c_h4_synthesis`: strict binary target、spec、word/PASS/guardの保存規約。`acquire`は一切呼ばない。
- `track_c2_b_strong_baseline.load_inputs`: hash/PASS/guard≤epsilon・T field一致の読取検証パターン。Full Y4/LD12の価格/予算・整数ceil/gcd下界は移さない。
- `track_c2_b_verify.lower_native`: negative angleはpositive wordのactual adjoint、Wを含むstrict phaseの条件。回路は生成していない。
- `pr_reference_rotation_v3`: 保存signed角度、role、有理multiplicity、D融合補正。分類を再実行せず、同じkeyのsigned補正を合算して非負の使用域を確認した。
- `pr_reference_transfer_error` / `pr_reference_bprime_accounting`: 補正済みraw disk・phase/radius/shot規約、naive vacuum加算の全support境界。旧compiler diskを再加算しない。
- `pr_reference_depth_v5` / 補強: 保存context、basis/inverse、実SOURCE inverse Dの20 native record。native depth保証を合成へ転用しない。
- `track_c_compiler_a_18round.I/upper/lower`をそのままimportし、192-bit外向き区間端点を再利用。

新[CT-0 SOURCE](../../../src/trottertracks/pr_reference_ct0.py)は、旧optimizer/acquisition runnerに接続しないsaved-data専用の小さな解析。旧全pipelineは不要なcompile/価格/合成を起動するため再利用せず、読取・key照合・headroomのみを実装した。[runner](../../../scripts/run_pr_reference_ct0.py)は保存STOPがあれば拒否する。

## 最小の後続案（未実施・要別scope）

現状から長round・全support総Tの取得には進めない。まず**GPT／ユーザー**が以下を判断する。

1. 全support位相のword価格資格を先に解くか、最初は有限の既知numeric keyだけの合成correctness pilotに限定するか。後者では全support・総RPE価格を主張しない。
2. 保存N維持か、同じenergy/全失敗率でのN再評価を認めるか。有限precision候補と追加raw disk上限を、その判断後に固定する。
3. guard方式metadataのない4 legacy recordを除外したまま進めるか、別途guard再資格を行うか。今回metadataを補完していない。

別途許可後の**Codex**の最小候補は、共通q1/r1のX/Yで、D Gaussian frameとactual inverse、Dのscalar/half-angle、RTE order0/order2・Z/ZZ中央、fixed H/Sdg、CSWAPの代表literal-native roleを少数選び、同じprecision候補・guard・新規key取得機会を両armへ与える局所pilot。coverage上の共通153 positive keyは保存wordの再利用候補だが、precisionに合うrecordだけを採用する。**具体key・precision・新規合成件数は今回固定しない。** 欠損一覧は機械可読結果に保存した。

strict guard、±角、word順序、W、actual adjoint、controlled relative phaseを局所検査し、U_N→U_CTだけを旧diskへ加える。価格が得られる場合も有理期待multiplicityで扱い、LP下界を実行可能価格とみなさない。Reference vacuum全support価格が未資格なら、その部分と全18round総Tを未資格のまま保つ。T-depthは別課題。

後続STOP候補：key/guard不一致、vacuum word価格境界欠損、raw disk接続不成立、正半径/beta/alias/failure条件不成立、公平条件や科学条件の変更が必要、又は許可範囲完了。新gadget・位相生成法・precision gridへ自動拡張しない。

## 技術検査

人工9検査PASS。公開前にSOURCE hashと保存結果の対応、JSON/schemaとSTOP、16個の文書リンク、旧manifest全entryの不変、索引/概要の追記だけであること、差分の空白検査を確認した。旧科学suiteは再実行せず、著者自身の技術検査を独立レビューとは呼ばない。

## 終了

**mandatory STOP**。Stage B価格未資格とCT-0範囲完了を記録した。単なる「不可能性」や研究no-goとは解釈しない。次担当は**GPT／ユーザー：全support価格の資格方針、precision/誤差予算、独立した最小後続scopeを判断**。Codexは新規合成・全18round総Tへ進まない。
