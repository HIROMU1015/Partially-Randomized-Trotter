# PR-DF LD2：Q3候補の保存価格照合とPF比較の次段階

2026-11-15 JST。保存資料・SOURCE・metadataの読取と判断のみ。起点 `85dab58d6d29d54502a7650c129bc38a9bb5ec99` から、専用branch `codex/d2-pr-df-ld2-saved-price-assessment-20261115` と専用worktreeを作成した。元LD1 branchは変更していない。

- **今PF比較へ使えるもの**：LD2自身のQ3適合済み物理候補・配分、source-bound request/profile、PF 2基準の固定総額。対象26軸に一致する有限直接価格・S/E[S]・残差・評価済みモデル・非自明期待価格boundは、確認範囲で**未同定**。価格を0として比較できない。
- **次の有限作業で使えるもの**：元H/state/snapshot・prefix2の入力出典、corrected scalar request、同path S/C取得経路と既存のLD2専用4セル案は再利用可能。ただし準備・SOURCE対応を確認して新たに評価する必要があり、小さい較正だけで26軸E[C]が揃うとはしない。
- **優先推奨**：まず **LD1 m9-Xの1軸だけで、対象Iの期待compiled価格に使える片側下界が作れるかを確認**する。新path/compile0のSOURCE論証を先行し、成立した場合だけ1軸下界とshot加重部分下界を評価する有限案。LD2の価格整備より入力・方法が揃っている。ただし成功は未確定で、LD1対PFが決まってもLD2やPR-DF方式全体の順位は決まらない。

[26軸正本](ASSESSMENT.json)／[26行CSV](AXIS_MAP.csv)／[旧raw照合](RAW_CATALOG_CHECK.json)／[次の一作業](NEXT_ONE_TASK.json)／[統合HANDOFF](HANDOFF.md)／[静的検査](CHECKS.json)。**新しい価格予測・総額・下界の算出、LD1数値のLD2への移送、Q3、sampling/build/compileは全て0。**

## 1. 固定LD2候補と価格quantity

B `a82b0f0ec7f0f11d9981ddb2343fcd5378a87503:docs/research/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json#candidates[1]` が正本。labelはPR_DF_LD2、statusは `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`、allocation IDは **PR_DF_LD2_TOTAL_Q3_SAVED_CONTROL_01**。NはX/Y各軸で **[64,60,56,52,47,43,38,34,30,25,21,17,12]**。同じ配分のLD1もQ3適合済みだが、同費用又はLD1が最安という意味ではない。

- original configuration：`1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`
- total decoder view：`e6fe7e6845dd5eae8829f54b8bbae655ccf1c81c320c4d0ee1bfed0399bb5583`
- 元candidateとtotal candidateのcanonical IDはAXIS正本の別欄へ保存。0b4b2d94のALIGNMENT rows338〜363、C85557306のPR_DF_COVERAGE coverage26〜51、D748be687のrequest26軸へphysical/m/axisで対応する。

対象は保存linear H4/1.00 Å/STO-3G/tol-only DF10、元H/state/snapshot、one-body＋元順DF prefix2を決定論、tail fragment2〜9をfinite RTE、S2/Δ=.1/K2/元exact κ、h=.1/M12である。Bのsimulation_policy.target_lawは **ideal_finite_RTE_law**。適合は保存ideal raw・IID二項モデルの固定配分Q3への適合で、実PCG64/十分N/最適N/global RMSEを認定しない。

価格は状態準備費P=0、scalar c+s_Rを総Tに一回、symmetric-directional、P0、**測定付きfull controlled Hadamard X/Y wrapperのcompiled RZ数**。基準はnew ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend/coupling/layout追加指定なし。旧requestのscalar/keyを後続runへcopyせず、e8273676のcorrected total-scalar bindingを使う。今回そのrequest生成やH/preparation復元は行っていない。

## 2. 26軸の証拠状態

下表の「未」は**確認範囲で対応記録が未同定**の意味で、価格0や世界中に保存物が存在しないという主張ではない。CSV/JSONでは価格・期待値・boundsをnull/空欄とし、有限観測・モデル参照・対象law期待値を別欄にした。全軸に同じ欠測状態が続くが、IDとSOURCE locatorは軸別に保持した。

| m | 軸 | physical ID | q / R | N | 直接C | S/E[S]・残差 | 保存モデル/予測 | 期待C/非自明bound |
|---:|---|---|---:|---:|---|---|---|---|
| 0 | X | ce7ef02e7926c1a6c924 | 1 / 1 | 64 | 未 | 未 | 未 | 未 |
| 0 | Y | ce7ef02e7926c1a6c924 | 1 / 1 | 64 | 未 | 未 | 未 | 未 |
| 1 | X | 8f889f60c3a813858814 | 2 / 2 | 60 | 未 | 未 | 未 | 未 |
| 1 | Y | 8f889f60c3a813858814 | 2 / 2 | 60 | 未 | 未 | 未 | 未 |
| 2 | X | 8a5aad480758a7cff292 | 4 / 3 | 56 | 未 | 未 | 未 | 未 |
| 2 | Y | 8a5aad480758a7cff292 | 4 / 3 | 56 | 未 | 未 | 未 | 未 |
| 3 | X | 334b1fd35cb2f1ad32b0 | 8 / 5 | 52 | 未 | 未 | 未 | 未 |
| 3 | Y | 334b1fd35cb2f1ad32b0 | 8 / 5 | 52 | 未 | 未 | 未 | 未 |
| 4 | X | cef49970538ca248d77e | 16 / 10 | 47 | 未 | 未 | 未 | 未 |
| 4 | Y | cef49970538ca248d77e | 16 / 10 | 47 | 未 | 未 | 未 | 未 |
| 5 | X | 5812e05afec3a9cdcbf4 | 32 / 20 | 43 | 未 | 未 | 未 | 未 |
| 5 | Y | 5812e05afec3a9cdcbf4 | 32 / 20 | 43 | 未 | 未 | 未 | 未 |
| 6 | X | fe234d98b289ae91869c | 64 / 40 | 38 | 未 | 未 | 未 | 未 |
| 6 | Y | fe234d98b289ae91869c | 64 / 40 | 38 | 未 | 未 | 未 | 未 |
| 7 | X | 8425b1e4da150d855b29 | 128 / 80 | 34 | 未 | 未 | 未 | 未 |
| 7 | Y | 8425b1e4da150d855b29 | 128 / 80 | 34 | 未 | 未 | 未 | 未 |
| 8 | X | 3f8def82f66ab24905a1 | 256 / 160 | 30 | 未 | 未 | 未 | 未 |
| 8 | Y | 3f8def82f66ab24905a1 | 256 / 160 | 30 | 未 | 未 | 未 | 未 |
| 9 | X | 4e84fb24b5fc0777a44b | 512 / 320 | 25 | 未 | 未 | 未 | 未 |
| 9 | Y | 4e84fb24b5fc0777a44b | 512 / 320 | 25 | 未 | 未 | 未 | 未 |
| 10 | X | 44077408e13b64b85978 | 1024 / 640 | 21 | 未 | 未 | 未 | 未 |
| 10 | Y | 44077408e13b64b85978 | 1024 / 640 | 21 | 未 | 未 | 未 | 未 |
| 11 | X | 93a7c992650daf2be173 | 2048 / 1280 | 17 | 未 | 未 | 未 | 未 |
| 11 | Y | 93a7c992650daf2be173 | 2048 / 1280 | 17 | 未 | 未 | 未 | 未 |
| 12 | X | f73c8ae19a25ef249f0a | 4096 / 2559 | 12 | T2 STOP/価格なし | 未 | 未 | 未 |
| 12 | Y | f73c8ae19a25ef249f0a | 4096 / 2559 | 12 | T2 STOP/価格なし | 未 | 未 | 未 |

m12の2軸は所在不明だけではなく、同candidateの旧T2取得試行が存在し、sampling前STOP・価格未取得と確認できた。元STOPを再開する許可にはしない。

### 公開済み台帳・raw・receiptで実際に照合したもの

| 保存物 | 照合結果／使えない理由 |
|---|---|
| C85557306 PR_DF_COVERAGE、元3ae2e605 COVERAGE、0b4b2d94 ALIGNMENT | LD2の全26軸でfinite_cohort/structural_meanが空、E_C=null。C1は `FIXED_PREFIX1_MODEL_OUTSIDE_PREFIX2_DOMAIN`、additiveは `IMPLEMENTED_NOT_EVALUATED`。後者は実装の存在で、評価済みLD2予測ではない |
| D748be687 PRICE_REQUEST_COVERAGE | LD2のq/R/physical/profile/measurement/law-selectorが揃う。compiled referenceと実preparation hashは未取得のmetadata準備。request-readyを価格取得済みとしない |
| carrier155529e2の旧価格raw 5台帳 | exploration481、confirmation39、fixed-delta1、reused203、extension22 recordをmetadata照合。**対象13 physical IDの一致0、別IDの同一(system/method/formula/prefix/T/q/R/K) tupleも0**。既存価格の平均・残差・統計は再計算していない |
| 同raw内のH4/PR/prefix2 | explorationに11 entry、confirmationに3、extensionに3がある。旧LD2価格そのものは存在するが、対象候補とT/q/R/K/公式が異なる。cohort重複entryを独立な新規標本として数えない |
| 例 `134088ad0d3333745e3a` | public exploration/confirmation recordはT=.8/q4/R4/K2。現LD2 m3はT=.8/q8/R5。元trial result.jsonのlocatorも同public rawに残る。これをm3価格へ移さない |
| 例 `f36dae42438bf274a1b7` | T=1.6/q16/S2/K2まではLD2 m4と一致するがR=4。現m4はR=10、physical IDも違うため不採用。Rだけ変えて価格を再ラベルしない |
| raw/provenance・compiler SOURCE | public scientific_provenanceのH4 DF10/元H-state-snapshot、full measured wrapper/P0とcompiler hashを確認。155529e2のcompose→providerに対してnew ZZの39ddd59eは別lowering。compiler設定が同じだけでは価格同等にならない。今回の除外はphysical tupleだけでも確定する |
| 92ff17deの136候補inventory | 全てmethod FULL、全rank prefix・決定論価格/信号のinventory。PR/LD2ではない。manifestにある上記carrier/rawへの入口は再利用するが、Full PF価格をLD2へ供給しない |
| T2 752b9747のm12 REQUEST/RESULT/WORKER_STOP | 同configuration・physical f73c8ae1…・q4096/R2559を確認。SOURCE8291f16c…でpreparation_contextのscalar照合に失敗、sampling前STOP、compile_log=[]、mean_compiled_price=null。e8273676の修正は技術準備であり再取得の結果ではない |
| C1 2e6a0cdf、P1/P2 39ddd59e、LD1 E[S]/凍結/range | C1のH4はprefix1/R4,8/旧lowering、P2/ES/凍結/rangeもLD1。prefix2のS observable・head/tail・law・coefficientsとは一致しない。LD2の値・幅・補正として移さない |
| df5e5c45の「LD2全round価格計画」 | **PR-Pauli LD2**の計画。configuration d7dd4f52…、Pauli辞書/ordered-product/phase/loweringが別。PR-DF LD2の1fa33556…とは別であり、そのCAL/holdout/m9等を採用しない |

5 rawの固定commit/path/key・metadata候補・trial locatorと除外理由はRAW_CATALOG_CHECKへ保存した。公開rawに記載された元receipt locatorは追跡したが、非公開receipt本文を公開artifactへ複製する操作は自動承認レビューで拒否されたため、本文の複製は行っていない。採否の根拠は公開済み物理metadataで足り、非公開本文を新しい価格入力にしていない。

この結果は「現LD2と同じwrapperの価格が絶対に無い」ではなく、**同一候補への受入可能な価格を確認範囲で同定できない**という結論である。特定の旧価格は条件不一致で不採用、未列挙cache/archive/他serverは所在未確認として分ける。fingerprintだけから候補を逆算する新しい探索・H4 buildはしていない。

## 3. PF 2基準へどこまで比較できるか

B2 `1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/RESULT.json` の値を引用し、再集計しない。

| 基準 | 固定総額 (RZ) | 現LD2との順位 | 一つ得られれば比較が前進する量 |
|---|---:|---|---|
| B0I p7 S2（16h/7） | 593,555,352 | 未判定 | LD2の同lawで正当な期待費用の**部分下界**がこの額を超えればPF側が低費用。LD2側を示すには全体上界又は必要な期待価格表がこの額より小さい根拠が必要 |
| Full PF S2（16h/7） | 838,693,580 | 未判定 | 同じ部分下界がこの額を超えればFull PF側が低費用。LD2側を示すには同様に全体の上側が必要 |

任意の部分集合Aについて、有効な `ell[m,a] <= E_I[C[m,a]]` があれば、価格の非負性から

```text
sum_(m,a in A) N[m,a] ell[m,a] <= T_LD2
```

とできる。**全26軸の点期待値が揃う前に、片側順位が決まる経路はある。** ただし現LD2ではその非自明ellが未同定。有限pathの最小値、旧prefix2の最小観測、構造S又はE[S]をそのままellにしない。compiled後の相殺があるためC>=Sは仮定できない。部分費用の上界だけでは、残る正費用を無視してLD2全体が安いとは示せない。

PF側の比較も採択済みcompiler閉包・各固定shot配分に条件付く。両PFを一括してFull PFと呼ばず、未観測60軸の誤差/2%保証・十分N/global RMSE/5方式winnerを認定しない。LD1/LD2のQ3 MSE差、Rが小さいこと、headが大きいことだけから費用の順序は出ない。

## 4. LD2を将来評価する際に再利用できるもの

① **今使える**：Q3の同一物理候補・配分・law、A保存H/state/snapshot/prefix2 λ/tail入力の出典、corrected total-scalar request、compiler/wrapper契約、PF比較の相手の値。価格としては未接続。

② **有限評価を加えれば使える**：Dのprefix別Runtimeとnormal→同path S/C→X/Y個別full-wrapper compile、Cの既存予測/残差/reader。C85557306のPR_DF_VALIDATION_PLAN §最小の次段階4には、**LD2 m0 q1/R1、m3 q8/R5、m6 q64/R40を較正、m4 q16/R10を別holdout**とする既存案がある。LD2 E[S]がないため、先にLD2自身の構造期待値を評価するか、full Cモデルを設計する必要がある。これは未採択の案で、今回nや追加compile件数を新設定していない。少数pathを取得しただけでは対象Iの母平均、長round移送、片側順位は認定できない。

③ **不足**：LD2自身のprepared component/basis/hashと保存law、現条件の直接paired S/C、E_I[S]又は直接E_I[C]の基準量、同lawの残差推定と不確かさ、長round/全supportのcompiler根拠、必要な片側平均bound/精度。元入力出典や同じsource名は、生成・計算・資格取得済みの証拠ではない。

## 5. 次の一作業の選択

| 候補 | 答えられる問い | 今回の判断 |
|---|---|---|
| LD2の価格評価を先行 | LD1を代表にしたことでPR-DF対PFの結論が変わり得るか、LD2がLD1/各PFより安いか。同lawの価格/上下界が得られた範囲で初めて分かる | **代表選択の影響を解消するには必要**。しかし現LD2は26軸の基準S/価格/モデルが未同定で、少数短roundの観測だけではこの問いを解決しない。入力復元・独自較正/構造/lawからの整備が必要 |
| **LD1片側下界を先行** | 少なくとも固定LD1と2つのPFのどちらが低費用かを、全26軸の期待値を埋めずに決着できるか | **推奨**。固定S/kernel・word/new-ZZ/compiler調査・26軸対応が揃っており、1軸のSOURCE論証と有限評価を試せる。成功未確定。LD2の価格順位を解消する作業ではない |

### 推奨scope：LD1 m9-X、1軸の下界成立可否

- 対象：**LD1 m9-X、physical `06a78c7f9856f373336f`、T51.2/q512/R1020、N=25、保存I**。LD2 m9（4e84fb24…/R320）とは別。LD1の既存数値は優先順位判断だけに用い、LD2の26軸へ入力しない。
- 選定理由：85dab58dの既存RESULT.rows[18]には、この1軸の**モデル参照加重項2,828,507,555.268906326 RZ**が保存され、両PF基準を上回る参照規模がある。m12より短いq/Rで一つの軸に絞れる。これは成功を示す下界ではなく、SOURCE確認を試す位置の選択理由に限る。
- 再利用：同RESULTのidentity/structural固定locator（C9d95d3ea COVERAGE.values[18]）、旧950577f4の軸対応、dcf905fcのSOURCE inventory、39ddd59eのnew-ZZ、元Accounting/compose/provider、標準Qiskit1.3.0の固定SOURCE。m4独立holdout・凍結原本は変更しない。
- 必要な新規作業：**SOURCE下界論証1件**。全supportで `h(omega)<=C(omega)` を示せるならE_I[h]、又は `C−S>=l` を示せるなら保存E_I[S]とlから期待C下界を得る。compilerのgate数上界/観測した相殺量を下界へ反転しない。lawに関して平均的な下界を使うなら、その確率weightと期待値の根拠も必要。
- 論証が成立したときだけ新規算出：軸別下界最大1件、`L_partial=25*ell_9X` 最大1件、PF 2基準との比較最大2件。**新path/H4 build/compile0、再fit/Q3/全13round総額0**。今回これらの新下界・積は算出していない。
- 得られる判断：L_partial>838,693,580なら固定LD1は両PFより高費用、593,555,352<L_partial<=838,693,580なら少なくともB0I側を判定できる（Fullの等号/未判定は分ける）。小さい又は不明な下界からLD1が安いとはしない。
- **STOP**：正当な非自明下界を作れなければUNKNOWNで停止。自動でY/m12/LD2/追加pathへ広げない。law/physical/SOURCE/compiler不一致、保存不能では部分論証と失敗箇所を保全して停止。成立時の1軸結果を公開して停止。実行は別指示。

LD1の大きい参照総額も、m4の小さい記述的差も、真のLD1価格がPFより高い証明ではない。今回の推奨は**少ない追加作業で一つの固定比較を決着できる可能性**を優先する判断で、PR-DF方式全体をLD1一候補で代表し続ける決定ではない。LD1がPFより高いと判定できても、LD2次第でPR-DF対PFの結論が変わる余地は残る。

## 6. 重複照合・ロードマップ差分・終了

専用branchの開始HEADは85dab58d、既存LD1 branch/旧STOPは保全。最新Cは61f249e6からa1aa38dfへ進んだがFull-H F1診断、Aはdf5e5c45から2f3bf20cへ進んだがPauli LD2のS0監査準備であり、現PR-DF LD2の同等調査ではなかった。旧T2 m12は同候補の停止試行として追加確認した。関連Git履歴、C/D coverage、既知raw5台帳、136候補manifest、関連local job名を確認し、同じPR-DF LD2価格評価やLD1片側下界の公開完了/同名jobは確認範囲で未同定。他担当のworktree/jobは変更せず、未確認の外部実行まで不存在とはしていない。

ロードマップへの差分：LD2の「価格未同定」を26軸のQ3/physical/台帳/raw/除外理由付きで具体化。旧prefix2価格が存在することと、今回候補へ再利用できることを分離した。次はLD1 m9-Xの片側下界の成立可否を限定確認し、LD2固有の価格整備は代表選択の影響を残す独立課題として保持する。ロードマップ本体は統合担当が更新する。

静的検査は26軸のID/N/候補・固定locator、rawのmetadata一致/不一致、文書/JSON/CSV、旧証拠保全に限る。科学module import・価格予測・総額/下界の算術・Q3/回路/sampling・回帰suiteは起動しない。commit/push・remote SHAと公開成果物一致を確認して **mandatory STOP**。
