# H2・H4・H6の高次PF費用変化：承認済み保存値解析 run01

固定12候補の保存値解析を一回正常終了した。費用分解22件の有理数恒等式と、44 endpointのshot関係がすべて整合した。主ε=.0005ではH2はFullがS2・PRがYoshida4で低費用、H4/H6はFull/PRともYoshida4で低費用だった。**H6 PRは単価Cが上がってもN削減がそれを上回る**。また、H4/H6の方向関係は登録精度によって変わった。

これは固定候補・条件付き保存値についての会計上の説明であり、qから独立したPF次数の因果効果、母平均優位、strict identityの確定ではない。旧strict COMPAREはBLOCKEDのまま。新しいsampling/build/compile/simulation、Hamiltonian/reference再計算、整数shot採用、標本分散/SE計算は0。追加解析へ進まずmandatory STOP。

## 実行条件・承認・SOURCE

linear H2/H4/H6、隣接1.0 Å、STO-3G、tol-only DF1e-8、rank3/10/19。T=.8、α_axis=.025、P=0、USUAL_SIGNED、既存empirical allowance/reference/normalization、symmetric-directional、状態準備を除くfull cosine/sine Hadamard wrapperのcompiled RZ費用。H2 Full prefix3/q4・q2、PR prefix1/q4・q2（n33/34）。H4 Full prefix10/q8・q4、PR prefix3/q8・q4（n48/33）。H6 Full prefix19/q8・q4、PR prefix5/q8・q4（各n33）。すべてS2/Yoshida4の順、PR K4・ABS_TIME_PER_OUTER32。

ユーザーは固定契約に対し「この固定契約の数値再解析のみ承認する」と回答した。[承認記録](approval_receipt.json)。旧H2/H4/H6 grantを再利用せず、新しいH246専用grant/rootで実行した。

| 項目 | 値 |
|---|---|
| 科学解析SOURCE | 2e59a55beec8159c7832ff4ff02513b8976c0eb9 |
| 固定契約SHA256 | 2478152414dce3c468e48c17e04d0addf3143794190be686593bc94eae1708e8 |
| 入力manifest SHA256 | 30979e34bda28d0ed70fc53b8a8004e22cd94a77d45a45c4101c08a7cb42355a |
| 結果SHA256 | 3614edf7afb06891428b48a0a8969a56b7bf699b3647cf7a1cdb3cd0df3ab6c0 |
| 実ledger | bootstrap1 / bundle_load1 / 分解22 / shot関係44 / 構成12 / 精度整理6 = 86 event |
| native終了 | exit0・reaped、約2.171秒、attempt1/retry0 |
| 保全 | SOURCE/input/runtime1,333 file、既知1,622 file変更0 |

[準備REPORT・入力索引・人工テスト](../../track_a_h246_order_cost_analysis_ready/2026-10-24/REPORT.md)、[fresh preflight](fresh_preflight.json)、[native completion](native_completion.json)、[wait記録](native_wait.json)、[ledger receipt](execution_ledger_receipt.json)、[SOURCE](SOURCE.json)。封印Python3.11で既存shot関係・schedule/allocation・sealコードを使用し、元の数値SOURCEは変更していない。既存人工10件・封印runtime人工86 event PASSの後、今回の承認済み実行を一度だけ行った。

## A：主精度のC/N/G分解

表はYoshida4/S2の比。1より小さいほどYoshida4が低い。主比較では保存平均C_cos=C_sinなので、軸別一般式のC_eff比と共通C比が一致する。

| 系 | 方法 | C比 | N_total比 | G比 | 低費用 |
|---|---|---:|---:|---:|---|
| H2 | Full | 1.488218 | 0.680909 | 1.013341 | S2 |
| H2 | PR | 0.748759 | 0.706199 | 0.528773 | Yoshida4 |
| H4 | Full | 1.499081 | 0.658015 | 0.986417 | Yoshida4 |
| H4 | PR | 0.966765 | 0.702462 | 0.679116 | Yoshida4 |
| H6 | Full | 1.507583 | 0.467147 | 0.704263 | Yoshida4 |
| H6 | PR | 1.014120 | 0.548725 | 0.556473 | Yoshida4 |

H2 Fullは単価増加をN削減が補い切れずG比が1を越える。H4 FullはN削減が単価増加を僅かに上回り、H6 FullではN削減がより大きい。PRはH2/H4でCとNが共に低下する一方、H6ではC比が1を上回る。したがって「PRは高次化すると必ず単価も下がる」という説明は今回の3系を一律に説明しない。保存Nの削減との組合せでGを説明する必要がある。

一般の軸別会計 `G=Σ N_a C_a` を正本とし、`R_G=R_Ntotal×R_Ceff=Σ w_S2,a R_N,a R_C,a` を全22比較でexact確認した。加法項はS2基準の `ΣN0ΔC`、`ΣC0ΔN`、`ΣΔNΔC` と定義し、cross項を一度だけ数えた。

| Full・主精度 | C項 | N項 | cross項 | ΔG |
|---|---:|---:|---:|---:|
| H2 | 1.806888e11 | −1.180951e11 | −5.765612e10 | 4.937562e9 |
| H4 | 6.872509e12 | −4.709249e12 | −2.350297e12 | −1.870364e11 |
| H6 | 5.037260e13 | −5.288037e13 | −2.684118e13 | −2.934895e13 |

表示は丸め値。全Full/PRのexact分解・元G・保存paired G SEは[22比較CSV](cost_decomposition_22.csv)、主精度は[6比較CSV](main_C_N_G_6.csv)。元Gのfloat演算順を保持し、別fieldのexact C×N再構成と8 ulp以内で整合した。PRのpaired SEは同一trajectory内の費用変動であり、異なる公式をpairingせず、差のSE・比のCI・Fullの不確かさを新しく推定していない。

## B：q・RTE構成と回路単価

元SOURCEはS2の1 signed pieceとYoshida4の正・負・正の3 signed pieceを使用する。各比較でqは半分になるため、Fullのpiece数はH2で4→6、H4/H6で8→12になる。これは単価増加と対応する構成上の変化だが、compiled RZの操作別寄与を測ったものではない。

PRのABS_TIME_PER_OUTER32は、S2では各outerの1 pieceへ32、Yoshida4では3 pieceへ10/12/10を配分する。各PR候補の最初の保存trajectory6件で、SOURCE scheduleと原eventのpiece数・各pieceのRTE数が一致した。

| 系 | S2→Y4のq | 代表軌道のRTE event数 | basis switch数 | 同じbasisの隣接数 |
|---|---|---|---|---|
| H2 PR | 4→2 | 128→64 | 8→12 | 120→52 |
| H4 PR | 8→4 | 256→128 | 72→47 | 184→81 |
| H6 PR | 8→4 | 256→128 | 128→58 | 128→70 |

これら6軌道では保存Taylor orderは0、event phaseは+1、identity applicationは0だった。値は**最初の登録軌道だけの記述**であり、全214軌道のoperation stream保証や未観測event費用0の主張ではない。元cohortの保存stratum統計は結果JSONに引用した。

RTEイベント数の減少、deterministic prefix pieceの増加、Gaussian basis reuseが同時に関わるため、event数だけからCを予測しない。特にH6 PRのC比>1は、RTE数が半減すればcompiled単価も下がるという単純な説明では捉えられない。basis adjacencyをcompiled RZ削減数へ換算したり、次数単独の因果効果へ外挿していない。[代表RTE CSV](representative_RTE_6.csv)、[全構成と保存統計](analysis.json)。

## C：bias・normalization・headroomとshot

元 `explanation_side` による連続式 `N* = 2 B_upper² / h² × log(2/α_axis)`、`h=ε/√2−|bias|−u` と保存整数の整合は44/44だった。保存u/log_B_margin/referenceを変更せず、整数Nを再採用したものでもない。[88軸CSV](shot_relationship_axes_88.csv)。

主精度ではFullのbase log_Bは0のまま。H2 Fullのcosine headroomは0.000273249→0.000337925、H4 Fullは0.000247923→0.000343971、H6 Fullのsineは0.000211996→0.000340657だった。保存bias減少に対応するheadroom増加が、確認されたN削減と整合する。

PRではYoshida4のbase log_BがH2で0.000477350→0.018531064、H4で0.000850509→0.033017948、H6で0.002107803→0.081810694と増加する。式のnormalization側はshotを増やす方向に働くが、主精度ではheadroom側の変化と合わせたN_total比がいずれも1未満になる。これは保存値間の会計関係であり、biasやnormalizationを別々に操作する反実仮想を実行した結果ではない。

H4/H6 PRでは保存outer_PF_or_discard_biasとfinite_RTE_biasを同じ保存総biasに結合できた。Yoshida4の有限RTE成分はH4で約1e-14〜1e-13、H6で約1e-13の保存診断値だった。主精度の総bias変化は主にouter側の保存成分に現れるが、この診断を厳密な系統誤差boundへ格上げしない。H2の成分は旧保存signalと現在DIAGNOSTICSのbit差を保持し、Fullの未保存成分を0として扱っていない。

## D：固定候補の精度依存性

同じ候補/prefix/q/cohortの登録5精度を比較した結果であり、各εでのbest候補再選択ではない。

| 系・方法 | ε=.05 | .01 | .005 | .001 | .0005 |
|---|---|---|---|---|---|
| H4 Full | S2 | S2 | S2 | S2 | Y4 |
| H4 PR | S2 | S2 | Y4 | Y4 | Y4 |
| H6 Full | S2 | S2 | S2 | S2 | Y4 |
| H6 PR | S2 | S2 | S2 | Y4 | Y4 |

H4では登録ε=.005/.001、H6ではε=.001で、FullはS2・PRはYoshida4が低費用という方向関係が現れた。主ε=.0005だけでは両系ともYoshida4が低費用なので、方向逆転の有無を分子名だけに帰属させることはできない。

H2は主ε=.0005でFull S2・PR Y4。他4精度16セルは完了比較とallowance政策が一致する保存endpointを未取得のため未評価のまま。H2の精度依存性がないと結論していない。登録点の間の境界・交差εは推定せず、補間しない。[22精度方向CSV](precision_directions_22.csv)。

## 説明できた範囲・残る条件

固定保存値の範囲では、C/Nの積恒等式と加法項によりGの変化をexactに説明でき、保存bias/u/log_B/headroomとNの整合も確認できた。q/RTE構成が変わり、精度によって会計方向が変わることを記述できた。一方、PF次数のみの因果効果、全軸operation別のcompiled費用寄与、母平均・希少event・厳密bias保証は未確定である。

H6 Fullの旧中断stage由来complete pairの受入と、元parent exit/reap不明は保持する。H2のstrict identity残条件、保存referenceの条件付き精度、Full/PR間の末尾差、有限探索・事後固定選択も残る。今回の検算を独立した外部再現や詳細GPT/Claude科学レビューと呼ばない。

追加が必要とユーザーが判断した場合の最小候補は、まずH2の受入済み政策による他4精度の保存入力会計を別契約で準備すること。次数単独の説明を求める場合も、同じq/prefixの保存証拠が揃う範囲を先に確認する。新規compile/samplingや全候補再計算を既定の要件にしない。[限界と後続候補](limitations_and_next_steps.json)。いずれも今回の承認には含まれず未実行。

## 終了監査と次の担当

[最終検証](final_verification.json)、[保全](preservation_audit.json)、[file inventory](file_inventory.json)。既知1,622 fileと、完了runの認可/ledger/marker/resultを含む93 fileを文書化後にもSHA照合した。元SOURCE・入力・結果・既存grant/STOP・H6資産を変更せず、他worktreeのHEAD凍結も要求していない。公開対象は集約結果・出典・SHAであり、private raw/event/seed/内部path・有効grantを含めない。

次はユーザーが結果と追加解析の必要性を確認する。Codexは今回の解析を終了し、追加計算・詳細科学レビュー・最終freezeを自動開始しない。[STOP](STOP.json)を保存しmandatory STOP。
