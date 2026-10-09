# Track B G4：条件付き分離認証・matched CTS結果／GPT handoff

**G4-Aは条件付きB2分離を認証、G4-Bは同targetのCTS比較を完了した。科学実行はmandatory STOP。**
x=1/4の保存J1 lawが限定B2 classの下界を下回る一方、取得したCTSの一つのlawは
そのJ1 lawをT/CX/1Qで下回った。この二つを混同せず、研究採否・次scopeをGPTへ返す。
Tをprimary、1Qをauxiliary/exploratoryに維持する。研究GO、新規性成立、RA-RTE採択とは判定しない。

## 場所・source・認可

- execution branch: `track-b-g4-conditional-separation-cts-20261009`
- independent worktree: `.worktrees/track-b-g4-conditional-separation-cts-20261009`
- base/G3 immutable result: `3b0fa70b47848e72ef9a9c9e13afc3164962f7cd`
- G4-A source: `dbc9c43ec7d991dee006b1f7c25e7b8ad009f345`
- G4-B source/specification freeze: `954571951e23eacc33ab4cca1629c94f32f9292c`
- user authorization: 「これに沿って進めて」＋[G3再レビューv2](../../research/track_b_G3_scientific_review_20261009_rereview_v2.md)§13。
  A成功時だけB、Cはdesign only。技術手順ごとの再承認を要求しない指示に基づく。
- v1/v2 reviewだけを選択してbyte-exact copyし、元path/size/hashは
  [review_input_identity](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/review_input_identity.json)に保存。

旧G1/G2/G3/RA-D0のsource・contract・authorization・result・marker・STOPを変更しない。
新しいG4 markerを各phaseにexclusive createし、一回/retry 0で実行した。
Track Aのworktree・status・artifact、root worktreeの未commit資料は編集していない。

## 固定taskと証拠区分

2-qubit distinct-basis controlled finite P3(-ixR)、R=3/4 Q0+1/4 Q1、
Q0=ZI、Q1=V†IZV、V=exp(-iπXX/16)、σ=+1、既知x={1/8,1/4}、入力|00〉。
分子・geometry・DF rank・split L_D・PF delta windowは対象外。
full operator meanを保つtaskで、状態限定の古典解への置換は行わない。
追加workspace 1、axis accuracy 1/200、failure allocation 1/5280、shot cap10^9/axis。
native IRの共通exact cancellation後のadditive synthesized-primitive Clifford+T count。
G_1QにはRe/Im readout合計5/axis-shot-pairを含める。

known/development条件を再利用したsource-bound local evidence。
外部再現、immutable CI、independent held-out、DF/PR総費用評価ではない。
original D0のU3<L2 witnessや旧結果の再分類ではない。

## G4-A：独立算術と比較class

[独立証明](g4_A_independent_proof_20261009.md)、
[frozen scope](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/scope_A.json)、
[result_A](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/result_A.json)。

ordinary9＋PTSC-K0 27＋A27=63 pure profiles/xを、元event/cost/errorから別実装で再構成した。
計126 profiles、T/native1Q/readout1Qの378価格計算。旧G2/G3算術helperはimportしない。
normalized方向intervalは平方比較、各source eventとsynthesis identity/count/error boundは旧R1 rawへ照合。
G2保存区間とのoverlapも成立したが、その区間を新下界の入力として使っていない。
13 off-domain/synthetic focused tests PASS。

認証classは、固定辞書・内部IID label・3 precision・既存三representationの非負配分と任意凸混合、
同じ辞書上の任意full-support proposal、固定Bernstein sufficient-shot policy。
pure profilesへの凸還元とCauchy–SchwarzからG≥4 ln(10560) r²を証明し、
exp(37/4)<10560の有理証明による保守的下界37 r_lo²を使用する。
zero T-costはそのまま扱い、最適ISのinfimumの達成を要求しない。

digital classは、あるideal B2 coefficient cに対応する非負tilde cを持ち、
e≥||tilde c−c||1をbiasへ戻し、remaining accuracy>0とする同辞書実装。
全eventでh_i+r d_i≤rを実データ上で認証し、下界を延長した。
degree residualだけを許す旧K2/K3の任意点には延長していない。
別dictionary、Pauli cancellation、stratification、state-dependent variance、別confidence方式を含まない。
下界はこの**固定予算設計policy**の下界で、物理的必要shot数の下界ではない。

| x | T lower | 保存J1 1Q-selectedのT | readout付き1Q lower | 同一J1 lawの1Q | 認証した分離 |
|---|---:|---:|---:|---:|---|
| 1/4 | 176,461,958.04 | 174,820,322.72 | 465,306,606.84 | 461,712,911.04 | T/1Qともstrict |
| 1/8 | 178,806,307.49 | 179,183,039.36 | 462,249,688.60 | 463,389,609.27 | この下界では分離せず |

exact rational符号で判定し、表だけ表示を丸めた。x=1/4のT gap≈1,641,635.32、1Q gap≈3,593,695.80。
1Qで抽出した同一lawの構成例であり、T-primaryの新独立実験へ遡及変更しない。
x=1/8でlaw・precisionを増やして救済していない。

status: `G4_A_STATED_CLASS_SEPARATION_CERTIFIED`。A wall0.4434s/CPU0.4433s、peak RSS約44MiB。
matrix/circuit/synthesis/LP/sampling/DF/GPUの新規呼出し0。
独立した**実装・算術チェック**であり、旧R1保存cost/error sourceへの依存は残る。外部再現ではない。

## G4-B：文献CTSの同target finite specialization

[synthesis前contract](g4_B_matched_CTS_contract_20261009.md)、
[scope_B](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/scope_B.json)、
[result_B](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/result_B.json)。

Peetz/Smart/NarangのCTS、出版本文Theorem1/Eq.(5)–(6)とSupplementary Note5 Eq.(13)–(15)を
M=3へ有限化したoperator ensembleを使う。
[一次文献](https://www.nature.com/articles/s41534-025-01168-w)。
coherent first-moment adapterは文献のSCU channel taskと区別し、同じfull P3 meanをexact Pauli algebraで確認。
I−5x²/16 I−3cx²/16 ZZ+iΣb_kP_kを、real correction二event＋common-angle四rotationで実装する。
負のidentity補正とrotationのidentityを融合する別法は採用していない。
controlled phaseはstrictに保持し、Ls rational approximationとcoefficient L1 errorも課金した。
I1 Pauli情報を使う対照であり、この小toyでI1が取得不能だとは主張しない。

旧fixed runtime（Python3.10.12、pygridsynth2.0.0、mpmath1.3.0、package lock/.py tree一致）、
up_to_phase=false、旧precision三つとsynthesizer optionsを保持。
17 focused off-domain tests PASS後にsource固定。

12 synthesis keys（2x×3ε×signed angle）、28 native event条件、162 pure profiles、
4,686 deduplicated finite proposals、486 profile/axis winnerを取得した。rejected0。
6 selected complete lawsを保存し、q/weight/full Pauli mean/m2/L/bias/integer-shot/resourceを認証した。
各axisのminimaは有限poolだけの選択であり、全CTS/proposal最適性ではない。

### T-primaryの有限pool比較

以下は各methodの**Tで最適化した保存law**のT座標。全座標を合成した架空lawではない。

| x | ordinary | PTSC-K0 | A | J1 | return | matched CTS |
|---|---:|---:|---:|---:|---:|---:|
| 1/8 | 191,903,766 | 189,163,062 | 179,989,518 | 179,160,992 | 187,808,511 | 150,889,450 |
| 1/4 | 212,872,955 | 203,581,579 | 177,552,401 | 174,720,368 | 189,832,929 | 160,268,880 |

CTS/J1 T比は約0.84220（x1/8）／0.917288（x1/4）。記述値であり、後付けmateriality/GOへ使用しない。
G3のall-axis winner pool集計とは別に、ここではsame optimized axis onlyで揃えた。
全exact vector・precision・proposalは保存JSON、36行の
[display CSV](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/resource_comparison_display.csv)を参照。

### 同一lawの三座標

次は各armの**1Qで選ばれた一つの完成law**の全座標。1Qはauxiliary。

| x | law | n/axis | G_T | G_CX | G_1Q（readout込） |
|---|---|---:|---:|---:|---:|
| 1/8 | 保存J1 | 831,327 | 179,183,039 | 4,465,885 | 463,389,609 |
| 1/8 | matched CTS | 1,038,128 | 151,106,411 | 4,365,462 | 388,150,680 |
| 1/4 | 保存J1 | 1,114,220 | 174,820,323 | 5,661,820 | 461,712,911 |
| 1/4 | matched CTS | 970,447 | 164,669,576 | 3,712,301 | 431,125,042 |

両xでこのCTS pointは指定J1 pointを三座標で下回り、workspaceは1で同じ。
全J1の劣位、一般I0への優越、PR全体・DFでの改善は結論しない。
x1/4ではこのCTS pointのshotsも少なく、同一の非負common overheadを各shotに加えても
この二点の順位はJ1側へ反転しない。方法別付帯費用を新たに仮定していない。

B status: `G4_B_MATCHED_CTS_COMPLETE`、wall38.0747s/CPU38.0726s、peak RSS218,732KiB、single process。
wall600/CPU480/RSS512MiB、per-key wall30/CPU20、32MiB output capを通過。
新しいscience Hamiltonian signal、quantum shots、trajectory、LP、DF/molecule/NPZ、GPUは0。
認可された12個の2×2 primitive strict guardsと28 native event IR取得はBの実行として明示する。
wrapper-level実行や4×4 Hamiltonian signalを生成したことにはしない。

## 保存照合・provenance・STOP

[saved-values audit](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/saved_values_audit.json)では
126 profilesの下界式、12 sequence identity/count/strict-bound binding、28 source-derived native cost、
486 winnerの保存会計、6 complete lawのcertificateを確認した。監査の新規取得0。
strict operator guardの再合成・再matrix evaluationは行わない。
phase source/scope/runtime identitiesとmarkerは保存し、retry0。

- A marker SHA256: `b05f20ce52bae8e8e34e743870b62167eedb0ba5d11ba75a88054cac32a8f3b7`
- B marker SHA256: `e0caf44c712c4d7b970334f26958227f404b817b14d0824fe8a78d983a739dd9`
- evidence inventory: [G4 manifest](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/evidence_manifest_v1.json)
- old inventory: 819 protected paths。index/synthesisのTrack B追記だけはprefix保持として別記し、
  旧科学source/result/contract/authorization/markerはbyte-exactを維持する。
- new STOP: [STOP.json](../../../artifacts/track_b_g4_conditional_separation/2026-10-09/STOP.json)

## GPTに戻す判断

[G4-C設計比較](g4_C_next_validation_design_20261009.md)で、common wrapper、p/basis transfer、
synthesisの離散費用、情報取得費用の四候補を比較した。どれも新条件は未実行。

GPTには、B2限定分離をどの学術的主張として残すか、I1 CTS後にI0構成の研究価値があるか、
理論/mechanism noteへ区切るか、次の独立検証一つにどの情報価値があるかを判断してもらう。
Codexは研究RQ・新規性・論文着地点・主algorithmを採択していない。
このhandoffの後に追加grid/angle/precision/synthesis/LP/scienceへ進まない。
