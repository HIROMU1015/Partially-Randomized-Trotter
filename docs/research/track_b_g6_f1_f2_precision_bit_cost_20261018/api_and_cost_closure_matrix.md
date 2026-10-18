# F1＋F2：今回の技術的閉鎖と残件

W1表の再掲ではなく、**今回具体化した受入条件/費用式**を記す。`CLOSED_BY_ROUTINE_DERIVATION`は既存lemma・SOURCEの直接整理/標準算術の合成であり、新理論受理ではない。複数statusは数理/実装/費用の異なる次元に付く。[機械索引](closure_index.json)

| ID/項目 | 既存保証・SOURCE | 今回のAPI条件/通常導出 | 実装状態・費用・未閉鎖 | 分類／次作業・受入条件 |
|---|---|---|---|---|
| F01 入口/domain | [MATH]§1、K138–145、G103–143/216–225 | [F1 §1](preflight_precision_contract.md)：exact Rat/正規化、σ、型・分母・bit長、Full m≥1対canonical m≥3、直接helperの契約を一つの入口へ整理 | 型の包括検査は未接続。読込/正規化をsetup Qモデルへ計上 | `CLOSED_BY_EXISTING_LEMMA` / `CLOSED_BY_ROUTINE_DERIVATION` / `SPECIFIED_NOT_IMPLEMENTED`。Codex実装は別許可後。全前提が失われず、低水準indexingをvalidatorとしないこと |
| F02 identity | [MATH]§1、K221–222、G105/109/237–251 | exact x=0をpositive sampler前に分岐、deterministic identityとして保持 | 数学は既存、G7 dispatcherは未実装。sampling/weight/angle費用を架空に発生させない | `CLOSED_BY_EXISTING_LEMMA` / `SPECIFIED_NOT_IMPLEMENTED`。新入口実装後、random zeroと区別しbit消費なしを受入要件にする |
| F03 order | [BIT]§5/K45–82/208–212、G221 | [F1 P1/R](preflight_precision_contract.md)：Δ/hを既存rootとnormalized intervalへ代入し、全orderのrelative/support guardに対する十分述語を具体化 | 述語のpreflight証拠生成/検証は未実装。費用は3n rootsとD(n,K_root+ℓ_m)のsetup項 | `CLOSED_BY_ROUTINE_DERIVATION` / `SPECIFIED_NOT_IMPLEMENTED`。新H/K selectorを研究成果としない。固定値が受入述語を満たす証拠が必要 |
| F04 labels/child全support | [BIT]§5、K230–231、G222 | P2のh≤ηp_minとh≤ηp_min/2。R(0,q_min)の直接適用 | 全到達parentに同じ十分条件、全語列挙なし。child lawはtrial中生成。親ごとの訪問PASSだけで事前保証にしない | `CLOSED_BY_EXISTING_LEMMA` / `CLOSED_BY_ROUTINE_DERIVATION` / `SPECIFIED_NOT_IMPLEMENTED`。実lawと同じidentityで固定Hの証拠を保存する |
| F05 d midpoint | [BIT]§5、K224–227 | P3：Δ<d_minとΔ/2≤ρ(d_min−Δ)が全uでpositive lo/relative midpoint guardを満たす | 未実装preflight。dとacceptanceの2 rootをlocal費用に計上 | `CLOSED_BY_ROUTINE_DERIVATION` / `SPECIFIED_NOT_IMPLEMENTED`。条件が不成立ならSTOP、丸め値の事後救済なし |
| F06 acceptance | [MATH]§7/[BIT]§5、K85–94/228–229 | P4：Δ<1/4、(2−η)Δ+h≤η/4。既存A_u下界とclip/floor誤差の直接合成 | 受理coinはpacket後。coin rejectでもchild law/再sum費用を支払う | `CLOSED_BY_ROUTINE_DERIVATION` / `SPECIFIED_NOT_IMPLEMENTED`。全uの相対threshold条件の事前証拠が必要。AをZとしない |
| F07 canonical全law | [G7C]、G119–160/162–212 | group setup guardに加え、存在するfirst/root-child exact lawの有限最小値とconditional≥p_minで全sample/event再構築へ適用 | 有限input/groupの条件。Greenなし。包括preflightは未接続。各modeのD/S/再構築を費用に含む | `CLOSED_BY_ROUTINE_DERIVATION` / `SPECIFIED_NOT_IMPLEMENTED`。不存在groupの分母を評価しない。Full下界の無断転用なし |
| F08 zero/failure/confidence | [MATH]§8、[G7C]、K/G guard | [F1 §7](preflight_precision_contract.md)：raw/coin zero、technical、invalidを分離。全law条件/IID/正remaining/全N完了を固定confidenceの前提にする | 原因別例外/証拠接続は未実装。fixed αと任意αを区別。失敗prefixを完了confidenceにしない | `CLOSED_BY_ROUTINE_DERIVATION` / `SPECIFIED_NOT_IMPLEMENTED`。B/CをX=0にしない。新retry/failure budgetは採用しない |
| F09 mean/budget | [BIT]§§3–4/[U]§7、K239–254/G258–284 | 実πW̃=c̃、M̃のmean、Mへの係数bias、B_o/Uのhi、κ/range、既存factor2・ceil Nを維持。C_budgetとw_bを具体化 | 数学式は既存。入力/精度/測定とのbindingは仕様条件、任意α APIは未完成だが今回必須でない | `CLOSED_BY_EXISTING_LEMMA` / `CONDITIONAL_COST_BOUND`。未知B_RA/Zや観測varianceでNを減らさず、残余幅が非正ならSTOP |
| F10 setup/bit | [BIT]§2、K147–160、G19–41/216–225 | [F2 §§1–4](full_bit_complexity_accounting.md)：b_c、Q/R/D/S、3n roots、pow/factorial、input/Green/U/roundingを含むC_setupを提示 | 標準モデル下の費用は具体化。実CPU/allocator/platformの保証ではない | `CLOSED_BY_ROUTINE_DERIVATION` / `CONDITIONAL_COST_BOUND`。採用モデルと入力/精度変数を保持。CPython実測への換算は別許可/証拠 |
| F11 packet/3経路/weight | K121–133/194–254、G237–251 | [F2 §5](full_bit_complexity_accounting.md)：L²再sum、2 roots、child D、πの(m+2)H bit、C_raw/C_packet/C_coin/C_eventと最大trialを具体化 | 現SOURCEの再計算を削除せず計上。raw zero/coin zero/acceptedのprefixが異なる。新cacheなし | `CLOSED_BY_ROUTINE_DERIVATION` / `CONDITIONAL_COST_BOUND`。最悪上界を期待実費としない。仮想高速selectorへの置換なし |
| F12 angle/provider | [BIT]§6、K129/235、CP2–58 | [F2 §6](full_bit_complexity_accounting.md)：symbolic ratio費用と、既存atan手順をJ/K_φで評価する条件付きC_angleを分離。phase/σ/actual adjointを維持 | angle APIは未実装。query意味論は条件付き既存。native/strict physical単価は未認証 | `CLOSED_BY_EXISTING_LEMMA` / `SPECIFIED_NOT_IMPLEMENTED` / `CONDITIONAL_COST_BOUND` / `PHYSICAL_PROVIDER_UNCERTIFIED`。angle精度証拠と物理証拠を混同しない |
| F13 memory/全N/出力 | [BIT]§2、G247、既存全trial会計 | [F2 §7](full_bit_complexity_accounting.md)：generator peak bits、保存容量別項、C_setup+N最大trial+出力、経路別期待会計を具体化 | record retain/serialization/実測outcome形式は別。N・angle/provider等の未取得情報を0にしない | `CLOSED_BY_ROUTINE_DERIVATION` / `CONDITIONAL_COST_BOUND` / `NOT_VERIFIED`（具体出力形式/実bytes）。出力方針の固定後だけ全runの容量/実費を認証可能 |
| F14 物理/応用/比較 | [B-README]/[B-COST]/[B-HANDOFF]、今回のユーザー指示 | DF/PR/RTE adapter・backend/state/accumulator、P_U/一般次数比較は除外。渡すのは同じ精度/費用の前提だけ | 再監査・実装・新証明なし。現STOP/claim保持 | `OUT_OF_SCOPE` / `PHYSICAL_PROVIDER_UNCERTIFIED`。別対象・承認なしに進めない |

今回`REQUIRES_NONTRIVIAL_PROOF`へ確定分類した行はない。追加の非自明理論を必要としない範囲でF1/F2を具体化した。未実装と物理未認証は残る。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[B-COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/measurement_and_cost_contract.md
[B-HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/handoff.md
[B-README]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/README.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
