# H4/H6/H8：共通資源比較へ向けた対応表

2026-10-13。[H6希少費用の技術解析](track_a_h6_rare_order_cost_reliability_v1.md)の補助資料。新しい比較契約・候補grid・予算・H8 grantを確定しない。

| 条件 | H4旧M1/PM保存系列 | H6探索・独立32保存系列 | H8で比較前に必要な固定 |
|---|---|---|---|
| Hamiltonian/geometry/basis | linear H4/1.00Å/STO-3G。M2は既観測1.30Å固定移送5構成 | linear H6/1.00Å/STO-3G | geometry/basis/electron数・Hamiltonian snapshot・指定state。未生成 |
| system/sector | 8 modes、Nalpha=Nbeta=2、sector36 | 12 modes、3/3、sector400 | 16 modes/4/4ならsector4900という組合せ候補。実state/資源検証なし |
| DF | 固定rank12、generation prefix、旧snapshotとrank方針 | tol-only1e-8/rank19/cutoff0、signed generation、weighted projection | H6と共通のDF生成/切断/順序政策の採否、実rank・phase/signのreceipt。rank未確定 |
| T/delta | T.8、q1/2/4/8・delta.8/.4/.2/.1 | 同じT/q/delta | 同じfinite-time task。coverage規則を結果前に固定 |
| PF | 旧M1はS2中心、full-rank4 q候補 | full19 S2＋global Yoshida S4各4 q | 同じS2/S4構成とprimitive/controlのscopeを固定 |
| partition | M1 L_D0/3/6/9/12、PM-1のB0近接prefix4/5 | B0/B2共通prefix5/10/15、B1 full19 | サイズに対応するprefix生成規則。単純コピーや25%最適則としない |
| random grid | 保存されたq/r/K、B2/B3、M1 random194 | B2 q1/2/4/8,r1/2/4,K2/4（72）。B3 normalization84だけ | qとR=qrを区別して適格coverage/登録上限不足の扱いを固定 |
| 精度 | M1単一ε.05、後のPM-2/AX-1bは保存値の多精度解析もある | signal ε.05/.01/.005/.001 | 同じcomplex-signal精度・各軸配分・headroom。energy精度としない |
| wrapper/control | 旧controlled wrapper系列 | symmetric_directional primary、ordinary補助4 wrapper | H6と同じcontrol、phase、測定、state-prep除外を固定 |
| compiler | Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、couplingなし | 同じ主要compiler options、native v5 full-wrapper | profile/native lowering/dependency/sourceを同時固定。version一致だけで全output互換としない |
| 費用 | full measured wrapper RZ/CX/depth等×解析shot | 同じ大枠、native mapはRZ/CX/depth/size | cost単位・axis pairing・shot×one-shot定義・未取得値を固定 |
| 数値/shot | 旧signal/allowance/PM会計、AX-1bはCONDITIONAL_ORACLE | 独立作用診断の経験的u、α_axis.025、Hoeffding整数shot | 同じ診断scopeと余裕/不適格判定、厳密certificateの有無を明記 |
| 費用標本 | M1 random194×32、旧selector訂正保全 | 全72×探索8、事前選択8×独立32 | 希少層の評価方法と候補選抜/確認の規則。追加計算前に固定 |
| 準備費用 | P感度、実準備は未取得 | P=0と固定pairの共通P交点 | 必要なら同じ共通Pモデル。実準備込みRPE結果と混同しない |

H4 M1の約65%とH6の約54–63%の低減をサイズスケーリングへ直接つなげない。現在のH4 production geometry/structured-Gaussian/resume系列も旧M1とは別のsource・回路系列であり、継続中の結果をこの表へ勝手に混合しない。

AX-1bのSINGLE→FEWの保存OOF改善はH4内部の校正診断。回路費用featureを対応付ける参考にはなるが、H6/H8のrare費用上界、独立予測精度、bias/shot予測には使えない。今回はモデルを再fitしない。H6を見て変更するモデルはH6 developmentモデルとして記録し、H8予測claimを行う場合だけ事前に係数/feature/取得費用を固定する。

H8の新DF/state/signal/transpileは未認可・未実施。H6のq8が登録上限にあることから、H8ではq/Rのcoverage規則が重要だが、今回qを追加採用しない。候補不適格、数値判定不能、資源STOP、方法の精度限界を区別する。PRが勝つことを成功条件やgeometry/seed変更の理由としない。

根拠：[H4 M1](../pr2_matched_accuracy_m1_b1_result_validation.md)、[AX-1b保存モデル](../../artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/execution_audit_v1/result_review_report.md)、[H6科学契約](track_a_h6_matched_supplement_execution_contract_v3.md)、[今回引継いだ科学レビュー](track_a_h6_independent_confirmation_scientific_review_20261013.md)。
