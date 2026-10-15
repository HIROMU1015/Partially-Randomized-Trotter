# Parent allocation S4：完成版科学レビュー受領・C1追補注記

2026-10-15 JST。利用者の「こんな感じで進める」と指定された完成版を受領し、同書§10.2 **C1（技術文書の整理）だけ**を実施した。新しい科学レビュー・数値分析・実験・研究主線の選定は行っていない。

## 受領原本と責任範囲

- [科学レビュー完成版の原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3086f48c5598c8bb31b0587e807ca6c1b3d2754/docs/research/parent_allocation_s4_scientific_review_revision_20261015.md)
- 原本だけの固定commit：`e3086f48c5598c8bb31b0587e807ca6c1b3d2754`。
- 110,900 bytes、SHA256：`642095bdef9f146f6a29205157e295f14a907d062123a77fb770f94e8b1630de`。
- ローカル提供原本をbyte一致で保存し、GitHubから再取得して照合済み。原本の内容・著者表記・推奨の条件付き表現は変更していない。

原本はClaude独立レビューを基にGPT-6が改訂した完成版と自己記録している。Claude本人による改訂承認を意味せず、Codexによる全論文・数式・派生集計の独立再認証でもない。
原本が「〔派生集計〕」とするfloat64のNC proxy、価格分布、post-hoc選択等はレビュー側の探索的分析であり、S4の元ledgerや事前登録された認証結果に追加・置換しない。

## 比較集合と二つのproxy目的を区別する

| 区分 | 結果正本と扱い |
|---|---|
| v3の50 control law | 元v3 namespaceの保存値をそのまま再利用 |
| S4の訂正KKT／halfway20 law | v4 namespace。config_04／05／10／11で採択目的の候補を補完 |
| v3の旧raw-C proxy20 law | 元v3 namespaceで履歴を保持。訂正後70 lawに含めず、採択KKTの成功／失敗へ読み替えない |
| analytic IS正常適用外5 slot | lawではなく正常適用外。失敗・性能negativeやゼロ費用へ変更しない |

旧proxyは`sum C_i sqrt(z_i²+b_i²)`、S4は採択済みの`sum sqrt(C_i) sqrt(z_i²+b_i²)`に対応する。
同じconfig IDでもcommit・namespace・row path・SHA256で識別する。訂正前後は異なるproxy候補であり、同一KKT手法の反復試行とは扱わない。
S4の保存配分は固定proxyの区間・有理化付き候補であり、native総費用の最適点を認定するものではない。

[訂正済み70 law Evidence Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md)と
[70 law索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/artifacts/track_b_parent_allocation_review_packet/2026-10-15/s4/law_comparison_index.csv)に元rowの対応がある。
旧v3 Evidence Packetの内部候補部分は、旧proxyの履歴として参照する。

## 頂点＋ISの保存T利益には条件を併記する

レビューが引用する次の約14%／19%は保存済みのbest-common／best-noncommon比較の丸めた記述を引き継ぐもので、本追補で新たな比率を計算していない。

| arm | 保存common law | 保存noncommon law | レビューの限定的T利益 |
|---|---|---|---|
| ordinary | config_00:regularized_16 | config_03:regularized_1/16（vertex_2） | 約14% |
| closed P5 | config_06:regularized_1 | config_07:analytic_IS（vertex_0） | 約19% |

適用条件は、**h=0、固定十分Nによる期待T費用、r=1、development synthetic、固定native tool**。
具体的には3-qubit distinct-basis、p=(1/5,3/10,1/2)、x=tau=5/7、状態000、sigma+1、finite P5。
Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4)R_ZZ01(pi/4)、Qi=Vi†ZiVi。
分子geometry／basis／DF rank／split L_Dは対象外。primitive1e-6、pygridsynth2.0.0／seed0／strict phase、axis精度1/200、alpha_axis1/40。

- 同じproposal選択機会を与えたbest-to-bestで、左右のproposalは異なる。配分だけの効果へ全差を帰属させない。
- 上の二比較ではnoncommon側の**N・CX費用が増加**している。T利益を全資源座標の優越と呼ばない。
- Nは共通信頼保証の十分試行数で、実測量子shotsや必要最小Nではない。費用は加法的nativeモデルで、Tはnative1Qと重複合算しない。
- h>0の指定二lawの交点は、全70 lawの下側包絡線ではない。レビューが論じる最良領域の上限を、領域全体の認証へ拡張しない。
- 多stepのnormalization／二次momentに関する条件付き議論は、最適化後総T費用比の一般下界やno-goではない。
- 角度に依存する固定合成価格の差は、統計的な価格ノイズと同一視しない。別tool／precision／seedでの順位頑健性は未検証。
- classical online費用は`NOT_MEASURED`。PR／RPE／QPE全体・分子DF・whole-circuit最適化へ14%／19%をそのまま移さない。

正本：[S4結果・保存監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe0f3f077342bf7a557033e4dfe2734b7ead6091/docs/tracks/algorithm_codesign/parent_allocation_s4_supplement_results_and_gpt_handoff_20261015.md)。
S4=`4ccc1832f63cd323b23fc05623f7e0b58cf34f37`、A4=`71cf1b76492ed01fc5bb8363246b5f13f7e166d7`。原source・結果・classification・contract・authorization・消費済みmarker・STOPを変更していない。

## 今回の扱いと次の承認境界

完成版はD＋E、すなわち限定成果の整理と同一fixtureでの追加探索の当面中断を第一候補として**条件付きで推奨**している。
今回進めたのはC1の記録・条件の併記までで、既存mandatory STOPを維持する。Track B全体の恒久的終了、全内部配分のno-go、新しい研究主線・閾値をCodexが採択した意味ではない。

| 作業案 | 今回の状態 |
|---|---|
| C1：原本保存・技術的追補注記 | 本文で実施。旧資料の書換え・新しい数値集計なし |
| C2：NC proxyのexact再集計 | 未実施。新しい科学的数値分析として別途範囲・承認が必要 |
| C3：価格の索引／感度分析 | 今回追加しない。単純な転記と、新しい分布・反事実的価格分析・閾値判断を区別 |
| C4：全h最適包絡線 | 未実施。保存係数を使っても新しい派生数値分析なので別途承認 |
| C5：多stepの適用境界 | 未着手。解析仮定の検討と登録入力の数値forecastを区別し、別途指示を待つ |

r=4、h=50 T、「価格差の2倍」、0.5%等は原本で未採択とされた設計例であり、実行条件・成功閾値へ転用しない。
今回の原本保存・C1完了を理由に、新しい科学計算・重要科学レビューを自動開始しない。
レビュー側の推奨を踏まえた今後の研究採択・変更や、C2–C5の具体化は必要な別途指示・承認の下で扱う。

**mandatory STOP。** 新規synthesis／compile／matrix／guard／worker／N forecast／NC集計／全h包絡線／sampling／LP／DF／分子／GPUは0。
本追補と受領原本以外の研究資料・原記録は変更しない。
