# 既存研究と探索範囲の対応表

2026-10-16 JST。新しい科学的GO/STOP評価ではなく、固定資料から確認できる**検討の範囲**を分類する。今回のH1〜H4一時停止は [追記記録](research_state_transition.md)、資料identityと取得区分は [provenance.json](provenance.json)。本文中の S/G IDは同JSONの `fixed_repository_sources` に対応する。

## 分類の意味

| 分類 | 意味 | 含意しないこと |
| --- | --- | --- |
| ① 検討済み | 明記した問い・入力・比較・実験／静的監査の範囲で検討済み | 一般化学・全精度・最終資源が解決したという意味ではない |
| ② 生成規則未特定 | 検討したが、独立した有利な生成／情報取得規則を特定できなかった | 未検討でも一般的不可能性でもない |
| ③ 既知構成 | 近接する構成原理・恒等式・手順が既存文献にある | 費用込みの全タスクが既に解決したとは限らない |
| ④ 重要条件未確認 | 調査済みだがアクセス・仮定・有限保証・費用等が閉じていない | 問いを新しい名称で再開する認可ではない |
| ⑤ 検討記録不足の可能性 | 今回読んだ資料群で、その狭い問いの十分な検討記録を確認できない | 世界で未解決、プロジェクト全履歴で未検討、新規、有望という認定ではない |

分類は重複する。失敗や未実装だけを理由に⑤へ置かない。「未解決」は、文献全体のopen problemを証明した意味ではなく、**現在の研究目的への接続が取得資料で未成立**という意味で用いる。

## 既存研究の範囲

| 問い・系列 | 分類 | 実際に検討した内容・既知法との対応 | 残る条件／今回の扱い | 固定出典 |
| --- | --- | --- | --- | --- |
| 固定DF-prefix／finite-RTE／RPE費用接続 | ①③④ | norm・PF誤差・tail・回路価格・shot配分の交換は研究目的と実装基盤で整理済み | 全round絶対energy総費用、state準備、元H固有位相との関係は未完成。一般的split調整を新問題としない | S4、G9 |
| 旧候補C1 macrocycle・符号相殺 | ①②③④ | composition、ground-connected branch、保存oracle診断・強対照まで統合済み | 固定catalog内の否定的結果とcatalog外の未判定を分離。新catalog生成へ進まない | G2 |
| 旧候補C2 branch付きPF外挿 | ②③④ | Richardson型bias相殺とbranch/remainder・精度配分の条件を監査済み | 安い同branch情報・証明・複数幅control費用が未取得。外挿一般は未検討ではない | G2 |
| 旧候補C3 self-energy／active space | ②③④ | Schur/Feshbach、resolvent/root、Neumann、downfoldingとの対応を監査済み | 高body項・normalization・state/control費用。既存QSVT/LCU中心の実現はD1内新案に流用不可 | G2 |
| 旧候補C4 cost-aware sampling／whole path | ①②③④ | signed mean、proposal/rejection、有限状態会計、importance samplingとの対応 | inverse stack圧縮・range/tail・online古典費用。単にsamplingを変えるのは新領域ではない | G2、S6 |
| N1 表現・Gaussian completion／cluster | ①②③④ | 近似biasを元精度へ戻すselector、費用込みexact fallback、共有basisを検討済み | 固定比較の不成立を「representation未検討」と言い換えない。N1 STOPを保持 | G2、S12–S13 |
| N2 signed charge／算術 | ①③④ | signed表現・fitとcompute/phase/uncomputeの費用を検討済み | fit改善と回路費用改善は別。N2 HOLDを維持 | G2 |
| N3 occupation／縮約certificate | ①③④ | occupation・Schur・Fantope/SDP/SOSとactive-energy保証を検討済み | 一般二体・全energy/solver費用は未解決。N3 CLOSEDを保護 | G2 |
| 対称性保護・leakage／mixed可解block | ①②③④ | 群平均／echo、圧縮空間漏洩、conditional SU(2)、小Lie closureをtoyで確認。Patel、THRIFT等との重複もレビュー済み | 安価なcoherent mixed oracle、分子でのスケール、controlと総費用が未成立。代数・対称性全般を⑤としない | S12 §§4–6、S13 §§7–8、S14 §3 |
| 一体再配分・厳密ground情報利用 | ①②③④ | BCH期待値・既存compact BCHとの同情報比較、finite-step現固定構成まで検討 | 情報取得と費用改善の独立性が未成立。現固定構成CLOSED。未実装だから新しいとはしない | G6、G7 |
| H-E1 ground固有値特化補正 | ①②③④ | 2blockの外側commutator消去・Hejazi/processed PFとの対応 | 多fragment生成・control・finite-stepへの移送。現定義CLOSEDを維持 | G5、G7 |
| M-a/M-b 固定回路PF小改変 | ①②③④ | corrector／counterterm等を既知回路との差として監査済み | 追加探索STOP、K1未認可。係数・fragment配置の小変更を新領域として数えない | G8、G7 |
| SOS-PF K0/K0.5 | ①②③④ | PSD化、DF/BLISS/Gram、SOS gapとfinite-step、Kang–Su等の対応を調査済み | 有用なgap/remainder証明・新生成・総費用は未成立。現探索CLOSED、A2 NOT_ADOPTED、技術判定UNDETERMINEDを区別 | G1、G3–G4、G12–G13 |
| H1 先導固有値補正 | ②③④ | 固有状態commutator・低body表示と多fragment残余を静的監査済み | 先導係数とfinite-step実用性は別。高body表示から4-RDM不可避の一般下界は主張しない | S1–S3 |
| H2 混合相関／filter | ②③④ | filter ratio、元H測定、leakage・分母・shot会計を静的監査済み | 個別UNDETERMINEDを維持。同identityだけで新情報取得法としない | S1–S3 |
| H3 frame／interaction picture | ②③④ | Gaussian共役、DF-square、THRIFT/CFQM/SPRINTとのアクセス比較 | sampleされた和の指数やcontrolを無料としない。現探索PAUSED | S1–S3 |
| H4 copies／phase reference | ②③④ | uncontrolled H+cIの位相不変性、既知reference、ASP等の追加アクセスを区別 | direct measurementを禁止するno-goではない。copiesから絶対energyを得る全方式の下界でもない | S1–S3 |
| Track A native実装・費用研究 | ①③④ | DF/Gaussian、full controlled wrapper、保存native比較。後続H6のB0+I対照整備も存在 | finite complex-signal taskの経験的比較であり絶対energy全費用ではない。V2 bindingは準備／人工検証まで | G9、S5、S10–S11 |
| Track B D1–D3・M1-S | ①③④ | finite-RTE同mean会計・strict-phase合成・boundary cancellation、および後続shared ±の限定C-sec理論 | M1-S G0-T1は独立に正式採用された限定理論契約。Checkpoint受理範囲と未採用addendum草稿、未承認Gate 0を分離。DF一般・full RPEへの有用性は未確立 | S6、S8–S9 |
| 高次PF・第1研究 | ①③④ | 有限時間校正、state proxyとPF固有値差、model-fit、予算安全性とregretを分離 | S0–S4完了、固定S4 no-benefit。PF固有値知識を使うモデルを実QPE回路と同一視しない。S5を再開しない | G9–G11 |
| 高次PF・第2研究 | ①③④ | finite-time情報のdecision window、cheap predictor安全余裕、abstentionの説明的機構研究 | Direction Cは既存正式方針。新selector・rigorous certificate・end-to-end資源達成ではない。今回の観点C1–C5と別の名称 | S7 |

旧統合資料G2のTrack A/B欄は当時のsnapshotである。後続S8〜S11を追加して、強いbaseline整備や限定理論成果を「未実施」と誤認しない。S8からの理論受理状況はその既存記録による帰属であり、今回Codexが証明や新規性を再審査したものではない。

## 検討記録不足の可能性を残す狭い範囲

| 狭い問い | 分類と根拠 | 既知の対照と留保 |
| --- | --- | --- |
| symmetry/fermion encodingの選択が、元Hのphaseを保ったcontrolled回路生成・測定までの費用へどう影響するか | ③⑤。G9の基盤表はJW/DF/sector中心。S12–S14の表現研究は豊富だが、GSE/tapering等を同一absolute-energy contractで比較した記録は今回確認できない | encodingやtapering自体は既知。sector行列の古典縮約と量子encoded回路は別。⑤は新規生成則の証拠ではない |
| exact ground copies、coherent preparation/inverse、複数copy同時操作を区別した、元Hのenergy読出しの費用・情報条件 | ③④⑤。H4は位相参照、H2はmixed ratioを扱った。fermionic shadows／joint POVMを含む同一条件の全費用比較は確認できない | direct energy測定・RDM・単一copy上のjoint observable測定は既知。複数copyのcollective操作と混同しない。情報理論的新規性は未確認 |
| full interacting化学Hの効率的代数構成クラスを、係数から認識し、前処理も含めて保証できるか | ②③④を主とする。旧mixed-block研究と重なるため、広い問いとして⑤には置かない | 今回のCartan文献照合は時間非依存depthと効率性の違いを明確化した。既知Patel/THRIFTの未閉鎖条件を再発見しただけなら新研究にならない |

**確実に新しく有望な未探索領域を特定した、とは結論しない。** 最初の2項目も、比較・入力契約の欠落である可能性があり、新アルゴリズム研究へ移す価値はGPTの判断事項である。

## 範囲確認の方法と限界

固定commit `0d460c54…` の `docs/research/`、`docs/tracks/`、`docs/reviews/` について、Cartan/Lie、Majorana、encoding、shadows、copy access、swap network等の用語を補助検索し、該当する旧レビュー・toy報告の文脈を読んだ。実装側の補助検索では `sparse_no_collective` はN2の行列族の名称で、collective-copy測定の実装を意味しなかった。

検索語に一致しないことを未検討の証明にしない。別名の議論、未取得branch、非公開文書、取得できていない原本が残る。対象は固定した資料群であり、更新中の全repositoryの最新性や文献の網羅性を主張しない。今回照合した一次論文8件の限定比較と未取得事項は [問題領域表](open_problem_landscape.md)に整理した。

## 固定資料への入口

以下は原文への固定リンク。全27取得資料のbyte数・SHA・取得区分はprovenanceを正本とする。

- [研究目的 S4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/研究目的・研究課題.md)、[統合引継ぎ G2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md)、[実装基盤 G9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/acfa8674d51ede1d5cd50f1b49873e67de5b71b9/docs/research/qc_existing_technical_foundation_map_20261016.md)
- [SOS-PF K0 G3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)、[K0.5 G4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/technical_report.md)、[H-E1 G5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_he1_stage0_technical_prevalidation_20261015.md)、[一体再配分 G6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_exact_ground_pf_repartition_technical_preaudit_v1_20261014.md)、[M監査 G8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_m_fixed_circuit_k0_technical_audit_20261015.md)
- [旧表現レビュー S13](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d460c54f3bf54ea46d4d8129f6702703c4bb217/docs/research/representation_exploration_inputs/PR_representation_codesign_research_review_2026-10-10.md)、[toy検証 S12](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d460c54f3bf54ea46d4d8129f6702703c4bb217/docs/research/representation_exploration_initial_validation_20261010.md)、[既存方向レビュー S14](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d460c54f3bf54ea46d4d8129f6702703c4bb217/docs/research/quantum_chemistry_ground_energy_algorithm_direction_review_2026-10-13.md)
- [Track A H6強対照 S10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd695e6584f15b88341d2446ed010c7bdfef1666/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)、[V2準備 S11](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd695e6584f15b88341d2446ed010c7bdfef1666/docs/research/track_a_v2_h6_cost_binding_readiness_20261016.md)
- [Track B既知法対応 S6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)、[M1-S正式理論契約 S9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/373b8e4b9831f4d00bdc00ffc741ac61e03f0841/docs/tracks/algorithm_codesign/m1s_g0t1_s0_formal_theory_contract_20261016.md)、[Checkpoint後整理 S8：草稿](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05513eea807fbfb841c6012d2fe2133d70b5f1f9/docs/tracks/algorithm_codesign/m1s_g0t1_post_checkpoint2_theory_addendum_draft_20261016.md)
- [高次PF第1研究 G10](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/cd81018932b9557b7a55e2e1cd8c5e3988245409/PF_first_study_final_synthesis_20260925.md)、[第2研究方針 S7](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/1d01e77247364b49dbce2e9ce2fe7c32af0fb937/docs/second_study_v2/budget_safety_mechanism_20261005/final_direction_and_claim_scope.md)

今回の観点C1〜C5は、旧候補C1〜C4、Track B内部D1〜D3、第2研究Direction C、固定条件D1/D2とは別のラベルである。名前から研究状態や認可を移送しない。
