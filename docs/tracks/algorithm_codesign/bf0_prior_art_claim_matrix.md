# BF-0 prior-art claim matrix — Track B

日付: 2026-10-04  
status: `PROCEED_BF0_DESIGN_AND_NOVELTY_AUDIT` / `DRAFT_FOR_REVIEW`  
関連: [数学契約](bf0_mathematical_contract.md)、[BF-1提案・実行未認可](bf1_minimal_pilot_proposal.md)、[BF-0外部レビュー依頼](bf0_external_review_request_20261004.md)。

## 1. 結論と監査範囲

**B-Fの狭い問題設定はreview候補として残るが、新規method、有用性、BF-1実行GOは未確定である。** PF係数探索、errorとformula lengthの公平比較、absolute tail timeによるRTE費用評価、比例allocation、hybrid法の選択は既知である。単にこれらを組み合わせた実装、違う係数の取得、高次PFの適用だけならreplication/application noteへ縮小する。

今回確認した一次資料の関連本文では、同一の五stage係数familyを、有限RTEのbias・normalization・整数配分を含むcoherent-signal resource objectiveで再最適化し、同予算の通常PF設計および既知leading modelと比較する具体的methodを確認できなかった。これは**scopedな未確認**であり、文献の不存在、世界初、研究採択の証明ではない。以下の最後の二claimと強いbaselineのadequacyは外部reviewへ残す。

入口READMEやabstractだけで判断せず、下記の本文・定理・付録を確認した。数値benchmarkの再実行、著者実装の実行、引用網を網羅したsystematic reviewは行っていない。PDF pageは1-basedである。

## 2. 一次文献と確認箇所

| ID / 一次文献 | 固定版 | 確認した関連本文 |
|---|---|---|
| M: Morales et al., [Selection and improvement of product formulae for best performance of quantum simulation](https://arxiv.org/pdf/2210.15817v3) | arXiv:2210.15817v3 | §§II.B/D/E、III.A/B、IV.A/B、V.A/B、VI、Tables I/II/IV。係数探索、general forward/reverse sweep、長さと誤差の比較、finite-step挙動、customization |
| P: Günther et al., [Phase estimation with partially randomized time evolution](https://arxiv.org/pdf/2503.05647v2) | arXiv:2503.05647v2 | §V Theorem V.1、VI.B、VII.B、Appendix A.3 Lemma A.3 pp.21–24、D pp.34–35、E.3 pp.42–43。signed stage、tail time、RTE配分、S2/S4比較、gate/shot費用 |
| S: Casares et al., [Theory and practice of Trotter product formulas for quantum chemistry — SPRINT / GRADE](https://arxiv.org/pdf/2606.30741v1) | arXiv:2606.30741v1 | §§III.B pp.7–9、IV pp.13–14のgrouping/order/outer-composition/remainder選択、Appendix E Proposition 3 p.33。near-integrable・processed公式のerror構造 |
| C: [Composite Quantum Simulations](https://arxiv.org/pdf/2206.06409v3) | arXiv:2206.06409v3 | §§5.1–5.3 pp.12–21、Theorem 2.1、Lemma 2.1、Lemma 10、§6 p.27。higher-order Trotter/qDRIFT、partition、random budget |
| H: [Hybridized Methods for Quantum Simulation in the Interaction Picture](https://arxiv.org/pdf/2109.03308v3) | arXiv:2109.03308v3 | §§3–4 pp.12–19、Theorems 3.2/4.2/4.3。interaction-picture hybrid、residual norm・commutatorとquery cost |
| I: [Resource-Optimal Importance Sampling for Randomized Quantum Algorithms](https://arxiv.org/pdf/2603.13495v1) | arXiv:2603.13495v1 | §II Theorem 1/Eqs.7–13、II.A、III Theorem 2、V.A。arbitrary positive whole-circuit cost、重み付きmean/bias、qDRIFT例 |
| O: Ostmeyer, [Optimised Trotter Decompositions for Classical and Quantum Computing](https://arxiv.org/pdf/2211.02691v4) | arXiv:2211.02691v4 | §2のmany-generator変換と証明、§3.1、§3.2.7–11/Eq.40、§4、§5。optimized general fourth-orderとfinite-step performance |
| B: Blanes, Casas, Murua, [Composition methods for differential equations with processing](https://www.ehu.eus/ccwmuura/research/asf2.pdf) | SIAM J. Sci. Comput. 27(6), 1817–1843; DOI 10.1137/030601223 | §4.1/Eqs.4.1–4.4、pp.1828–1829。absolute coefficient sumとweighted leading errorの目的関数、problem-dependent weights、自由係数探索 |
| J: Hejazi et al., [Better product formulas for quantum phase estimation](https://arxiv.org/pdf/2412.16811v1) | arXiv:2412.16811v1 | §II、Eqs.10–12と周辺導出。two-block customized composition、eigenvalueに寄与しないleading outer commutator。五exponentialと五S2 stageは異なる |

M/O/Bを加えた理由は、通常PF設計を弱いstage-count baselineへ落とさないためである。M Table IVには五S2 composition以外のgeneral fourth-orderとprocessed法も含まれる。Iの最適性はwhole-circuit positive costに対するものであり、additive component costに限定されない。

## 3. Claim単位の比較

| Claim | 先行研究で既知の内容 / locator | Bが追加し得る内容 | 今回主張しない内容 | 未解決の証明・実装・比較条件 | P-D/R3等との差分 |
|---|---|---|---|---|---|
| 四次対称compositionのorder条件 | M II.B、O 3.1。self-adjoint kernelの係数条件 | 固定native kernelへの正確な意味論の適用 | 新order条件、新しい五stage family | 数値係数精度、signed-time adapter | 係数条件自体は旧P-Dにもある。新規性にならない |
| 制約を満たす係数探索 | M III.A、B 4.1。数値解探索、自由係数、目的関数 | 同じfamilyをRTE taskへ戻す候補 | 係数探索そのものの新規性、global optimum | 同じ枝・自由度・評価予算、optimizer仕様 | 固定既存formula名の選択から連続係数設計へ変える |
| PF cost-aware design | M IV.A/B、O 3–4。長さと誤差、有限step挙動 | finite taskのdecision/resource deltaの検証 | stage数削減だけによる優位性 | 原論文の強い通常PFと再最適化した対照 | 旧B1aの内部work省略を繰り返さない |
| coefficient normとerrorのtradeoff | B 4.1 Eq.4.4、M VI。\(\ell_1\)小化とweighted error | finite normalization/roundingが追加の意思決定を変えるか | \(\sum\lvert w_j\rvert\)を小さくするだけの新method | 既知leading-norm modelを含む強い対照 | P-Dのleading model=B4というnegative evidenceが直接の注意点 |
| deterministic/random hybridとorder選択 | C 5.1–5.3、P V–VII、S IV、H 3–4 | 今回はsplit/kernel/random法を固定する | hybrid導入、split/order選択の一般的新規性 | task/error norm/cost oracleの一致 | broad R3/旧Track B selectorの重複を解消する必要がある |
| signed PF stageのtail absolute time | P A.3 Lemma A.3。signed stageとabsolute-time負担 | finite taskで係数設計へfeedbackする候補 | tail absolute timeの発見 | stage list・identity phase・time signの一致 | fixed-formula候補評価だけならP-Dの反復 |
| RTE allocation | P A.3。timeに応じた連続配分・ceil型budget | lower bound付き固定roundingの影響の機構確認 | 比例配分の新規性、整数最適性 | rounding tie、ゼロstage、同じ総budget | allocation自由度を増やして旧STOPを回避しない |
| finite RTEとnormalization | P Aと現行source。有限Taylor/RTE、normalization補正 | biasとshotsを係数設計に同時に返す候補 | finite RTEの発明、\(B_K\)式の発見 | 現行K規約、meanとcorrected operator、propagation | finite correctionだけで選択が変わるとは旧P-Dが支持しない |
| near-integrable customized composition | S III.B/E、J II。構造を使う公式・error取消 | 固定DF構造でtail実装が設計に効くか | SPRINT/GRADE、perturbative法の名称変更 | 何をexactに実装できるか、general DFへの適用 | 理論差分のないgeneral co-designはR3と同じ問題 |
| exact-stage fusion | M IVで隣接exponential融合を数える | F/Sを別finite algorithmとして定義 | 融合自体の新規性、compilerで有限meanを変えてよいというclaim | finite化前後、full-q list、controlled phase | work構成差をmethod deltaと取り違えない |
| whole-circuit importance sampling | I II/III。\(q^*\propto p/\sqrt C\)、\(J^*=(E_p\sqrt C)^2\) | 保存済みsampleのcost dispersion診断のみ | cost-dependent samplingの発明、実測resource改善 | cost oracle費用、support/weights、finite confidence、sampling実装 | B-Sは別候補。B-F不成功の自動escapeにしない |
| task固有の同一family再設計 | M/P/Sの上記関連本文に、BFの限定taskを解く具体的探索法は未確認 | O/L/Fを同自由度・同予算で比較する限定候補 | 最後の二行に空白があるとの断定 | \(\min_w G_{\rm PR}(w)\)を既存methodが実質解いていないか外部review | R3の一般selectorではなく係数生成問題。ただし目的関数変更だけなら弱い |
| 係数生成→finite signal→compiled resource | 各資料には部分的なerror/cost設計がある。BFと同一pipelineは未確認 | 同じtaskでmaterialなresource decision deltaを閉じる目標 | 現時点のpipeline完成、compiled advantage、利用可能I0/I1 algorithm | native wrapper、baseline、実測cost、oracle依存、設計費用 | P-D/FRのproxy・oracle機構のみGOという限界を越える必要がある |
| **一文差分** | 上の既知要素は研究claimから除外する | **固定native五stage familyを、randomized tailの有限bias・normalization・制約付きallocationを使って同予算で再設計し、通常PF設計と既知leading modelの共通task再最適化を越えるmaterialな資源選択差を示せるか。** | 答えがyesであるとの主張 | 最終claimにはcompiled resource、利用可能情報、closest-art reviewが必要 | この一文が単なる既知最適化の適用に尽きれば縮小・停止 |

最後の二技術行は`OPEN_NOVELTY_REVIEW`である。ここで「未確認」を「先行研究では扱っていない」と書き換えない。

## 4. 重要な二問の横断表

`既知`は本文で確認した範囲、`未確認`はscopedな未確認、`提案`は未実装・未実行を意味する。

| 項目 | Morales等の通常PF設計 | Günther PR | SPRINT | B-F |
|---|---|---|---|---|
| 係数探索 / order条件 | 既知 | 固定公式の利用・条件 | 構造を使う公式設計 | 既知familyを使用 |
| 長さとerror/cost | 既知 | 既知 | 既知 | 共通taskで再評価する提案 |
| eigenvalue / coherent error | 両者を区別 | phase-estimation task | energy/error構造 | coherent signalへ限定 |
| randomized tail | 主対象外 | 既知 | remainder strategyに含む | current finite-RTEを固定 |
| tail absolute time / allocation | coefficient norm設計は既知 | 既知 | 関連するfactorization設計 | 新規性としない |
| finite \(K,r,B_K\) | 主対象外 | 有限RTEを扱う | random remainderに関連 | 現行source規約を使用 |
| **同一familyを有限RTE-task用に再設計** | この限定taskは未確認 | 具体的係数探索法は未確認 | この限定taskは未確認 | **提案・新規性未確定** |
| **生成→finite signal→compiled resource** | 同一pipelineは未確認 | stage/cost評価は既知、同一探索pipelineは未確認 | 同一pipelineは未確認 | **目標・未達成** |

例えば既存の\(\ell_1\)-error objectiveにPRの既知shot factorを代入するだけでBFの設計とdecisionを説明できるなら、method claimは残らない。BF-1にはその説明を検査するL対照を必須にする。

## 5. 過去STOPとの差分と維持する判断

| 系列 | current sourceによるSTOP理由 | BFで変える研究対象 | 維持する制約 |
|---|---|---|---|
| P-D | fairness再最適化後、内部workを数えるB1b、leading B2、finite B4の主選択が一致。finite補正固有の選択改善が得られずS2へ進まなかった | 固定formula catalogの選択から同じorder manifold内の係数設計へ | L対照・内部DF work・共通再最適化を必須にする。catalogを増やしただけならSTOP |
| R3 / R3-S | hybrid/split/cost designの重複が強く、一般multi-fidelity選択を越えるquantum-specificな認証差分を固定できなかった | 今回はsplit探索・一般認証則を主題にしない | coefficient searchが一般最適化の適用に尽きるならmethod claimを縮小する |
| FR / FR-R | oracleや共通scalarによる機構と、同情報で使える片側認証・decision改善は別。current practical GOは成立していない | 新phase/radius boundを提案せず、finite signalをI2 benchmarkとして使う | I2機構をI0/I1 design methodへ昇格しない。FRのSTOP解除なし |
| P-A | interval DP固有のplan/compiled metric差を明示的一区間baselineに対して示せなかった | interval joint synthesisを採用しない | compiler fusionをBFの新規性に数えない |
| P-B | current H4範囲のsignal-weight主題は停止 | sample-weight探索を主題にしない | 旧weight方法をB-Sとして名称変更しない |
| P-C | current geometry familyでtracking固有差分・外挿/差分予測gateが成立しなかった | geometryやtracked splitを探索しない | 新geometry・splitでpositive resultを探さない |

P-D/R3の直接資料、およびP-A/P-B/P-C/FRの現行総覧は下のsource identity表を参照する。過去のSTOP、negative evidence、Aの研究statusは一切変更しない。

## 6. B-S: 保存済みsample-level costだけのpost-hoc診断

status: `POSTHOC_EXISTING_COST_DIAGNOSTIC`。M2 result JSONの`candidate_results[].compiled.paired_trajectory_rows`には、三つのrandom構成ごとに32 sampleのcosine/sine `rz_count`が保存されていた。index 0–31、正の整数cost、paired evolution fingerprintを読み取って確認した。Aのruntime/cacheを読んでいない。

経験分布\(\hat p_i=1/32\)に対するplug-in

\[
\hat\rho=\frac{(32^{-1}\sum_i\sqrt{C_i})^2}{32^{-1}\sum_iC_i},
\qquad\text{headroom}=100(1-\hat\rho)\%
\]

だけを計算した。Iの既知理想J比を、この保存済みsample集合へ当てはめた診断である。Re/Imは別々に計算し、この記録では同じ値だった。

入力は**H4 linear 1.30 Å、STO-3G、8 qubit、DF rank 12、T=0.8**。geometryは既に開封済みでありBのdevelopment/post-hoc referenceである。deltaは\(T/q\)で、下表の三点だけを参照した。

| 保存済み構成 | \(L_D\) | q / delta / r / K | samples per axis | mean RZ | min–max RZ | \(\hat\rho\) | 理想J headroom |
|---|---:|---|---:|---:|---|---:|---:|
| B2-rank3-q1-r4-K2 | 3 | 1 / 0.8 / 4 / 2 | 32 | 6242.53125 | 6103–6851 | 0.9996926435 | 0.03073565% |
| B2-rank3-q1-r8-K2 | 3 | 1 / 0.8 / 8 / 2 | 32 | 6650.46875 | 6206–7652 | 0.9992329957 | 0.07670043% |
| B3-rank0-q8-r32-K4 | 0 | 8 / 0.1 / 32 / 4 | 32 | 35178.1875 | 31102–42150 | 0.9987590242 | 0.12409758% |

観測された三sample集合では、B-S開発を支持するmaterialなheadroomは見えない。これはpopulation headroomの上界や不可能性証明ではない。32 sampleからのplug-inであり、sampling変更の実行結果、実際のfinite-confidence shots×cost改善、cost oracleや重み処理を含む総cost評価でもない。新trajectory、compile、sampling実装、別条件への展開は行わず、B-Sは保留する。必要なsample-level recordがない条件では`MISSING_EVIDENCE`で停止する。

sourceはM2 result v2のcommitted bytesと一致し、SHA-256は下表に記録した。Aの`TRANSFER_SUPPORTED`判定を再分類しない。

## 7. 推奨とreviewで閉じる項目

今回の判定は利用者承認済みの**`PROCEED_BF0_DESIGN_AND_NOVELTY_AUDIT`**の範囲に留める。三文書をreviewへ提出してSTOPする。BF-1 proposalは作成するが`PROCEED_MINIMAL_PILOT_DESIGN`、science execution、algorithm採択へ自動昇格させない。

reviewが狭い差分と比較契約を支持する場合だけ、BF-1のpreregistration/authorizationを別途固定する。reviewで既存methodが実質的に同じ\(\min_wG_{\rm PR}\)を解いていると判明した場合は、実装整理として価値がある範囲を`NARROW_TO_REPLICATION_NOTE`へ縮小し、差分も記録価値もなければ`STOP_DUPLICATIVE`とする。

reviewで残る主な問いは次の通り。

1. 最後の二claimに、単なるobjective置換を越える方法上の差分が残るか。I0/I1で利用可能な設計則へ到達する見込みがあるか。
2. M/O/B/Sの強い通常PF・general sweep・processed/near-integrable法に対し、五S2 unprocessed class限定が適切か。BF-1でそれらを全て追加する指示にはしない。
3. action proxyによる機構検査をBF-1の到達点として認めるか。compiled-resource claimへ進むための別reviewが十分に明示されているか。
4. Fのadapter、signed time、phase、baseline identity、数値guard、資源上限をreview可能な実行contractへ閉じられるか。

## 8. Worktree分離とsource provenance

| 区分 | path / branch / identity |
|---|---|
| Aの読取り元 | `/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928`; branch `pr2-v4-s2-parallelization-20260928`; 監査開始時HEAD `b6e65c6123475add5e620ec1064f361378bead95`、最終確認時HEAD `fd7552edc0334ccf57ecf501a128c85c8d22822a`（A側の並行更新、§8.3参照） |
| Bの独立編集先 | `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign`; branch `track-b-algorithm-codesign`; 同じM2 result commitをbaseとした |
| rootのproposal読取り元 | `/home/abe/Project/Partially Randomized Trotter`; branch `all-r-coherent-opt2-reoptimization`; HEAD `e098c54c78f589055082f9cfc2b13de50c90ca94` |
| remote | `git@github.com:HIROMU1015/Partially-Randomized-Trotter.git` |

BはMarkdown/Python sourceだけのsparse checkoutとした。M2 artifact、分子NPZ等は展開していない。必要な既存Markdownと一つの保存済みJSONをA側からread-onlyで参照した。新しい三文書以外の資料を一括copy/stage/commitしていない。

Aの`PROJECT_MAP.md`、総覧、Track構造資料はbase commitだけでは再現できないworking bytesを参照した。Bにcheckoutされた古い総覧を最新正本とは扱わず、下の監査開始時identityと、§8.3の最終確認時の更新を区別する。新しいB文書は提案であり、A正本やvalidation manifestの更新ではない。将来のB固有codeは`src/trottertracks/algorithm_codesign/`へ提案し、共有`src/trotterlib/`をtrack別コピーしない。

### 8.1 Repository / user text identities

SHA-256は2026-10-04に明示したtext/JSONだけから取得した。この表は**監査開始時snapshot**である。`same`は当時のA HEAD（b6e65c6）blobと同一、`modified`はそのtracked blobとworking bytesが異なる、`absent`はそのHEADにないことを示す。最終確認時のA checkout全体をこの表と同一とは主張しない。分子NPZのresolve/stat/hash/loadは行っていない。

| root | relative path | bytes | SHA-256 | A HEAD |
|---|---|---:|---|---|
| A | `AGENTS.md` | 3694 | `3e1e4a8aaab395fc9aca03d1f925f45a73502b8d47588c9588c0d5ab368bcbd5` | same |
| A | `PROJECT_MAP.md` | 42135 | `506e41407dbf421d71c9e13c5d72150c3b6ea777e8cfa6381c4f5d9a5b4e4cdc` | modified |
| A | `docs/research/研究概要・現状.md` | 120700 | `681c095491821009a30dfdd4c85a61afc19ed66e447993813589dca58e423855` | modified |
| A | `docs/tracks/README.md` | 3399 | `5ecb5785d141215c311b46b75ec4d5f91e002b7373abb210dff884eece60da4e` | absent |
| A | `docs/tracks/algorithm_codesign/README.md` | 8371 | `07037d5a6eab8a05c3a428b7b8f82762500b6a77a030d813163c4fa9da16df8e` | absent |
| A | `docs/tracks/migration_policy.md` | 6366 | `2d3e190277834350ba7354062d80827e3fc9351e16faa6369c49847ea3915a93` | absent |
| A | `docs/tracks/repository_structure_audit_2026-10-04.md` | 10346 | `0c2d102a5c0d1187067e8ce46e49bcbef38afa700ecbe0ba972bb60db622e9a2` | absent |
| A | `docs/tracks/resource_applicability/README.md` | 6689 | `c615808c2e1c9a7b4371da051e72ef715db1644588c1576153334ee687f5232f` | absent |
| A | `docs/pr2_matched_accuracy_m1_b1_result_validation.md` | 8343 | `4b619186a35e0c46fa7574757686ac2cbd47b8aa1ddd43ba76d2a6db5aa23b71` | same |
| A | `docs/pr2_matched_accuracy_m2_transfer_result_validation.md` | 11246 | `59d4e4ffc2c7b348ef592a21af6da7c51482e892166576ce2feafe92165ab42d` | same |
| A | `docs/research/r3_prior_art_and_minimal_contract.md` | 17240 | `d857aa74e2cce99d8cefc3bb512fe149d8c49fe3e7c1e1fdf92c6f317c09ad8a` | same |
| A | `docs/research_direction_pd_s1_posthoc.md` | 6501 | `8a2916b7f91b55e4a5d169943e0ab193741d2e7ada87ec2b54a49c87b05aae85` | same |
| A | `src/trotterlib/product_formula.py` | 4572 | `39c4e03748fcab03ea7e64936a19372cb5f1bd1bc9f72a4f09cef58c3c202eca` | same |
| A | `src/trotterlib/rte.py` | 77103 | `620a70077eb691281c01706ed726a91e625a98261c6e581c2d96502bc541e618` | same |
| A | `src/trotterlib/df_partial_s2.py` | 42929 | `6821823749b903ccd1dcb57803f27459f2b4fe265ca53175373fe55d00423d9e` | same |
| A | `src/trotterlib/df_partial_s2_repeated.py` | 30013 | `65b5ebd842a972735c7aa1e7ba4d11d8265ab586119f29bf5acc8631700c7d6d` | same |
| A | `artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json` | 170061 | `f41a92beb57e59cddc8c063b061c40acd4da50cb76ac0698efc2bce004937931` | same |
| root proposal | `track_b_algorithm_codesign_research_plan_20261004.md` | 45408 | `0b455ba98d16fdc4ae940b6593d5f5911ba45d5a3f0cf56e8d0be826d553beb2` | proposal |
| root proposal | `pr2_post_m2_research_redesign_20261004.md` | 46777 | `0819a0f358cdbb2029baa9e83c004b870c819100053ae209c9399c6e9707eb78` | proposal |
| user | `/home/abe/.codex/attachments/e36b28da-a6a4-4f72-b2e9-66c6c883cfa7/貼り付けたテキスト.txt` | 10461 | `9099b2f21dde2b0a3fda777d5be38cb59d42404305b832d948ab30a3f894d15e` | BF-0 authorization |

### 8.2 Downloaded primary PDF identities

PDFはrepository外の一時読取り領域に保存し、repositoryへcopyしていない。以下は今回読んだ版のcontent identityであり、immutable scientific evidenceとは別である。

| ID | SHA-256 |
|---|---|
| M | `dc108a7a7769315b3ccb61717754e5406d204355eab58a333bf2a8241a1123cf` |
| P | `000068f58332b5f24c6f17bd3b08ea22de1908e1ef2242a4ffc6f644aaf9aa61` |
| S | `47aa9742c3a137aa5e61e6ecdc2494d73e5c2e03f0717237ab2da212ac4a4453` |
| C | `839af3aefd947436547258ed3818d00f4baadab357e551ceaa5adf3359a8ca6a` |
| H | `8ccc2446366780cfbfb6f48da34931e2e85d9829e6527ed1acf796b0d82b2f3a` |
| I | `821771d1b0e5679d0dcdf741cd13b89fb97076e8a6661dae2c671e838f5974f6` |
| O | `485fdc8d291bd5c8216aa1d12419b7ccdc2022b4e090b5f84bcd5b949d9fddad` |
| B | `e30b20b86e78a8e763cc2ab48e400c2ca56de0ad3af2a6765e2c1bc85c846fba` |
| J | `dd15c958444600b816176aa31dbe99e8ca07d9ff5a5cbf917599e27da8e6e804` |

今回の作業は文献/ソースの読取り、上記保存済みcostの軽量集計、独立worktree作成、三Markdown草案の作成と静的確認までである。新しいHamiltonian、signal、sampling、circuit、compile、GPU操作、full test suite、artifact移動/再生成、stage/commit/pushは行っていない。

### 8.3 前回BF-0報告の最終確認時に検出したA側の並行更新

一次静的確認時は§8.1の全identityが一致していた。その後、最終確認でA HEADの進行と次の三textの更新を検出した。索引の追加箇所、総覧の最新停止位置、Track A README全文をread-onlyで再確認した。追記はPM-1の契約・source準備と実行未認可状態についてであり、Bの新しい科学証拠やauthorizationとして使用しない。Aの作業をBへcopyしたり巻き戻したりしていない。

| A relative path | 最終確認時bytes | 最終確認時SHA-256 |
|---|---:|---|
| `PROJECT_MAP.md` | 42722 | `090a368164af3e91f3f7340e2746905c7f55839c195fd65c8df2c71b89b626e2` |
| `docs/research/研究概要・現状.md` | 121717 | `af482e837fc5d6e6f3cc3994d92c28109f6c9b5493c443c4722227ac61806a39` |
| `docs/tracks/resource_applicability/README.md` | 7239 | `fa614119919092994090cd20b7e987f6f75c110ae0232d9b0042fccd187face8` |

上記以外の明示したsource、M1/M2報告、M2 result JSON、共通PF/RTE/DF Python source、root proposal/user指示のbytes/hashは開始時と一致した。Bのbaseは変更しない。

静的確認では、三文書間のlocal link、Markdown tableの列数、display-math delimiter、全表記source/PDF hashを確認した。BのHEADはbase commitのまま、indexにstaged変更なし、tracked Markdown/Python sourceのdiffなしで、新規三Markdownだけがuntrackedである。これはresearch testやscientific validationの実行結果ではない。

## 9. 2026-10-04進行方針の追記：BF-1後に全面再評価する

利用者は、BF-0 reviewの前に研究方針をもう一度全面再設計する運用を採らず、三文書review → 必要な最小修正 → BF-1事前登録・authorization → 一回限りのpilot → 全結果後mandatory STOP → 研究方針の全面再評価、を指定した。BF-1の判別対象はmethod deltaとdecision relevanceの二点であり、係数の違いだけではGOにしない。

狭い候補差分が残ることはpilotの情報価値をreviewする理由であり、methodの新規性が確定したことや実行許可を意味しない。reviewで既知methodとの重複、自由度/予算の不公平、受け入れられないoracle依存、F/Sの曖昧さ、過剰なparameter探索、positive-resultを拾うだけのGO条件が判明した場合はpilot前に修正または停止する。

BF-A（実質同じ）、BF-B（機構差はあるが資源差が小さい）、BF-C（decision-relevantな差がある）を、BF-1提案§10に整理した。**どのcaseでも先に停止し、その後にBの主研究としての継続・縮小・停止を再評価する。** BF-CからBF-2/BF-3を自動認可せず、BF-A/BからB-Sを自動開始しない。

[外部レビュー依頼](bf0_external_review_request_20261004.md)には三文書のreview対象bytes/hashを記録する。上の§8.1–8.3は前回監査時のsource観測履歴として保持し、今回のreview準備でAのstatus・source・artifact・pathを更新しない。今回の追加作業はB側Markdownの最小修正と依頼文の作成だけであり、レビュー結果はまだ受領していない。
