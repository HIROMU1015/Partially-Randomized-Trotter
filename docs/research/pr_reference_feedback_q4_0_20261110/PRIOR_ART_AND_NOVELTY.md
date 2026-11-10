# Q4-0：一次文献とQ方式の対応

2026-11-10 JST。これは依頼された技術調査・比較設計であり、独立科学レビュー、新規性の正式判定、研究claimの採択ではない。Qの正本は[`08f9cbe77456ec29d50a992d7165c4d726df7bf7`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/08f9cbe77456ec29d50a992d7165c4d726df7bf7)。新しい科学数値計算は行っていない。

## 1. 確認した一次資料と読んだ範囲

既存[`e8b19ff6`の文献調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md)のO6・L18を再利用し、以下の必要箇所を追加照合した。世界の全論文・引用網を網羅した調査ではない。PDFのページは先頭を1とする。

| ID | 著者・題名・出版情報／確認版 | 今回確認した本文・根拠 |
|---|---|---|
| P1 | B. L. Higgins, D. W. Berry, S. D. Bartlett, H. M. Wiseman, G. J. Pryde, **Entanglement-free Heisenberg-limited phase estimation**, Nature 450, 393–396 (2007), DOI [10.1038/nature06257](https://doi.org/10.1038/nature06257)。[arXiv:0709.2996v2](https://arxiv.org/abs/0709.2996v2), 2008-06-12 | PDF pp.2–3、Fig.1c/2：過去の全測定結果でBayesian feedback位相を更新し、次の測定後の期待位相分散を小さくする。複数pass数も用いる。Qと共通なのは可変reference位相による測定軸変更。全policy、分散指標、resource定義まで同一ではない |
| P2 | B. L. Higgins, D. W. Berry, S. D. Bartlett, M. W. Mitchell, H. M. Wiseman, G. J. Pryde, **Demonstrating Heisenberg-limited unambiguous phase estimation without adaptive measurements**, NJP 11, 073023 (2009), DOI [10.1088/1367-2630/11/7/073023](https://doi.org/10.1088/1367-2630/11/7/073023)。[0809.3308v4](https://arxiv.org/abs/0809.3308v4), 2010-08-31 | PDFのChernoff/Hoeffding式(1)、固定X/Y頻度の複素vector、複数passでのalias解消。有限確率保証とRMSEの区別も既知。異なるqを使う全protocolのscalingを、固定m17の定数shot比較へ流用しない |
| P3 | Shelby Kimmel, Guang Hao Low, Theodore J. Yoder, **Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation**, PRA 92, 062315 (2015), DOI [10.1103/PhysRevA.92.062315](https://doi.org/10.1103/PhysRevA.92.062315)。[1502.02677v3](https://arxiv.org/abs/1502.02677v3), 2021-10-21、[Erratum PRA104,069901](https://doi.org/10.1103/PhysRevA.104.069901) | Theorem1、Eqs.(1.1)–(1.3)、先頭Errata：cos/sinの確率への有界additive errorとresource scaling。旧§Vのbranch条件π/(2k)はπ/(3k)へ訂正。旧解析の定数を再使用しない。Q3の独自branch証拠は保持し、今回それを再判定しない |
| P4 | Antonio E. Russo, William M. Kirby, Kenneth M. Rudinger, Andrew D. Baczewski, Shelby Kimmel, **Consistency testing for robust phase estimation**, PRA103,042609 (2021), DOI [10.1103/PhysRevA.103.042609](https://doi.org/10.1103/PhysRevA.103.042609)。[2011.13442v1](https://arxiv.org/abs/2011.13442v1), 2020-11-26 | §II–III、App.B/C：候補位相集合と履歴整合性、robust scalingの補強。観測後のconsistency検査を、全成功履歴で数値実装が拒否しないことの証明や未知誤差への無条件certificateと見なさない |
| P5 | Nathan Wiebe, Christopher E. Granade, **Efficient Bayesian Phase Estimation**, PRL117,010503 (2016), DOI [10.1103/PhysRevLett.117.010503](https://doi.org/10.1103/PhysRevLett.117.010503)。[1508.00869v1](https://arxiv.org/abs/1508.00869v1), 2015-08-04 | PDF§II–IV、Eqs.(1),(2),(4)–(6)：Bernoulli cosine likelihood、rejection filtering、posteriorから選ぶreference位相とM≈1.25/σ。decoherenceモデルではT2を既知とする。credible region／中央値誤差を今回の一様failure保証へ移さない。arXiv v1と出版版全差分は未照合 |
| P6 | Thomas E. O’Brien, Brian Tarasinski, Barbara M. Terhal, **Quantum phase estimation of multiple eigenvalues for small-scale (noisy) experiments**, NJP21,023022 (2019), DOI [10.1088/1367-2630/aafb8e](https://doi.org/10.1088/1367-2630/aafb8e)。[1809.09697v3](https://arxiv.org/abs/1809.09697v3), 2018-12-17 | [本文](https://arxiv.org/html/1809.09697v3)§II、III.2/III.3 Eqs.(32),(33)、V.1/V.2：single-ancilla回路、古典time-series/Bayesian処理、kとreference位相、decoherenceとH4応用。複数固有値／noise-model推定であり、今回の元H厳密ground・PF/PR bias集合の保証と異なる。HTMLに別の生成日が表示されるため版年はabsを採用 |
| P7 | Brennan de Neeve, Andrey V. Lebedev, Vlad Negnevitsky, Jonathan P. Home, **Time-adaptive phase estimation**, PRResearch7,023070 (2025), DOI [10.1103/PhysRevResearch.7.023070](https://doi.org/10.1103/PhysRevResearch.7.023070)。[2405.08930v3](https://arxiv.org/abs/2405.08930v3), 2025-05-05 | [本文](https://arxiv.org/html/2405.08930v3)§II Eqs.(1)–(5)、III.1/III.2 gain-rate、IV、Code availability：contrast/asymmetryを含むlikelihood、control位相とk、期待情報獲得／時間。Qの回路尤度を含む既知familyだが、Qの具体的minimax区間policyと同一とは断定しない |
| P8 | Federico Belliardo, Valeria Cimini, Emanuele Polino, Francesco Hoch, Bruno Piccirillo, Nicolò Spagnolo, Vittorio Giovannetti, Fabio Sciarrino, **Optimizing quantum-enhanced Bayesian multiparameter estimation of phase and noise in practical sensors**, PRResearch6,023201 (2024), DOI [10.1103/PhysRevResearch.6.023201](https://doi.org/10.1103/PhysRevResearch.6.023201)。[2211.04747v2](https://arxiv.org/abs/2211.04747v2), 2024-06-21 | [本文](https://arxiv.org/html/2211.04747v2)Precision bounds、App.A Eq.(6)、App.B Eqs.(20)–(25)：cos/sin、未知visibility、particle posterior、nuisance重み、期待varianceを用いる測定選択。phaseのみ重視する場合もvisibilityを同時に扱う。中央値二乗誤差／asymptotic CR比較は今回のworst-case有限CIと別指標 |

P3/P4は既存文献調査の訂正確認を引き継いだ。P1/P5/P7/P8の関係づけは今回の本文確認による。全定理の独立再証明・外部実装の再現実験は行っていない。

追加の限定確認：E. Roccia, V. Cimini, M. Sbroscia, I. Gianani, L. Ruggiero, L. Mancino, M. G. Genoni, M. A. Ricci, M. Barbieri, **Multiparameter approach to quantum phase estimation with limited visibility**, Optica5,1171–1176 (2018), [DOI](https://doi.org/10.1364/OPTICA.5.001171)。著者機関の[metadata](https://air.unimi.it/handle/2434/609502)と検索可能な原稿冒頭は確認したが、PDF本文取得は403となった。節・式・保証詳細は未確認として、主要なunknown-visibility比較には本文取得済みP8を使う。

2026年の追加一次資料として、B. Akoramurthy, B. Surendiran, Xiaochun Cheng, S. N. Mohanty, **Bayesian quantum phase estimation beyond the eigenstate approximation: sample complexity, Cramér–Rao bounds, and adaptive measurement design for non-eigenstate and mixed-state inputs on NISQ devices**, Front. Quantum Sci. Technol.5 (2026-08-11), [出版本文/DOI](https://doi.org/10.3389/frqst.2026.1862834)のAlgorithm1と§11.5を部分確認した。既知overlap/purityを置く解析で、その推定費は残ると同論文も述べる。漸近効率・MSEの比較を未知ρの一様CIへ転用せず、oracle診断側へ分類する。全定理・code・実験主張の整合性は未監査。

## 2. Qと既知方式の数学的対応

Qはprefix中心Ecから `φ*=−t17 Ec+π/2` を選び、`P(−φbin)→H→Measure`を340回固定して実行する。測定は `Xφ=cosφ X+sinφ Y`、meanは `Re(e^(−iφ)z)`。これはP1/P5/P7の可変reference位相を持つRamsey/Hadamard実験と同じ**測定family**。prefix中心の近くを高感度なquadratureで読むという原理の発明を主張できない。一方、各論文の逐次policy、interrogation長の変更、Bayesian目的量までQと同じとはしない。

P7 Eq.(1)で対称readoutのλ=1、contrast ζ=ρと置き、kφへ今回の `−tE+b` を対応させれば、QのBernoulli応答形が得られる。この対応は**形の対応**であって、P7が未知ρ／bounded bのminimax保証を与えるという意味ではない。P8はρをlatent parameterとして推定し、Qは範囲[ρL,1]を残してconfidence集合を射影する。未知ρを扱うこと自体は既知、扱い方・保証基準が異なる。

P3のadditive errorは測定**確率**の単位。例えばphase bias bと半径ρによるcos確率の変化は `|ρ cos(x+b)−cos x|/2 ≤ (1−ρ)/2+ρ|b|/2` と評価できるが、これだけで全roundの閾値・shot定数を採用しない。Qのraw disk→phase/radius変換と同じ式ではなく、operator normをradとする根拠にもならない。

未知ρの一方向応答は、一般にρとphaseを同時に点同定しない。Qは事前区間を使って必要な有限幅を保証する方式であり、visibilityの一致推定器ではない。Q0に保存された無限標本時の正の区間幅もこの位置付けと整合する。Qの340は現CI familyでの十分数であり、局所Fisher情報やCR boundから全推定器最小値と認定しない。

## 3. 何が既知で、何が貢献候補か

| 論点 | 既知の部分／対応 | 今回残る貢献候補と、認定していないこと |
|---|---|---|
| adaptive測定原理 | P1/P5/P7のreference位相制御 | 新原理というclaimは支持されない。Qはprefixで一度選んだ方向のbatchであり、m17内feedbackはしない |
| confidence inversion | likelihood/CIをparameter集合へ逆写像しnuisanceを射影することは標準的な集合推定 | 特定のannulus＋bounded bias＋全count幅の解析・有限実装接続は貢献候補。同一の定理が既存文献にないとの網羅的証明はない |
| unknown visibility | P8のjoint Bayesian推定、P7のcontrastモデル | Qは校正を追加せず保存ρLだけで有限保証を作る。その条件で強い既知CI方式より有利かは未評価 |
| 数値・native保証 | RPEのbranch、系統誤差への頑健性はP2–P4で既知 | PF／finite-bit／SOURCE／native／prefix受理／feedback丸め／hull丸めを同じ元H精度・failureへ接続した固定caseの証拠。一般定理や既存保証の完全包含を認定しない |
| total cost | P6/P7にもinterrogation長や古典処理・時間を考慮する発想がある | 実Compiler Aのwhole interrogation価格境界とshotsを資源別に結ぶことが今回の具体的証拠。費用意識そのものの発明ではない |
| PR固有性 | 同じ複素meanと誤差域があれば別PFにも使い得る | PR固有の利益ではない。B′にも同じQを与えた比較と、PR vs Full/Truncatedを分ける |

候補claimの表現例は「固定PR実装で、既知の適応quadrature測定を未知半径・有界biasの有限標本保証およびnative総費用に接続した」。強X/Y／matched CIの比較完了前に「既知最良位相推定より優れる」「新しいadaptive PEを発明」「世界初」とは書かない。論文化でどこを主claimとするかはGPT／ユーザーの正式判断事項。

## 4. 実装の取得・再利用可能性

P8が挙げる[著者code](https://github.com/fedebell/AppuntiDottorato/tree/620a3cca53635c6542d90b8ad63ebdab78a07bfb/bayesian_multiparameter)の`bayesian_multiparamter.ipynb`を静的に読んだ（2026-11-10、Git blob `4fa6535686940f84eaeb664780efed054b2aed19`）。`prob_outcomes`、`bayesian_update`、`estimate_covariance`、`utility`、`optimize`、`resample`があり、phase/visibilityのparticle処理と実験選択を再利用できる候補。Torch/NumPy等を用い、実行entryにはsimulation/ground-truth入力もある。関数定義とlikelihood・utilityの該当箇所の読取だけで、notebookのimport/実行、数値再現、coverage資格化はしていない。そのままPRのstrict CI providerになるものではない。

P7のCode availabilityは[Bitbucket qtape](https://bitbucket.org/brennann/qtape/src/main/)を指すが、今回のアクセスでは本文を取得できなかった。gain-rateのSOURCE全体を読了したとはしない。P5/P6を含め外部policyの完全な再実装はせず、まずrepoの既存CI／branch／accounting資産を使う[最小比較](NEXT_NUMERICAL_VALIDATION_PLAN.md)を優先する。

未取得本文・部分確認・版差・最新追加資料の未監査を残す。特定の同等手法を発見しなかったことは不存在の証明ではない。**公開・照合後mandatory STOP。新規性と主要claimの正式採否はGPT／ユーザー。**
