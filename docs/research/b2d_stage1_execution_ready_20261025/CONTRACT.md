# B2-D Stage 1：最終科学実行契約案・承認手順

2026-10-25 JST。**未承認の提案書。科学実行・正式grant発行・claim・消費は未実施。**
一括準備指示の起点は `a9b53854b283b7e84f7ffe5ba7284ab004c3fc57`。
技術接続は準備済みで、次はユーザーの契約採択・実行承認と手動承認記録の発行へ渡す。
本書・sample・未承認packetの作成自体は実承認にならない。

## 1. 最終SOURCE・契約binding

| 対象 | 固定値 |
|---|---|
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| 最終科学SOURCE候補 | **`348e08cbc44fb0a81cfc4b6932914f7b146a5b9c`** |
| 旧科学SOURCE候補 | `97a449ad53c74361ffc5c4cffd6de5f3f0f0a72e` は履歴として保全。今回の実行対象にはしない |
| 実行checkout | `/tmp/prt-b2d-stage1-science-20261025`、上記commitのclean detached worktree。準備のみ、未実行 |
| Published PF | HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae、`20d64c2fea1ed2d2188777df4f3d962ee94df486` |
| 契約原本 | [contract.proposal.json](contract.proposal.json)、statusは `UNAPPROVED_TEMPLATE`。SOURCE closure122ファイル・recipe・layout・scope・環境・出力を保持 |
| 正式採択後のcontract SHA-256 | **`26aaf07cf0e74eb60263d01a0bfa91ccc1a3c623d8795a18299aaaa3ac80ff6a`**。既定の採択処理によるbytesのdigestを事前計算したもの。採択済みという意味ではない |
| ローカル未承認packet | `/tmp/prt-b2d-stage1-draft-20261025`。proposalと承認文面sampleだけ、authorization.jsonなし |
| 正式packetの予定先 | `/tmp/prt-b2d-stage1-approved-20261025`、未作成 |
| Fresh科学出力 | `/tmp/prt-b2d-stage1-science-20261025/artifacts/research_b_b2d_stage1/20261025/once`、親を含め今回未作成 |

SOURCE closureは全src Python、fixture、3つのB2-D入口script、環境bindingを含む。後続の文書公開commitと科学SOURCEを区別する。
契約・出力先・環境・SOURCEを変える場合は、同じ承認を流用せずbindingを作り直す。コード変更後も348e08と同じSOURCEだと扱わない。

## 2. 科学条件

| 項目 | 最終承認を求める基準案 |
|---|---|
| Fixture／系 | `tests/test_df_partial_s2_repeated.py::_nontrivial_basis_case` のconstructor literal、synthetic 2-mode one-body＋3 DF因子。geometry／AO basis／分子DF rank探索は非該当 |
| Fixture blob | `0c92ae9c1e8c5721d9c8629a83435b855f5d6043`。元helper自体はimport・呼出ししない |
| Split | L_D=2、lambda_frobenius_squaredのweight-ranked prefix、one-body→選択DF順。generation-prefixではない |
| Sector／state | N=1完全粒子数sector、sorted OpenFermion JW順、mode0 occupationの共通physical状態 |
| Arm | A0 partial-S2、A1 standalone D-S2、A2 Yoshida4、A3 published Morales Y8m10b、A4 published YP8m8 |
| Grid | δ=±0.09、q=(2,3)、n_D=(1,2)はA1〜A4のみ。A0は4行、他各8行、計36行 |
| RTE | r=2、K=2、seed41はmetadata、analytic momentsのみ、samplingなし、truncation_tolerance metadata=1.0 |
| Scalar／threshold | extract_identity_phase、coefficient_atol=0 |
| Precision／correctness | complex128、絶対atol=1e-12。科学的改善の閾値ではない |
| Signal／phase | X/Y・半径・直接複素signal誤差。phase diagnostics無効、radius_floor／branch_margin=None |
| 判定 | descriptive_only_no_scientific_go。自動的な科学的GO／NO-GO・総資源優位の認定なし |

実H／preparation／tail／array hash、選択因子、N=1適格性・非可換性、H identity、A4数値correctness、complex128識別力は未確認。
固定recipeをclaim後に一回だけ生成し、合法domain、resolved identity、配列hash、同じsector／basis順とstate、resolved graphを照合する。
未生成の入力hashを架空値で固定しない。条件が不適格なら保存・STOPし、N=2／別state／別fixture／別精度／追加gridへ移らない。

## 3. 全scopeとcorrectness順序

[scope.json](scope.json)の `b2d_stage1_scope_v2` 全文を契約へ束縛する。
科学的処理境界のbudget vectorであり、親子の回数を足した「全job総科学call数」を主張しない。
NumPy／BLASの各内部演算は、それを内包する固定関数のSOURCEと入力recipeで管理する。

- 入力境界10回：Hamiltonian1、split1、preparation1、sector1、full H1、D block3、retained tail1、occupation state1。
- preparation内のone-body Gaussian1、全DF Gaussian3、残差DF Gaussian1、および既存basis fingerprintのOperator等を含む。retained tail内のbasis QuantumCircuit／Operator・sector変換・leakageも必須入力処理として含む。
- runtime domainは2 modes、N=1、D3、選択因子2／残差1、retained非identity成分1〜3／basis1。Z0／Z1／Z0Z1の合法な展開だけを許し、実成分・basis hash・operation数をreceiptへ保存する。
- 共通DAGは36行・941 dispatch。backendの実exponentialは共有cacheの448 unique keyまで、scopeに列挙したlabel/timeのみ。
- RTE config／moments各2、signal expectation144。Taylor・power・normalizationは既存関数内の固定scope。phase演算・samplingなし。
- **追加のpublished列規約gate4回**をA4 correctness内に登録。DAG941とcache key448は維持。最大logは3134 recordに更新。
- 完走時は登録回数・resolution全件を照合。未登録処理・重複・超過・完了不足はSTOP。

将来の実行順序は次を必須とする。

1. Input identity・sector／basis・state norm・Hermiticity・H=D+R+cIを検査。
2. retained residualとD内部／D-R非可換性を確認。
3. 全4件の独立数学的A4 `P K^n P†`・actual inverseを検査。
4. 固定upstreamの `_get_s2_sequence` に基づくfull mathematical productとの独立operator照合を全4件実施。
5. 全gate通過後だけA0〜A4の固定operator／signal比較へ進む。

Upstream [product_formula.py](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py) は列を数学的 `P K^n P^-1` と記述する。一方、[旧validator](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/validate_morales_y8m10b_h2.py) はlistをそのままappend相当の左積で消費する。
今回の対象は文書化された数学的積であり、実appendではその順を反転する。この規約対応を明示的に照合し、旧validatorの直接append operatorと同値だとは主張しない。
係数generatorは既存固定版を共有し、full列constructorの3関数bodyは固定upstreamから変更なしで移植。独立参照はDRun展開や `p @ k**n @ p†` を使わないflat S2数学積である。
この照合に失敗した場合はpublished法としての比較を開始しない。今回は数値照合未実行。

Full controlled Hadamard wrapper、native compile／transpile、processor価格、shot認証、全RPE総資源評価は対象外。

## 4. 実行環境

[environment.json](environment.json)に実在する環境を記録し、契約とSOURCEへ束縛した。既存venvは変更しない。

| 項目 | 固定値 |
|---|---|
| 実行interpreter | `/tmp/pr-phase2df1pre-private-20261023/venv/bin/python`、常に `-I -B` |
| Runtime | CPython 3.11.9 final。venv／base prefix、interpreter bytes、pyvenv.cfg、package import元も記録 |
| NumPy／SciPy | 1.26.4／1.14.1 |
| Qiskit／Aer／Nature | 1.3.0／0.15.1／0.7.2 |
| OpenFermion／OpenFermion-PySCF | 1.6.1／0.5 |
| PySCF／Numba／Matplotlib | 2.7.0／0.60.0／3.9.2 |
| 依存closure | 全50 distributionの実version・RECORD digest、package originのhash。依存requirementsの整合もmetadataで確認 |
| 環境変数 | PYTHONPATH／PYTHONHOME／LD_PRELOAD／LD_LIBRARY_PATH／OMP／OPENBLAS／MKL／NUMBA thread指定は全て未設定の状態にbinding |

環境metadata・import元を確認したもので、実科学backendやGaussian回路の動作を実証したものではない。
version・interpreter・import元・bindingが変わればpreflightで拒否する。未導入時のGaussian fallback、環境の黙示更新なし。
実資源は32 CPU、RAM約62 GiB（available約52 GiB）、GPU driver利用不可。2×2 sector・共有cacheを再利用し、人工的なCPU／RAM／worker／wall-time capを追加しない。共有環境・他研究は変更しない。

## 5. 実ユーザー承認方式と、ユーザーが行う操作

外部の独立したchat承認APIは確認できなかったため、**ユーザー自身が承認内容を手動照合し、管理者権限で保護された記録を発行する方式**を採用案とする。
これはchat identityの暗号学的・プログラム的証明ではない。ユーザー／管理者の手動照合とOS所有権に依存することを明示する。

承認recordは固定先 `/var/lib/prt-b2d-stage1/approvals/<grant-id>.json` にroot所有・agent非書込で置く。
Codexが書けるworkspace／tmpのpacket、任意文字列、自己申告authority、hash一致だけでは実承認にならない。
receiptは実際の承認本文hash・参照、SOURCE、recipe、layout、scope、環境、output、packetの絶対path、authorization digest、一回実行／STOPを束縛する。
実行workerは非root。protected receiptが無い／user所有／symlink／内容不一致ならclaim前に拒否する。

**ユーザー操作は次の二段階。今回はどちらも未実施。**

1. 研究Bチャットで下記契約・scope・環境・grant／STOPを確認し、実際に明示承認する。その実承認メッセージをユーザー自身で `/tmp/b2d-stage1-user-approval.txt` に保存する。sampleを承認済み記録として代用しない。
2. ユーザーが内容・SOURCE・digestを照合して、次の管理者用commandを自分で実行する。これは**新grantの正式発行**であり、Codexに今回実行させない。

```bash
sudo /tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-b2d-stage1-science-20261025/scripts/issue_b2d_stage1_approval.py \
  --draft /tmp/prt-b2d-stage1-draft-20261025 \
  --packet /tmp/prt-b2d-stage1-approved-20261025 \
  --approval-file /tmp/b2d-stage1-user-approval.txt \
  --approval-reference '研究Bチャットの実承認メッセージ日時またはID' \
  --confirm-contract 26aaf07cf0e74eb60263d01a0bfa91ccc1a3c623d8795a18299aaaa3ac80ff6a \
  --acknowledge-manual-user-review
```

管理者操作では、既存形式のcontract.json／authorization.json／user_approval.txtとprotected receiptを新規作成する。science・claimはしない。
ユーザーにsudo権限が無ければ、この記録発行だけを信頼する管理者へ依頼するのが最小解消方法。保護を外した自己承認へのfallbackは設けない。
発行後にユーザーが次のCodex実行指示を出した場合だけ、非root workerで次を実行する。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-b2d-stage1-science-20261025/scripts/run_b2d_stage1.py \
  --execute --packet /tmp/prt-b2d-stage1-approved-20261025 \
  --output /tmp/prt-b2d-stage1-science-20261025/artifacts/research_b_b2d_stage1/20261025/once
```

**承認文面の完成案（実承認ではない）：**

> 科学SOURCE `348e08cbc44fb0a81cfc4b6932914f7b146a5b9c`、正式採択後contract SHA-256 `26aaf07cf0e74eb60263d01a0bfa91ccc1a3c623d8795a18299aaaa3ac80ff6a` のB2-D Stage 1条件を採択し、固定scope・環境・fresh outputで一回だけの限定科学検証と新grantの発行を承認します。retry／resume、条件変更、追加gridは認めず、完了・失敗ともmandatory STOPとします。

## 6. Claim・保存・STOP・残る未確認事項

既存のconsumed marker／`claim_packet` のatomic exclusive-createを再利用する。grant再利用、marker削除、旧STOP解除なし。
Fresh outputの親とoutputは将来の承認済みlauncherだけが生成する。今回はscience artifact directoryを作っていない。
binding、input provenance／配列、resolved graph、event、partial／full result、status、STOPの9ファイルを保存し、失敗時も取得済み情報・理由codeを保持する。
全A4／published gate失敗時は性能比較なし。scope逸脱・途中例外・非有限数値・保存失敗もSTOP。秘密の例外本文・他研究記録を出さない。
Log3134 record、record64 KiB、JSON1 MiB、総出力16 MiB、STOP reserve256 KiB。一回実行・自動retry／resumeなし。

complex128の識別力は未資格で、曖昧な有限差を改善認定しない。必要な非可換性・残差が残らない場合やprecision不足はINCONCLUSIVEとして停止し、条件を探索しない。
科学SOURCE・scope・環境・launcher・手動承認接続は準備済み。**残る実行境界は、ユーザーの最終承認とprotected receiptの手動発行である。**
それらの前には科学実行不可。実科学経路・H identity・A4数値correctness等の未検証は、準備完了と区別して維持する。

担当：研究B／ユーザーが最終採否と手動記録発行、Codexはその後の別途明示実行指示で固定Stage 1だけを一回実行する。
C1最優先・他研究非干渉を維持し、通常のB進行で統括報告は必須にしない。今回の公開確認後 **mandatory STOP**。
