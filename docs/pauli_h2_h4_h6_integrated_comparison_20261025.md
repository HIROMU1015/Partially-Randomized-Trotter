# H2・H4・H6 Pauli-JW：保存PF/B0/PR比較の最新版統合

2026-10-25。専用branch `codex/pauli-h2-h4-h6-integration-20261025`。
基点 `8fd1ceb283309a7e515ff4a533a253260bdd0f78`。

**3分子×5要求精度の保存比較を15行に統合した。H4だけを最新141構成探索の主結果へ更新し、H2/H6の候補ID・G・G比は変更していない。**
全数値は固定GitHub refの保存値の引用であり、新しい科学計算・数値再解析・候補選抜・再ランキングを行っていない。
「最良」は元研究の有限登録集合・選抜手順内の表現であり、母平均winnerや全パラメータ最適性の認定ではない。

## 1. データ入口と固定条件

- [最新版15行CSV](../artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/H2_H4_H6_optimized_PR_comparison.csv)
- [同内容JSON](../artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/H2_H4_H6_optimized_PR_comparison.json)
- [列の説明・使い方](../artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/README.md)
- [行ごとの出典](../artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/row_provenance.csv)、[列ごとの転記対応](../artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/field_provenance.json)
- [取得元commit・Git blob・SHA256](../artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/source_manifest.json)、[転記照合](../artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/verification.json)

| 条件 | H2 | H4 | H6 |
|---|---|---|---|
| Geometry / basis | Linear、隣接1.00 Å、STO-3G、中性singlet | 同左 | 同左 |
| 非identity項数 M | 14 | 184 | 918 |
| System qubits / ancilla | 4 / wire4 | 8 / wire8 | 12 / wire12 |
| 表現 / PF | Pauli-JW / S2 | 同左 | 同左 |
| T / compiler | 0.8 / Qiskit1.3.0、rz/sx/x/cx、opt1、seed17 | 同左 | 同左 |
| 制御 / 測定 | central-rotation、cosine/+Im sine | 同左 | 同左 |
| 回路価格 | 状態準備を除くfull Hadamard wrapperのcompiled RZ | 同左 | 同左 |

Backend/Target/coupling/layout/routingの指定なし。各分子内で保存Hamiltonian、raw state/ray、full-H reference、正規化規約を共有する。
分子間でstate/referenceは別であり、H4の局所Arb保証をH2/H6や新H4候補へ移していない。
Pauliではconstantを常に保持し、同じ切り捨て構成のB0/B0Iを重複させない。DF-nativeの結果はこの主表に含めない。

## 2. 候補探索数と確認標本

以下は元報告・catalogの保存記載の引用である。新たに候補数やtrajectoryを集計していない。

| 分子 | 登録候補数 | 決定論的側 | PR側 | 凍結確認集合と別seed標本 |
|---|---|---|---|---|
| H2 | 120 | 56（B0 52、full PF 4） | 64 | 2候補、各96 paired trajectory |
| H4 | PR141、決定論的106は別保存探索 | 保存106構成を引用 | 初期100＋適応30＋11 | 4候補。p36/q2/r8のK2/K4は各48、p0/q8/r8のK2/K4は各384。保存確認計864 |
| H6 | 215 | B0 110、full PF 4 | 101 | 4候補、各48 paired trajectory |

主表の `confirmation_n` は当該候補のpaired trajectory数であり、量子測定shot数ではない。
同じ確認候補を複数精度に使った行を足して、新しい標本数にしてはいけない。

**H4は探索段階でnomineeを固定し、その候補の独立確認標本から主表Gを評価する。**
保存された確認後の最小候補は別結果であり、ε=.05のK4、ε=.01のK2を主表へ差し替えない。

**H2/H6は確認対象集合を事前凍結したが、主表はその確認集合内の観測G最小候補である。**
確認標本を見た後の選抜を含む。H4と同じ統計的手順として扱わない。
両系列のSEや名目95%区間を、選抜後の厳密な同時信頼保証・母平均winner保証へ格上げしない。

## 3. 保存exact値による15行比較

Gの単位はRZ×shot、主表示は仮想ρ=0。PRのGは有限確認標本の価格に基づく条件付き会計値である。
この表は元CSV文字列を転記したもので、Gや比を再計算したものではない。
候補IDのpは保持prefix、q/r/Kは元候補のパラメータ。決定論的側のH4 IDは元正本の表記を保持する。

| 分子 | ε_sig | 保存選択PR | 確認n | PR G exact | 保存最良決定論的PF | 決定論的G exact | PR / 決定論的G exact |
|---|---|---|---|---|---|---|
| H2 | 0.05 | PR_p0001_q1_r4_K4 | 96 | 1251625/2 | PF_p0014_q1 | 1353240 | 250325/541296 |
| H2 | 0.01 | PR_p0001_q1_r4_K4 | 96 | 250331975/16 | PF_p0014_q1 | 45976320 | 50066395/147124224 |
| H2 | 0.005 | PR_p0001_q1_r4_K4 | 96 | 500687975/8 | PF_p0014_q1 | 293617350 | 20027519/93957552 |
| H2 | 0.001 | PR_p0001_q1_r4_K4 | 96 | 25043990925/16 | PF_p0014_q4 | 15183677784 | 8347996975/80979614848 |
| H2 | 0.0005 | PR_p0001_q1_r4_K4 | 96 | 200448054525/32 | PF_p0014_q4 | 77733393012 | 66816018175/829156192128 |
| H4 | 0.05 | p036_q2_r8_K2 | 48 | 143838793/12 | p132_q1 | 23237760 | 20548399/39836160 |
| H4 | 0.01 | p000_q8_r8_K4 | 384 | 60500170885/192 | p144_q2 | 1440845160 | 12100034177/55328454144 |
| H4 | 0.005 | p000_q8_r8_K4 | 384 | 242000558639/192 | p176_q2 | 9734669448 | 34571508377/267008076288 |
| H4 | 0.001 | p000_q8_r8_K4 | 384 | 2016670655853/64 | p180_q8 | 579283719324 | 672223551951/12358052678912 |
| H4 | 0.0005 | p000_q8_r8_K4 | 384 | 8066682748313/64 | p180_q8 | 2663993522484 | 1152383249759/24356512205568 |
| H6 | 0.05 | PR_p0684_q1_r8_K4 | 48 | 1380373939/8 | B0_p0724_q1 | 182956812 | 1380373939/1463654496 |
| H6 | 0.01 | PR_p0078_q8_r8_K2 | 48 | 23330078720/3 | B0_p0830_q2 | 10708083010 | 2333007872/3212424903 |
| H6 | 0.005 | PR_p0078_q8_r8_K2 | 48 | 766686975275/24 | B0_p0830_q4 | 65486795360 | 153337395055/314336617728 |
| H6 | 0.001 | PR_p0078_q8_r8_K2 | 48 | 8036333388915/8 | B0_p0864_q8 | 2989511806608 | 2678777796305/7972031484288 |
| H6 | 0.0005 | PR_p0078_q8_r8_K2 | 48 | 22593289103905/4 | B0_p0864_q8 | 12486761806824 | 22593289103905/49947047227296 |

H2の決定論的選択はfull PF、H4/H6はB0/B0I。H4の保存正本は、全5精度でPR主候補が決定論的より低い条件付きGを報告する。
H2/H6も元報告の有限確認集合内比較を保持する。今回新しい順位判定はしていない。

H4の比は `PR_over_best_deterministic_exact`、H2/H6は `PR_over_D_exact` をそのまま使用した。
旧H4決定論的CSVの逆向き比を反転する処理は行っていない。
H4のratio小数表示は主CSVに存在しないため空欄とし、exact分数を保持した。
`PR_C_cos_SE` / `PR_C_sin_SE` は価格標本SE、`PR_G_SE` はGのSEであり、同じ量ではない。
H4価格SEは保存candidate catalogから取得し、SEや信頼区間を新しく計算していない。

## 4. PR方式分類とH4の更新

| 分子・条件 | L_D/q/r/K | 方式分類 |
|---|---|---|
| H2・5精度 | 1/1/4/4 | Hybrid PR（0<L_D<14） |
| H4・ε=.05 | 36/2/8/2 | Hybrid PR（0<L_D<184） |
| H4・残る4精度 | 0/8/8/4 | Fully randomized RTE（L_D=0） |
| H6・ε=.05 | 684/1/8/4 | Hybrid PR（0<L_D<918） |
| H6・残る4精度 | 78/8/8/2 | Hybrid PR（0<L_D<918） |

L_D=0でもconstant phaseは保持する。L_D=Mは全項保持で非自明なランダム化tailがなく、今回の選択PRには該当しない。
したがって、PR候補集合の条件付き低G観測をハイブリッド化自体の効果と同一視しない。
Hybrid対fully randomizedの新しい最小費用比較・順位付けは行っていない。

旧H2/H4/H6 CSVのH4は固定27/8/4/4・256 trajectoryだった。
これを新しい主表では最新141構成探索の保存主結果へ置き換えた。
旧固定PRに対するε=.05のB0優位という過去の記述と、最適化候補を用いた最新の条件付き比較は対象が異なる。
旧CSV・旧報告・旧STOPは変更していない。旧固定PRは取得した原資料snapshot内だけの歴史的baselineとして保持する。

## 5. 使用範囲と未保証事項

主表示は仮想ρ=0・状態準備を除くRZ×shotであり、実compiledρ=0の証明ではない。
PR価格は有限確認標本の平均で、未観測希少order/eventによる母平均費用tailは未保証。
元の条件付き会計・数値包含・局所Arb・正式certificate受理を区別し、保証状態を変更していない。
大域最適性、母平均winner、全law実装残差、無条件end-to-end資源優位は未証明である。
探索候補集合とstate/referenceは分子ごとに異なり、3分子から一般的なサイズ依存則は確定しない。
H4の `guard_order_saved` は元資料の保存判定の引用であり、新候補の実ρがguard内にあるという保証ではない。

保存Gの引用だけで要求された有限候補の比較表は完成する。科学的保証の強化は別の研究判断・別の明示実行承認を要する。
論文原稿、主要claimの正式採択、新しい詳細科学レビュー、次の数値実行は今回行わない。

## 6. 固定出典・SOURCEと技術確認

以下の固定refからGitHubファイルを実際に取得し、保存10ファイルのGit blob SHAを照合した。
比較表の行は出典CSV、H4 summary、catalogの同一ID/εと文字列で照合した。

| 正本 | 固定commit | 主な保存資料 |
|---|---|---|
| H2/H6 | `8fd1ceb283309a7e515ff4a533a253260bdd0f78` | [報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8fd1ceb283309a7e515ff4a533a253260bdd0f78/docs/pauli_h2_h6_resource_comparison_20261025.md)、H2_H6_best_comparison.csv、H2_H4_H6_comparison.csv、batch_catalog.json |
| 最新H4 PR | `d3fdfad880902b7ff35b710d22e1c9759f7087c2` | [報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md)、best_by_precision.csv、result_summary.json、candidate_catalog.json |
| H4決定論的PF | `fbfbfd21a252a4a32f52e3b40e50acd91b3c626a` | [報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbfbfd21a252a4a32f52e3b40e50acd91b3c626a/docs/non_df_h4_pauli_b0_comparison_20261025.md)、best_by_precision.csv |

科学SOURCEは結果公開commitと別であり、元報告の記載を保持する。

- H2/H6：`d17780fcb57bcff344325a8632f718c0cc56f1dc`、`0880c38e3e8a6ea57882ec66e6366a8b86e408e2`、`a8c7c7dca5a03aae1b008a09048d89c216bbf89c`。元研究のnative SOURCE履歴は同研究のSOURCE_history.bundleに保存されており、各SHAのGitHub通常commitページの存在を新たに認定したものではない。
- H4 PR：`52865921930b3afd11b1447dff0a6cbcb6085af8`、`4a7f91fa7ba8aa3ae1ce4ccb30bde9b973869410`、`a17ad381b6ff2ac615e3cd930c641947c551bcbc`。
- H4決定論的：`47384f5698094e466e83027a5a54751e0cf6000e`、`8ab472db8f90d6f64f6cc626584655c5a7487a56`。

[統合スクリプト](../scripts/integrate_pauli_h2_h4_h6_saved_results.py)は標準ライブラリのCSV/JSON文字列転記だけを使う。
既存packagingのDictReader/DictWriter規約を再利用し、科学・会計runnerはimportしない。
保存decimal tokenをfloat変換せず、G・比・SE・名目区間・順位を計算しない。
巨大raw/QPY/trajectoryは複製せず、固定refから取得した要約・catalog・報告だけを小さいsnapshotとして同梱した。

15行の一意性、全5精度、exact値/IDの転記、H4旧固定PRの不混入、方式分類、選抜方式の保持を確認した。
人工テスト3件（prefix境界分類、候補IDの一意join、JSON数値tokenの文字列保持）はPASS。報告書の15行もCSVと文字列一致を確認した。
著者Codex自身の構造検証であり、独立科学レビューではない。
旧科学SOURCE/raw/grant/ledger/marker/STOPと旧比較CSVは変更せず、別branchをmergeしていない。
公開commitとremote照合状態は[publication_status.json](../artifacts/pauli_h2_h4_h6_integrated_comparison/2026-10-25/publication_status.json)を参照。

新科学call・数値再解析・再ランキング・新統計区間はすべて0。公開照合後 **mandatory STOP**。
次の担当はユーザー/GPT：この条件付き比較をどう研究上位置付けるかを判断する。Codexは追加指示なしに科学計算へ移行しない。
