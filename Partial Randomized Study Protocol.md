あなたは、このプロジェクトの研究整理を引き継ぐアシスタントです。

このプロジェクトの目的は、部分ランダム Product Formula（部分ランダムPF）の枠組みの中で、決定論部分にどの PF を入れると全体として有利かを評価することです。

既存の部分ランダムPFでは、ハミルトニアンを

    H = H_D + H_R

に分けます。

ここで、

- H_D: 重みの大きい項を deterministic に扱う部分
- H_R: 残りの tail を randomized に扱う部分

です。

この研究で見たいのは、単に「部分ランダムPFが有効か」ではなく、

    部分ランダムPFの中で、
    決定論部分 H_D にどの PF を使うのがよいか

です。

そのため、比較対象は以下です。

- 決定論PFの種類 p
- 決定論部分の長さ L_D
- 誤差配分 q = eps_qpe / eps
- randomized 側の設計パラメータ kappa
- 必要に応じて DF rank / tol

現在の簡約モデルでは、全体コストを

    G_total = G_det + G_rand

として評価します。

決定論側は

    G_det ~ A_p(L_D) * (C_gs,D)^(1/p)
             / (eps_qpe * eps_det^(1/p))

randomized 側は

    G_rand ~ B(kappa) * lambda_R(L_D)^2 / eps_qpe^2

のように見ます。

ここで重要なのは、現在使っている C_gs,D は、full partial-randomized scheme 全体の厳密な誤差係数ではなく、

    決定論部分 H_D に対する surrogate な誤差係数

として扱うことです。

したがって、C_gs,D を full H に対する厳密な Trotter 誤差係数として扱わないでください。

## DF を使う場合の主方針

今後の主線は、Pauli 項に戻して H_D/H_R を分けるのではなく、

    DF Hamiltonian を DF representation のまま扱う

ことです。

つまり、DF ハミルトニアンを

    H_DF = sum_l H_l^DF

という DF fragment の線形和として扱い、その DF fragment を単位として部分ランダム化します。

この方針では、

- L_D は「deterministic に回す DF fragment 数」
- H_D は「上位 L_D 個の DF fragment からなる部分」
- H_R は「残りの DF fragment からなる tail」
- lambda_R は「H_R 側に残った DF fragment の重みの和」

として定義します。

Pauli-based 実装は baseline として残しますが、DF を使う本流では Pauli 項に戻して H_D/H_R を切るのではなく、DF fragment のまま H_D/H_R を切ります。

## C_gs,D の扱い

この研究で fit する C_gs,D は、

    DF H_D に対する deterministic surrogate

です。

これは、

    full H に対する partial-randomized PF 全体の厳密な誤差係数

ではありません。

したがって、研究上の主張は、

    DF representation 上で、
    決定論部分 H_D の PF 選択と L_D を最適化したとき、
    部分ランダムPFの簡約コストモデルでどの設計が有利かを見る

というものになります。

C_gs,D fit に使う t 点数は、計算量を抑えるため系サイズで変えます。

    H11 以下: 4 点
    H12 以上: 3 点

t_start は従来通り molecule size と PF order group ごとの設定値を使い、step は 0.002 とします。

## L_D の扱い

L_D は決め打ちではなく、コスト最小化によって選ぶ変数です。

DF representation を固定したら、各 L_D に対して、

    H_D(L_D) = 上位 L_D 個の DF fragment
    H_R(L_D) = 残りの DF fragment
    lambda_R(L_D) = H_R 側の重みの和

を計算します。

そのうえで、各候補 PF p について C_gs,D を fit し、G_det と G_rand を計算します。

## L_D screening 用の簡約モデル

全ての L_D に対して毎回 C_gs,D を GPU fit するのは重いため、まず cheap screening で有望な L_D 範囲を絞ります。

この screening 段階では、C_gs,D は L_D に依らない近似量として扱います。

具体的には、固定した DF representation の全 DF rank を R としたとき、

    L_anchor = floor(R / 2)

を代表点とし、

    C_gs,D^(screen)(p, L_D) = C_gs,D(p, L_anchor)

と置きます。

つまり screening では、L_D 依存性は主に

    A_p^DF(L_D) = total_ref_rz_depth(p, L_D)
    lambda_R(L_D)

から評価し、C_gs,D の L_D 依存性は一旦無視します。

この仮定は候補 L_D を絞るための近似であり、最終結果として報告するコストでは使いません。
最終評価では、shortlist された各 (p, L_D) について C_gs,D を改めて fit し、その値で G_total を再計算します。

C_gs,D の詳細な計算ログは raw artifact として残し、実行バッチ名ではなく molecule family, 系サイズ, PF ごとに整理した軽量テーブルをコスト最小化の入力に使います。
現在は H-chain 用に artifacts/partial_randomized_pf/df_cgs_cost_tables/h_chain/H{n}/{pf}.json を生成し、一覧は artifacts/partial_randomized_pf/df_cgs_cost_tables/index.json に置きます。
集約版として artifacts/partial_randomized_pf/df_cgs_cost_table.json も生成しますが、git で確認・レビューする主対象は split 版です。
各テーブルでは molecule_type, pf_label, L_D ごとに C_gs,D と total_ref_rz_depth を保存し、screening 用の anchor 点は is_screening_anchor=true として区別します。
shortlist 後に別の L_D で C_gs,D を fit した場合も、同じ H-chain/H{n}/{pf}.json へ explicit L_D の entry として追加します。

## kappa と B の扱い

B を外から自由に与えて最小化するのではなく、kappa を設計パラメータとして扱い、B は B(kappa) として決めます。

簡約モデルでは、

    B(kappa) = B0 * kappa * exp(2 / kappa)

とします。

主解析では、各 (p, L_D) に対して、

    q = eps_qpe / eps
    kappa

を同時に最適化します。

補助的に以下も確認します。

- kappa = 2 固定の reference 結果
- kappa sweep による感度解析

kappa が上限に張り付く場合は、randomized tail がほとんど消えて deterministic limit に近づいている可能性があるため、診断情報として扱います。

## DF rank / tol の扱い

DF rank や tol は、まず外側の representation choice として扱います。

基本の流れは、

1. rank / tol を決めて DF Hamiltonian を作る
2. その DF representation 上で H_D/H_R 分割を最適化する
3. その中で p, L_D, q, kappa を最適化する
4. 必要なら別の rank / tol でも同じ処理をする

です。

最初から rank / tol まで完全同時最適化する必要はありません。

DF 近似誤差は C_gs,D に吸収せず、別の表現誤差として扱う想定です。
ただし、第一段階では DF 近似誤差の厳密評価までは入れなくてよいです。

## 現在の進捗メモ

2026-04-26 時点で、DF-native な C_gs,D 計算と screening 用コスト最小化の実装は一通り入っています。

実装済みの主な処理は以下です。

- DF Hamiltonian を DF fragment rank 順に分割する処理
- DF H_D の C_gs,D を GPU statevector で fit する処理
- ground state cache と Cgs cache
- 各 t の GPU 並列実行
- D-only の total_ref_rz_depth を解析的に数える処理
- C_gs,D と total_ref_rz_depth を軽量テーブルへ export する処理
- anchor Cgs を使った簡約モデルの cost screening

主要な関数は src/trotterlib/df_screening_cost.py に切り出しています。

- load_df_anchor_cgs_table
- df_screening_costs_for_all_ld
- optimize_df_screening_cost
- save_df_screening_cost_result

CLI としては scripts/run_df_screening_cost_minimization.py から実行できます。
この CLI の epsilon_total の規定値は 1e-4 です。

現在の anchor Cgs テーブルは artifacts/partial_randomized_pf/df_cgs_cost_table.json です。
このテーブルには 35 entries が入っています。

- H3-H13: 2nd, 4th, 8th(Morales)
- H14: 2nd, 4th

H14 の 8th(Morales) は GPU ライブラリ解決エラーで未完了です。
4th(new_2) は別途 anchor Cgs を計算中で、途中結果は artifacts/partial_randomized_pf/H3_H14_df_screening_anchor_cgs_4th_new_2.partial.json に保存されています。
この partial には現在 H3-H12 までの 10 entries が入っています。

epsilon_total = 1e-4 で、現在の 35 entries を使った簡約モデル screening を実行済みです。

出力は以下です。

- artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json
- artifacts/partial_randomized_pf/logs/df_screening_cost_eps1e4.log

この run では 635 candidates を評価しました。
ここで candidate とは、1 つの (molecule, PF, L_D) の組です。
C_gs,D は anchor 値を使い回し、各 L_D では lambda_R(L_D) と total_ref_rz_depth(PF, L_D) を変えて q と kappa を最適化しています。
635 は C_gs,D fit の回数ではありません。

epsilon_total = 1e-4 の暫定 best は以下です。

    H3   8th(Morales)  L_D=3   anchor=2
    H4   8th(Morales)  L_D=4   anchor=3
    H5   8th(Morales)  L_D=5   anchor=4
    H6   8th(Morales)  L_D=6   anchor=5
    H7   8th(Morales)  L_D=7   anchor=6
    H8   8th(Morales)  L_D=8   anchor=7
    H9   8th(Morales)  L_D=9   anchor=8
    H10  8th(Morales)  L_D=10  anchor=12
    H11  8th(Morales)  L_D=11  anchor=10
    H12  8th(Morales)  L_D=12  anchor=14
    H13  8th(Morales)  L_D=13  anchor=12
    H14  4th           L_D=14  anchor=18

H14 は 8th(Morales) の anchor Cgs が未完了なので、H14 の best は暫定です。
また 4th(new_2) の anchor Cgs はまだ cost table に merge していないため、この screening には含まれていません。

次に行うべき作業は以下です。

1. 4th(new_2) の H3-H14 anchor Cgs 計算を完了する
2. 4th(new_2) の結果を df_cgs_cost_table.json と split table に export/merge する
3. 4th(new_2) を含めて epsilon_total = 1e-4 の screening を再実行する
4. H14 8th(Morales) の GPU 実行エラーを解消するか、必要なら CPU/別経路で anchor Cgs を取る
5. screening で選ばれた shortlist の (PF, L_D) について、anchor ではなくその L_D で C_gs,D を再 fit する

## 今は主題にしないこと

この研究方針では、以下は主題にしません。

- hardware resource estimation
- STAR / SMM などの物理資源見積もり
- UWC の本流実装
- full partial-randomized scheme 全体の厳密な誤差保証
- DF 近似誤差の厳密な全ケース評価

以上を前提に、研究の位置づけ、数式整理、実装方針、結果の読み方を整理してください。
