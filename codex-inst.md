あなたは、このプロジェクトの実装を進める Codex です。
以下の方針で、DF-native な GPU 処理を実装してください。
API の細かい設計は任せますが、研究方針と実装の向きは崩さないでください。

# 実装目的

今回やりたいのは、DF Hamiltonian を使った H_D に対して、C_gs,D の fit を高速化することです。

具体的には、

    各 (p, L_D, t_values) に対して、
    DF representation 上の H_D を Trotter 実行し、
    perturbation error を計算し、
    C_gs,D を fit する

処理を GPU statevector 実行に置き換えたいです。

既存の参照コードには、DF Trotter 専用の block/circuit builder と GPU runner がすでにあります。
そのため、今回移植すべきなのは、

    参照コードの DF-native GPU runner 部分

です。

Pauli terms を Qiskit circuit に変換する GPU path を主実装として作るのではなく、
DF block / DF circuit builder を使う path を優先してください。

# 実装方針

## 1. DF-native な C_gs fit path を作る

現在の Pauli-based C_gs fit とは別に、DF Hamiltonian 用の C_gs fit path を用意してください。

この path では、

- DF representation を入力にする
- DF fragment 単位で H_D を構成する
- DF 用 circuit builder で Trotter 回路を作る
- GPU statevector runner で t_values ごとの実行を行う
- CPU path と同じ形式の perturbation error を返す
- 最終的に C_gs,D を fit する

という流れにしてください。

## 2. Pauli baseline は残すが、主線にしない

既存の Pauli-based partial randomized PF 実装は、baseline として残してください。

ただし、DF を使う本流実装では、

    Pauli 項に戻して H_D/H_R を切る

のではなく、

    DF fragment のまま H_D/H_R を切る

ようにしてください。

Pauli path は以下の用途に限定してください。

- 既存結果との比較
- sanity check
- 小さい系での baseline
- 従来実装の互換性維持

## 3. DF fragment の重み定義を一貫させる

H_D/H_R を分割するには、DF fragment ごとの重みが必要です。

最初は、各 DF fragment の大きさを表す自然なスカラー量を定義し、それを使って降順に並べてください。

重要なのは、どの重みを使う場合でも、

- 全 PF
- 全 L_D
- 全 molecule
- 全 rank / tol

で同じルールを使うことです。

重み定義は後で差し替えられるようにしてください。

## 4. L_D の最適化は DF fragment 上で行う

DF representation を固定したら、各 L_D に対して

    H_D(L_D) = 上位 L_D 個の DF fragment
    H_R(L_D) = 残りの DF fragment
    lambda_R(L_D) = H_R 側の重みの和

を計算してください。

そのうえで、各候補 PF p について C_gs,D を fit し、G_det と G_rand を計算します。

L_D は決め打ちではなく、コスト最小化によって選ぶ変数です。

## 5. kappa と B の扱いを入れる

B を外から自由に与えて最小化するのではなく、kappa を設計パラメータとして扱い、B は B(kappa) として決めてください。

簡約モデルでは、

    B(kappa) = B0 * kappa * exp(2 / kappa)

とします。

主解析では、各 (p, L_D) に対して、

    q = eps_qpe / eps
    kappa

を同時に最適化します。

また、補助的に以下も残してください。

- kappa = 2 固定の reference 結果
- kappa sweep による感度解析

kappa が上限に張り付く場合は、randomized tail がほとんど消えて deterministic limit に近づいている可能性があるため、診断情報として出力してください。

## 6. DF rank / tol は外側ループとして扱う

DF rank や tol は、まず外側の representation choice として扱ってください。

基本の流れは、

1. rank / tol を決めて DF Hamiltonian を作る
2. その DF representation 上で H_D/H_R 分割を最適化する
3. その中で p, L_D, q, kappa を最適化する
4. 必要なら別の rank / tol でも同じ処理をする

です。

最初から rank / tol まで完全同時最適化する必要はありません。

DF 近似誤差は C_gs,D に吸収せず、別の表現誤差として扱う想定です。
ただし今回の GPU 実装では、DF 近似誤差の厳密評価まで入れなくてよいです。

# キャッシュと再計算防止

GPU 実行は重いので、C_gs fit の cache key には、少なくとも以下の情報を含めてください。

- representation_type = "df"
- molecule / geometry / basis など対象系の識別情報
- DF rank または tol
- PF order / PF name
- L_D
- t_values または t window
- evolution_backend = "gpu"
- GPU 実行設定
- chunk_splits
- circuit builder の設定
- 重み定義

Pauli path と DF path の cache が混ざらないようにしてください。

# 出力で必ず分かるようにすること

各候補について、最低限以下が確認できるようにしてください。

- representation_type
- DF rank / tol
- PF name / order
- L_D
- lambda_R
- C_gs,D
- A_p
- q_opt
- eps_qpe
- eps_det
- kappa_opt
- B(kappa)
- G_det
- G_rand
- G_total
- kappa が境界に張り付いたか
- C_gs,D が DF H_D に対する surrogate であること

# 今回やらないこと

今回の GPU 実装では、以下はやらなくてよいです。

- hardware resource estimation
- STAR / SMM などの物理資源見積もり
- UWC の本流実装
- full partial-randomized scheme 全体の厳密な誤差保証
- DF 近似誤差の厳密な全ケース評価
- Pauli terms 用 GPU path の主実装化

# 最終的に欲しい実装の状態

研究方針に沿った最終的な構造は以下です。

- Pauli-based partial randomized PF:
  baseline として残す

- DF-native partial randomized PF:
  主線として実装する

- 共通最適化部分:
  q, kappa, L_D, PF 比較のロジックは共通化する

- 表現依存部分:
  Pauli と DF で分ける

- GPU 処理:
  DF-native C_gs,D fit を高速化するために使う

# 重要な解釈

この実装で得られる C_gs,D は、

    DF H_D に対する deterministic surrogate

です。

これは full H に対する partial-randomized PF 全体の厳密な誤差係数ではありません。

したがって、コードコメント、README、出力名ではこの点が誤解されないようにしてください。

このプロジェクトの主張は、

    DF representation 上で、
    決定論部分 H_D の PF 選択と L_D を最適化したとき、
    部分ランダムPFの簡約コストモデルでどの設計が有利かを見る

ことです。

実装もこの研究目的に合わせてください。