# N5-B2：m17のlogical word policy要件

2026-11-08 JST。[REPORT](REPORT.md)。現policyの静的な適用範囲を整理し、新b／precision／辞書／providerを採択していない。保存条件の正本は[N5-B0 M17_INPUTS](../../../artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json)。

## 1. 現在の固定policy

| 処理 | 固定SOURCEの動作 | m17に対する状態 |
|---|---|---|
| 入力 | 有限のactual binary64 float。abs(theta).hexをpositive keyに使う | Signed literalを保つ。近接key統合／π snap／2π同一視なし |
| Domain | **C0 lookupより前**にabs(theta)<=Lを検査 | 固定scalarは拒否。旧recordが別poolにあってもinterfaceは変わらない |
| C0優先 | 固定299 record。Guard＋保存64uで既存10^-11へ資格 | Wordは再利用候補。M17のlogical targetへの保証と供給域は別 |
| C0 miss | b36、as_integer_ratioと整数演算による最近接ties-to-even | θとn2^-36のquantizationを残す。今回nを計算していない |
| Basic | j=-36…6の43 word、各保存guard、ascending jのwritten product | 保存wordは取得済み。全選択chunk和・m17 precisionは未資格 |
| Negative | Positive complete word全体のactual adjoint | Emission側でinverseを一度。Signed wordを先に渡さない |
| 停止 | Domain内で固定有限bit走査／文字列連結、backendなし | 有限にrejectすることと、全必要角へ適格wordを返すことは違う |
| Certificate | C0はguard＋emitter、missはquantization＋basic guard和＋emitter | 既存strict値は維持。Gによる新数値会計へ自動置換しない |

このSOURCEを新しいdomain／bへ単にparameter変更することはできない。Constructorもb、指数域、内部割当、最終10^-11を固定している。無断の定数変更、C0優先の解除、より安い／精密な別recordへの差替えはしない。

## 2. 既存証拠の再利用

- **C0299**：保存文字列、exact W、guard、hash、negative actual adjointを再利用できる。保存11件のlogical guardだけはm17両armY cap内だが、library件数であって全必要位置の資格ではない。
- **43 basic**：取得済みwordと保存outward guardを再利用できる。N5-B0照合はbasic guard単独でB′0件／Reference1件がY cap内。要求ラベル10^-13だけで全部を一律に排除／採用せず、actual guardと選択和を確認する。
- **85 certificate**：84 frame＋B′ b2の保存quantization／word成分、文字列と正負対応を再利用できる。旧最終10^-11の資格はm17全roleを覆わない。
- **Emitter／consumer**：同じ文字列、named gate、whole inverse、inplace経路を再利用候補にできる。N5-B1の技術証拠は再実行せず引用する。

現証拠が足りないことは、実word誤差が大きいという下界ではない。新guard評価なら必ず改善する、又は新規合成なら必ず解決するとも判断しない。

## 3. Logical・emitter・shellの型

C0についてlogical上界をg_C0、missについて量子化上界Q(theta)、選択basic集合J(theta)と保存logical guard g_jを記号で表す。

`e_log,C0 <= g_C0`,

`e_log,miss <= Q(theta) + sum_(j in J(theta)) g_j`。

後者は同じ固定written productのunitary telescopingに対するlogical上界である。Exact Wを含む各basicのstrict意味を保ち、projective phase fitへ変更しない。負側のactual adjointは同じnorm上界を持つ。

既存emitted certificateでは上記へemitter項を加え、whole側で別shellを加える。候補Gを適用するなら、同じlogical productからの**pure common scalar**だけを信号側へ分離する可能性がある。Logical e_logが0になるわけでも、既存strict値が減額されたわけでもない。

本番接続の候補式は、適格な全位置iのlogical上界を使って

`epsilon_log,joint <= sum_i e_log,i`、

`epsilon_pair >= max(epsilon_log,joint,X, epsilon_log,joint,Y)`、

`d_syn <= 2sqrt(2)*epsilon_pair`、`d_total = d_old + d_syn`

とする方法である。これはGの適用条件と誤差規約を採用した**後**の会計候補で、今回の採択／数値実行ではない。Gで処理できない残項があれば別に保持する。Compiler／SOURCE／vacuum／finite-bitの旧diskは同じraw単位で一度だけ加え、PF位相予算の控除やX/Y maxを重複／省略しない。

## 4. 保存m17十分条件の位置づけ

M17はq131072／r32／88 bits、B′626／Reference630 shots/axis。保存Y gate capはB′6.902168833913054e-16、Reference2.5648979454057876e-15。これは旧一様worst-case会計の**保存Nを保つ十分割当**であり、最適precisionや必要条件ではない。X cap／RZ数／d_old等は[SAVED_METADATA](../../../artifacts/pr_reference_n5b2_static/2026-11-08/SAVED_METADATA.json)へ引用した。

保存b36 quantization上界は約5.1448789686e-12、辞書emitterは45/140737488355328、C0 emitterは1/140737488355328というN5-B0の証拠を保持する。今回新しい和や比較値を計算していない。Uniform boundがcapに入らないことから、実m17 finite targetでの不可能性を断定しない。

量子化はrelative角の差であり、Gでは消えない。同じgrid cellの非C0入力へ同じwordを返す仕様の分解能はbasic guardだけを改善しても変わらない。ただし実targetのcell内位置をまだ評価していないので、m17の新しい下界を主張しない。

## 5. 完成policyへ必要な条件

現policyはB′ scalarの実literalでdomain rejectするため、**そのままのm17全support実行policyにはならない**。これはRPEの科学的失敗／研究no-go認定ではなく、固定interfaceの適用限界である。

新policyを選ぶなら、先に全必要角の保証域、actual literalの扱い、C0の再利用／優先、quantization、基本guard、固定word順序、complete adjoint、有限停止、候補Gの前提、同じpolicyのpaired arm適用を固定する必要がある。必要なbや内部精度の選択規則を未決のまま新規合成へ進まない。価格比較時には変更policyで旧anchorを無条件流用しない。

**新規合成が必要かは未確定**。選択した仕様の有限key／precisionについて保存wordで保証できないことを特定してから判断する。最初の有限診断候補は[NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)に固定し、今回は実行せずmandatory STOP。
