# アルゴリズム変更の二案と成立条件

2026-11-09。以下の式は静的な条件整理／提案で、採択された新しい成功保証ではない。[保存証拠・重複一覧](REPORT.md)、[次の限定計画](NEXT_VALIDATION_PLAN.md)。削減率を予測していない。

## 1. Q：前roundの区間を使う接線方向の読出し

### 変更と、既知法との関係

現在のwrapperは同じ時刻のcoherent mean `z_m` の実部・虚部を、別々のfresh trajectoryと±1測定で推定する。案Qは、前roundまでに得た**保証付きenergy区間**の中心から測定位相を決め、最後の高価なroundだけ一つの感度のある方向を測る。他round、H、PF、r/K、event law、ReferenceのF/Gは変更しない。最初の診断ではm17だけを候補にする。

これは[Higginsらの適応測定](https://arxiv.org/html/0709.2996v2)や[単一ancilla IPEA](https://arxiv.org/html/quant-ph/0610214v3)に属する既知のfeedback原理の応用である。新しいphase estimation原理とは呼ばない。通常RPEの「直前の推定に最も近いbranchを選ぶ」既存`reconstruct_rpe_phase`とも区別する。Qでは**実際に測定するobservable**が履歴によって変わり、旧X/Y用のshot十分式をそのまま使えない。

今回読んだO2のA1/SCVはpath情報による古典補正、A2は有料LCU差分、adaptive confidence単独は停止保証の共通部品だった。Qはどれも再実行しない。round別PF／α再配分とも操作が異なる。ただし全branchの未実施を証明したわけではなく、既存の適応PEに対する独立新規性も未確定である。

### 信号の意味と不足する保証

無重みcanonical lawに限り、物理ancilla meanを

\[
z_m=\rho_m e^{i(-t_mE_0+b_m)},\quad
\rho_m\in[\rho_{L,m},\rho_{U,m}],\quad |b_m|\le B_m,
\quad t_m=q_m h
\]

という**保存保証を含む許容集合**で扱う。ここで`b_m`はPF位相と残りのSOURCE／native／finite-bit等のdiskを一度だけ位相へ換算した合計。保存ρ下界は実信号の真の半径ではない。無重み±1の物理平均にはρU=1という安全な外側制約を使えるが、weighted Closed P3の補正平均へ同じρUや二項則を移さない。

前段の成功event上で`E0∈J=[Ec−w,Ec+w]`とする。`θc=−t_m Ec`、測定方向を`φ=θc+π/2`とすると、observable `Xφ=cosφ X+sinφ Y` の期待値は

\[
\mu_\phi=\operatorname{Re}(e^{-i\phi}z_m)
=\rho_m\sin[-t_m(E_0-E_c)+b_m+e_\phi],
\]

となる。`eφ`はfeedback位相の算術・生成・lowering誤差を包含する符号付き項。測定前のancilla P(−φ)→H→readoutでこのobservableを実現できるが、回路を今回は生成していない。既存YのSdg→Hとφ=π/2の対応、複素信号の符号は後日のcorrectness対象である。

必要な一つの条件は

\[
t_m w+B_m+\overline\delta_\phi<\pi/2.
\]

これは正の半径でsinの単調枝を使える条件で、**shot削減の十分条件ではない**。前段の点推定を真値と置かず、J全体を使う。未知E0、gap、state vector、真のradiusはpolicyへ与えない。φは過去の実outcomeだけから決め、その後は固定回数のfresh independent trialを取る。trajectory間でnoiseや測定outcomeを共有しない。

±1標本の区間`[l,u]`を得たとき、

\[
\mathcal J_m=\{E\in J:\exists\rho\in[\rho_L,\rho_U],\ |b|\le B_m,
\ |e_\phi|\le\overline\delta_\phi,
\ l\le\rho\sin[-t_m(E-E_c)+b+e_\phi]\le u\}
\]

を外向きに囲む。空集合、複数枝、幅不足を都合よく除外しない。返す値をhullの中点とするなら、成功event上でhull径≤2εEという別の確認が必要。旧RPEの.4 rad capに対応する、より強い径≤`2·.4/t_m`を最初の診断基準にする。

**大きな障害は未知半径。** 仮に統計誤差0・b=eφ=0でμ>0でも、可能な位相には`asin(μ/ρU)`から`asin(μ/ρL)`までの幅が残り得る（両比が[−1,1]にある部分）。これは局所精度を上げても消えない識別問題である。原論文の単位半径を代入して回避しない。PF bias／feedback誤差も床を追加する。今回その幅を保存数値で評価していない。

### 費用が下がる条件

旧m17費用は`NX μX,j + NY μY,j`。Qの固定回数をn、新readoutの価格をδcj、追加の前段費用をAjとすれば、必要条件は資源jごとに

\[
n(\mu_{\rm evolution,j}+\delta c_j)+A_j
 <N_X\mu_{X,j}+N_Y\mu_{Y,j}.
\]

nを軸ごとの旧Nと比較して「半減」とは言わない。旧両軸の合計と比較する。追加の学習／半径測定が必要なら、そのshotsと全interrogation価格をAjへ戻す。既存m0〜16を本当にそのまま使える最初の案だけAj=0の可能性がある。

任意feedback Pはnativeで1 RZ相当でも、strict Tで安いとは限らない。新しいangle domain・精度・word coverage・operator diskが生じる。RZ削減を総T削減とみなさず、将来は`Σ N E[T_full]`で判定する。追加qubit/CXを必要としない測定構成を目標にするが、compiler後の価格・depthやfeedforward latencyは未取得である。

### 強baseline、再利用、STOP

- B′／Referenceの両方にQと同じ固定phase選択・統計保証を与える。Referenceだけが古いHoeffding対照と戦う比較にしない。
- X/Y対照にも同じm17失敗予算、利用できる既知二項confidence、整数配分の機会を与える。単に統計不等式を鋭くした効果と、observable変更の効果を分ける。
- 最終比較では同じ元H、精度／failure／prep除外の強Full・Truncated PFにも同じ読出し機会を与える。PR内の利益とReference固有差とPFへの優越は別結果。
- 再利用：`rpe_four_round_phase_validation.reconstruct_rpe_phase`（旧branchの対照）、`pr_reference_interference_pilot.wrapper/reference_core`、`pr_reference_v6q_*`のcore・phase・law接続、`pr_reference_bprime_accounting.error_and_shots`のdisk単位、既存二項tail関数。旧consumerのaxisはX/Y限定なので、φをYラベルに偽装しない。新しい薄いwrapper／区間反転adapterが必要。
- STOP：ρの不確かさだけで目標幅を超える、単調枝が閉じない、同じ保証を与えたX/Yよりshot負担が下がらない、未知state情報・半径oracleが必要、追加feedback価格／校正費で利益が消える場合。別角度／別precision／追加radius測定を自動探索しない。

最初は回路不要の有限識別診断。成功しても新しいRPE保証を採用せず、GPT／ユーザーが理論・公正性を判断する。新科学再解析／confidence評価には別途実行許可が必要である。

## 2. I：最新のPR構成でReferenceの増分を測る

### 変更とSOURCE上の具体的接口

[旧横断整理§5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/OVERLAP_AND_NEXT_SCOPE.md)に既にある接続案を具体化する。独立した新機構として再発行しない。既存Track C／Dの担当・STOPはそのまま、新Aが接続条件を比較できる材料に限る。

Track C c5a3c102の最終policyにはClosed P3、joint basis catalogue、効率control／diagonal network、調整Dとh/rate/α選択が既に入っている。この強いcontrolled実装に対して、同じlogical trajectoryをuncontrolled F/Gへ分けるReferenceを比較する。lawと回路基盤の改善をReference追加効果から分ける。

静的に確認できた具体点：

1. `track_c_closed_p3.application_events`はphase、signed child angle、productの適用順、coefficient＝proposal×weightを保持する。`flattened_runs`が同basis runを決める。
2. `track_c_joint_isometry.operations/decode`と`track_c_bottleneck_basis.partial_ops/circuit`はRZとXXPlusYYによるpartial Gaussian completionを使い、inverseは実gate列の逆順inverse。これは真空rayを調べる根拠にはなるが、別completionへ旧vacuum位相値を移せない。
3. `track_c_diagonal_fusion.emit`は**既に制御された**`diag(I,word)`を出す。networkはsystem maskとancilla付きmaskに±angle/2を配置し、productsもcontrolled Zを使う。native回路からancilla線だけ削除すると、元のuncontrolled word／scalarにはならない。
4. 必要差分は、同じlogical rotations/productsからuncontrolled runを出し、旧runのbasis／mapping／exact rational合算を再利用する限定emitterと、同じD recipeをF/Gへ接続するadapter。SE gadget／Gaussian合成器／ReturnKernelを作り直さない。
5. 新Aの`assemble_reference_for_saved_round`はh=.02/K2／旧r/bits／旧cacheを固定する。h=.32やClosed weightを通すためにこの関数の条件を緩めない。新しい接続点は別の限定adapterとし、旧consumerを保全する。

### 信号・価格の因果経路

pathωごとのReference coherenceがstrong controlled回路の`〈ψ|Vω|ψ〉`に、既知のvacuum補正後に等しくなることが先の条件。その後に同じπ(ω)w(ω)で平均する。channel一致、unitaryのglobal phase一致だけでは不十分。両armでweight積、range、finite-bit/root/angle誤差、actual inverse、cutの境界融合損失を払う。

強controlledの単価から、制御解除分をSj、追加Reference／cut／補正分をHjとすれば`μR,j=μC,j−Sj+Hj`は会計上の記号である。Sj,Hjを未取得価格で埋めない。shotも異なると

\[
G_{R,j}-G_{C,j}
=\sum N_C(-S_j+H_j)+\sum(N_R-N_C)\mu_{R,j}.
\]

D0と同じく二項とも必要。Closedのrange低下をReferenceだけへ与えず、joint／fusionの率とV6-Pの率を足さない。Reference固有の差は、同じ最新law・同じbasis・同じD選択を固定したcontrolled対照から測る。

### 公平性と停止

最小ablationはcanonical／Closed×strong controlled／Reference。phase/weight条件と同一compilerをそろえて、同じSOURCE familyの固定fixtureを比較する。Full／Truncatedのcontrol最適化も保持する。短fixtureの資源差は全law期待値や18round資格ではない。

既存強controlですでに消えたgate、joint basisの節約、D融合、Track Dのhead-tail融合／cut selectorを新利益に数えない。IはTrack D C1を実装する指示ではない。価格によるpath/cutの事後選別をしない。

STOPは、vacuum ray／scalar／ordered wordが一致しない、同じ強control改善後に追加差がない、Referenceのcut損失／CSWAPが節約を消す、weighted disk/rangeでNが増えて総差が消える場合。SOURCE適合不足ならUNQUALIFIED。別law／分割／PF／cutへ救済探索しない。新しいscientific build/compile・correctness・価格・RPE再会計は別許可が必要。

## 3. 比較・優先度と総Tへの出口

| 評価軸 | Q：readout変更 | I：最新回路との接続 |
|---|---|---|
| 大きな費用項への作用 | 長roundのD・basis等を含む回路反復数 | RTE／Dの制御・rotation単価。ただしbasis共通費は残る |
| 既知性／独立性 | 既知適応PEの応用。確認資料のlaw/CV／α最適化とは操作差がある | 組合せは既提案。新規原理とは扱わず増分だけ問う |
| 最初の実行費 | 保存ρ・位相域を使う有限区間診断、量子回路不要 | 保存logical IRとのbinding後、限定wrapperが必要 |
| 主な科学的危険 | 未知radius、prefix失敗、feedback誤差、単一方向の識別不能 | weighted coherent mean、vacuum phase、actual inverse、cut損失、新context |
| 再利用性 | 現PF・finite-bit producer・F/G不変。統計後処理とreadoutが差分 | Cの完成pipeline／lawと新Aのgadget。旧callerの制約はそのまま |
| Tへの注意 | 新feedback角のstrict価格・誤差を払うと利益が消え得る | 最新native fusionの角度とword policyは旧CT辞書で未資格 |

Qを先に診断し、Iは独立した比較案として残す。どちらにも数値的優位の裏付けはない。強Full／Truncatedを上回る独立新規機構が得られた、という結論は現時点では出せない。

将来の採択順は「同じenergy成功条件 → shotsと全wrapper native資源 → 必要なstrict CT資格／総T」の因果関係による。native段階の否定結果を新規合成で救済せず、既存CT資産は適合するkey/contextだけ再利用する。任意角数×一定T、有限m7のT比×全round、異条件の率加算は行わない。
