# PR-DF LD1：全13 round期待価格の構成と次の有限作業

2026-11-15 JST。保存成果・SOURCE・metadataの読取と計画作成のみ。開始HEADは `dcf905fc5c0abb05966019bcae8b0206af9aa952`、既存D2 branchを継続した。

- **保存証拠で評価できる範囲**：Q3固定配分、保存ideal-source構造期待値、有限S/C、凍結残差モデル、F0の条件付き残差差上界を26軸で結合できる。**対象lawの期待残差の値又は有用な推定区間がなく、期待compiled価格の点値・精密な区間は全26軸で未確定**。全supportの粗い価格envelopeは保存済みだが、期待値を特定するものではない。
- **主な不足**：law間の差上界とは別に、基準lawでの `D=E[C−S]` が必要。有限cohort/係数の誤差、未観測roundへの移送誤差、運用seed ensembleを主張する場合のjoint-law接続が残る。F0上界をDへ代入してはいけない。
- **推奨する次の1作業**：**凍結モデルの26軸参照価格と、PF総額593,555,352 RZを跨がないために必要な「未評価残差平均の誤差予算」の有限評価**。新path/compileは0、再fitなし。残差誤差を0とは置かず、参照値・既知law上界・未評価項を分ける。これは正式E[C]/Gの計算ではなく、後続取得に必要な精度を総費用差から設計するための作業である。今回は数値評価しない。

[26軸metadata join](AXIS_JOIN.json)／[次の有限案](NEXT_FINITE_PLAN.json)／[統合担当HANDOFF](HANDOFF.md)／[静的照合](CHECKS.json)。既存F0・上界表・価格を再集計/再作成していない。新科学計算、予測、path生成、H4 build/compile、CI、正式E[C]/Gは0。

## 1. 固定配分・物理候補・価格範囲

対象は保存H4直線1.00 Å/STO-3G/tol-only DF10、PR-DF LD1/S2/Δ=.1/K2/元exact κ、元H/state/snapshot・fragment/basis順である。価格Cは状態準備費P=0の測定付きfull controlled Hadamard wrapper、scalar c+s_Rを総Tで一回、symmetric-directional control、新ZZ `CONTROLLED_ZZ_CX_CRZ_CX_V1`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。構造Sは元Accountingの構造量を維持する。

Q3正本 `a82b0f0e…:docs/research/pr_df_fullh_q3_fixed_results_20261115/C_RESULT_JOIN.json#candidates[0]` は `PR_DF_LD1_TOTAL_Q3_SAVED_CONTROL_01`、shot policyはUSER_SUPPLIED、状態は `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。各軸の配分は **[64,60,56,52,47,43,38,34,30,25,21,17,12]**。十分N・最適Nの認定ではない。

| m | N_X=N_Y | q / R | 元physical ID | 各軸の主な有限価格の役割 |
|---:|---:|---:|---|---|
| 0 | 64 | 1 / 2 | 53e00d8b07d1547d19a7 | P2 fresh4：較正。P1/replay：参照 |
| 1 | 60 | 2 / 4 | c00e00dc1e9c414fb7d9 | 該当cohortなし |
| 2 | 56 | 4 / 8 | f2fcd66b1ef9ae2da5df | 該当cohortなし |
| 3 | 52 | 8 / 16 | 045843b2429bece6f29b | fresh4：較正 |
| 4 | 47 | 16 / 32 | 3df0cfd3356c358aeda8 | fresh4：独立holdout |
| 5 | 43 | 32 / 64 | 57d641a831d76360b570 | 該当cohortなし |
| 6 | 38 | 64 / 128 | 2a8c6718220eec59d419 | P2 fresh4：較正。P1/replay：参照 |
| 7 | 34 | 128 / 255 | b42287268e5249dfb4d1 | 長round、該当cohortなし |
| 8 | 30 | 256 / 510 | 224aef0a626068b43184 | 同上 |
| 9 | 25 | 512 / 1020 | 06a78c7f9856f373336f | 同上 |
| 10 | 21 | 1024 / 2040 | b872ad1347a35c8f4315 | 同上 |
| 11 | 17 | 2048 / 4080 | d48210cc27c599e268c2 | 同上 |
| 12 | 12 | 4096 / 8160 | b55c521fdce6a448dbcb | 同上 |

AXIS_JOINはX/Yを別々の26行で保持し、旧26軸表 `950577f4…:…/AXIS_EVIDENCE.json` を再利用する。各行の `Q3_row → prior_axis → finite_evidence`、`structural`、`F0_C_row`、`residual_law_difference_bound` から固定原本へ戻れる。S/C値やTVを新表へ再計算していない。

元configuration `a4dda8f7…`、total decoder view `9c3d392e…`、旧価格configuration `587d20f7…` は別欄のまま。同一physical/m/axisと元canonical/total candidateの対応を照合した。旧receiptのIDや実行SOURCEをconsumer commitへ付け替えていない。量子shot Nと古典path数nを区別し、X/Yは同一pathの別応答とする。価格にnormalization B、q/R又はnを再乗算しない。

比較先は `1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/HANDOFF.md` の **B0I p7 S2（16h/7）、593,555,352 RZ** の転記。PF自身の評価済み配分と採択済みcompiler閉包に条件付く。PFの配分をPRへ流用せず、PFの条件付き最小を5方式の正式winnerとは扱わない。今回差額や順位は計算していない。

## 2. lawと量を対応させる

`f=C−S`。F0内のP/Q/F/Gと、元normal samplerのQ、総資源Gを混同しないため、ここでは次の記号を使う。

| 記号 | 確率測度と既存SOURCE | 得られたもの／未評価 |
|---|---|---|
| I | 保存SOURCEの理想finite-RTE law。BのQ3 targetは `ideal_finite_RTE_law`、CのSも同宣言。F0 reference Pは保存binary64 raw weightの正確な比を正規化したlaw | 保存E_I[S]の数値enclosure。元の因数分解/IEEE伝播誤差は保存数値契約の範囲。解析的な実数重みから保存値への誤差とは別 |
| J | 同じCDFへ独立一様53-bit入力を入れる条件付きpath law（F0のgrid G）。order条件付きcomponent draw、outerを含む全位置の独立性又は履歴条件付き距離の根拠を仮定 | I↔Jの条件付きpath TVと期待残差差の上界。E_J[S]、E_J[f]、E_J[C]の値は未取得 |
| Q_η | seedに確率測度ηを定め、元PCG64 normal samplerとseed派生規則を適用したpath law | ηは未採択。固定cohortの軌道/receiptは保存済みだが、E_Qη[f]やJとのjoint距離は未評価 |
| A | 解析的な実数のτ・係数から定義するlaw | A→保存SOURCE Iの誤差は今回のF0/上界に含まない。これを比較対象に追加するなら別の入力誤差根拠が要る |

固定根拠は `law_math`（d72dc4e6のMATH.md §1/5）、B `Q3_result#record.deterministic_input.statistical_contract.simulation_policy`、C `E_I_S#values[i].law_binding/numerical_enclosure`、F0受入259a0b2d、上界dcf905fc。完全commit/pathはAXIS_JOIN.source_refsにある。同名lawのmetadata一致だけを、解析的実数lawや実PCG64との一致の証明としない。

### 期待値の恒等式と、差上界が制限する項

同一law L、同一path関数なら `E_L[C]=E_L[S]+D_L`、`D_L=E_L[f]`。保存source Iの対象では、

```text
E_I[C] = s_I + D_I
       = s_I + D_J + Δ_IJ,       Δ_IJ = D_I−D_J, |Δ_IJ| ≤ b_IJ.
```

`s_I=E_I[S]` は保存区間がある。`b_IJ` はdcf905fcの26軸の `conditional_RZ_shift_upper`。その数値表を再作成しない。必要なのは**D_J又はD_Iの基準値/区間**であり、`D_J=b_IJ`、`D_I=b_IJ`、`D_J=0`のいずれも導けない。

もしD_Jの有効な区間[d_lo,d_hi]が得られれば、

```text
E_I[C] ∈ [s_I_lo+d_lo−b_IJ, s_I_hi+d_hi+b_IJ] ∩ [0,U_C].
```

現状d_lo/d_hiの有用な期待値評価がないため、全supportの粗い `−U_S≤f≤U_C` と `0≤C≤U_C` を超える価格精度は認定していない。保存上界の小ささは、残差期待値を推定できたことを意味しない。

J自体を価格対象にするなら `E_J[C]=E_J[S]+D_J`。保存s_Iを使う表記では `E_J[C]=s_I+D_J+(E_J[S]−s_I)` となり、構造lawの差項も必要である。dcf905fcの残差差上界を、この構造差項へそのまま代入しない。Iに対する上式なら構造期待値をJへ変更する必要はない。

### 保存モデルを置くと、何が不足しているか

凍結モデルの参照関数を `dhat_a(m)=θ0,a+θq,a q_m+θL,a q_m R_m` とする。係数の根拠は元PCG64/CDF有限cohortで、J又はIの母平均として凍結されたものではない。

```text
e_J = D_J−dhat
E_I[C] = s_I+dhat + e_J + Δ_IJ.
```

`s_I+dhat` は構成可能な**モデル参照値**。`e_J` は未評価であり、有限cohort誤差、Jと取得lawの差、関数形/長round移送の不足を含む。これらが重なるなら二重に加算しない。e_J=0という全26軸モデル仮説を別途採択した場合だけ、s_I+dhat±b_IJ（保存構造数値幅を別に含む）を条件付き価格として使える。それは既存holdoutから証明された事実でも、今回の採択でもない。

運用lawの分解を明示するなら、

```text
e_J = (D_Qη−dhat) + (D_J−D_Qη).
E_I[C] = s_I+dhat + (D_Qη−dhat) + (D_J−D_Qη) + Δ_IJ.
```

最新上界が制限するのは最後のΔ_IJだけである。有限pathのfitは最初の括弧の母平均誤差を認定せず、PCG64 SOURCE/seed再現性は二番目の括弧を0にしない。Q_ηを対象にした `E_Qη[C]` には `E_Qη[S]` 又は `E_Qη[S]−s_I` も必要である。同じQ_ηからCを直接評価すれば構造lawの置換項は不要になるが、推定誤差とQ3のlaw変更は別に残る。

## 3. 実運用seed ensembleに移すための具体的な不足

既存SOURCEで確認できるのは、carrier155529e2の `trial_seed` がcohort・physical ID・index・outer/piece等をhashして先頭16hexからseedを決定し、provider c31ef424の `iter_sample_rte_events` が `Generator(PCG64(seed))` でorderをchoiceし、続いてorder+1個のcomponentをchoiceする経路である。orderによって乱数消費数も変わる。

不足するものは、(1)ηが一様seed、random index、固定有限index/cohortへの一様測度等のどれか、その重みと再利用/再始動規則、(2)その規則が作る同一stream内・outer間のjoint law、(3)Jとの距離/coupling又は一致履歴に一様な条件付き距離の根拠。period、hash、stream非重複、NumPy再現性、周辺CDFが同じことだけでは独立性は成立しない。固定seedの一列は決定的で、ηの定義なしに母平均を主張しない。

価格の期待積和は、各軸の母平均が定まれば線形性で構成でき、X/Y独立性を要求しない。一方、Q3のshot/axis/round IIDモデルを実運用へ移すには、信号・decoder側の確率契約との整合も要る。価格だけをQ_ηへ変えてもQ3適合が自動的に移るわけではない。

| 今後の対象lawの選択 | 比較の意味・必要な判断 |
|---|---|
| **保存Iを維持（推奨、既存契約）** | Q3と同じ保存ideal-lawの条件付き比較。Iの再採択は不要。期待残差の根拠又はモデル仮説/誤差評価が要る。actual PCG64の保証を先行必須条件にしない |
| Jを直接対象とする | 丸めCDF＋IID格子の期待価格。E_J[S]/D_J又は直接E_J[C]と、BのI-Q3からJへの信号/精度接続が必要。価格側だけで変更しない |
| 運用Q_ηを対象とする | 明示したη・運用規則に限った期待費用。ηの採択、joint-law/信号/精度の根拠が必要。有限cohort4本への一様測度を選べばその集合の平均に限定され、全lawの保証や現Q3の証明にはならない |

今回ユーザー判断を必須とするlawの変更はない。運用費用へ主張を拡張する場合に初めてηと接続水準を選ぶ。必要価格精度・信頼水準・許容誤差は未採択。PFの2%、他方式の基準、便宜的なpath数を流用しない。

## 4. m0/m3/m4/m6の有限証拠が使える項

| 保存証拠 | 使えるもの | 埋まらないもの |
|---|---|---|
| m0/m6 P2 fresh各4 path、m3 fresh4 pathの同path S/C | 指定Q cohortの残差、3点の軸別モデル係数。Cの凍結正本の入力/残差/セル平均へ戻れる | D_I/D_Jの値、母平均推定誤差、全support範囲、長round関数形 |
| P1各2 path、P2 replay各1 path（m0/m6） | 元lowering/再現/開発参照 | 新たな独立較正標本としてpoolしない |
| m4 fresh4 path、C最終受入9edfa5d3 | 観測前予測と未使用m4有限Q cohort平均の記述的差。両軸約−0.544502%は保存報告の引用 | law差、標本のばらつき、モデル誤差を分離しない。CI・母平均誤差・m7〜m12の移送上界ではない |
| 凍結3cac53daの2軸モデル | `[1,q,qR]` の参照関数。較正3点/rank3/係数3/lack_of_fit_DOF=0、m4の2予測を保存 | 較正算術残差0を予測保証としない。未観測roundの価格はまだ評価していない |

m4は独立holdoutのまま保つ。今回も次案もm4をfitへ戻さず、係数・予測を観測後変更しない。X/Yの一致は独立8 pathの証拠ではない。full-wrapper statevector=false、元P2/m3/m4で未実施の作用検証を新PASSに変更しない。

## 5. 次の有限作業1件：参照価格と必要誤差予算

総費用の目的量は `T_I=Σ_(m,a) N_(m,a) E_I[C_(m,a)]`。比較先F=593,555,352とする。これを今回計算していない。

新compileより先に、保存された凍結モデルを参照点にして「どの程度の未評価残差平均誤差なら比較が変わらないか」を求める。**モデルが真との採択を先に要求しない**。未知誤差を明示した感度評価なら、PCG64証明の成否を待たず、後続取得の精度を費用目的に沿って決められる。現状の情報だけで長roundの重要度やnを推測して取得するより、直接の不足を数値化できる。

```text
p_ref = s_I_saved+dhat                 （母平均の認定ではない）
T_ref = Σ N p_ref
K_IJ = Σ N b_IJ                       （保存上界を再利用）
v_lo/hi = Σ N(s_I_lo/hi−s_I_saved)     （保存数値区間の伝播）
E_res = Σ N(D_J−dhat)                 （未評価、nullを維持）
T_I = T_ref + E_res + ξ_S + Δ,
ξ_S ∈ [v_lo,v_hi], |Δ|≤K_IJ.

M = F−T_ref
PR側が低いことの十分条件：upper(E_res) < M−v_hi−K_IJ
PR側が高いことの十分条件：lower(E_res) > M−v_lo+K_IJ.
```

右辺の境界は、仮の2%・95%・path数を置かずに計算できる。Nは非負なのでKの積和は三角不等式で成立し、軸/round独立性を要求しない。共通係数の誤差を独立と仮定した二乗和やCIへ置き換えない。E_resの上下限が未取得なら総額比較の判定はUNRESOLVEDのまま。境界の符号からモデルの正否や実際の優劣を断定しない。保存数値区間を超える解析実数lawの変更は含めない。

| 項目 | 有限scope |
|---|---|
| 対象law・量 | 保存Iの期待費用を将来評価するためのモデル参照値と未評価E_resの比較境界。Jは既存F0の条件付き補助law |
| round・軸 | m0〜m12×X/Y、26軸。Nは§1を固定 |
| 再利用入力 | 保存E_I[S]/数値区間26、凍結2モデル、較正残差平均6＋保存m4残差予測2、m4 compiled参照予測2、dcf905fcの上界26、PFの固定総額。m4観測は検証記録としてだけ参照 |
| 新しい数値評価上限 | 残差特徴量評価18件。残る8値は保存済みから再利用。full-price参照は新24＋保存m4の2＝26。加えて重み付き参照/法則差予算/数値区間/比較境界の小さいexact集計1組 |
| 新path／実compile | **0／0**。sampling・H4 build・F0再実行・fit・CI・Q3・正式E[C]/Gは0 |
| 実装経路 | d83737b6の `price_model_validation.predict` と既存Git reader/Exact Fractionを再利用。freeze callerのcalculate/mainは再fitするため呼ばない。tolerance必須のvalidate APIも呼ばない。新runnerは読取＋既存predict＋exact積和だけ |
| 出力 | `docs/research/d_pr_df_ld1_reference_budget_20261115/run01`。現時点未使用。参照価格と誤差境界を保存、E_res/CI/正式E[C]/G/winnerはnull |
| 得られる結論 | 参照modelとPFの差を埋めるために、どの符号・大きさの未評価残差平均誤差が問題になるか。次のlaw整合した残差推定又はモデル検証の精度設計。全26軸の母平均・順位の認定は得ない |
| STOP | 入力/axis/physical/law/scalar/lowering/compiler/SOURCE不一致、欠測、重複未解決、保存m4参照値不一致、非有限/負の参照価格、保存失敗で該当処理停止・部分結果保全。補正/clamp・仮tolerance・再fit・追加path・retryなし。公開/remote照合で停止 |

実行前にreader/入力/使用SOURCEをcommit固定し、§10.5は今回からの差分だけを照合する。小さいexact算術と読取cacheを用い、巨大event列・QPYを作らない。資源の見積りや性能倍率を発明せず、後続実行で読取/評価/保存のwall・RSS・終了codeを記録する。

**採択が別途必要なこと**：この有限数値再解析の実行許可。さらに、将来E_res=0等を採用して条件付きモデル総額を比較したい場合は、その全26軸モデル仮説の採択が必要。今回も次案も未知誤差を勝手に0へ補完しない。参照値と境界の計算自体はその仮説を必要としない。

追加取得を直ちに推奨しない理由は、対象Iは定まっている一方、母平均の精度要求が未定で、今少数のnormal pathを増やしてもD_J/Iの資格や長roundの誤差は定まらないため。正式E[C]の推定へ進む際は、law整合した推定器、精度/信頼性、必要nの設計を境界に照らして定める。PCG64運用lawを選ぶ場合のみηの判断を先行させる。

## 6. 重複照合・公開・ロードマップ差分

旧26軸表・Cの元receipt索引・m3/m4・凍結モデル・F0/上界調査を再利用した。新規差分はNを含む26軸metadata join、項の分解、有限な参照/誤差予算案。dcf905fc以後の関連C公開c383105bの差分はPF60価格/引渡しであり、PR-DFの26軸参照価格/残差母平均評価ではない。B2の1002f1dfはPF集計、統合ad6ab45bは方針整理であり、どちらも再実行しない。既存freezeにはm4の2価格予測だけがあり、同条件の26軸参照表は確認範囲で未同定。関連local job名で同じLD1評価を確認せず、出力先競合もなかった。未列挙archive/他serverの不存在までは主張しない。

ロードマップad6ab45bの§0/§7/§8.2/§10へ渡す差分は、**I維持の初回比較にPCG64証明を先行必須とせず、「期待残差の基準値/誤差が未評価」と「law差の上界は保存済み」を分離したこと**。次の有限作業は0 compileの参照/誤差予算評価。PF総額・Full-H/Pauli作業・新上界のC受入完了を待つ必要はないが、上界の条件確認を省かない。正式E[C]/CI/価格精度/長round・actual law/十分N/5方式winnerは未解消。ロードマップ本体は統合担当が更新する。

今回の検査は26軸の固定N/physical/canonical対応と既存locatorの読取確認、文書/JSON整合、旧証拠保全に限る。科学module import、保存価格/残差/上界の新算術、テストsuite起動は行っていない。固定SOURCE/raw/凍結正本/m4独立比較/旧STOP・Pauli/PFのworktree/jobは保全。commit/push・remote SHAと公開成果物一致を確認して **mandatory STOP**。後続数値評価は別指示。
