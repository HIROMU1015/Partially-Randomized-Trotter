# Closed P3導入後のh・resolution・α共同選択

作業日：2026-11-08。独立後続研究。**最終値の正本はextended/FINAL.json。全18roundの独立評価を完了。旧固定Closed P3 policyに対してRZ 6.0327%、CX 5.2350%減。公平に再最適化したcanonical比ではRZ 2.1150%減。強Full PFはなお低コスト。**

## 1. 固定条件と基点

基点は `2e38af3b12200c48c4f1b793638d223baff24e76`。Linear H4、1.00 Å、STO-3G、保存DF12/generation order、Partial LD3の218 signed components、Full LD12、保存元Hのexact groundを利用可能とする。同じ18round、δ0=.02、energy精度1.5936e-4 Ha、総α≤.05、各roundの位相予算.4 rad。状態準備費用を除外する。Qiskit1.3.0、compiler A、D内部BCH係数、P3次数、Closed P3 law/weighted estimator、共同isometry、単一Pauli control、対角合成を維持した。

変更するのは、既存の適格なD構成・h・resolution選択と36軸のα配分である。旧研究のSTOP、SOURCE、raw、個別契約は維持し、旧runnerを再開しない。

## 2. 再利用と有限menu

- 保存信号：D内部PFの `SIGNALS.json` 11,988行。ここにある診断値だけではnative資格としない。
- 元のnative資格：QUALIFICATION、QUALIFICATION_EXTENSION、REGISTERED_QUALIFICATION、bottleneckのJOINT_NATIVE_SOURCEから227個の異なる条件を取得。
- 小さいwrapped終点位相だけに依存する候補は除外する。支配分枝重みp>.5、支配固有エネルギーbiasの累積絶対値＋asin((1−p)/p)<.4という既存nonresonant診断を維持。残るmenuはPR171条件、Full51条件。
- 各条件のname/m/h/rateだけでなくD caseも元native資格と照合する。元のSOURCE→保存H、D時間角生成、有限RTE、alias、数値diskを同一条件のまま再利用する。
- PRには625 joint-basis catalogueの一様source/native supplementと対角合算diskを適用する。FullへRTE側の誤差を付加しない。前回固定PR点の資格は同じ原値を保持する。
- `track_c_bottleneck.allocate`、directed `rigorous_shots`、Closed P3の`qualify`、`paired_paths`、compiler A、既存D/R circuit buildersを呼ぶ。新コードはmenu接続、選択、価格データの役割管理と集計に限定する。

初期menuには31個のRTE occurrenceセル、57種類のD case/h群がある。同じexact rational λt/r・確率p・次数・256-bit law・eta/rho・angle SOURCEであれば、microstep係数とoperator mean証明を再利用できる。h、step数、resolutionが異なる場合の全path rangeやPF誤差、shot数まで同一視しない。

初期menuの疎さを点検すると、m15の調整D/h=.32/rate16など、同じ科学条件の有望な組合せがnative未資格のため含まれていなかった。そこで**局所拡張を一回だけ**行った。後半m12–m17、既存調整D0123/0321、trainingで有望な同構成のhの1/2・1・2倍（既存dyadic h範囲内）、隣接rateと中間rateを対象とした。新しいrate3/6/12/24/48を含む128条件のSOURCE/native資格を実際に評価し、すべて閉じた。新しいD係数、family、分割、分子、Taylor次数、sampling lawは追加していない。

新primitive20 keyについて128 compiler contextsを検査し、重複を除いた27 loweringを区間評価した。新しいexact microstep幅は5種類で、全218条件付きlawとgrouped P3平均を検証した。旧証明と合わせた最大operator mean差は1.67e-15。

価格の事前screeningだけには、既存MCの近傍cellのmicrostep当たり価格を使った。これは未検証のproxyであり、最終資源値には使わない。各round・各armの新候補上位2点の和集合12点を直接MCへ進め、元の222条件も残した。**両P3が同じ183 PR条件、Fullが元の51条件を持つ計234条件menu**で最終選択した。native資格済みだがproxyでscreenされた残り116条件は、`extended/MENU.json`に「資格あり・価格未検証」として残す。不適格や負の資源結果に読み替えない。

11988診断行すべての再資格、連続最適化、全順序探索は行っていない。最終結果はこの限定menuとscreening範囲の座標解でありglobal最適性を主張しない。

## 3. Weighted誤差・統計会計

同じQ_j、正規化した正有理p_jに対してA=Σp_j Q_j、x=λ_R t/rとする。両方式の理想補正平均は同じ有限Taylor P3：

\[
P_3(-i x A)=I-i x A-x^2A^2/2+i x^3A^3/6.
\]

Closed P3の各eventはG7のproposal πとweight wを持ち、πw=係数midpointをexact rationalで満たす。bit/root/angleの差は同じ既存certificateで囲む。各trialで全microstepのweight積を掛け、全trialで平均する。今回のclosed P3はrejection/zero massが0、acceptance1。技術失敗をzeroへ置換しない。

eventの十分rangeは W*≤B_RA(1+rho)/(1−eta)^4。全pathではこれらの積に、各weightのbinary64丸めと区間積最終丸めの膨張を含めたLを使う。測定値はY=(全path weight)×(±1)。**Y/L∈[−1,1]にHoeffdingを適用するのであり、通常の無重みHadamard信号と同一視しない。**

既存canonical資格の半径下限ρ_cと統計位相余裕β_cを、同じ候補のcanonical正規化B_cから再scaleする：

\[
\rho_w\ge\rho_c\exp\{\sum M\log B_c-\log L\}-\Delta,
\quad
\beta_w=\operatorname{lower}\{\beta_c-\arcsin(\Delta/\rho_{\rm scaled})\}.
\]

Δには新しいroot/angle bias、weight binary64化と圧縮区間積の誤差を含める。旧一様basis/source/native diskは保守的に保持する。D・Rの境界順序、scalarのcontrolled相対位相は保持する。

各軸の十分trial数は192-bit directed intervalで

\[
N_{m,b}=\left\lceil\frac{4\log(2/\alpha_{m,b})}
 {\rho_m^2\sin^2\beta_m}\right\rceil
\]

を上側丸めする。αのbinary64値をexact rationalに戻して総和≤1/20を確認する。整数切上げを目的関数にも反映する。これは十分条件であり、情報論的に必要最小のtrial数という主張ではない。

## 4. 回路価格・共同選択・データ分離

同一のfull/singleton/joint basisと効率的なcontrol、既存reference/term/merge/networkを両P3に与える。各trajectoryまたはD群でRZ優先、CX・depth・sizeの順にtieを解く一つの回路を選ぶ。他指標の最良回路を混ぜない。

D_bulkにD/D境界のsigned補正を一度だけ加え、D＋RTE＋wrapperを合計する。RZ/CX/sizeはそのmodular native実装の加法値。全体depthはserial upper boundで、長roundの全DAG深さではない。

RZ目的のcost-sensitive α提案と、各roundの整数shot×期待RZが小さい候補選択を交互に行う。初期αを変えた二つの開始点、同一停止規則を両P3とFullへ与える。最良の訪問policyを保存する。有限menu座標探索でありglobal optimumの証明ではない。

初期trainingでは、連続α提案が整数切上げ後に既存αより約0.1%高価となる例があった。このため元の提案器を保ち、旧αも開始候補に加え、directed逆shot式で整数Nに必要なαを上側評価する小さい局所改善を追加した。高価な軸のtrialを一つ減らすために安い軸の追加trialでαを捻出し、総RZが減る交換だけを採用する。一提案の追加trialは64まで。総αと各軸の十分shotを元の区間式で再確認する。量子sampling lawや統計不等式は変更しない。

前回のtrainingとholdoutは今回の選択用trainingへ明示的に役割を変更して再利用する。新しいセルには新規trainingを取得する。今回のholdoutは新namespaceの独立seedで生成し、policy・36軸α・整数trialをfreezeした後にだけ読み込む。

期待価格はrare/no-rare層別MC。両方式に共通の乱数streamを与え、各方式自身のrare確率を使う。この価格MC用層別samplerと、shot保証のfresh independent uniform-bit条件を区別する。共通セルを複数roundで再利用する場合は、全roundの係数を先に合算してからcell/stratumの共分散を取る。

初期は各cell/層32標本。m17で32/64 resolutionとh=.16/.32の費用差がこの精度と競合したため、trainingの3セル(.32,512)、(.32,1024)、(.16,512)だけを各層256へ増やして再選択する。初期POLICY.jsonを保全し、最終選択をPOLICY_REFINED.jsonへ保存する。これはholdoutを見る前の追加である。

この初期menu研究ではtraining3328軌道、独立holdout1280軌道を取得した。その後の一回限りの局所拡張では、この両方を**明示的にtrainingへ再分類**した。新しい6セルの直接MC384軌道を足し、拡張選択用は計4992軌道となった。最終の独立評価は別seed名`extension_holdout`で取得した。旧holdoutは最終評価へ混ぜない。最終policyは`extended/POLICY.json`でfreezeし、変更前後のdigestも照合する。高価格cellは各層128、他は32標本とする。

直接AのBasisTranslatorをexact context/angleでmemoizeする技術helperも検査した。人工16 contextと保存8軌道でnative指標が一致し、完全occurrenceのX/Yも通った。ただし今回の実測では高速化を確立できなかったため、**全training・最終holdoutは従来の直接compiler Aを使用した**。このhelperのCPU時間を量子資源の利益に換算しない。

## 5. 限定検証

m16/m17、調整D0321、h=.32におけるresolution候補で開始した。初期pilotではm17の64→32にRZ約6.0%減、差/paired SE≈2.2の候補を認めた。m16の32→16は悪化した。このpilotは選択用の予備結果であり最終推定ではない。

初期trainingのm17ではClosedだけrate32、canonicalは64を選んだが、3セルの精度追加後は両方h=.16/rate64へ変わった。したがって初期のlaw固有の選択差を科学的結論にしない。局所拡張の直接MCでは、両方式ともm14の調整D/h=.64/rate8、m15のh=.32/rate16、m16のh=.32/rate24、m17のh=.32/rate48を選んだ。m12/13ではproxy上位の新tuned候補が採用されず、従来order/vertexが残った。

完全なn=512/1024 occurrenceを持ち、noncommuting rare prefixとroot eventsを含む正確にn個のevent列を用いた。1/2 D steps、scalar、controlled X/Yとweighted積を独立70次元sector演算子へ照合した。最大readout差6.52e-15。人工testは共有cell共分散の係数合算とsource diskによる半径/位相余裕減少を確認した。

全supportの資格は有限回路例だけでなく、元の128 compiler contexts・625共同basis・全218条件付きlawの一様証明から得る。新しく選んだh/rateのPF/shotを、前回選択点の数値から外挿しない。

新たに選んだn=768の完全occurrenceも、1/2 D steps、weighted X/Yを独立sector行列へ照合し、最大差9.82e-15だった。全方式の全36軸について、正のrho/beta、exact α和≤1/20、directed式以上の整数trialを確認した。関連人工testは8件PASS（データ分離testを含む）。

## 6. 最終結果

同じ最終holdoutで旧policyも再評価した。前回公開値1.394938574e12から新しい乱数実測値を直接引いた差を、追加改善とはしない。下表の価格は状態準備を除いた期待native数で、MC標準誤差と成功保証は別である。

| 方式 | 整数trial合計 | 総RZ | 総CX | serial depth上界 | size | RZ MC SE |
|---|---:|---:|---:|---:|---:|---:|
| 旧固定Closed P3（今回holdoutで再評価） | 17028 | 1.387608544e+12 | 1.925041360e+11 | 5.312907122e+11 | 2.025658744e+12 | 7.110533e+09 |
| A：最適化Canonical P3 | 16946 | 1.332071607e+12 | 1.863565000e+11 | 5.100098315e+11 | 1.945479172e+12 | 4.943179e+09 |
| B：最適化Closed P3 | 16791 | 1.303898609e+12 | 1.824266292e+11 | 4.992174341e+11 | 1.904338949e+12 | 4.834427e+09 |
| C：強Full PF | 14225 | 5.647934021e+11 | 1.012313194e+11 | 2.088520659e+11 | 8.371684838e+11 | 0.000000e+00 |

旧固定Closed→Bの削減はRZ **8.370993524e10 ± 8.581509829e9（paired MC SE）**、約9.75 SE。割合はRZ **6.03267655%**、CX **5.23495599%**、depth上界 **6.03686032%**、size **5.98915268%**。A→BはRZ **2.817299755e10 ± 1.089716414e8**、**2.11497621%**。

B/Cの費用比はRZ **2.308629**、CX **1.802077**、depth上界2.390292、size2.274738。Fullへ同じ効率control・対角合成・整数α改善を与えたため、旧Full基準を不利にしていない。Fullの旧RZ5.660906974e11に対し今回は5.647934021e11で、αの整数改善分だけさらに安い。

最終A/Bは**全18roundで同じD構成・h・resolutionを選んだ**。Closedをcanonicalの選択へ移した登録済みablationは最終Bと同一結果だった。このmenuではClosed固有のh/resolution選択差や相互作用の追加利益は観測しない。samplingの利益と共通の再選択効果を区別する。

### 選択が変わった支配round

| m | 旧固定ClosedのD / h / rate | 新ClosedのD / h / rate | 旧→新trial (X,Y) |
|---|---|---|---|
| 14 | order_0321 / 0.08 / 8 | tuned_0321 / 0.64 / 8 | [430, 430] → [527, 526] |
| 15 | vertex_j0 / 0.08 / 8 | tuned_0321 / 0.32 / 16 | [619, 619] → [339, 338] |
| 16 | tuned_0321 / 0.32 / 32 | tuned_0321 / 0.32 / 24 | [299, 299] → [355, 355] |
| 17 | tuned_0321 / 0.32 / 64 | tuned_0321 / 0.32 / 48 | [253, 253] → [303, 302] |

全方式の18round h/rate/step数、36軸αと整数N、rho/beta、PF位相、総指標は`extended/ROUNDS.csv`。D/RTE/wrapperの各指標は`RESOURCE_COMPONENTS.csv`、詳細誤差diskはpolicy/FINALのqualificationに保存した。

m17はh=.32を保ち、n=1024→768、全microstep 8388608→6291456。PF位相上界は.0336685391 radのまま。log rangeは.270397071→.360529421へ**増加**し、rho下限は.762978948→.697246003へ低下するため、trialも[253,253]→[303,302]へ増える。一方、RTE event数が25%少なくなる回路価格との総合で利益が残った。normalizationを小さくしただけの改善ではない。

m16もrate32→24でeventを25%減らし、trial増を受け入れる。m15はvertex/h=.08/rate8から調整D/h=.32/rate16へ移り、PF bias・attenuationが改善してtrialが減る。m14はhを大きくしてPF予算を多く使う代わりにD stepを減らす。

### 寄与と条件付き保証

| BのRZ内訳 | 旧固定Closed | 最適化Closed |
|---|---:|---:|
| D | 3.373971816e+11 | 3.439699811e+11 |
| R | 1.050211269e+12 | 9.599285356e+11 |
| wrapper | 9.365400000e+04 | 9.234400000e+04 |

全体のD RZは**6.572799426e9増加**、RTE RZは**9.028273335e10減少**した。誤差・signal・trial・単価を合わせたtradeoffであり、全成分が下がるという主張ではない。

登録済みablationで、旧αのままh/resolution/D選択だけ変更するとRZ **5.92696158%減**。その選択でαを再配分するとさらに **0.11237542%減**、両方で6.03267655%減となる。旧h/resolutionを固定したα単独の改善は0.12518128%。これらは異なる条件での経路依存の差であり、百分率をそのまま足さない。Closed法自体の2.1150%は、再選択済みA/Bの比較である。

| 方式 | 最小rho下限 | 最小beta_stat下限 (rad) | mean decoder energy誤差 (Ha) |
|---|---:|---:|---:|
| A | 0.689573863 | 0.334203408 | 1.284014863e-05 |
| B | 0.697246003 | 0.334203408 | 1.284014863e-05 |
| C | 0.996960423 | 0.290725234 | 5.825898290e-06 |

全36軸のα exact和≤.05、選択SOURCE/native/finite-bit/weighted予算は閉じ、元H energyをdecoderへ渡さないmean分枝検査も通った。総位相予算.4に対応する最終energy boundは **1.52587890625e-4 Ha < 1.5936e-4 Ha**。保存Hermitian-H・exact ground・ideal native・fresh independent bits条件で成功率≥.95の十分条件を満たす。価格差のMC SEをこの成功確率に混ぜない。

最終独立holdoutは16セル2176軌道、64 worker、1299.44秒。coarseのholdout、最終選択のtrainingとは別seedである。量子backend shotは0。

## 7. 再現・SOURCE・raw

新SOURCE：`track_c_closed_p3_joint_opt*.py`と`track_c_closed_p3_joint_extend.py`。runnerは`run_track_c_closed_p3_joint_opt.py`、`run_track_c_closed_p3_joint_opt_extended.py`。数値表の入口は本研究の`artifacts/`。**最終値はextended/FINAL.json、extended/POLICY.json、extended/ROUNDS.csv**。rootのFINAL.jsonは初期menuの独立結果として保全し、取り違えない。

固定モデルは基点のMODEL.json.gz/MODEL_ARRAYS.npzで再生可能。非公開pickle原本を公開しない。SCF・DF fit・ground solveを再実行しない。元科学SOURCE履歴は新しいbundleへ保全する。既存の統合総括は上書きしない。

SOURCE.jsonは科学SOURCE head・各file digest・段階ごとの実行commit・公開input identityを結ぶ。SOURCE_HISTORY.bundleは元の6科学commitを保持し、GitHub APIの公開用commitと区別する。再現時は基点2e38af3がbundle prerequisite。公開用commitへ原実行のSOURCE SHAを付け替えない。

元のprivate common.pklは保全し、public codecの9科学fieldだけで再生できる。新規SCF、DF fit、ground solve、量子backend shotはいずれも0。原rawは専用folderに残し、公開用RAW_DATA.tar.gzとRAW_INDEX.jsonで全memberのbytes/hashを照合する。

再現用のPythonは3.12.3、Qiskit1.3.0、numpy1.26.4、scipy1.14.1、mpmath1.3.0。新しい実行を許可されたfresh checkout/outputでのみ、root runnerのinventory/pilot/correctness/training/refine/selected-correctness/holdout、続いてextended runnerのinventory/training/correctness/holdoutを用いる。保全済みrawを入力として使う場合は、fresh checkoutでarchiveからcoarse training/holdoutを復元する。最新SOURCEで初期α処理を再実行すると整数改善も有効になるため、初期stageを原形通り復元するにはSOURCE.jsonの段階別commitを使う。最終extensionは保存input・raw・固定policyの組から再評価できる。

## 8. 限界とSTOP

保存Hermitian H、exact ground仮定、ideal native、fresh independent bitsに条件付けたRPE十分条件と、価格MCの不確かさを分離する。元の連続積分Hの無条件包含、実機性能、状態準備、T合成、別分子・分割・P5・sampling lawへ一般化しない。

Closedの最適h/resolutionがcanonicalと異なることを必要条件にしない。今回の最終trainingでは両者の選択が一致した。normalization改善と、共通のh/resolution・α再選択の利益を分離して報告する。整数交換と座標探索はlocalであり、別の既知feasible ablationがごく僅かに安い場合も、holdoutを見てpolicyを選び直さない。

技術上、cache helperの初回検査はまだ生成されていないtraining行を参照して停止したため、取得済みj=0行へ直して再検査した。失敗logを残す。bundle梱包はSHA単体の指定ではrefを収録できないためHEAD refへ修正した。科学SOURCE・入力・数値結果の変更ではない。

科学計算は全18round評価で完了し、以降は報告・梱包・公開照合だけを行う。公開照合後mandatory STOP。次の担当は**ユーザー**で、本結果の採否と今後の研究方針を判断する。Codexの後続科学実行はない。今回のmenuをさらに細分する探索、P5、別sampling family、別分子、新PF、Track B単独研究、詳細GPT科学レビューを開始しない。
