# N3占有条件：次の限定比較の結果前設計案 v1

2026-10-12 JST。設計案を保存する段階。実行認可=false、source生成器未実装。
現scopeの認可は理論・保存証拠の検証と本比較設計まで。
実行前に本案の科学判断、version付き生成器と独立checker、synthetic検査、
source/input SHAを持つexecution plan、新しい明示認可を必要とする。
同じ内容のGPT再レビューを自動で要求するものではなく、ここが次の進行判断点。

## 1. 判別する仮説と成立範囲

占有情報を残すσν条件が、強い同情報の既存保証の安全なORでも取りこぼす縮約を
有限予算内で認証できるか。外部下界強化だけの利益と条件式の利益を別に評価する。
新定理・world-first・大系での多項式算法・量子総資源削減を問うbatchではない。

Hamiltonianは実対称one-body＋Jij≥0のunordered density、外部1modeを空にする固定ν-sector。
全候補は利用者指定、dやmをtruthで選ばない。ν≥2の4合成baseを用いる。
geometry/化学basis/DF rank/L_D/時間発展/PF窓は適用なし。
目標δ=scale/250は基底エネルギーの投影幅で、operator time-evolution errorではない。

## 2. 結果を見る前に保存した新入力

[pack](../../artifacts/n3_occupation_redesign_preparation/2026-10-12/proposed_transfer_inputs.json)
と[receipt](../../artifacts/n3_occupation_redesign_preparation/2026-10-12/input_preparation_receipt.json)。
実行時はreceiptのSHAとexecution planのSHAの両方を照合する。
係数生成[script](../../scripts/prepare_n3_occupation_transfer_inputs.py)はbounds/受理/truthを計算しない。

| base | active m | ν | full modes | external index | scale | active hopping graph |
|---|---:|---:|---:|---:|---:|---|
| C1 | 6 | 2 | 7 | 1 | 1 | chain |
| R1 | 7 | 3 | 8 | 4 | 3/2 | ring |
| L1 | 8 | 3 | 9 | 0 | 3/4 | ladder |
| S1 | 9 | 4 | 10 | 8 | 5/4 | star |

各baseにnormが等しいedge_pair/distributed_threeの2結合方向、計8入力。
係数はすべて有理数、結合振幅scale/30。δ/振幅/graphを結果後に変更しない。
入力の寸法・粒子数・scale等は同時に変わるため、baseをiid標本とした一般化確率や
個々のfactorの統計効果を主張しない。新分子分布の無作為抽出でもない。
旧T1/T2/T3の12入力と今回T3証拠はdevelopment。旧証拠の再利用はidentity検査用だけ。
既知の成功点を生成器へ埋め込まず、新入力のpredicateを結果前に計算しない。

## 3. 共通trial上界U

各inputのactive determinantをcanonical occupation順に全列挙（最大126、hard cap256）。
one-body hoppingで直接つながる異なる二determinantを、occupation tuple対の
lexicographic順で最初の128対まで取る。明示trialのRayleigh値
(Hxx+Hyy)/2−|Hxy|を係数から計算し、relative signとnormを保存する。
全determinantとこの有限pair集合の最小値を共通Uとする。
最低値の証拠はdeterminantまたは二determinantの明示状態で、dense ground stateではない。
候補trialの上限や順序を結果後に広げない。任意fillingで安価とは主張しない。

## 4. 外部下界の三行

ν−1=p、Jd=diag(Jid)、O=hA−diag(hA)。

1. c0=hdd+pℓ、ℓは従来の係数one-body lowerをexact PSDで確認。
   Gershgorin、従来5個λと追加dyadic6候補を元契約同様に使う。
2. c_occ=hdd+underline σp(hA+Jd)。density active内の正項は下界へ捨てる。
3. c*=max(c0,c_occ,c_dh)。c_dh=min_{|S|=p}Cdiag(S)+underline σp(O)。
   Cdiagは全external occupationの係数式を列挙（今回最大choose(9,3)=84、cap256）。

各固有値和証拠はZ≽0,G−τI+Z≽0、lower=pτ−trZ。
c_dhの列挙は今回p≤3だけの限定方式。一般fillingの多項式性は仮定しない。
有効な候補のmaxは下界として安全。数値eigをそのままcに使用しない。
生成失敗はvalid lowerだけへ戻し、失敗理由を保存する。
三行が同じ値でもduplicate rowとして識別し、証拠費用をcacheする。

## 5. 各下界行で同じU,c,δを渡す四判定

| method | exact受理条件 |
|---|---|
| NORM | c>U、δ(c−U+δ)≥||v||² |
| FIVE | 従来5点λでκ=vᵀ(hA−λI)⁻¹v、μ=κλを作り元3条件を確認 |
| OLD-DIRECT | 旧Mκ≽0を一変数proposal＋有理PSDで確認 |
| OCC | σν(κhA−vvᵀ)≥κU−δDの有理trace証拠 |

三下界×四判定×8inputs=96 primary records以内。
各行にBASE-OR=NORM∨FIVE∨OLD-DIRECTをderived recordとして保存する。
主比較は最強係数下界c*でOCC固有証拠対BASE-OR。
下界変更の寄与は同methodのc0→c_occ→c*の採否差、判定式寄与は同c行の差で見る。
下界・Uをnewだけへ優遇せず、全methodへ同じ値と同じ合法係数情報を与える。
OCC運用fallbackにはnorm証拠・旧valid証拠を厳密に移せるが、
raw_OCC_generated、baseline_fallback、OCC_operational_unionを別fieldへ記録する。
fallbackをOCC固有gainとして数えない。

OLD-DIRECT/OCCの数値目的は正の共通scaleでconditioningをそろえた凹margin。
各c/methodに端点を含む最大128回の小行列eig proposal、bracket幅2^-40。
best x∈[0,1]を分母2^16/24/32/40へ下・上丸め＋端点、最大10有理候補。
旧five点をnew proposalの固有候補へ追加しない。
OCC候補のτ/Zはmid-spectrum数値proposalをdyadic化し、
bits24/32/40ごとに初期inflate=4mE/2^bitsの1倍/4倍、最大6組/κを固定。
E=max(1,max|Gij|)の有理scale、inflateは保証ではなく候補生成。
元係数から作る二つのPSDをexact checkerで確認して初めてlowerと認める。
同じκの数値固有分解は再利用し、new試行後のprecision追加・rounding変更を行わない。

## 6. LMI不可能性・効率の帰属

旧generatorが失敗しただけでは旧LMI族の数学的不可能性と言わない。
OCCがc*でBASE-ORを超えた場合、追加診断として固定16回以内の旧small-matrix
proposalから有理PSD W≠0を作り、
max(tr(WM0),tr(W(M0+DR)))<0をexactに示せれば全κ∈[0,D]の不可能性証拠になる。
レビューのzero-slope二outer mixtureはこの十分なdual証拠の特殊例。
separator生成に失敗しても入力やκ探索を再試行しない。
数値pipeline固有の追加受理と、条件族全体の厳密分離を別ラベルにする。

費用はU生成、三lower、κ search、τ/Z生成、exact verification、証拠bytes、
最大bit長、cache hits、各未判定理由、CPU/wall/RSSを分ける。
factorial実行の共有cache費用と、単独算法に必要な取得費用を区別する。
standalone BASE-ORは固定c0→c_occ→c*順の早期受理cost、
OCCのstandaloneは同lower順＋各行BASE-OR fallback＋必要時OCC追加costを記録する。
詳細比較のため後段を実行した費用を早期終了算法へ無条件に課さない。
数学的包含から計算効率やglobal optimumを推定しない。

## 7. 実装・resource・保存契約案

次の明示認可後にversion付きmodule/runner/独立checkerを新規実装する。
本保存証拠checkerを新しい生成器として使用しない。
synthetic fixtureは最大24、CAR/ordering、k=0/m、ν=1へのnorm帰着、
singular PSD/zero row、境界affine、invalid density/partition/oracle拒否、
旧証拠fallback、改変τ/Z/U/c拒否、reference対cached経路の一致を含める。
既存12入力の科学scanをsource調整用に再実行しない。

性能設定：CPU最大2case workers、BLAS/OMP/MKL/Numba/Rayon各1thread、GPU0。
immutable係数、basis/index、old λ/LDL、trial diagonal、数値小行列とPSDをcase内cache。
有理数PSDを中心にし、full many-body行列を生成器に渡さない。
最大m=9、ν≤4、trial det256/pair128、external diag256/case、
数値eig850/case（3c×2search×128、rounded-κ spectrum30、bound3、dual16等を含む）、
exact PSD512/case、dyadic spectral pairs180/case。
有理中間値4096bitsを全演算でguardする新実装を要件とする。
今回の保存checkerの一時値監視範囲を、その完成保証の代用にしない。

worker address-space4 GiB、aggregate8 GiB、JSON shard32 MiB、全output64 MiB、log8 MiB。
phase/totalのwall-time/CPU-time上限は置かない。tool polling timeoutは科学上限と別。
全8inputsのproof ledgerを閉じhashする。独立checkerはstdlibのみで検査する。
新batchにmany-body truth診断、量子compile、sampling、QPE/PR総costは含めない。
source/pack/plan SHA、call counts、bit/output/memory cap、attempt1/retry0を保存する。
必要な許可がある場合はHIROMU1015専用branchへ公開しremote保存証拠監査を行う。

## 8. 結果前の判別とSTOP

soundness違反、改変証拠の受理、truth流入は`SOUNDNESS_STOP`。
schema/resource/source不整合は`TECHNICAL_STOP`、有理capだけの個別失敗はUNRESOLVED。
c≤U、有限proposal失敗は不受理であり、物理的縮約不能ではない。

最強共通c*でraw_OCCのvalid受理がBASE-OR不受理を変えるbaseが2群以上なら
`FINITE_PIPELINE_TRANSFER_GAIN`。その追加受理のうち2baseで旧族のexact separatorも
検証できた場合にだけ`CERTIFIED_FAMILY_TRANSFER_SEPARATION`を併記する。
前者だけを後者と呼ばない。これらは限定input・取得下界・有限予算の結果。
それ以外は`NO_OCCUPATION_SPECIFIC_TRANSFER_GAIN_IN_FIXED_SCOPE`。
費用と未判定率を必ず併記し、保証件数だけで効率的算法や中心テーマを自動採択しない。

正常終了・負の結果・技術失敗のどれでもMandatory STOP。
新入力追加、δ/amp/seed/precision/call予算変更、retry、N1/N2再開、
実分子、多外部、一般二体、逐次縮約は自動実行しない。
新しい科学結果を取得した場合に、その研究上の意味をGPTへ返す。
