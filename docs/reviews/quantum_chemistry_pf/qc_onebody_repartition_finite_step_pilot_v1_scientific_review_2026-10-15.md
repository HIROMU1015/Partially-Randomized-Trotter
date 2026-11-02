# 一体項再配分PF：固定H4有限step pilot v1の科学的レビュー

- 作成日：2026-10-15 JST
- 区分：ユーザーの「レビューを開始して　丁寧に考えて」に基づく重要科学レビュー。
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 結果正本：`31ea58449eb8e535b8f5d5f62956fb410d35f0ae`
- 数値source：`19d5bd74db020564ea9a4b1882e102911e983cdb`
- 継承する前提：厳密基底状態を利用可能とする。基底エネルギーの真値を構成選択へ与えない。block-encoding・qubitizationを候補から除外し、Track A/Bの担当研究をこの系列へ移管しない。
- 今回実施したこと：固定文書・契約・主要source・保存数値の該当箇所・監査記録の読取り、既存レビューとの整合性確認、関連一次文献の限定確認、解析的な費用・判別力の検討。
- 今回実施していないこと：新しいHamiltonian生成、RDM・期待値取得、固有値計算、PF数値評価、数値フィット・最適化、保存結果の新規集計、独立checker再実行、compile・T合成、repository書換え。

## 0. 最終判断

**現時点で一体項再配分の第一候補を終了する判断は採用しない。現在の再配分と分子入力を凍結したまま、許容時間刻みの境界を判別する、範囲を限定した追加検証の準備へ進む。**

ただし、旧pilotの登録結果 `NO_UNIFORM_REGISTERED_COMPONENT_GAIN` は変更しない。4/4のtarget×price条件で10%改善がなかったことは、その固定窓・費用モデルに関する有効な結果である。

継続理由は「未検証の領域があるから」だけではない。今回、新たに次が確認された。

1. E0や有限hの真値を使わず選んだ再配分で、主要係数と小さい有限hのraw S2 biasが実際に低下した。
2. 登録されたstatic primitiveモデルでは、再配分に伴う費用増が小さく、THRIFT旧templateのような大きな既知費用障害は観測されていない。
3. 元raw S2すら最大hで厳しい誤差予算を十分満たし、比較の最良点が時間刻み窓の上端に達している。誤差減少をquery数削減へ交換できる境界を、今回の窓は測定していない。
4. scalar-corrected S2・S4・Richardsonの残差は数値許容幅を下回り、これらに対する残差改善・高次係数の差を判別できていない。

したがって、前回レビューが最も重要とした「補正済みS2等の強い対照にも有限stepで利益が残るか」という問いは、まだ十分に試されていない。

**次の作業はCodexによる追加検証の設計・実装準備であり、追加数値実行ではない。** 研究上の変更対象は主としてhの評価領域とその検証規則であり、新しい分子、再配分係数の再探索、別fragment、別アルゴリズムへは拡大しない。必要な新scope・source・resource上限を固定し、別途のユーザー明示実行承認後に一回実行する。

| 対象 | 判断 |
|---|---|
| v1の登録10%改善未達 | 維持する |
| 主要誤差・小hのraw S2 bias低減 | この固定入力での成果として維持する |
| 強い対照に対する量子資源優位 | 未実証 |
| 現在の候補の全面終了 | 現時点では採用しない |
| 凍結した候補の許容h境界を測る限定検証 | 準備へ進む科学的価値がある |
| production、全面compile、大規模分子scan | 進めない |
| 独占的1-RDM応答生成高速化のclaim | 従来どおり採用しない |
| 閉式・特殊化の歴史的先行性 | 未確定。matching対照の存在から既報と断定しない |
| 第二候補のDF因子回転 | 保留。自動移行しない |
| 新しい科学計算 | 未認可。レビュー承認と別に扱う |

## 1. 研究全体における位置付け

### 1.1 継承する目的と役割分担

本系列は、厳密基底状態を利用できるという前提で、量子化学Hamiltonianに対するPF・時間発展回路の構成を改善する研究である。PRの資源効率・ground-energy task接続を担当するTrack A、PR内部のfinite-RTE・乱択・表現を担当するTrack Bとは区別する。

前回の統合方針でTrack Aの継続作業へこの系列を戻そうとした案は撤回済みである。今回も復活させない。対象は固定12 density fragmentの全決定論PFであり、PRのL_Dや残差samplingを変更する検証ではない。[G1, G2, G9]

厳密基底状態の準備法を新しい研究課題へ戻さない。機構検証では理想的な状態由来情報を使ってよい。一方、古典RDMが無料で正確に与えられることと量子状態が供給されることは別なので、運用費用のclaimでは追加情報取得を別記する。完全な測定backendの完成を今回の比較準備の前提にはしない。

### 1.2 過去の判断から継承すること

- C1の旧macrocycle、N1–N3、THRIFT等の限定的結果・停止は保持する。それらの費用倍率を現在のH4再配分へ転記しない。
- 現一体再配分はHを厳密に保つため、新たなactive-space切断誤差を導入しない。ただし元保存DFの精度・由来を新たに認証したわけではない。
- 同じ恒等式・1-RDM・自由度・cacheを許したcompact BCH対照が同じ応答表を同等のscalingで生成できる、という限定監査は維持する。
- その監査は、有限stepの性能が悪いことや閉式の既報性を証明したものではない。
- 前回科学レビューは、独占的な古典生成高速化ではなく、再配分後のPFの有限step有用性を判別するためにpilotを勧めた。

今回の結果によって、機構の小h側は数値的に支持された。しかし資源交換の境界は取得できなかった。これが、今回の追加判断を必要とする新情報である。[G1, G3, G9]

## 2. 取得資料と確認範囲

### 2.1 今回実際に取得したGitHub資料

以下のG1–G9は結果commit `31ea58449eb8e535b8f5d5f62956fb410d35f0ae`、S1–S3は固定数値source `19d5bd74db020564ea9a4b1882e102911e983cdb` から取得した。

| ID | path／役割 | 確認範囲 |
|---|---|---|
| G1 | `docs/research/qc_onebody_repartition_finite_step_pilot_v1_results_20261015.md` | 結果、登録scope、表、限界、次の判断事項 |
| G2 | `artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/contract.json` | 入力・selector・評価・価格・query・STOPの固定契約 |
| G3 | `docs/research/qc_onebody_repartition_finite_step_pilot_v1_gpt_handoff_20261015.md` | 未解決の問いと今回の情報価値 |
| G4 | 同preparation directoryの`source_manifest.json` | source ref、21 files、contract/environment identity |
| G5 | `artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/independent_audit.json` | 独立JW/Padé照合PASS、回数と限界 |
| G6 | `artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/execution_audit.json` | 一回消費、call caps、selector seal、保全の保存記録 |
| G7 | 同attempt directoryの`raw.json` | 冒頭95行：h grid、truth、元S2/h0の保存recordとnodesの一部 |
| G8 | 同execution directoryの`publication/receipt.json` | 冒頭45行：remote取得PASS、payload/source/branchと一部blob identity |
| G9 | `docs/reviews/quantum_chemistry_pf/qc_onebody_repartition_final_audit_scientific_review_2026-10-14.md` | GitHub冒頭85行。全体は添付の同名Markdownも読取り |
| S1 | `src/trottertracks/ground_pf_repartition/qc_onebody_repartition_pilot_v1.py` | 応答・候補生成・H保存・Λ生成・PF構成・phase処理。末尾は270–345行の追加取得で確認 |
| S2 | 同directoryの`qc_onebody_repartition_resources_v1.py` | primitive会計、境界融合、query、repeat、eligibility、comparison |
| S3 | 同directoryの`qc_onebody_repartition_pipeline_v1.py` | seal→truthの順序、固定h、36セル、CS2/EXT2導出、費用比較 |

この科学判断に必要な結果・契約・主要計算source・監査記録にはアクセスできた。`raw.npz` の全行列をdecodeして再検査したわけではない。全276 cost rows・69 recordsを再ランキングしたわけでもない。個別checker、旧27検査、準備37人工検査を再実行していない。

GitHubから本文を取得したことと、Codexのremote全blob監査を自分で再実施したことは区別する。後者は行っていない。

### 2.2 重要identity

- 入力pack SHA256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`
- Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`
- source manifest SHA256：`8b4a3ed3034c0931dd33cff60cb27cdce974f8ca5dc316bab71d2d0ed29d8fe1`
- contract SHA256：`c3e9d2028ca74a65d7f781d48a02033755d92919a24be3415097fa573b97161d`
- selector seal SHA256（G7）：`98f6ff69923ceca02448f750cf387bfe443a8795cf66b45034a481e32e0bc008`
- raw NPZ SHA256（G7、ファイル全体は未decode）：`815e31a4c305a7bf13b380c53f1e99808920aaaf72d03836ee5af76dd058c81c`
- 結果報告のGit blob SHA：`a0a14bce61f95e14b0d3d0f4d9b1bd3a35fe5269`
- 契約のGit blob SHA：`6617f3d3190e36f78e5216497aacf5aadf6703f1`
- 公開receiptのverified payload：`4ae95ac0b2f045218743babe69a95609c567ad1f`

これらのSHA256は保存記録の値を転記したもの。Git blob SHAとSHA256は異なる識別子である。今回のレビューで全hashを再生成していない。

## 3. 登録pilotで得られた結果

### 3.1 入力と比較の単位

- 直線H4、隣接距離1.00 Å、STO-3G。
- 保存DF rank12、generation順0…11を固定。8 spin orbitals、Nα=Nβ=2、sector36次元。
- 全12 density fragmentを決定論PFに用いる。PRのL_D探索ではない。
- 元一体項Aと最初のdensity Bだけを、同じdensity frameに対角なXでA+=X、B-=Xと変更。
- spin-paired・mean-zeroの一方向、leading objective、固定bound、反復optimizerなし。
- original / response_selected / known_linear_extractionの3構成。
- S2と5-stage signed S4、6hの36 PFセル。CS2と隣接Richardsonを導出して69 records、276 cost rows。
- εE=10^-4 / 10^-5 Ha。deterministic 1/2、statistical 1/4、synthesis/information各1/8。
- PF ground-branch overlapの適格性は0.99。二つのstatic primitive価格P1/P2。

4 target×price条件は、4個の独立分子実験ではない。同じ分子・同じ36 PFセルから得た、二つの精度と二つの価格仮説に対する会計である。[G1, G2]

### 3.2 主要係数と最大hでの誤差

最大登録hは`0.002745119306432635`。元の6点は、これを1,2,4,8,16,32で割った点である。

| 量 | original | response_selected | known_linear_extraction |
|---|---:|---:|---:|
| leading d | 0.01336651855253499 | 0.008200683552525947 | 0.01031437351283108 |
| 最大hのraw S2 signed bias / Ha | 1.007256416e-7 | 6.179765899e-8 | 7.772542032e-8 |
| raw S2 numerical allowance / Ha | 1.000e-10 | 1.000e-10 | 1.000e-10 |
| raw S2 overlap | 0.9999999999999560 | 0.9999999999999845 | 0.9999999999999722 |
| CS2 signed bias / Ha | -1.650082696e-13 | -6.809674729e-14 | -3.979385828e-13 |

この表はG1の転記であり、新しい集計ではない。

主要係数とraw S2 biasは、おおむね同じ比で低下している。これは小hで期待される主要項の改善が、実分子の数値構成にも接続されたという肯定的証拠である。ただし、新しい収束次数が得られたこと、広いh領域で同じ比を維持すること、S4の誤差を最適化したことを意味しない。

最大hでのCS2/S4/EXT2のbiasは概ね10^-13 Ha、一方でnumerical allowanceは10^-10 Ha程度である。その領域で誤差値の小さい順に科学的順位を付けない。overlapの約10^-14の差も、資源上の有意な改善としない。1を僅かに超す値は丸め残差であり、物理的確率が1を超えたとは扱わない。[G1]

### 3.3 費用結果

| εE / Ha | 価格 | response/control | v1判定 |
|---|---|---:|---|
| 10^-4 | P1 | 1.000741289844 | 10%改善なし |
| 10^-4 | P2 | 1.000389256520 | 10%改善なし |
| 10^-5 | P1 | 1.000741289844 | 10%改善なし |
| 10^-5 | P2 | 1.000389256520 | 10%改善なし |

P1はGivens/linear/pairの価格(4,1,2)、P2は(8,1,4)。いずれも実T合成・compile・hardwareの測定値ではない。

保存された最良行の名称はCS2:response_selected / CS2:originalだが、元raw S2も同じhで適格かつ同回路価格である。CS2が保存されたのはlexical tie規則による。これを「元のbaselineもd0取得を必須とする」という意味へ読み替えない。[G1, S2]

## 4. 今回のno-gainが意味する範囲

### 4.1 v1の結果は正しく、しかし主要な判別境界に届いていない

厳しいtarget10^-5 Haのdeterministic budgetは5×10^-6 Ha。最大登録hで元raw S2のbiasは約1.01×10^-7 Haであり、budgetより約50倍小さい。緩いtargetでは余裕はさらに大きい。

結果として両方式とも最大登録hを選び、query数に差を生じさせる機会がない。今回の時間窓は、正しさ・小hの機構確認としては有用だったが、誤差低減を許容hの拡大へ交換する性能を十分に判別する窓ではなかった。[G1, G2, S3]

これは「実験が無効」「Codexが誤った範囲を勝手に計算した」という批判ではない。Codexは固定hとSTOPに従っている。問題は、研究上の中心課題に対する評価領域の不足である。

### 4.2 制約付き最適値と、必要なh領域を評価した最適値は別

方法i、h集合D、適格性条件E_i(h)、費用C_i(h)について、v1が比較したのは

C_i^(v1) = min { C_i(h) : h∈D_v1, E_i(h) }

である。

研究上知りたい許容h領域をD_fullとしたとき、D_v1⊂D_fullなら各方法の最小費用は拡張により下がり得る。しかし、下がる程度は方法ごとに違うため、C_R^(v1)/C_B^(v1)から拡張後の比の方向は決まらない。**再配分も対照も、拡張で改善できる。**

最大hで全て適格だったことは、同じ小h branchでより大きいhが適格な可能性を示すが、正当なalias・branch domainを越えて任意にhを大きくしてよいという意味ではない。

また、誤差は一般にhに対して単調とは限らない。一つの不適格点を見つけただけで、全てのより大きいhが不適格だと証明したことにはならない。後続検証は、定義した連続branch・有限領域・探索規則での結果として表示する必要がある。

### 4.3 なぜ費用差が誤差改善をほぼ反映しないのか

v1ではtargetのbudget fractionsを固定し、eligibleであることを確認した後のquery数は

N_q(h)=ceil(π/(h ε_stat))

で決める。deterministic biasがbudgetの内側でさらに小さくなっても、同じhのquery数は減らない。[G2, S2]

繰返し境界融合を含むsourceでは、固定したprimitive grammarの費用は、概ね

C_i(h)=r_i(h){a_i N_q(h)+b_i}

という形になる。a_iは反復ごとの定常primitive費用、b_iはentry/exit/scalar等の境界項、r_iは契約上のbranch-repeat factorである。この形はS2のsourceからの解析的整理であり、新たな費用値の計算ではない。

v1で同じh・同じquery・実質同じoverlap条件なら、主にa_i,b_iの差が結果を決める。responseの約0.074% / 0.039%の増価は、その固定モデルの小さなprimitive増であって、誤差改善を使い切った後の費用障害ではない。

## 5. 継続の見込みと、その限界を数式で分ける

### 5.1 未補正S2同士の主要項モデル

以下は本レビューによる解析的な見積りで、新しいPF評価・数値fitではない。

raw biasがb_i(h)≈d_i h²、overlap/repeatが共通、費用単価がhに大きく依存せず、境界・整数化を無視できる範囲なら、deterministic予算ε_detに対して

h_i≈sqrt(ε_det/|d_i|),

C_i∝a_i sqrt(|d_i|/ε_det)/ε_stat.

従って

h_R/h_O≈sqrt(|d_O|/|d_R|),

C_R/C_O≈(a_R/a_O)sqrt(|d_R|/|d_O|).

保存されたleading値をこの式へ概算として当てはめると、h比は約1.28という大きさになる。これは「実際に1.28倍のhを使える」と確認した値ではない。未補正S2同士の主要項だけのモデルであり、S4・CS2・EXT2への優位、finite-h保証、overlap、実合成費用、情報取得費用は含まない。

この限定的な見積りは、現在の小さいprimitive増に対して、誤差減少がquery数を変えるだけの余地を持ち得ることを示す。今回の追加判別が単なる希望的探索ではない理由の一つである。ただし強い対照に勝つ予測としては使用しない。

### 5.2 強い対照を戻すと、dの改善だけでは判断できない

局所的な孤立・非縮退ground branchで、対称S2の有効Hamiltonianを

H_eff,i(h)=H+h²D_i+h⁴F_i+O(h⁶)

と表せる範囲では

E_i(h)-E0=d_i h²+c_i h⁴+O(h⁶).

scalar correction後は

E_i(h)-h²d_i-E0=c_i h⁴+O(h⁶).

ここで

c_i=〈F_i〉0−Σ_{n>0}|〈n|D_i|0〉|²/(E_n−E0)

であり、1-RDM応答表で下げたd_iだけからc_iの大小は決まらない。overlapの主要変化もD_iの非対角成分に依存する。これは前回レビューの摂動整理を継承したもので、今回c_iや励起和を計算していない。[G9]

同じ条件の近似費用比較では、CS2同士の費用は

C_R,CS2 / C_O,CS2 ≈ (a_R/a_O)(|c_R|/|c_O|)^(1/4)

となる場合がある。しかし現データの小さい残差からc_iを安定に取得したとは言えない。

縮退系を研究全体から除外したわけではない。上の単一branch式の適用条件を示している。今回の固定H4では保存truth検証に非zero gapがある。[G1]

### 5.3 数値床から高次係数を作らない

r_i(h)=E_i(h)−E0−d_i h²をc_i h⁴として評価したいとき、r_iの数値幅η_iは、正規化後にη_i/h⁴へ拡大する。小さいhにするほど、高次係数の識別が悪くなる場合がある。

今回、CS2/S4/EXT2のbiasはnumerical allowanceより小さい。したがって、保存されている約10^-13の値を使って、四次係数の比、収束次数、最適h、資源勝者を新たに推定しない。

必要なのは、finite-h residualが判別できる領域か、必要箇所だけの適切な数値精度である。元の小h全点を際限なく高精度化することを第一選択にはしない。

## 6. sourceと契約から確認した科学的注意点

### 6.1 E0非利用と前向きな候補選択

S3はselector sealを書いた後にH truth eigensolverを呼ぶ。S1の候補生成は供給されたstateのγとd0を使い、E0やfinite-hのerror最小化を使わない。候補は一方向のzero/endpoints/root/vertexに限定されている。[S1, S3]

この分離は保持する価値がある。ただしH4は既存development入力であり、h windowの追加はv1を見て設計する。同じH4の後続結果を独立held-out証拠とは呼ばない。候補Xを凍結し直さずそのまま継承することで、追加評価のたびに候補を救済する循環を避ける。

### 6.2 principal phaseは大hでも自動的に正しいとは限らない

phase_recordはprincipal argumentからE=−arg(z)/hを作り、exact-input overlapが最大のclusterを選ぶ。現在の小h窓を前提とした処理である。[S1, S3]

h拡張では、元と同じコードのh個数だけを増やして実行してはいけない。phase wrap、cluster交差、continuationの一意性、数値誤差、S4の負時間stageを考慮したvalid domainを先に定める必要がある。

候補を選ぶためにE0を入力しないことと、独立参照側でbranchやbiasを確認することは別である。ただしE0に最も近くなるよう整数2π/hを各点で都合よく付ける処理を、運用可能なunwrapとして主張しない。

### 6.3 v1の比較対象は「再配分法全体」より狭い

S2のcomparison関数でprimary candidateに入るのはresponse_selectedのraw S2とCS2であり、それ以外のeligible methodはcontrolになる。従って、response_selectedのS4やEXT2も強いcontrol集合に含まれている。[S2, G2]

v1では最良controlがoriginalなので、この定義が今回のno-gainを生んだわけではない。しかし後続でresponse-S4が良くなった場合、それをprimary S2成功へ変更したり、逆に一体項再配分全体の失敗としたりしない。

後続でもprimaryは維持する。必要なら、同次数・同補正におけるoriginal/response/knownの対照差をsecondaryとして事前定義し、primaryの結果と混ぜずに報告する。

### 6.4 Richardsonの現会計は妥当な一案であり最適性証明ではない

S2のaccountは外挿weightsのL1を使い、各実験の統計幅へ均等な十分配分を与えている。二実験の費用とrepeatを含んでいる点は適切だが、この配分が最小費用とは限らない。[S2]

費用がA_i/ε_iと近似でき、Σ|w_i|ε_i≤ε_statと置ける場合には、既知の凸配分によりε_i∝sqrt(A_i/|w_i|)を使える。これは標準的な解析的比較であり、新しい外挿法ではない。

現primaryをそのまま引き継ぐ場合はv1配分を保存し、強い実用優位をclaimする前に、同じPF点から得られる合理的な精度配分に頑健か確認する。追加のChebyshev node群や全外挿methodの無制限比較を今回の必須条件にはしない。

### 6.5 情報取得ラベルと実費用

元raw S2/S4はγもd0も要求しない。元CS2はd0が必要。responseはγ+d0が必要。既知線形抽出rawはRDM不要で、補正版は自分のleadingを直接取得できる。[G2, S2]

公平な比較は同じ情報へアクセスできることを許すが、不要な取得費用を対照へ強制しない。current small-hでは元rawも最良価格で適格であるため、responseの追加情報費用を0に置いてもcomponent利益はなかった。この限定結果を保持する。

### 6.6 合成予算と実T費用を混同しない

v1の`uniform_synthesis_angle_allowance`は予算表示であり、native/T合成を実行した値ではない。energy、unitary error、各rotation errorの換算や制御位相精度を閉じた完全なQPE/RPE保証ではない。[G1, G2, S2]

これは現conditional proxyの比較を無効にはしないが、10%のproxy gainをそのまま実T削減へ格上げすることを防ぐ。大hでproxy gainが残ってから、必要なprimitiveへ絞って実装費用を取得する順序が適切である。

## 7. 先行研究と新規性の位置付け

今回の外部確認は、広い新候補探索ではなく、現在の判断に直接必要な一次文献の限定確認である。

- P1（FFF）は一体項を二体fragmentから移し、Hamiltonianの期待値を保ちながら測定varianceを変える自由度を扱う。再配分自体の発明をclaimしない。
- P2（Hamiltonian partitioning assessment）は量子化学PFの分解と回路・誤差を比較する既存研究である。本レビューでは書誌・概要を再確認し、詳細式の解釈は前回レビューの取得内容を継承する。
- P3（2025年Digital Discovery出版版）は、対称二次PFの主要固有値誤差を状態期待値で評価し、partition選択とQPE資源へ結び付ける。主要誤差予測の目的自体に新規性を置かない。出版版であって旧first-order preprintと同一視しない。
- P4（2026年compact BCH v1）はgrouped symmetric BCH、再帰、期待値評価を扱う。愚直な全交換子展開だけを比較相手にしない。

P3は、演算子normに基づく誤差指標と実際の固有値誤差が同じ順位を与えないことを報告している。ただし、この一般的な所見を「今回のH4は拡張すれば必ず勝つ」という根拠にしない。

今回の数値結果は、同情報・同恒等式のmatching compact BCHとの生成計算量同等性を覆さない。また、そのmatchingが存在することは閉式の既報性を確定しない。歴史的先行性は未確定のまま保持する。

今後の研究貢献の候補は、局所応答の情報次数を明示した再分配生成法と、その有限stepでの有用条件である。しかし「主要係数が下がった一件」だけで独立論文の主張が十分とは認定しない。強い補正対照、回路費用、適用条件、追加入力での再現性が必要になる。後続の数値結果が肯定的でも、新規性認定は別の判断である。

## 8. 代替方針の比較

| 方針 | 得られる情報／問題 | 判断 |
|---|---|---|
| v1の10%未達だけで全面終了 | capped h窓でのno-gainを、許容h全体のno-gainへ広げてしまう | 今は採用しない |
| 同じ小h窓・同じ候補で再実行 | 主に再現性の反復で、未取得の費用境界は分からない | 不要 |
| targetを小さく変更して現窓で勝ちを探す | 別の科学的要求へ変更することになり、登録targetの問いを残す | 今回の次作業にしない |
| selector・方向・first fragment・分子を追加探索 | 原因切り分けよりも救済探索になり、何が効いたか分かりにくい | 行わない |
| 小残差を全点で超高精度化する | 高次係数の記録には役立ち得るが、資源境界が遠い問題は残る | 主目的にしない |
| 全compile・T合成へ進む | 物理的step削減が未確認のまま詳細費用へ投資する | 行わない |
| **現在の分解を固定し、valid domain内でh境界を判別する** | 強い対照も含め、誤差低下を費用へ交換できるかを直接調べられる | **準備を推奨** |
| 第二候補DF等価回転へ移る | 今の候補より有望という新情報はなく、別の構成検討が必要 | 保留 |

限定継続は、現候補の成功を前提としない。既存36次元の保存入力・再配分を再利用し、未取得の一つの境界へ作業を絞れるため、情報価値と追加負担の関係が比較的明確である。具体的な実行資源はCodexが見積り、実行前に固定する。本レビューは所要時間や新call数を約束しない。

## 9. 追加検証で固定すべき科学的条件

以下は**新しい限定検証を準備するための提案**であり、v1の契約の無断変更ではない。v1のsource・結果・認可・STOP・10%未達は不変とする。

### 9.1 変更しないもの

- 同じ保存H4 pack、Hamiltonian hash、DF rank12、fragment順、scalar・normal-order規約、sector、厳密入力状態。
- 同じresponse-selected shift X。v1で選ばれた値を継承し、finite-h結果を見て再選択しない。
- original / response / knownの3構成、S2/S4/CS2/二点Richardsonという既存の主要比較。
- εE=10^-4 / 10^-5 Ha、P1/P2、primary 10%改善と二精度×二価格のuniform判定。
- 元の理想情報・static primitive/Heisenberg proxyという評価レベル。
- block-encoding/qubitization除外、Track A/B非移管。

主要な変更は、h評価領域と、そこに必要なbranch・数値精度・探索の契約である。新しいbudget配分や強対照最適化を追加する場合は、v1の比較と区別できるsecondaryとして事前定義し、成功後の指標変更にしない。

### 9.2 hの探索計画

単純に「前回hの2倍を追加し、未達ならまたレビュー」と繰り返さない。最初から有限の範囲・計算上限・分岐を一つのplanにまとめる。

- 現小hの保存点をanchorにする。
- 全方法が利用可能な共通の拡張領域と、必要なmethod別validity条件を定める。
- coarse探索と必要箇所の局所細分化を結果前に定める。物理的誤差が単調であることを無条件に仮定しない。
- branch/alias/numerical上限に達した場合の扱いを定める。
- 最良点が再び領域上端へ張り付いたら、その最小費用はwindow内の値であり、global optimumと表示しない。
- 10%効果の判別に対してgridの粗さが大きすぎないよう、費用比の識別と必要なh分解能を結び付ける。
- 十分な判別ができない場合も、新しいhを無断追加せず、未判定理由を保存して停止する。

この探索は性能評価側のoracleであり、真値なしの運用precision selectorが完成した証拠にはしない。h選択のoracleは全対照に同じ権利で与える。候補Xの生成は既にsealされたまま維持する。

### 9.3 branch・有限stepの安全性

hを大きくしても各指数はunitaryとして定義できるが、principal log、BCH局所展開、ground-connected phaseの解釈は別条件である。

共通の既知energy shift、係数由来の安全なphase範囲、または明示的なbranch continuation等から、どの範囲を比較するかを固定する。必要な参照診断を使う場合は、運用情報と独立監査情報を分ける。新しい真E0を候補選択へ入れない。

S4のsigned stagesに対する十分条件がS2と異なり得ること、near-degenerate cluster、overlap急変、phase-wrapの曖昧性を扱う。曖昧点を都合よく低biasのbranchへ接続しない。

この点の技術的実装・許容差・独立checkerはCodexがまとめて行う。元の小h専用処理を、そのまま正当化なしに大hへ流用しない。

### 9.4 数値精度とフィット

小hの数値床からc4を推定して後続hを保証しない。新しいhの直接PF評価と独立数値照合を使い、必要な箇所だけ事前に定めた精度を与える。

数値精度を段階的に上げる必要があるなら、それは最初から明示した一回の実行内分岐として、call・メモリ・出力上限に含める。未登録のretryや失敗後のprecision rescueにしない。

外向き区間証明を新しい絶対条件にはしない。引き続き経験的数値allowanceの範囲で評価する場合は、その性質と結論の限界を明示する。数値不確かさと価格モデル不確かさを同じ区間に入れたと見なさない。

## 10. 費用・情報の継続判断条件

### 10.1 component利益

候補Rと強い対照Bについて、同じモデルで適格なhを使った費用差

ΔC_component = C_B − C_R

を評価する。正なら、その値は追加情報費用に許される余地の一部となり得る。負なら、そのcomponent比較から追加情報取得を正当化できない。

ただし異なる資源を勝手に足し合わせない。量子queries・primitive vector・情報コピー数・古典時間を別に示し、換算する場合は価格を固定する。元raw対照には不要なγやd0の費用を強制しない。

### 10.2 hが大きいほど係数情報の要求精度は厳しくなり得る

CS2の補正係数dの誤差がη_dなら、追加energy誤差はh²η_dである。情報予算ε_infoに対する必要条件は

η_d ≤ ε_info / h²

となる。

これは標準的な誤差伝播であり、新測定法ではない。h拡張でquery数が減っても、補正情報の精度を上げる費用が増える可能性がある。機構検証ではideal dを使ってよいが、運用上の利益を主張するときはこの費用を戻す。

responseのγ誤差は、g/Q・選択X・leading予測へ影響する。今回は厳密状態由来のideal情報による凍結Xの検証を維持し、γ測定法開発を開始しない。必要precisionと許容取得費用を別建てで記録できればよい。

### 10.3 compileへ進む入口

finite-h / conditional proxyで強い対照を上回る兆候が残る場合に限り、次の科学判断で限定的なprimitive合成・compileの価値を検討する。今回の準備・実行承認へ自動的にcompileを含めない。

同じclassの複雑なGaussian/T角度依存で結果が変わり得ることは認めるが、「未知だから全compileが必要」とはしない。どの価格成分の変化が結論を変えるかを特定してから取得する。

## 11. 次のCodex作業

一つの依頼に、次をまとめることを推奨する。

1. 本レビューを原文保存し、v1の有効な10%未達と、h境界未評価を区別した研究判断を追補する。旧結果を書き換えない。
2. 現入力・selector seal・保存shift・leading値・frame・既存sourceの再利用可能性を確認する。準備中に新しいregistered expectationやPF性能を算出しない。
3. 許容h領域、branch/alias処理、数値精度、coarse/refinement規則、uniform効果判定、未判定・STOP条件を明示する。
4. 既存の強対照、情報要求の違い、S4/EXT2を含むprimaryとsecondaryの役割を固定する。
5. runner/checker・人工fixture・source/input/環境identity・resource caps・execution targetを準備し、GitHub公開・再取得確認を行う。
6. 準備完了後Mandatory STOPし、固定planへの新しい明示実行承認を待つ。

細かなvalidation、実装、効率化、call配分、source改訂は、この科学方針を変えない範囲でCodexに委ねる。sourceが更新されたという理由だけで、同じ詳細GPTレビューを何度も要求しない。

科学的な比較対象や真値アクセスの意味を実質的に変更する必要が出た場合は、その論点だけを明示して止める。別の分子や第二候補の探索を技術作業として紛れ込ませない。

準備後の数値実行は、古い36-cell one-shot認可を使わず、別の明示承認で一回だけ行う。事前固定された探索分岐を含む一回と、失敗後の無断retryは区別する。

## 12. 継続・終了・未判定の分岐

| 後続の結果 | 判断の方向 |
|---|---|
| raw S2間では改善するが、CS2/S4/EXT2等の強い対照に対する実質的利益が残らない | 現S2再配分候補の低費用研究は区切る有力な根拠となる。rawだけへbaselineを弱めない |
| 十分判別できる領域でprimaryの10%uniform改善がない | 現scopeの非達成として確定する。新しい分子・directionへ自動救済しない |
| 一部target・価格だけ利益がある | uniform成功とは呼ばない。条件依存の所見として保存し、後から主条件を都合よく変更しない |
| 条件付きprimitive利益はあるが、情報費用の許容域が小さい | 理想機構の成果と運用利益を分け、次に取得する情報の価値を科学判断する |
| 強い対照に対し意味のある差が残る | 一件developmentでの限定成果。独立入力への移送・実費・新規性の評価は別段階 |
| branch/数値/上限により境界へ届かない | 未判定理由を保存。一般不能性とせず、同じ探索の無期限延長もしない |

最初の不適格点以後に別の適格領域があり得る場合、検証したdomainの結果と、全hに対する最適性を明確に区別する。運用上使えるbranch領域を事前に定義したうえで、その範囲で判断する。

## 13. 今回支持するclaimと支持しないclaim

| Claim | 判定・根拠 |
|---|---|
| 今回の再配分が固定H4でleadingを低下させた | 支持。保存値・独立直接収縮照合 |
| 小hのraw S2固有位相biasが低下した | 支持。最大hの保存recordと独立JW/Padé照合 |
| v1の4条件で登録10%component改善なし | 支持。固定窓・価格・oracle h条件 |
| 再配分後のoverlapが実用上改善した | 未支持。差が丸め水準 |
| CS2/S4/EXT2の高次残差に確定的順位が付く | 未支持。数値許容幅以下 |
| 主要誤差低下が費用削減へ交換できない | 一般には未判定。v1は許容h境界へ未到達 |
| static比約1.0004–1.0007が実T・hardware比 | 未支持。compile/T合成なし |
| 最終QPE/RPEの総費用利益 | 未判定。固定Heisenberg proxyと情報・合成費用の限界 |
| 1-RDM応答生成の独占的計算量優位 | 従来どおり主張しない |
| 閉式・特殊化は既報で研究価値なし | 今回からは結論しない。先行性は別問題 |
| 厳密基底状態の仮定があるのでPF overlapは必ず1 | 誤り。元HとPFの固有状態は一般に異なる |
| v1が否定的なので第二候補へ自動移行する | 採用しない |

## 14. 論文化の着地点と研究資源配分

現時点の成果だけでは、「既存法より低費用な新しい量子化学算法を提案した」という主要claimを支えるには不足している。leading低下は既存の摂動的誤差最適化と近く、今回の登録費用は未達である。

限定追試が肯定的なら、低情報rankの局所応答、凍結した生成則、補正済み強対照を含むfinite-h費用の関係を一体として示す方向が候補になる。しかし一件の成功では一般性・独立性・総資源の頑健性を認定しない。

否定的なら、正確な局所応答や主要係数低減を保ちながら、強い補正PFを戻すと利益が消えるという限定的な知見として整理できる。ただし、その否定結果だけで独立論文が十分と保証しない。

重要なのは、同じ方法差監査と小hの準備を繰り返すのではなく、現在の候補に残る科学的な問いを判別できる設計へ一度まとめることである。次にまた上端だけを少し広げるためのレビューを要求する計画にはしない。

## 15. 承認と保全

本レビューは研究判断と記録の作成であり、数値実行承認ではない。

- v1のsource、入力、selector、結果、認可、STOPを保持。
- 既存global manifestのdescription欠落FAILを無断修正しない。
- N1 STOP、N2 HOLD、N3 CLOSED、THRIFT現template CLOSEDを保持。
- C1–C4を自動再開しない。
- Track A/Bのsource、契約、認可、結果を変更しない。
- DF等価回転へ自動移行しない。
- 新しい科学計算には新source・scope・capsに対する別途の明示実行承認を必要とする。
- 技術準備後はMandatory STOP。通常の準備完了だけで同じ詳細レビューを再要求せず、新しい科学的結果・契約変更の節目でGPTへ戻す。

## 16. 参考資料・リンク

### GitHub正本

- G1 結果：[結果報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/docs/research/qc_onebody_repartition_finite_step_pilot_v1_results_20261015.md)
- G2 契約：[contract.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/contract.json)
- G3 引継ぎ：[GPT handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/docs/research/qc_onebody_repartition_finite_step_pilot_v1_gpt_handoff_20261015.md)
- G4 source identity：[source_manifest.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/source_manifest.json)
- G5 独立監査：[independent_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/independent_audit.json)
- G6 実行監査：[execution_audit.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/execution_audit.json)
- G7 一次JSON：[raw.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/raw.json)
- G8 公開監査：[receipt.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/artifacts/qc_onebody_repartition_finite_step_execution_v1/2026-10-15/publication/receipt.json)
- G9 前回レビュー：[scientific review](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ea58449eb8e535b8f5d5f62956fb410d35f0ae/docs/reviews/quantum_chemistry_pf/qc_onebody_repartition_final_audit_scientific_review_2026-10-14.md)
- S1：[pilot source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/19d5bd74db020564ea9a4b1882e102911e983cdb/src/trottertracks/ground_pf_repartition/qc_onebody_repartition_pilot_v1.py)
- S2：[resources source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/19d5bd74db020564ea9a4b1882e102911e983cdb/src/trottertracks/ground_pf_repartition/qc_onebody_repartition_resources_v1.py)
- S3：[pipeline source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/19d5bd74db020564ea9a4b1882e102911e983cdb/src/trottertracks/ground_pf_repartition/qc_onebody_repartition_pipeline_v1.py)

### 外部一次文献

- P1: Choi, Loaiza, Izmaylov, *Fluid fermionic fragments for optimizing quantum measurements of electronic Hamiltonians in the variational quantum eigensolver*, Quantum 7, 889 (2023). [出版社](https://quantum-journal.org/papers/q-2023-01-03-889/)。今回は出版社ページの概要・書誌を確認。Trotterへの結論部の詳細は前回レビューの取得内容を継承した。
- P2: Martínez-Martínez, Yen, Izmaylov, *Assessment of various Hamiltonian partitionings for the electronic structure problem on a quantum computer using the Trotter approximation*. [arXiv:2210.10189](https://arxiv.org/abs/2210.10189)。今回は概要・書誌を確認し、全本文の再監査はしていない。
- P3: Mehendale et al., *Estimating Trotter approximation errors to optimize Hamiltonian partitioning for lower eigenvalue errors*, Digital Discovery 4, 3540–3551 (2025), DOI 10.1039/D5DD00185D. [出版社](https://doi.org/10.1039/d5dd00185d)。検索応答に掲載された出版社本文のII・III・Appendix F等を参照。直接openは一経路でInternal Errorとなったが、検索から得た出版本文は利用できた。図表の画像やPDFを今回新たに解析していない。
- P4: Maxwell et al., *Practical Estimation of Trotter Error for Hamiltonian Simulation*, arXiv:2606.30738v1, 2026-06-29. [HTML](https://arxiv.org/html/2606.30738v1)。III.1–III.2の関連箇所を確認。全定理の独立再証明、最新softwareのbenchmark、閉式の初出性証明は行っていない。

以上の文献と保存結果を根拠にした限定科学レビューであり、文献の網羅的調査や、sourceの全機能に対する独立認証ではない。
