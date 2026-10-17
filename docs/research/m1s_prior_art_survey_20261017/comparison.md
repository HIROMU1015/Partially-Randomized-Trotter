# M1-Sとの技術比較表

2026-10-17 JST。比較対象の定義差を記録する。非同値性の証明や新規性の採否判定ではない。
版・完全書誌は[文献一覧](bibliography.md)、数式の所在は[一次証拠](primary_evidence.md)。

## 1. Branch・乱択・観測対象

| 文献／方式 | 乱数を共有する対象・law | 正負branch／積順序 | 平均／測定対象 | M1-Sと共通する点 | 同一視できない点・未照合 |
|---|---|---|---|---|---|
| M1-S固定定義 | 同位置Pauli label共有、位置・step間IID | 両枝同順、gate sign reversal | $\langle\psi\vert\Psi^s(I)\vert\psi\rangle$ | 比較対象そのもの | fixed $r$、exact D、involution-tailという限定 |
| Campbell qDRIFT | IID Hamiltonian項、重み比例 | controlled HはApp.D。M1-Sのjoint lawの指定なし | 主に平均channel、controlled QPEも扱う | stochastic exponential、controlled実装 | system diamond boundだけをshared first momentへ転記しない |
| COS randomized PF | 一つのpermutationを対称積の前後で共有 | Eq.(6)で同じ符号の前後逆順積 | mean operatorを介したchannel bound | 回文形と順序相関 | IID replacementとpermutationの周辺分布・角度が違う |
| MLMC-qDRIFT | fine列とcoarse奇数subsequenceを共有 | 同じ時間方向、異なる深さ・角度 | observable差、coherent／augmented測定 | quantum branchesに共有labelを載せる | opposite-sign干渉ではない。shot-noise対策は別の非unitary構成 |
| SCU／CTS | CTSの左右propagatorは独立sample | Fig.1のV1／V2、cross term | $\mathrm{Tr}[Oe^{i\phi}V_1\rho V_2^\dagger]$ | 枝間cross termの確率的取得 | shared joint lawを持つとの主張ではない |
| PR partial RTE | 正負RTEを独立sample | directional deterministic control、App.A46 | normalized coherent signal＋RPE | partial分割、± control、RPE、資源評価 | RTE mean/normalizerとqDRIFT bias、branch共有の有無 |
| SYK/TETRIS | UとU′は独立 | controlled二枝、mirror-on-average | evolved observable、noise診断 | stochastic二枝干渉 | exact mirror、平均mirror、M1-Sを区別 |
| Proctor mirror | 元回路情報・乱択Pauli等を使用 | inverse／quasi-inverseを逆順配置 | known outputのbenchmark | 同じ回路情報の再利用 | energy相対位相を推定する構成ではない |
| Kim–García-Pintos | 挿入時間uと1−uをsymmetrize | 各step内で相関、step間IID u | mixed-unitary channel | D/Rに似たsplitと相関した対称構成 | label共有ではなく連続挿入位置。access modelも別 |
| PRHS／qSHIFT | step内path係数／履歴依存quasiprobability | 複数time slice間の相関 | channel／observableのorder matching | 相関乱択と誤差・shot交換 | 非負IID同位置共有lawと異なり、weight二乗会計が要る |
| Abbott coherent control | 実装・環境・相対位相まで指定 | two-channel interference | control-target channelのcross block | system channelだけでcoherenceが決まらない | M1-S固有rateやRPEの定理ではない |
| Wallman–Emerson RC | twirlingと補償gateを関連付ける | target理想回路を保つrandom dressing | noisy実装の平均channel | 相関したrandom gatesを使う | Hamiltonian項のsampling biasを改善するM1-Sとは目的が違う |

## 2. 理論・保証の型

| 資料 | 仮定と保証対象 | M1-Sへの対応候補 | 未完の照合 |
|---|---|---|---|
| MF Eqs.(16),(26),(35) | arbitrary Hermitian terms、IID rescaled exponentials。mean operator trace-normとHaar信号biasの非漸近bound | HS生成子の分散とmean信号解析に近接 | 固定eigenstate・有限減衰secular参照・共鳴floorを含む直接系か |
| Chen Algorithm0.1/Thms.1–2 | 独立random productの単一実現concentration。全状態／固定状態を区別 | 共有wordを一blockとみる適用候補 | word内の再使用labelを2N独立gateと置くことはできない |
| COS mixing lemma | 各unitaryの距離a、meanの距離bからdiamond bound $a^2+2b$ | ancilla込み回路に対する適用の検討候補 | 関連誤差が同じa,bと価格で閉じるか |
| Gu Thm.1/SI | Hermitian Kraus noise、unitary channelの固有operator phaseが先導次数で不変 | 散逸とphaseの摂動分離、左右mode | isolation distanceはphase差。M1-Sのpair-sum gap／HS normへの対応 |
| YC出版Lemma3/Thm.2 | deterministic unitary、low-dimensional support・gap・off-diagonal leading perturbation | 位相とeigenstate leakageの分離、outer PF bias | dissipative meanと具体三次係数・uniform secular boundへの対応 |
| KGW Thm.1 | Hamiltonian列の最小ground gapと隣接差、位相誤差・失敗率 | randomized evolutionをphase estimationへ接続する先行例 | 固定Hの状態・pair-sum・平均写像と同じ仮定ではない |
| Li Thm.4.1 | register QPE、random unitary近似、residual/gapと成功確率 | random simulationのQPE error budget | single-ancilla multi-round M1-S decoderへの移送 |
| T1a/T1b既存原本 | fixed finite dimension/r、小step、有限減衰予算、HS normとexact pair-sum pinching | 今回の照合対象 | 次元／gap／r同時変更、general DF-native、guiding state、全資源 |
| Davies 1974 | abstractは弱結合でのmemory消失とMarkovian極限 | secular平均化との背景的接点 | 全文未取得。特定定理との対応・直接系という認定はしない |

## 3. Shot・費用・推定task

| 既存手法 | 改善するもの／課金するもの | M1-Sとの共通点 | 比較で保持すべき差 |
|---|---|---|---|
| WBC random statistical PE | LCU normalization、sample数、controlled rotation数を交換 | 短い回路と多数測定を同時に最適化 | CDF threshold推定であり無補正M1-S Task Aそのものではない |
| PR | partial RTE、guiding overlap、RPE RMS、det/random gate費用 | 最も直接的なphase-estimation資源baseline | RTE normalizationはqDRIFT phase distortionと違う。費用単位を合わせる |
| Kanasugiほか | STAR論理rotation、qDRIFT/RTEの減衰、partial RPE費用 | visibilityとshot・費用の接続 | 同論文のγはrandom cost係数。M1-Sの許容位相γと別記号 |
| KISS importance-sampled qDRIFT | rescaled angle、bias・concentration、sample/experiment/time-step | 項別実装価格をsamplingに入れる | lawを変える方式。重みだけのproposal変更と同じでない |
| Cugini–Atif–Subaşı IS | 固定ensemble、$q\propto p/\sqrt c$、費用×second-moment proxy | 同じ精度での総費用比較 | bias不変のreweighting。shared lawを変えたM1-Sのphase biasを消さない |
| qFLO | 複数step-sizeのRichardson外挿と測定反復 | finite-step biasと測定数を交換 | 線形observableの外挿。Argの直接外挿・同じRPE保証ではない |
| qSWIFT | 高次channel補正とone-ancilla測定 | sampling・biasの高次改善 | gate scalingだけから同じconfidenceでの総shot費用を推論しない |
| MLMC | path差varianceとlevel別costの配分 | 共通乱数の資源価値 | coherent共有のみではshot noise floorが残る。augmented implementation費用は別 |
| M1-S γ V1 | $M=A^-\sin(\gamma-b^+)$に基づく同じ十分shot設計 | visibility対phaseの局所feasibility | bound比較、ceil後の整数shots、whole-wrapper単価、全RPEを分離 |

共通の情報access、物理Hamiltonian、state、時間、精度、confidence、control位相、
normalizer、状態準備、古典生成・補正費用を揃えなければ、上のcomplexityを横並びの勝敗表にできない。
本調査では新しいshot数・費用比・ランキングを作成していない。
