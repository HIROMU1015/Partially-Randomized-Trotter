# 未実行：H4 Full PF S2・Δ=0.1のH_eff最小検証案

[PLAN.json](PLAN.json)が候補・SOURCE・全round identity・数値条件のmetadata正本。[REPORT](REPORT.md)。**今回許可されているのは技術準備まで。本案の実科学実行には別途明示承認が必要。** 既存2f0のSCIENCE_PLANを具体化し、従来既定0であった元H Hermitian対角化を、本案では明示的に1回提案する。

## 固定候補と入力

Linear H4/1Å/STO-3G、保存tol-only DF10、sector(2,2)/36次元、元weighted-Hermitian completion・signed weights・fragment順を維持する。Full PF、prefix10、S2、Δ=.1、h=.1/M12/P0。binary64由来のexact h/Δは`3602879701896397/36028797018963968`で、canonical行のT/qを使う。κ/R/Kは対象外。

保存bindingのH=`c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`、state=`9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`、snapshot=`12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`。保存input indexは`7a8bc7f132e166390c943e6b32c92cc6abe1baf4`の`SOURCE_INDEX.json`、loaderはC86f37を使う。receipt/contextの検査を実行時に維持する。state生成・SCF・DF fitは行わない。

targetは**同sectorの元保存full DF-H**。DF近似前の分子Hamiltonianの厳密groundへの誤差を保証する案ではない。declared referenceは元SOURCEの保存solver proxyを保持し、新しい`eigh`の最小値で置換しない。元Hの対角化は有限行列内の比較energy/固有basis取得であり、保存stateを生成し直す処理ではない。厳密ground資格はfalseを保持する。

## 承認対象の科学scope

| 処理 | 上限・内容 |
|---|---|
| 保存context | 既存loader 1回。SOURCE/hash/sector/H/state/scalarを確認 |
| PF macro-step | direct expmのS2 step Sを1個、U=e^(−iΔc)Sを1個の対象として構成。cは元ham.constant、tail identity追加なし |
| effective spectrum | 既存Schur経路1回、principal H_eff/全固有energy・vector・state weights・数値診断1件 |
| 元full DF-H | 同sector Hermitian `eigh` **1回**、全energy・vector・元state weight・残差・overlapを取得 |
| 直接PF signal | m0/T.1/q1、m12/T409.6/q4096の2件。既存S^q state actionにe^(−iTc)を一回適用 |
| 全spectrum再構成 | 同二時点のΣ_j w_j exp(−i E_eff,j T)を計2 complex出力、直接PFと照合 |
| 新full-H reference action／保存mean・reference再解析 | **0／0**。保存energy/metadataのSOURCE付き転記のみ |
| normalizer・N・MC・RMSE・価格・G・幅探索 | 全て0 |

このscope内の通常の再構成・norm/残差計算は既存評価器を利用する。library内部の個別演算を別候補callとして数えない。旧PR全13roundやC baseline全体の計算、保存PF結果の受入診断を含めない。最小callerが必要なら本案の実行時に既存APIを接続するだけとする。固定数値SOURCEはこの準備成果のcommitと上記凍結kernelを記録し、fresh output directoryへ保存する。

## 順序・scalar・共通step

Cの`Context.pf_step`は元matricesの先頭one-body＋全10blockを元順に使用し、S2のsigned piece `[1]`を既存`expm_step`で構成する。step自体にはglobal constantが含まれないため、H_eff入力Uにはe^(−iΔc)を一回含める。直接signalは元Cの規則どおりS^qにe^(−iTc)を一回掛ける。**Uを既にq乗した結果へ再度total scalarを掛けない。** Full PFのblock中のidentityを新たに抽出して除去・加算しない。

同H/sector/fragment順/formula/実幅/scalar/recipeなら、全roundでU_m=U^(2^m)となる。本案のmetadataはこの条件を満たすが、C/Bのcanonical IDは変更しない。一般のceil-grid、初期例外、prefixや幅変更を同じstepとみなす処理は行わない。今回13個のstepを生成し直す必要はない。

## 分枝・状態対応・数値検査

- `PRINCIPAL`は初回の明示log規約。scalar込みの固有位相からenergyを返し、reference energyやRPE priorでunwrapしない。cut clearance、選択branch整数、再構成残差を保存する。Δ‖H‖<πの診断だけでPFの物理branchを証明しない。Δ=0からのcontinuationや別windowの探索はscope外。
- 元HとH_eff双方の全energy・保存state weight、元/effective固有basis overlap、近接energy groupを報告する。元最低groupと各effective groupの射影trace overlapを記録する。縮退内の個別vector対応はbasis依存、groupも指定resolution内の数値診断である。最小値同士を自動対応させず、重なり閾値を新設してground資格を付けない。
- δE_min=E_eff,min−E_H,min、proxyとの差、必要な対応energy差を区別する。対応が曖昧なら`matched_effective_group=null`と理由を保持する。全spectral measureによるsignal照合は一成分近似やRPEのbranch正解を意味しない。
- complex128。state norm1e−12、step/power unitarity・再構成・Gram・signal絶対差1e−9、Hermiticity/元H固有残差1e−9 Ha。energy cluster1e−8 Ha、branch guard1e−8 radは既存評価器の数値resolutionを明示転用する。RMSE予算や厳密誤差上界ではない。
- 失敗値・中間診断を保存し、非有限、input/SOURCE/scalar不一致、数値gate不成立、cut/分枝対応未解決でSTOP。threshold変更・state再正規化・unitary投影・自動retry・別Δ/別分枝へのfallbackはしない。

## 完了の意味と後続

数値再構成と二時点signalが一致しても、δEをRPEの厳密bias又はその上界と認定しない。physical alias、保存stateの元ground同定、effective stateへの支持、他spectral成分、全decoder/shot/lossとの接続は別の資格である。診断取得と未資格理由を併記して完了できるが、分枝や数値gateが不成立なら部分結果としてSTOPする。

結果はAがBへ全spectral/状態情報とSOURCEを渡す。Bがraw PF信号を直接使うriskには同じδEを追加しない。PF誤差を分離したMinkowski評価を使う場合は、その比較対象・state/decoder条件・残余成分の誤差契約を別途確認する。

公開・remote照合後mandatory STOP。Truncated、PR exact-tail、他Δ・他公式・全RPE統計への拡張は本案の承認に含めない。
