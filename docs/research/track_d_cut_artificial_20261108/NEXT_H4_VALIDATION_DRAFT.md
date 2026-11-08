# 候補Bの最小実H4移送案（未採択・未実行）

人工2caseのうち事前固定された不均衡例で追加depth減を観測したため、近傍cut proxyを実H4へ移送する価値はある。ただし、これは既知SE-QPE型Referenceの分割位置heuristicであり、独立した新機構・一般的資源優位は未認定。候補Cの実装やqモデル/全RPE会計へ進まない。

## 対象案と出所

Track DのLinear H4／隣接1.00 Å／STO-3G／Nalpha=Nbeta=2／元H／number-CDF8／元frame順L_D2／S2/s1/r32/K0/h=.1を保持。元H ground利用を仮定し準備費は除外。O2 legacy DF12/L_D3/.02/r1/K2とは異なる。新sampling・state/Born評価は不要。

基点は今回の公開SOURCEと、C6aの保存q4 event列・固定SOURCE `6b6667026ec8b670f610e41141d195883a9eb513` の組合せを候補とする。旧資料/入力hash・event payload・raw QPYの可用性は別途read-only preflightで確認し、正式採択時に一つの完全なinput manifestへ固定する。C7-4 descriptor関数は固定 `64072ba8` の同じ版を用いる。元cohortは既に探索に使われているため、未使用独立holdoutや全lawの確認実験とは称さない。

最初は**q4の事前指定された1 path**、中央2とdescriptor選択cut（合法候補1/2/3）だけの技術移送案とする。ordinalは価格を見て選ばず、採択時に保存順の固定規則で決定する。今回、実pathのrun数・proxy・価格を計数していない。実際のfile/path/call上限は科学実行を採択する時点で固定する。

## 再利用と追加処理の有限枠

1. 現人工専用adapterのcohort記述を、科学入力に対して明示的にbindingした専用SOURCEへ接続する。選択規則・負荷係数・近傍範囲は変更しない。
2. 保存event順序・符号・角度・scalar/identity・law・qubit order・P0/phase compilerを照合する。compiled価格を読む前にdecisionを保存。
3. Strong/中央の旧結果は、**同一input/event順序、canonical IR、全宣言phase、builder/最適化、P0、postcheck精度**が一致し、旧PASSとQPYを検証できる場合だけ継承。SHAだけ/同qだけで再利用しない。
4. 新しい選択cutは同じ `prepare_steps/physical_segment/reference_gadget` で最大1 bundleを構築し、既存C3/C5/B6b全入力・paired/χ checkerへ接続。中央を返した場合は追加効果0を保存し、別cutを探さない。
5. 選択が中央と異なる場合の新規P0 wrapperは通常X/Yの2本。中央/Strongも再利用不適格なら、その2方式×X/Yの4本を含めて**最大6本を提案上限**とする。Source互換/取得不能ならSTOPで、黙って予算を拡張しない。raw構築数、reuse判定call、postcheck数も正式契約で固定する。

測定結果・stateを必要としない全入力operator/observable検査を優先する。保存H4の実行入口では、既存bit/精度/actual inverse/χ/qubit mapping/真空条件を緩和しない。新SOURCEと旧SOURCE・結果・STOPを分離し、同一IRの重複compileは避ける。

## 棄却条件とSTOP

- 選択cutのcorrectness不成立、phase/identity/真空条件不一致、旧source不整合、資源非干渉不成立ならSTOP。
- 正しく移送できても中央を選ぶ/compiled depthが減らない場合は、その1 pathで追加効果なし。規則の事後調整や他pathへの探索拡大はしない。
- Depth改善があってもcount/width増が相殺する場合は利害を分ける。期待価格改善・全RPE・旧Nの資格には結び付けない。
- 実H4の32 event/stepと、今回の人工16 event/stepでは構造・Gaussian幅・closure・CSWAP費が異なる。人工削減率を移植しない。

新しい実H4 SOURCE/input、対象1 path、有限build/compile/postcheck枠への明示許可が必要。q64別担当の未完了結果を選択器の最適化資料やholdoutへ使用しない。今回はこの提案のみでmandatory STOP。
