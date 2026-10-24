# 第2研究C0：PF次数・q・partitionの資源効率予測と構成選択

2026-10-24。研究識別子 **R2-CS-20261024**。C0設計資料であり、科学仮説・新規性・独立論文化は未採択。
第1研究、歴史的PR-2、既存Track B Algorithm Codesignとは別の研究単位である。

## 現段階の判断材料

研究対象は、構成候補ごとの全signal・compileを終える前に、利用可能な情報と評価予算から、低い総費用Gを持つ適格候補を選べるか、である。量子実行費用と選択のための古典費用を別に評価する。

**有望さを検査する最初の問いは、次数やqだけの普遍則ではなく、情報取得量を明示した選択が固定構成より役立つか、である。** 構成長は事前に得られるが、shot数を支配する候補別bias・headroomは既存pipelineではsignal/reference評価後に得られる。構造proxyが単価を説明しても、総費用の良い選択になるとは限らない。

一方、既存R3は極めて近い問題を調べ、一般multi-fidelity法・partition最適化との差分不足で **STOP_R3_NO_METHOD_DELTA** となっている。本C0はそのSTOPを解除しない。「低費用候補を少数評価で選ぶ」「abstainを使う」だけでは独立新規性が成立しない。[再利用・否定的履歴](reuse_index.md)、[一次文献との比較](prior_art_delta.md)

## 調査で確定した境界

- PF588はFULL/B0/B0Iの決定論的候補集合であり、588個のPR候補ではない。対応するPR探索・独立確認は別データである。B0/B0Iのprefix gridをPRのpartition sweepへ読み替えない。
- H4 Full Yoshida4/q2には、主比較に未採用だった適格な保存候補がある。固定S2/q8対Y4/q4の方向逆転は、q最適化後の最適次数逆転を意味しない。今回その候補の新費用比較は行っていない。
- PRの対応系列は各系のprefix1/3/5、K4、ABS_TIME_PER_OUTER32が固定。次数・qに関する開発用比較は可能だが、PRのprefix/K/lawを同時最適化する教師データとしては不足する。
- normalizationは固定lambda・時間・cutoff・allocationからsamplingなしに求め得る。必要なlambdaの取得にはDF対角化等が含まれ、無料の情報ではない。保存signal由来のlog_Bをそのまま安価な特徴量と認定しない。
- bias、empirical allowance、N、C、G、既存最良順位は、それぞれ取得後の情報または評価専用Oracleである。保存されていることと、事前選択で利用できることは別である。

fieldごとの取得経路と条件は [information_tiers.csv](information_tiers.csv) に固定した。候補選択用viewと評価用labelを分離し、派生特徴量も祖先情報の制約を引き継ぐ。

## 仮説・実装資産

候補H1は構成長とheadroomの競合、H2はFull/PRで異なるnormalization・RTE・partitionの寄与、H3は少数の有料適格性照会を含む選択である。H1/H3の限定的な開発検査を優先する。H2のPR partition一般則・移送性は現在の保存域では検査できない。[仮説と反証](hypotheses.md)

既存decompose、u_aware_shots、headroom、schedule/allocation、exact weighted/choose、qモデル診断は再利用候補である。ただしdecomposeは事後説明、Backend.accountはH2精度固定・結果書込を伴うため、事前予測器や汎用入口には使わない。科学処理の再実装は不要で、必要なのは情報境界・取得記録・評価interfaceである。[コード/SOURCE索引](reuse_index.md)

## 最小の次期検証案

Option 0の構造整理は本C0で実施済み。次に数値実行する価値をGPT・ユーザーが認める場合だけ、**Option 1のP1：H4/H6 Full、S2/Y4、q1/2/4/8、epsilon=.001の16保存endpoint** を推奨する。

固定S2/q8、固定Y4/q4、構成長だけの選択、q昇順の有料適格性照会を使う既知型規則を、同一8候補域のOracleと比べる。G・regret・不適格/保留・照会数を返し、全候補を見た選択を安価な予測にしない。これは新アルゴリズムの試験ではなく、安価な情報だけの限界と追加情報の価値を確認する開発pilotである。[評価protocol](evaluation_protocol.md)、[Option比較・採否条件](options_and_pilot.md)、[非実行契約案](pilot_contract_DRAFT.json)

新held-outは自動的な次段階にしない。P1で利益がない、一般手法との差分がない、または安価な安全性判断にほぼ全signalが要るなら縮小・停止する。利益があってもH4/H6は既参照の開発条件であり、予測成功の独立検証ではない。

## 技術確認・保全・引渡し

科学runner、新費用/比/regret/順位計算、実データfit、sampling、compile、simulation、有効grant、詳細科学レビューは実行していない。今回の追加はこの研究ディレクトリだけで、第1研究のCEM・paper manifest・原稿・主表・Figure 2・SOURCE/結果/grant/ledger/STOPは変更していない。[技術検査](technical_checks.json)、[取得版記録・STOP](STOP.json)

SOURCE30件、field役割87件、PF索引と主比較CSVの全fieldを確認。人工・metadata/保全32検査がPASS。検査はprimary agent本人によるもので、独立科学レビューではない。再確認はrepository rootで python3 -B docs/research/r2_configuration_selection_c0_20261024/technical/check_c0.py を実行する。検査は新科学演算を含まない。宣言された特徴の祖先を検査するinterfaceであり、任意の外部proxyの独立性を自動証明するものではない。

正式共通指示の提示SHA原本とLevel1レビュー本文は未取得。手元の共通指示は別SHA・§10.3なし。今回明示された§10.3相当の原則をユーザー指示として適用し、読了やレビュー採択を捏造しない。

**次担当：GPT・ユーザー**がH1/H3の工学的pilotに価値があるか、R3との差分を研究目標として残せるか、P1を採択するか判断する。P1は未実装・未封印・未承認で、実行前にCodexが独立SOURCE/inputと入出力を固定し、ユーザーがその数値再解析だけを明示承認する。**CodexはC0完了後mandatory STOP。**
