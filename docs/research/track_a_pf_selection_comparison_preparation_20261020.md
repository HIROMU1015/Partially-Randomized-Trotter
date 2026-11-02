# qモデル使用時と直接実測時のPF選択比較：技術準備

2026-10-20。基点`7430a386bcd4dc545107e413cd022ab7cb261968`。
今回の承認は**準備・人工テスト・構造検証のみ**。保存shotと単価を掛ける処理、最小費用の計算、実候補の再ランキング、PR/PF費用比や順位逆転の判定は実行していない。

## 比較対象と固定snapshot

GitHubの指定基点と公開資料をread-onlyで確認した。公開directoryのGit blob/sizeと固定local Git bytes、および全文取得した資料を照合した62件を[GitHub検証receipt](../../artifacts/resource_applicability/track_a_pf_selection_comparison_preparation/2026-10-20/github_verification.json)に記録した。gzipの中身は同一Git blobのlocal全bytesを展開して読み、公開資料の代わりにprivate会計結果を取り込んでいない。

| 系列 | 元の公開会計snapshot | PF比較集合 | PR対照 |
| --- | --- | --- | --- |
| H6 tol-only DF19 | V1保存再会計、SOURCE `2261a3ad1d6c23521474b31204572a6c3a51258b` | old32 / fine S2 96 / S2+truncated S4 136 | 確認union8、各n32の保存費用・SE・保存選択 |
| H8 tol-only DF28 | six-axis置換後、SOURCE `aa984d98a7b438dc953a829a24fe3d7edc9d0a6b` | old32 / strengthened184 | 確認union6、各n32の保存費用・SE・保存選択 |

どちらもlinear 1 Å、STO-3G、T=.8、P=0、symmetric_directional。S2/S4、B0/B0Iおよびfull PF B1、登録済みq=1/2/4/8を保持し、prefixと候補を新しく選別しない。指標は状態準備を除くfull Hadamard wrapperのcompiled RZに、元のcosine/sine shot数を掛けた総RZ。旧shot政策のα_axis=.025、経験的allowance、倍率1/10を保全する。

各系列はbias2政策（通常signedとempirical noncancellation）×ε=.05/.01/.005/.001/.0005×allowance倍率1/10の20条件を持つ。H6は3scopeなので60件、H8は2scopeなので40件、計100件のscope別条件。system/bias/ε/factorだけなら40条件であり、scopeを重ねて独立な100条件と数えない。

元の「モデル使用側」は、**保存されたモデル単価と既存実測単価の混在政策**である。全PFを純粋なqモデルへ揃え直す反実仮想は今回設計の実行範囲へ含めない。元の混在結果を再現し、対応するモデル単価だけを実測へ置換する前後比較である。

## 来歴・shot対応

[packet](../../artifacts/resource_applicability/track_a_pf_selection_comparison_preparation/2026-10-20/README.md)には保存会計6,680行（H6 PF2,720＋確認PR160、H8 PF3,680＋確認PR120）を転記した。探索PR n8は除外し、独立確認Pのn48/n104とも混合しない。

`saved_condition_rows.json/csv`は候補/PF/prefix/q/軸、bias、ε、倍率、適格性、保存shot、元の総RZ・PR SE、cohort、SOURCE、入力file SHA/locatorを記録する。H6のshotは元accountingのreal/imagからcosine/sineへ対応。H8通常biasは元six-axis input manifestに固定されたD0、B0I会計、Phase Aのprecision行へ照合し、noncancellationは保存accountingへ照合した。適格行の保存shotは両出典で一致することを確認した。shotの再計算・総和からの復元はしていない。不適格/未解決のnullを保持する。

`axis_price_lineage.json/csv`は320候補×2軸＝640 slot。実際に保存適格行で単価が存在するのは560 slotで、残80 slotの単価はnullのまま。この80を新しいモデル評価や0補完で埋めない。保存適格候補に必要な価格が揃うかを条件別に検査するため、全slotの値が揃うことと、ある条件の比較入力が揃うことを区別した。

| 単価slotの扱い | H6 | H8 | 計 |
| --- | ---: | ---: | ---: |
| 新しい450軸と対応するモデル単価 | 192 | 258 | 450 |
| 450軸外だが過去の公開実測が存在 | 4 | 4 | 8 |
| 元から実測で、そのまま保持 | 52 | 50 | 102 |
| 保存適格行に単価がない | 24 | 56 | 80 |

8軸はH6 `B0I_p7_q1_2nd`、`B0I_p11_q4_4th`、H8 `B0_p15_q8_2nd`、`B0I_p16_q4_4th`の両軸。過去に取得・公開済みの結果をIDと保存予測文字列で照合した。新規コンパイルではない。各result file SHAと元SOURCEを保持し、8軸を前回450軸診断の結果に混ぜたり、今回新たに誤差解析したりしていない。

旧profile照合待ち48軸（H6 32/H8 16）は元から保存比較に含まれる旧実測の来歴としてflagを保持するだけで、再取得・profile同一性の認定は行わない。これを含む全登録集合比較は条件付きであり、全軸が同一profileで取得された比較とは呼ばない。

## 比較案と入力準備状況

`execution_contract_DRAFT.json`は未認可・未seal。`grant_INVALID.json`はapproved=false、allowed_calls=[]で、有効grant・launch command・本番runnerはない。

| 案 | 準備済みのscope別条件 | 未解決/単価不足 | 適格候補なし | 解釈 |
| --- | ---: | ---: | ---: | --- |
| NEW450_ONLY | 38 | 62（適格性未解決30、単価不足32） | 0 | 元の集合のまま450軸だけ置換 |
| ALL_SAVED458 | 70 | 30（適格性未解決） | 0 | 元の集合のまま450＋既存8軸を置換 |
| MATCHED_NEW450_SUBSET | 72 | 0 | 28 | 同じ限定subsetで基準/置換の両側を再選択する案 |

実行するなら推奨は**ALL_SAVED458の入力準備済み70件**を独立した承認範囲にすること。うちH6は通常signedの30件、H8は両biasの40件。scope重複を除いた準備済み条件はH6 10＋H8 20＝30。無条件にprofile/runtimeが一致した比較可能条件は0で、70件もすべて条件付き診断になる。

H6 noncancellationの残30件では、元のB0I p5/p10/p15、q1/2/4/8の12候補の適格性が未解決である。単価を置換しても解決しないため、全domainの最小値・費用比・順位はnull/未解決とする。12候補を黙って落として全domain最良と呼ばない。

限定subsetはH6 old32では0候補、fine S2では59候補、S2+S4では96候補、H8 old32では4候補、strengthened184では129候補。72件の入力が揃うが、H6 noncancellationも含めて**限定集合の結果**にすぎない。基準側も同じsubsetで選び直し、保存full-grid winnerを片側だけ使わない。28件の適格候補なしは欠測/該当なしとし、費用0・比0にしない。

## 将来の数値解析手順

別途明示承認後、次の順に新しいone-shot契約をsealする。今回この手順を実データへ実行していない。

1. 本packetのcommit、使用strategy/condition ID、candidate-domain、入力SHA、技術SOURCE、算術runtimeを固定する。新grant/ledger/marker/STOPを用い、旧科学承認を流用しない。
2. 元のPF費用を`G=N_cos*RZ_cos+N_sin*RZ_sin`で有理数として再現し、保存`G_exact`と完全一致することを確認。不一致は該当conditionをSTOPし、自動修正・shot再計算・retryをしない。
3. 同じ保存shot・bias適格性・候補集合を保ち、model-priceの対応軸だけを保存実測へ置換する。旧実測はそのまま、PR費用/SE/cohort/選択もそのままコピーする。
4. 元scopeごとに基準/置換の最良PF・q・prefix・PF次数・総RZを計算。有理数費用→候補ID辞書順の既存tie規則を使い、全exact tieも保存する。
5. 保存PR費用/最良PF費用の比と、PR_LOWER/PF_LOWER/TIEの三状態を両variantで比較する。best候補変更、q変更、PF/prefix変更、全候補順位の厳密な逆転、tieの出入りを分ける。PRを別cohortや新候補に置き換えない。
6. CSV/JSON/図/報告書、baseline replay照合、別の算術表現による検算、条件coverageと出典/SHAを保存してmandatory STOPする。

`output_schema.json`と空のCSV template3個に候補別総費用、PF選択/PR比の前後比較、順位逆転のfieldを用意した。既存の保存winner・総費用・比は`baseline_saved_comparisons.json`へ元行ごと転記しただけであり、今回の新しい結果ではない。

人工prototypeは`kind=ARTIFICIAL`のみ受け付け、実データは拒否する。本番`--run`も入力を読む前に拒否する。新18検査で人工のshot重み付け・q/候補変更・順位逆転・PR/PF逆転・exact tie・入力不変・欠測拒否、および実保存表の一意性/shot文字列/flag/入力coverageを確認した。実データでのprototype実行、weighted cost、minimum、rankingのcallは0。

処理は入力を一度indexし、元会計行/単価/条件のID照合を共有する。将来も同一candidate/endpointの会計をscopeごとに重複実行しない。小規模な標準ライブラリ処理に独自CPU/RAM/worker上限を設けず、共有設定・他ユーザーの環境やprocessを変更しない。

## 残る制約と承認事項

H6のprofile差、H8等のruntime同一性未証明、S4合成モデル仮説、初回572軸のrun全体最終gate欠測を継承する。今回対応する新450軸のうち428軸が初回SOURCEであり、後続の最終gateを代用しない。bias/shotの経験的allowanceを厳密証明へ変換せず、有限PR標本の点比較を母平均優位と主張しない。

最新の2026-10-18 D_ACCOUNTING公開projectionは保存120比較の要約に限定され、全shot/会計原本はprivateと明記されている。今回の契約はその要約から不足数値を推定せず、上記の公開会計snapshotを基準とする。最新Dや論文主表の独立P4条件そのものを再評価する契約ではない。そこまで同一比較に広げる場合は、該当cohort/shot/単価の来歴を追加で固定する準備と、別の承認が必要。

残338軸の追加モデル予測（未保存78軸を含む）、H2旧signal20軸の修正、旧48軸の再取得、新sampling/compile/refit、PR候補や比較集合の拡張、主表/Figure 2改訂、詳細科学レビュー、最終evidence freezeは対象外。単価誤差2%未満という前回の事実だけから、最良PFやPR/PF順位が不変と結論しない。

次担当**ユーザー**が、strategy（推奨ALL_SAVED458）、条件範囲（入力準備済み70件/unique30）、上記の欠測/条件付き政策を確認し、**保存shot込み費用の再会計・PF再選択・PR/PF比/順位比較**を明示承認する必要がある。限定subset案を実行したい場合も別に指定する。新しい公開範囲は未承認で、今回は専用branchのlocal commitのみ。次担当**Codex**は新たな承認範囲だけを本番接続・sealして実行する。現時点は**mandatory STOP**。
