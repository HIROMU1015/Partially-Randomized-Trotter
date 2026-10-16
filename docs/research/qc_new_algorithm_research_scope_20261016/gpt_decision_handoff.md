# GPT向け：新アルゴリズム研究の探索範囲選定資料

2026-10-16 JST。Codexによる既存資料・一次文献の限定技術整理。**新アルゴリズムの成立、科学的優先順位、GO、論文化価値は判定していない。** この資料の作成・公開はGPT/Claude科学レビューの開始認可ではない。

## 今回確定した行政的な研究状態

ユーザー承認に基づき、H1〜H4を対象とする現探索を **PAUSED** として [新規記録](research_state_transition.md)に追記した。静的K0 COMPLETE、総合 `NO_GO_NO_INDEPENDENT_GENERATION_RULE`、H2個別 `UNDETERMINED_DATA_REQUIRED` は維持する。当該K0.5/K1未認可・未実施、新アルゴリズム正式採択なし。研究全体は終了せず、研究課題選定段階へ移る。一般的なD1/D2内no-goではない。

## 調査で増えた材料

1. [検討範囲表](research_coverage_map.md)で、既存の旧C1〜C4、N1〜N3、mixed-block、H-E1、一体再配分、M、SOS-PF、H1〜H4、Track A/B、高次PFを区別した。PF外挿・保証・代数／対称性・表示変更を広い意味で「未検討」とする根拠はなかった。
2. 過去の表現レビュー・toy結果には、低次元Lie閉包、symmetry依存可解block、coherent mixed accessの条件が明記されていた。今回の観点C1を、全く新しい探索領域として扱わない。
3. 追加対照8論文から、Cartanの時間非依存depthと前処理効率、Zeno法のblock-encoding依存、sample-based simulationが要求するρ-copies、既知のfermion encoding・直接energy測定の構成を区別した。文献の節・定理・条件は [問題領域表](open_problem_landscape.md)。
4. 旧統合表以後のTrack A H6強対照整備とV2準備、Track B M1-S限定理論契約／Checkpoint後の草稿整理を参照した。これらを未検討・未実施とせず、受理範囲を新算法一般へ拡張もしない。

**具体的な独立新回路生成規則は得られていない。** 今回の成果は、既知性と検討履歴、入力条件、検討記録の不足を区別したことにある。

## 十分な検討記録を確認できない狭い問い

| 問い | 現時点の証拠 | 未確定の意味 |
| --- | --- | --- |
| fermion encoding／exact symmetry sectorから、元Hのabsolute-phase回路・測定までの費用構造 | 基盤はJW/DF/sector中心。GSE/taperingは既知の対照として確認 | 同一条件での全費用比較の記録不足。新しいmappingや新算法を発見した意味ではない |
| ground copies／coherent preparationと逆／複数copy操作を分離したabsolute-energy情報取得 | H2/H4は一部アクセスを監査済み。fermionic shadow/joint測定は既報 | 強い既知測定対照を含む、同じ入力契約での比較が未成立。新しい測定や優位性は未提示 |

これらも既存研究の別名や実装比較に留まる可能性がある。**有望な未探索領域が確立したとは報告しない。** 代数構成・finite-step保証は既に検討した未閉鎖条件を主に含み、③既知／④前提未確認として扱う。

## GPTによるLevel 1レビューに必要な具体的判断

| 判断点 | 判断に必要な条件 | Codexが今回確定していないこと |
| --- | --- | --- |
| 新しい問いと既存系列の区別 | どの入力・出力・構成原理が、旧mixed-block、SOS、係数調整、frame、H2/H4と異なるか | 名称だけで新規性を認定しない。停止系列を再開しない |
| simulation算法としての研究範囲 | D2は絶対E0を固定したまま、測定／情報取得の改良だけを今回の目的に含めるか | 測定研究への目的変更を自動採択しない |
| 許す入力情報 | exact copies、coherent preparation/inverse、reference、ρ-copies、symmetry sectorのどれを与え、どの取得費用を計上するか | oracleを追加した比較を同条件のspeedupとしない |
| 必要な独立構成証拠 | 係数→primitive/回路の具体則、既知法への包含、前処理・ancilla・control・phase・readoutの費用 | 今回は新構成を設計していない。既知encodingの採用だけを新算法としない |
| 保証理論を独立主題にするか | 有用なfinite-step認証が入力から安く得られ、回路または情報取得費用を変えるか | A2 NOT_ADOPTEDを維持。上界改善を新生成算法と混同しない |
| 次の調査の情報価値 | 具体的な構成／適用クラス／未確認定理が提示されたとき、何を確認すれば研究選択が変わるか | ⑤の記録不足だけで新K0/K1や網羅的文献探索を開始しない |

上表は判断事項の列挙で、重要度順ではない。Codexは正式採択・優先順位・研究継続／終了・論文化価値を最終判定しない。

## 技術的な確認と、追加科学成果が必要な事項

既存文献の定理条件・版、既知構成への還元、明示input/output、phase/sector対応は、具体的な次対象が与えられれば静的に確認できる。未取得出版版・原本はそのまま未取得として扱う。今回の限定調査を反復するためだけの次段階は設定しない。

新しい数学的成果が必要になり得るのは、対象化学クラスの効率的認識・生成、同入力での非還元性と費用改善、有用なcertificate、制約されたenergy情報取得の上限／下限である。現資料はこれらの存在も不存在も証明していない。

数値が必要な条件は、実分子での構造・norm/gap/overlap、測定variance／copies、code準備・controlled回路・readout、強い対照との同精度成功確率での総費用である。ただし、**まず具体的な構成と研究問いが必要**。数値取得自体が独立アルゴリズムの欠落を埋めるわけではない。今回すべて未計算であり、今後必要なら対象と範囲を別途承認してから行う。

## 参照・取得限界・保全

旧K0の起点は固定 `0d460c54f3bf54ea46d4d8129f6702703c4bb217`。27件のrepository資料のcommit・byte数・SHA・GitHub取得または前回取得原本の再利用区分、一次文献8件の版と読取り範囲は [provenance.json](provenance.json)に記録した。索引の全branch最新性は主張しない。

Kang–Su ACM出版版のK0.5時点403と版差分未確認、旧レビューのPR新版全文の留保、完了K0のClaude `レビューoutter(3).md` 未取得は保持した。過去の要約から原本を再生成していない。

既存のCLOSED・STOP・PAUSED・HOLD・NOT_ADOPTED、SOURCE・results・authorization・manifest・旧STOPは変更しない。H1〜H4の一時停止だけを今回の追記記録で明示した。新規科学計算・保存科学数値再集計・新規実装・数値テスト・回路生成／compileは0件。原文レビューの転載・改変も行っていない。

公開対象はこのディレクトリの指定5資料のみ。GitHubの固定commitからの独立再取得・byte/SHA照合は公開物の同一性検証であり、科学的妥当性の新たなPASSではない。最終commitと照合結果は完了報告を参照する。

**公開・独立照合後 Mandatory STOP。** 新アルゴリズム採択・設計・実装、新K0/K1、科学計算、停止系列再開、GPT/Claude詳細科学レビューは開始しない。
