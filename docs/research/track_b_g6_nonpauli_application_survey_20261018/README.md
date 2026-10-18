# G6 非Pauli involution応用先・実装可能性：独立技術調査

2026-10-18 JST。ユーザーが指定したread-only調査。科学レビュー、新設計、科学実行ではない。

**量子化学への具体的な対応は既存資料にある。既存DF乱択tailの符号付き軌道回転Z・同一基底ZZはHermitian involutionであり、係数とcontrolled primitiveのSOURCEもある。一方、G6の局所packetを現行PR/RTEへ接続した実装・認証・資源改善の証拠は確認できない。**

元PR論文v2 §VII.Bの末尾、印刷p.14にも、異なる軌道基底で共役されたPauliへの乱択法の適用が明記されている。したがって、この演算子クラスやその共役回転構成自体はG6の新規性ではない。[一次文献L01](https://arxiv.org/pdf/2503.05647v2)。G6固有のreturn集約については既存監査を参照するだけとし、独立新規性・投稿GOを判定しない。

## 固定条件と証拠境界

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 作業branch：`track-b-g6-nonpauli-application-survey-20261018`。
- 作業worktree：同名の独立worktree。
- **作業基点**：`101bf3426fcd04cc0c01c0d202d0916268765d97`。他branchのmergeなし。
- **資料取得元**：ユーザー指定のG6 kernel `28cfabb1…`、監査 `bc649c59…`、Gaussian監査 `e6530c05…`、既存監査が指定するDF SOURCE `321cd8c3…`。完全SHA、path、Git blob、bytes/SHA256、remote一致は[source_inventory.json](source_inventory.json)。今回取得した28資料はすべて作業基点内の同pathともbyte一致した。基点と取得元commitを同一の役割として記録しない。
- 共通運用指示：親checkoutの`research_project_instructions_unified_2026-10-17.md`を参照した。取得元・hashとユーザーの追加解釈は[evidence_index.json](evidence_index.json)。未commitの科学資料を取り込んでいない。
- 科学的scope：既存DF抽出・PR/RTEの演算子とinterfaceの静的照合。新しい分子、geometry、basis、DF rank、`L_D`、delta窓、seed、precisionは選定していない。分子係数表の新規生成・NPZ開封・保存数値再解析を行っていない。

## 成果物

| 文書 | 内容 |
|---|---|
| [候補演算子](candidate_operator_inventory.md) | Z/ZZ、projector、DF全block、異基底積、決定論側の役割を分離 |
| [PR/RTE接続とaccess](pr_rte_connection_and_access.md) | 係数・符号・shift・正規化、D/R、G6接続の未成立部分 |
| [Controlled/native成立条件](controlled_native_feasibility.md) | 回路順序、signed angle、ancilla位相、workspace、費用と誤差 |
| [既知対照](pauli_and_known_method_baselines.md) | Pauli/CTS/SCU/DF-nativeとの同条件比較、一次本文の確認箇所 |
| **[適用可能性の一覧表](application_feasibility_matrix.md)** | 最重要成果物。多軸評価と総合分類 |
| [引き継ぎ](handoff.md) | G6 Evidence Packetへ渡せる事項・不足・次の担当 |
| [SOURCE索引](source_inventory.json) | 28件の固定GitHub資料の取得・hash・読取範囲 |
| [証拠索引](evidence_index.json) | 主張と根拠、7候補の独立status、権限と資源設定 |
| [文献scope](literature_scope.json) | 固定版、節・頁・式、取得失敗と代替資料を区別 |
| [技術検査](verification.json) | 文書・JSON・リンク・hash・変更範囲・保全検査 |
| [公開記録](publication_receipt.json) | 内容commitのremote照合。自己参照hashの限界も記録 |
| [metadata検査器](verify_metadata.py) | stdlibのみ。科学SOURCEをimport・実行しない |

## 分類と停止

`SUPPORTED`はその列で証拠があること、`NOT_ESTABLISHED`は未取得・未確認、`NOT_APPLICABLE`は記載したscopeで条件不適合が確認できること。数学・取得・SOURCE・controlled実装・現行R接続・対照・資源実証を別々に判定する。

総合分類はG6への**実際の接続**まで含める。C01/C02は`ALGEBRAICALLY_VALID_IMPLEMENTATION_INCOMPLETE`。Dの演算子をRへ移すC07は`REQUIRES_NEW_DECOMPOSITION_OR_RESEARCH_CHOICE`を優先する。単にSOURCEにZ/ZZがあるだけで`DIRECTLY_SUPPORTED_BY_EXISTING_SOURCE`にはしない。

技術調査の完了後に行うのは専用文書のcommit・push、remote bytes/SHA256照合、保全確認だけ。**その完了をもってmandatory STOP。** Track BのD＋E、現在数値探索のSTOP、Gaussian/DF correction closure、他研究の状態を変更しない。並行中G6 Evidence Packetへ自動merge・自動統合しない。重要科学レビュー、SOURCE実装、科学計算へ進まない。
