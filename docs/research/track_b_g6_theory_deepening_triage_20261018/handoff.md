# ユーザー・GPTへの引き継ぎ

2026-10-18 JST。**最優先は、既存の有限bit保証を実効的な精度・全処理費用の契約まで追えるか（C1）。次に、一般次数で追加return集約が強いpartial/closed対照後に何を保証できるか、その成立域・限界（C2）を提案する。** 肯定結果だけでなく、全てroutineに閉じるという結果、取得費用で利点が消える条件、比較命題の否定も判断を前進させる。

## 今回特定した区別

既存の必要bit数の多項式的取得可能性と、自動H/K selector未実装は別である。未実装を未証明の存在問題にしない。C1は、root・丸め・相対誤差・support・angle・fail-closedを実際の呼出条件と全費用へ接続する不足を閉じる問題に限定する。

G7は試行中にも精度不足で例外になり得る。確率的rejectのNoneと例外は異なり、例外をzeroに読み替える既存契約はない。x=0の数学的identity別扱いと、G7にidentity返却分岐がないことも区別する。これは今回見つけた定理の反例ではない。[proof_and_contract_risks.md](proof_and_contract_risks.md)

三次のnormalizer順序、m=1/L=1の特殊例は既存。C2はその再証明ではなく、一般次数でU、finite-bit N、event費用のどの比較が成立するかを分ける。P5にはGreenを使わない既存closed生成があるため、それを除く比較は採らない。moment、量子前zero、期待量子回数、hard cap、全rejectの古典費用を保持する。

## 既存資料で解消できるもの／新研究が必要なもの

| 種類 | 対象と今回の位置付け |
|---|---|
| 既存記述の確認で足りる | L=1,m=1,x=1、挿入上界が等式でないこと、valid parent/child、operator mean/channel、phase/order、SOURCEの固定α、exact provider仮定 |
| 通常の技術仕様化の候補 | 既存lower bound→root/rounding検査、APIごとのdomain/精度不足、setup/selector/reject/angleの費用scope。新しいselectorを実装することとは別 |
| 新しい理論が必要となり得る | C1で既存lemmaだけでは閉じない全処理保証、C2の一般次数の比較・成立限界、保証を保った既知法への非自明な還元。今回新証明なし |
| 別の応用研究 | DF入力誤差・G6 adapter・実測interface・strict native費用。既存Rの共役Z/ZZとprimitiveは確認済みだが、全接続は未確立 |

## 優先順位と最初の一件

[中心文書](candidate_priority_decision.md)は8候補を比較し、優先2件へ絞った。C1を先にする理由は、C2や既知法比較で未取得precision/bit/angle費用を無料にしないため。容易に実装できるからではない。C1が既存lemmaのroutineな適用で閉じた場合は、仕様完成として止め、新規理論を増やしたとは扱わない。

次の最小作業案は[W1](minimal_next_work_proposals.md)：既存lemma→APIの充足条件→費用項の差分表を作り、各穴を直接仕様化／新証明／実装／物理取得へ分類する。H/K selector、angle計算、科学test、samplerは作らない。

C2は、Uと対照normalizer、十分N、event依存費用という3つの問いを分けたW2へ限定する。上界がわずかに改善するだけ、弱いbaselineでしか差がない、未知量が必要という場合は投資を止める。普遍的資源改善を構成成果の必要条件にはしない。

## 担当・承認と未取得資料

- **ユーザー**：C1→C2という技術的優先提案を採るか、次のW1静的調査を指示するかを判断する。今回の選別は研究方針の正式採択ではない。
- **Codex**：明示指示後、W1の静的仕様調査は通常の技術作業として進められる。重要科学レビューの繰返しを形式的な前提にしない。ただし今回のmandatory STOP後には自動続行しない。
- **GPT**：新しい比較命題・非自明保証へ投資する範囲の判断が必要な場合、その限定課題を扱う。新たな重要科学レビューを開催するなら、共通運用§4のLevel 1／GPT-6 Pro推奨に該当する中心的判断として、別の開始承認後に行う。前回の構成価値判断全体を再開催しない。
- **ユーザー／Codex**：新証明・SOURCE実装・科学test・sampling・保存値再解析・回路/compile/合成には、対象に応じた別の明示許可が必要。新しい科学実行contract/grantは今回発行していない。

指定された直近GPTレビュー原本はrepositoryと添付領域で見つからず、`NOT_RETRIEVED`。ユーザー提示の「構成的価値を評価済み」という研究状態は保持するが、原本固有の推論・条件は未確認である。既存公開2026-10-10レビュー[U]の§7はUの出典としてのみ読み、直近原本の代替にしない。私的レビューは取得・転載・公開していない。

## 公開・保全と終了

作業基点は5f17a754。旧Packet、非Pauli branch、既存SOURCE/results/grant/ledger/marker/claim/STOPを変更していない。出典と実読範囲は[source_inventory.json](source_inventory.json)、gap・候補は[evidence_index.json](evidence_index.json)。今回の科学実行・SOURCE変更・科学testは0件。検査は文書・JSON・リンク・Git・hashのみ。

公開照合は[publication_receipt.json](publication_receipt.json)、文書検査と保全は[verification.json](verification.json)。receiptは内容commitを記録し、それを追加する最終commitはpush後に別途全公開bytesを照合する。保全範囲は親tracked/直下untrackedのbytes、登録済み他worktreeのHEAD/index/lock。非登録環境や他worktree全untrackedの監視まで主張しない。

**公開・remote・保全確認後mandatory STOP。** 選別した問題の証明・実装・数値計算・SOURCE変更・新レビュー・論文執筆・他branch統合は開始しない。

<!-- FIXED_REFERENCES -->

[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
