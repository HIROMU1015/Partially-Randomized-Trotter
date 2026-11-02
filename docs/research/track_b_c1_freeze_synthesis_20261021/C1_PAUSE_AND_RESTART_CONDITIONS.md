# C1一時凍結と将来の再開判断条件

状態：**`C1_PAUSED_AFTER_EXPLORATORY_RESOURCE_ASSESSMENT`**。

これは今回ユーザー・GPTが採択した今後の研究進行方針を記録するもの。C1を恒久終了・放棄せず、G6の数学的・方法論的研究は維持する。[機械可読の方針記録](research_state.json)。新しい科学的価値判定・論文採否をCodexが行った結果ではない。

## 凍結する範囲と理由

C1の追加科学計算、追加native保証整備、大規模資源比較を一時停止する。新分子・時間・次数探索、十分N取得、shots、native synthesis、全round RPE、追加Pilot/Full-N、新しい証明・samplerも自動開始しない。

採択判断を支える既存事実は、3方式完走後も現在条件でlogical差が小さく、zeroによる回路回避を観測せず、degree0 parentが大半であったこと、query/normalizer等の古典費用が無視できず、strict native/全task保証も未取得であること。[固定値・引用範囲](C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)、[既存比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/comparison.md)。これをRA一般の資源劣位・G6数学的無効性・原著価値の不存在へ一般化しない。

保持するものはR1入力/basis/label packet、adapterと選択correctness、whole-path技術接続、探索的3方式観測、過去の失敗と修正の証拠。G6の既存構成/保証・条件付きnative会計は[既存v1.1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md)の範囲で維持する。成果は将来の補助実験・limitations・supplement候補に使えるが、候補の原稿採用は別判断。

この「凍結」は研究方針上の停止であり、他worktreeの通常作業を禁止したり、共有サーバー・ファイル権限・job設定を凍結したりする操作ではない。既存のgrant/claim/ledger/authorization/marker/STOPは一切書き換えない。新grant・attempt marker・実行SOURCEは作らない。

## 再開判断条件の候補（未採択・未実行）

|判断材料|誰が確認・採否を決めるか|今回の境界|
|---|---|---|
| 非空parent/非自明returnが増える具体的機構・対象を、必要な情報アクセスとともに説明できる | ユーザー・GPT | 新候補の探索・計算・機構の新証明は行っていない。degree0にも回転はある |
| quantum resource削減の可能性と、追加証拠がどの研究判断を変えるかを示せる | ユーザー・GPT | 新改善率・順位・十分N・native価格を作らない |
| classical setup/query/selector/cache/Evidence費用を抑える実装上の見通しがある | ユーザー・GPTが必要性を判断、承認後Codexが技術作業 | 既存P5R最適化を新手法・最適性能と扱わない |
| 強いbaselineにも同等の前処理/cache/selector/basis reuse/相殺機会を与え、target・精度・費用境界を揃えられる | ユーザー・GPT | 新しい比較契約は今回採択していない |
| 目的に必要なmodel/native/state/readout/全trial保証の到達層と、不足証拠を明示できる | ユーザー・GPT | 既存保証の条件充足と新定理必要性を混同しない。全不足を新証明不足とはしない |
| 新作業の範囲とSOURCE/input/environment/停止条件を固定し、必要な承認を得る | ユーザーが明示承認、Codexがその範囲だけ実施 | 研究再開、詳細レビュー開始、証明/実装/科学計算の実行承認は各範囲で別 |

これらは再開を検討するための候補で、満たしたと判断して自動実行するgateではない。再開対象が決まっても旧one-shotを再利用・再発行せず、必要なら新しい独立した契約と承認を用意する。

## 次の担当

ユーザー・GPT：G6論文の中心claimと補助C1節の採否を別途判断する。重要な詳細科学レビューを始める場合は開始範囲の明示承認を得る。G6本体の維持は今回の新証明・実装・実行承認を意味しない。

Codex：今回の新namespace文書/provenanceを公開・照合してmandatory STOP。正式改稿・再開準備・追加検証は新しい指示が来るまで行わない。
