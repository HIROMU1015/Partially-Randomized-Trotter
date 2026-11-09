# 次段階の最小案：未採択・未実行

[今回の結論](REPORT.md)。native emitterと1Q橋は再利用可能になり、固定54人工caseでは全資源が保存P0と一致した。技術的継続には価値があるが、全P0を再実装する計画へ拡大しない。以下はユーザー／GPTの判断材料であり、実行承認ではない。

## 第一候補：保存H4の限定照合を一回

ユーザーが保存H4再解析を明示許可する場合だけ、既存C6a `q4_sample00`／中央cut2、C1 head-tail追加融合OFF、Strong／Reference・X/Yの既存raw4／既存compiled4を原本hashで固定する。固定条件はREPORT記載の元H／number-CDF8／LD2／S2/s1/r32/K0／h=.1、元H厳密ground仮定・準備費除外、保存P0 Qiskit1.3／opt1／seed17／rz,sx,x,cx／topology-free／位相保持を維持する。

新sampling／回路生成／compile／ground solve／fit／N／全RPE／q64以上は含めず、保存raw→native→partial bridgeの7資源を既存P0と一度比較する。新dense CARや全入力simulationを追加しない。SOURCE／入力・wire・phase conventionを先に固定し、推論不能なgate、hash不一致、unsupported、価格差は隠さず記録する。作用保証を追加する必要が生じたら、そのcheckerと許可範囲を別途具体化する。未知の作用をPASS扱いしない。

不一致があっても、原本や保存P0を変更したり、価格を合わせるため角／cut／rule／閾値を事後調整したりしない。対象を増やさず残差domainを報告し、公開照合後STOP。通常の技術bug修正・再実行の可否も次の指示に明記する。

## 必要になった場合だけ：最小人工P0判別

第一候補の残差またはSOURCE調査で、既存54caseが識別できない具体的なtranslation／pass-order差が見つかった場合、最大5qubitの一つの固定native wordと対照をPLANに列挙して新P0の必要call数を限定する。今回のP0 callは0であり、この案を先行実行しない。未知角を多数走査するcalibrationやP0全体の複製へ拡張しない。

## 方針の選択

| 担当 | 次の判断／作業 | 条件 |
|---|---|---|
| ユーザー／GPT | 保存H4限定照合を一回行うか決める | 対象・SOURCE・再解析範囲・STOPを明示許可 |
| Codex | 許可された場合だけ上記限定照合、時間／peak記憶の実測、残差報告 | 新P0／H4 build／長q追加へ自動進行しない |
| ユーザー／GPT | 部分構造モデル＋既存A/F/S、またはPR全体共同最適化へ重点を移すか判断 | 一般P0閉包に広範compiler再実装が必要なら、構造評価器を技術資産として保全 |

現時点の提案は「native／部分bridgeは残し、H4の有限移送性だけを低コストで確認する」。その結果が悪ければP0全互換の追究を一区切りとし、既存A/F/Sを条件付きモデルとして使う案を優先する。A/F/S自体の変更、PR条件の共同最適化、新しい科学的GOは今回行っていない。
