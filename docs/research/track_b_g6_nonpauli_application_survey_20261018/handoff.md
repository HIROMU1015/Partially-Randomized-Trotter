# G6 Evidence Packetへ渡す技術情報

2026-10-18 JST。独立branch/worktreeで作成した技術調査。並行中の構成論文Packetには書き込まず、自動統合しない。

**G6の一般Hermitian involution条件に対応する量子化学の具体例は、既存DF乱択tailの符号付き`B_l Z_p B_l†`と同一基底`B_l Z_pZ_q B_l†`で確認できる。係数・basis操作・controlled primitiveの既存SOURCEもある。ただしG6局所packetと現行PR/RTEの接続、有限精度とnative費用の統合認証、資源改善は未確立である。**

## 1. 具体的に渡せる材料

1. [演算子棚卸し](candidate_operator_inventory.md)：number projectorとreflection、同一basis可換parity、異基底積、DF全blockを区別した。
2. [係数と接続](pr_rte_connection_and_access.md)：既存`exact_df_diagonal_coefficients`のI/Z/ZZ係数、`Q=sgn(c)T`、`p=|c|/Λ_R`、定数shiftとidentity policyを対応させた。C01/C02は現在Rに存在し、新D/Rを要しない。
3. [controlled条件](controlled_native_feasibility.md)：既存の無制御basis＋中央CZ/controlled-RZ/RZZ、signed angle、parent phase、control相対位相を特定した。G6の新しい回路は作っていない。
4. [最重要の多軸表](application_feasibility_matrix.md)：C01/C02はINCOMPLETE、DからRへの移送C07はNEW_CHOICE、projector等の不適合と未確認pairは別statusとした。
5. [既知対照](pauli_and_known_method_baselines.md)：元PR v2 §VII.B p.14は異なる軌道基底のPauli involutionを明示する。応用クラス自体をG6独自の一般化とは書けない。G6固有のreturn集約の意味は既存数学監査へ接続する。

構成論文の応用節へ提示できるのは、**既存DF表現の成分にG6の仮定が対応する、という条件付きの具体例とinterface上の不足**である。実装済みの分子G6、native性能比較、独立新規性、投稿GOの証拠としては使えない。

## 2. 決定的に不足する証拠

| 不足 | 現在の状態 |
|---|---|
| 分子入力identityとexact rational p/xのbinding | 今回の分子固有bundleは未取得。SOURCE上のfloat係数経路だけでG6の有理入力・元H誤差を認証しない |
| G6 packet→ordered event→測定 | 実装・既存統合証拠は確認できない。ordinary法のlawをG6のlawと読み替えない |
| finite-bitと全trial estimator | zero-fillの分母、実proposal/weight、range、biasを現行measurementへ接続する証拠がない |
| strict controlled/native | primitive SOURCEはある。G6 word依存角度、basis/control位相、全event誤差・価格は未認証 |
| 同条件の既知法対照 | ordinary DF-native、Pauli-aware CTS/SCU、合法なGaussian共有に同じ取得・精度・compiler機会を与えた比較がない |
| 資源改善 | 古典取得・normalizer・拒否・shot・状態準備・全wrapperまでの実証なし。新しい率・順位は作成していない |

thin orbital列が未整備であることと、既存full-basis primitiveの存在は両立する。前者の不足を後者の不可能性へ読み替えない。旧Gaussian/DF correctionのclosureを解除しない。

## 3. 次の担当・最小作業・承認

| 担当 | 提案する次の最小作業 | 承認・境界 |
|---|---|---|
| ユーザー | 両方の独立Packetが完了した後、本調査を構成論文の条件付き応用例へ統合するか判断する | 今回は統合未承認。mandatory STOP後の別指示による |
| Codex | 統合の指示があれば、今回の固定リンク・多軸表・不足条件を指定された文書へ反映する | 文書統合の別指示が必要。新SOURCE・計算には拡張しない |
| Codex | 既存の分子入力bindingやG6接続文書が他の公開固定commitにあると指定された場合、その資料だけread-onlyで追加照合する | 今回の完了後には別指示が必要。未公開係数の生成は行わない |
| GPT／ユーザー | 実装や性能検証へ進める価値・scopeを判断する必要が生じた場合に判断する | 重要科学レビューは明示的な開始指示後。SOURCE実装・新計算はさらに別の明示承認が必要 |

これは次の実験や新しいalgorithmを採択した表ではない。今の資料不足を埋めるために実装・分子計算を自動開始しない。

## 4. 公開・保全・停止

作業基点は`101bf3426fcd04cc0c01c0d202d0916268765d97`。取得元commitは[source_inventory.json](source_inventory.json)に別記。28固定GitHub資料は各Git blobとremote bytes/SHA256を照合した。一次PDFの取得不能箇所は[literature_scope.json](literature_scope.json)に残した。

親worktreeの全tracked file、親直下の既存untracked file、登録済み他worktreeのHEAD/index/lockをread-onlyでsnapshot比較する。非登録の並行環境や他worktreeの全untracked bytesを監視したとの主張はしない。実際の観測範囲・差分は[verification.json](verification.json)。保全は「他の作業が停止していた」という意味ではない。

公開receiptは[publication_receipt.json](publication_receipt.json)。内容commitをanchorとし、receiptを含む最終commitの自己hashは文書内に再帰的に埋め込まない。最終remote HEADと最終全fileの照合結果は完了チャットでも報告する。

**公開・remote照合・保全確認を終えたらmandatory STOP。並行G6 Packetへのmerge、新しいSOURCE、科学計算、重要科学レビューへ進まない。**
